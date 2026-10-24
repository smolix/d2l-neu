# D2L Writing Guide

Scope: all prose in Basics (ch. 1–8), Advanced (ch. 9–17), and Mathematics for
Deep Learning (ch. 24–29): chapter introductions, sections, captions, code
commentary, and summaries. It replaces the abandoned `attic/style-guide.md`
and `attic/language-guide.md`. Code follows `code-guide.md`, slides
`slide-guide.md`, figures `figure-style-guide.md`, and exercises
`exercise-catalog/CATALOG_RUBRIC.md`. The companion review
`reviews/writing-review-2026-10.md` applies this guide to the current text.

The guide is built on four chapters that teach well, read side by side with
the current book:

- Gilbert Strang, *Linear Algebra and Its Applications*, 4th ed., ch. 5,
  especially §5.1 "Introduction" and §5.3 "Difference Equations and Powers
  $A^k$" (pp. 260–296).
- David MacKay, *Information Theory, Inference, and Learning Algorithms*,
  ch. 20, "An Example Inference Task: Clustering" (pp. 284–292).
- Thomas Cover and Joy Thomas, *Elements of Information Theory*, 1st ed.,
  ch. 2, "Entropy, Relative Entropy and Mutual Information" (pp. 12–49).
- Bernhard Schölkopf and Alexander Smola, *Learning with Kernels*, ch. 1,
  "A Tutorial Introduction" (pp. 1–22).

Quotations below are short excerpts from these chapters. Examples of defects
are quoted from the D2L sources with file and line as of 2026-10-23.

## 1. The Standard

D2L should be the book a student reaches for to understand deep learning, and
the book a lecturer reaches for to teach it. The test of every section is
whether a reader who has just finished it can answer four questions in their
own words:

1. **What problem** did this section solve?
2. **Why** is that problem worth solving?
3. **How** does the solution work, in one or two sentences?
4. **Where** does it fit: what came before it, what it enables, and what it
   leaves open?

The current text often fails this test in five ways:

- **Density.** Paragraphs introduce several new concepts at once, often before
  the reader knows what they are for.
- **Distant references.** Arguments rest on equations and coined terms from
  sections hundreds of pages away. The reader has to flip back or give up.
- **Language.** Sentences are overloaded and mannered, with stock phrases,
  dashes, and clever headings.
- **Missing motivation.** Mechanisms arrive before the problem they solve.
- **Length without focus.** Sections cover a field instead of teaching an idea.
  The median section runs to 3,000 prose words and the longest to almost
  10,000.

Good teaching is not long. MacKay teaches K-means, soft K-means, the failure
cases of both, and the questions they leave open in nine pages. Strang's
section on difference equations solves compound interest, Fibonacci numbers,
and a Markov chain with one idea. Clarity comes from choosing the few things
that matter and explaining each of them completely.

## 2. Lessons from Four Chapters

The four chapters share one method: a concrete problem, the simplest case
worked to the end, then the general idea, then its limits. Each also has a
habit worth copying.

### 2.1 Strang

Strang opens chapter 5 by placing it in the course: "This chapter begins the
'second half' of linear algebra. The first half was about $Ax = b$." The
reader knows at once what is new.

He motivates eigenvalues with a problem, not a definition: two coupled
differential equations with initial values 8 and 5. He reassures the reader
about prerequisites in one line ("If you can differentiate $x^n$, $\sin x$,
and $e^x$, you know enough."). Then he reduces the problem to the case the
reader can already solve: "If there were only one unknown instead of two,
that question would be easy to answer." The scalar solution $e^{at}u(0)$
suggests the guess for the vector case, and the guess turns the differential
equation into $Ax = \lambda x$. He finishes the example with actual numbers
($\lambda = -1, 2$; $c_1 = 3$, $c_2 = 1$) before stating the general recipe.

Section 5.3 does the same for recurrences. It starts with \$1000 invested at
6%: \$1338 after five years compounded yearly, \$1349 monthly, \$1349.83
daily, \$1349.87 continuously, so "the improvement over compounding every day is only
four cents." Discrete and continuous time are linked through money before
any matrix appears. Then comes a question the reader can feel: "How could we
find the 1000th Fibonacci number" without computing the 999 before it?
The answer, $u_k = A^k u_0 = S\Lambda^k S^{-1}u_0$, follows, and Strang checks
it against intuition: the formula is full of $\sqrt 5$ yet must produce
integers, so the second term must round the first to the nearest integer.

Strang also decides what to leave out, and says so. On matrices without a
full set of eigenvectors: "Certainly they have to be discussed, but we will
not allow them to take over the book."

**Take from Strang:** place the chapter in the course; motivate with an
application; solve the scalar or $2\times2$ case completely, with numbers;
generalize by keeping the form of the simple solution; check the result
against intuition; state scope decisions openly.

### 2.2 MacKay

MacKay's clustering chapter begins with people, not algorithms: "Human brains
are good at finding regularities in data." The biologist sorts the world into
"things that are brown and run away, and things that are green and don't run
away." He then gives four reasons to cluster (prediction, compression,
spotting anomalies such as ships in satellite images, and models of learning
in the brain), each with a concrete image, and states the task precisely:
group $N$ objects into $K$ clusters. He also sets aside what he will not
cover: "we won't be talking about hierarchical clustering yet."

The algorithm appears in a box with its edge cases answered ("What about
ties?"). Toy data show that it converges and that the result depends on the
initialization. Then MacKay turns critic. He asks what is ad hoc ("Where did
the distance $d$ come from? ... How do we choose $K$?"), shows two data sets
where K-means fails, and names the cause: "The K-means algorithm has no way
of representing the size or shape of a cluster." "These criticisms of K-means
motivate the 'soft K-means algorithm'," which he presents as a small change to
a known algorithm: "The update step is identical; the only difference is that
the responsibilities $r_k$ can take on values between 0 and 1." He ends by
admitting what is still unsolved: "Adding one stiffness parameter $\beta$ is
not going to make all these problems go away."

His humor stays in the margin: the "K" in K-means is "a silly name, but we are
stuck with it."

**Take from MacKay:** motivate from human-scale examples; give several
concrete reasons a problem matters; present an algorithm with its edge cases;
show it failing on a picture; let the failure motivate the next method; describe
the new method as a difference from the old one; say what remains open.

### 2.3 Cover and Thomas

Cover and Thomas introduce a dense set of definitions without ever feeling
dense, because each one follows the same rhythm. A one-sentence meaning
precedes the formula ("The relative entropy is a measure of the inefficiency
of assuming that the distribution is $q$ when the true distribution is $p$").
A convention follows with its justification ($0\log 0 = 0$ "by continuity").
Then comes a small example with numbers: a fair coin has entropy 1 bit, a
four-symbol source has entropy 1.75 bits, and 1.75 is also the expected
number of yes/no questions needed to identify the symbol. For KL divergence,
$D(p\|q) = 0.2075$ bits while $D(q\|p) = 0.1887$ bits, which settles that KL
is not symmetric.

Every derived identity is followed by one plain sentence: "Thus $X$ says as
much about $Y$ as $Y$ says about $X$." "Thus functions of the data $Y$ cannot
increase the information about $X$." They also tell the reader when nothing
new is happening: "There is nothing really new in this definition because
$(X, Y)$ can be considered to be a single vector-valued random variable."
Claims are fenced by counterexamples: conditioning on a downstream variable
cannot increase dependence in a Markov chain, but with $Z = X + Y$ for
independent fair bits it can.

The chapter's opening is direct about what kind of chapter it is: "It is
irresistible to play with their relationships and interpretations, taking
faith in their later utility." It also says how the definitions will be
judged: they will reappear as answers to questions in communication and
statistics, and "that will be the ultimate test of the value of these
definitions." Headings are nouns: "Entropy", "Joint Entropy and Conditional
Entropy", "Data Processing Inequality".

**Take from Cover and Thomas:** meaning before formula; a numerical example
after it; one plain sentence after every identity; say what is and is not
new; delimit claims with counterexamples; name sections by their subject.

### 2.4 Schölkopf and Smola

The tutorial chapter of *Learning with Kernels* states its intent in plain
terms: "Rather than going immediately into mathematical detail, we introduce
kernels informally as similarity measures." The learning problem comes first
in words (two classes of objects; assign a new object to one of them),
then as notation, with a reminder that the objects could be anything: "the
task might be to categorize sheep into two classes, in which case the patterns
$x_i$ would simply be sheep."

It says how it will proceed: "General similarity measures of this form are
rather difficult to study. Let us therefore start from a particularly simple
case, and generalize it subsequently." The simplest classifier assigns a point
to the class with the closer mean. Written with kernels, this classifier
turns out to be a Parzen-window Bayes classifier, and it already has the form
of a kernel expansion. The rest of the chapter changes only which training
points carry weight and how much, which leads to support vectors. The reader
meets the general form inside an algorithm they already understand.

After each formula the authors stop for intuition ("Let us now try to get an
intuition for why we should be minimizing the length of $w$"). They anticipate
the reader's objection (why the $+1$ in the constraint is needed) and answer
it with a two-line argument. They interpret a bound by an extreme case:
labels independent of inputs, where zero training error is still possible and
the capacity term shows that it means nothing. An analogy appears only where
it clarifies a name: a ball in a box rests against some walls and not others,
as a solution rests on its support vectors. References to later chapters
offer more depth. They are never prerequisites: "Abundant references to later
chapters will enable the interested reader to fill in the gaps at a later
stage, without losing sight of the main ideas."

**Take from Schölkopf and Smola:** state the problem in words, then in
symbols; begin with a simple method that already contains the general form;
reveal its connections; describe advanced methods by what they change;
follow formulas with intuition; anticipate objections; use references to add
depth, not to supply missing steps.

## 3. Motivation

### 3.1 Openings

Every chapter, section, and major subsection opens by answering the four
questions of §1, in this order and in plain words, before any new notation
appears.

- **Problem.** Give a concrete instance, ideally with a number. "A model that
  writes a 2,000-token answer must read its whole context 2,000 times" is a
  problem. "Inference efficiency is an important consideration" is not.
- **Why it matters.** Say what fails, costs, or becomes impossible if the
  problem is left unsolved.
- **How.** Give the idea of the solution in one or two sentences, before the
  details.
- **Where.** Connect the idea to the previous section in one sentence, and
  name what remains open.

A section opening that cannot be written this way is a sign that the section
has no single idea (§4).

### 3.2 Order of Problem and Mechanism

Never introduce a mechanism, a name, or an equation whose purpose the reader
cannot yet see. If a section needs a tool (a scan, a Lagrangian, a change of
variables), first show the obstacle that the tool removes. MacKay shows
K-means failing before introducing soft assignments; Strang asks for the
1000th Fibonacci number before diagonalizing.

### 3.3 Closings

A section ends by saying what the reader can now do and what is still
unsolved, so the next section has a reason to exist. One or two sentences.
MacKay's "Adding one stiffness parameter $\beta$ is not going to make all
these problems go away" is the model. It admits a limit and poses the next
question.

## 4. Choosing What to Teach

### 4.1 Teachable Core

A topic belongs in the main text if it passes all three tests:

1. A reader will use it, or meet it repeatedly in the rest of the book or in
   practice.
2. It can be explained from what the reader already knows, in the space
   available.
3. Understanding it changes how the reader thinks about something else.

Topics that fail the tests go to an exercise, to the chapter's Further
Reading notebook (§12.3), or nowhere. A topic does not belong just because
courses traditionally teach it, or because a recent paper uses it.

### 4.2 One Idea per Section

Each section teaches one central idea, stated in one sentence without jargon
(for example "a linear state update can be computed for all positions at
once"). Everything else in the section develops that idea, applies it, or
shows its limits. Material that serves a different idea belongs in a
different section or nowhere.

### 4.3 Families of Methods

When a field has many variants, teach the mechanism through one
representative, worked completely. Mention a relative only if it changes one
identifiable thing, and then say what it changes in one sentence. Do not list
models by name: RetNet, GLA, Longhorn, and Titans mean nothing to a reader
until each one is tied to a difference that matters. Lists of shipped
systems ("from Jamba to Kimi Linear") date quickly and teach little. They
belong in the chapter's Further Reading notebook (§12.3), if anywhere.

### 4.4 Length Budgets

| Unit | Target | Ceiling |
|---|---|---|
| Part or chapter introduction (`index.md`) | 250–400 words | 500 |
| Section opening, before the first `##` | 150–300 words | 400 |
| Section (prose, excluding code and exercises) | 2,000–3,500 words | 4,500 |
| Paragraph | 3–6 sentences | 8 sentences |
| Sentence | 12–25 words | 40 words |

These numbers are guides. A section may run longer when the length is
content: a derivation the reader needs, a worked example, an experiment that
teaches. The test is whether anything could be removed without losing a
teachable point; if so, remove it. Length must come from examples and
explanation, not from additional topics. Balance is judged per chapter as well
as per section: a chapter with three sections (Information Theory) can have
longer sections than one with eight, as long as each section still teaches one
idea.

### 4.5 Material Written for the Authors

The following are written for the authors, not the reader, and are removed:

- The history of the draft or the build: "In an early draft of this cell the
  allocator reported twice the formula" (`kv-cache.md:510`), "verified to
  work under this book's build" (`multi-gpu-practice.md:93`).
- Audits of papers: errata ("In Eq. 12 of ... This is a typographical
  error", `convergence.md:127`) and code-versus-paper comparisons. If a
  discrepancy matters to the reader, it gets one footnote.
- Narration of what each framework's stored run did ("The stored PyTorch
  run reaches the data region ... The stored JAX run makes partial
  progress", `gan.md:650`). The prose states what holds in every tab. If the
  tabs disagree, the experiment does not support a conclusion yet.
- End-of-section disclaimers about experimental scope. State the strongest
  claim the experiment supports, once, where the result is read (§9).
- References to lecture slides, course notation, or other material the
  reader does not have ("The lecture notation writes $f_\theta = -E_\theta$",
  `limits.md:156`).

## 5. Building an Explanation

The default sequence for a new idea follows the four model chapters:

1. **Concrete problem**, with a number or a picture (§3.1).
2. **Smallest case**: one variable, one token, a $2\times2$ matrix, two
   clusters. Solve it completely.
3. **A running example** carried to a number the reader can check.
4. **The general form**, obtained by keeping the shape of the small solution
   (Strang's "look for pure exponential solutions").
5. **What changed**: name precisely what the general case adds ("the update
   step is identical; the only difference is...").
6. **Limits**: a case where the method fails, shown, with its cause named.
7. **What remains open**, which motivates the next section.

### 5.1 Example: Teaching a Linear Recurrence

The current `ssm.md` reaches linear recurrences through the GRU equations,
a cited simplification of them, and a forward reference to scans
(`ssm.md:63–140`). A Strang-style treatment starts with one number:

> Consider a state $h$ that keeps a fraction $a$ of its value at every step
> and adds the new input: $h_t = a h_{t-1} + x_t$. With $a = 0.9$ and a single
> input $x_1 = 1$ followed by zeros, the state reads $1, 0.9, 0.81, 0.729,
> \ldots$: an input from $k$ steps ago survives with weight $a^k$. Unrolling
> the recurrence makes this exact,
> $$h_t = \sum_{s \le t} a^{t-s} x_s,$$
> so the state is a weighted sum of past inputs in which the weights decay
> geometrically. Two consequences follow. First, all $h_t$ can be
> computed at once, as a sum, without waiting for $h_{t-1}$. Second, $a$
> sets the memory: with $a = 0.9$ the weight halves about every seven steps,
> with $a = 0.999$ about every 693.

Only then does the vector case follow, then gates that make $a$ depend on
the input, then the scan that evaluates the sum in parallel. Each step adds
one thing to an example the reader has already computed.

## 6. Density

### 6.1 Introducing Concepts

- Introduce at most one new technical concept per paragraph.
- Define a term in words the first time it appears, in the same sentence or
  the next. Give the idea first and the name second: "We can fold all the
  inputs into one running sum. This operation is called a *scan*."
- Do not use a term before its definition, even in a roadmap.
- Do not introduce a name, an acronym, or a citation that the section does not
  use again.

### 6.2 Roadmaps

Chapter and section roadmaps tell the reader which questions will be answered,
in words the reader already knows. They do not preview terminology, results,
or numbers.

Defect (`ssm.md:21–29`):

> First we reduce the GRU down to a minimal gated cell whose update is
> *linear* in the state, which makes the whole sequence computable by a
> parallel *scan* in logarithmic depth. We then derive the linear recurrence
> from a continuous-time *state space model* (SSM): discretization provides
> step-size gates, a stable parameterization, an exact equivalence between
> recurrence and convolution, and, through the HiPPO theory, a principled
> answer to what dynamics make a fixed-size state a good memory of the past.

This passage introduces ten technical concepts in two sentences, none of which
the reader can understand yet.

Revision:

> We first simplify the update so that it is linear in the state. That one
> change lets us compute the state at every position at once, which is what
> GPUs need. We then ask how such a linear memory should forget: which
> fraction of the past should it keep at each step? Physics and control
> theory have studied this question for a century, and their answer gives
> both a good initialization and a second, equivalent way to compute the
> model, as a convolution.

### 6.3 Placement of Results

Do not report numbers from experiments, capacity laws, or comparisons before
the reader has met the experiment. The SSM chapter introduction
(`recurrent-modern/index.md:21–63`) reports "the expected squared read error
is $(n-1)/d_k$", "recall ... roughly halves by two writes per key", and
"roughly 0.92 to 1.00 across the sweep" before the reader knows what a write,
a key, or the sweep is.

## 7. References and Dependencies

A reader should be able to follow any section without opening another one.
References add depth. They must not supply missing steps.

### 7.1 Restating Facts from Elsewhere

When an argument uses a fact from elsewhere, state that fact in one sentence
or one equation, then cite it.

Defect (`ssm.md:129–131`):

> the eigenvalue analysis of :numref:`subsec_bptt-gradient-pathologies`
> applies with the eigenvalues in plain sight.

Revision:

> Because every $a_t$ lies between 0 and 1, an input from $k$ steps ago
> reaches the state scaled by a product of $k$ numbers below one. This is the
> vanishing-gradient problem of plain RNNs (:numref:`subsec_bptt-gradient-pathologies`),
> except that now we can read the decay rate directly from the gates.

### 7.2 Distance Rules

- **Within a section:** reference freely.
- **Earlier in the same chapter:** if the referenced item is more than one
  section back, restate it.
- **Another chapter:** always restate the fact needed. A bare
  `:eqref:` to another chapter is never the only carrier of an argument.
- **Forward references:** at most one per section, and only to promise that
  a question raised here will be answered ("we return to the choice of $a$ in
  :numref:`sec_mamba`"). A forward reference never justifies a present claim.
- **Equation references** (`:eqref:`) point to equations on the same page.
  To use an equation from elsewhere, display it again.

### 7.3 Coined Terms

Terms coined inside the book must not be used outside the section that
defines them as if they were standard. The current text has many: "census"
(16 uses across ch. 9 and 11, never defined in prose), "testbed" (23),
"seam", "ladder" and "rung" (38 uses, naming two different objects in ch. 12),
"crowding", "the classical spine", "the shared scoreboard". Name things by
what they are: "the character-level language model trained on *The Time
Machine* in :numref:`sec_rnn-scratch`", then "this model". If a coined
name is needed for a recurring object, define it in one sentence where it
first appears, and restate the definition in each chapter that uses it.

### 7.4 Prerequisites

A chapter introduction may say what the reader should already know, as
knowledge, not as a list of labels: "We assume you know how a GRU uses gates
to decide what to keep." Sections do not carry prerequisite lines. If a
section needs something from far away, it restates it (§7.1).

### 7.5 Running Examples and Shared Models

Shared test models and datasets are reused across chapters. `TinyLM`,
`TinyCharLM`, the language models trained on *The Time Machine*, and
Fashion-MNIST let results in different chapters be compared, and spare the
reader from learning a new model for every experiment. Three rules make the
reuse readable:

- **Define each shared model once,** in a place of its own: a short section,
  or a subsection whose heading names the model. It is not built in the
  middle of a section about something else (the `TinyLM` of ch. 9 is
  currently built inside the Adam section and used by four later sections).
- **Restate it where it is reused,** in one or two sentences: what it is,
  its size, its data, and why it suits the experiment at hand. Then cite the
  section that defines it (§7.1).
- **Call it by its name.** The code name (`TinyLM`) is a real object and may
  be used. Nicknames for it ("the testbed", "the capstone scoreboard") may
  not (§7.3).

Running *stories* are different. A question opened in one chapter and closed
in another ("closing a loop opened in Chapter 10") forces the reader to
remember it, and that reader has usually forgotten. If a later chapter
answers an earlier question, it states the question again.

### 7.6 Use of the Appendix

The Mathematics for Deep Learning appendix is optional reading for readers
who already know a topic, and a first textbook for readers who do not
(decided by Alex, 2026-10-24). The main text states each result it uses,
with the intuition, so that its argument can be followed without the
appendix. It points to the appendix for proofs and for details that would
distract from the flow of the chapter: the properties of entropy, for
instance, are derived in the appendix, and the main text states them and
points there. It never asks the reader to "recall" appendix material, and
never hands a demonstration to the appendix ("A 25-parameter network demonstrates the phenomenon, and
:numref:`subsec_mdl-quadratic-model` does exactly that",
`optimization-intro.md:259`). The Preliminaries alone point into the
appendix 34 times.

### 7.7 Repetition

Explain each fact once, in the section where it is first needed. Later
sections restate it in one sentence (§7.1); they do not derive it again.
The current text repeats itself often: the same NCCL fallback bug is told in
four sections of ch. 13, the $1/b$ variance law in four sections of ch. 9,
the inverted-dropout formula twice within `dropout.md`, and double descent
in both `generalization.md` and `generalization-deep.md`.

## 8. Mathematics

### 8.1 The Equation Protocol

For every displayed equation:

1. **Before:** say in words what the equation will express or what we are
   about to compute.
2. **The equation**, grammatically part of the sentence.
3. **After:** one plain sentence that reads the equation, in the style of
   Cover and Thomas's "Thus $X$ says as much about $Y$ as $Y$ says about $X$."
4. **When it helps:** a small numerical instance.

### 8.2 Symbols and Notation

Introduce the object in words, then its symbol. "Let $\mathbf{S}_t$ be the
state" is acceptable only after the reader knows what the state is for.

One symbol means one thing within a chapter, and core symbols keep their
meaning across the book: $\mathbf{W}$ has one layout, $\sigma$ is either an
activation or a standard deviation in a given chapter, never both. The
current text has many clashes. In ch. 15, $\tau$ is both a trajectory and a
target-network rate, and $\alpha$ both an entropy weight and a step size. In
ch. 17, $\alpha$ has three meanings and $\sigma$ four. In ch. 12, $C$ is
both the SSM readout and the chunk length. A sentence such as "not the
per-step $\alpha_t$ or the reverse-step $\sigma_t$" (`flow-matching.md:163`)
signals a clash to be fixed by renaming, as does a glossary box that lists
"four meanings of 'stable'" (`ssm.md:627`). If a chapter needs more than
about ten symbols, give a short notation table at its start.

### 8.3 Definitions

A definition has five parts: name, one-sentence meaning, formula, a small
example, and a boundary (what it is not, or when it fails). Say what is new.
If nothing is ("this is the same definition applied to a pair of
variables"), say that too.

### 8.4 Derivations

State the destination before starting. Name each step that is not routine
("substitute the update into itself", "the cross terms cancel because the
noise has mean zero"). End by saying what was obtained and why it matters.
Long algebra that teaches nothing goes to an exercise or an appendix
paragraph.

### 8.5 Intuition, Objections, Counterexamples

After a formal result, give the intuition (Schölkopf and Smola's "Let us now
try to get an intuition for why..."). If a reader would reasonably object,
raise the objection and answer it. Delimit claims with a counterexample when
one exists. Short proofs, led by a picture or an extreme case, follow the
`**Proposition.**`/`**Proof.**` convention in `CLAUDE.md`.

## 9. Code and Experiments

The code itself follows `code-guide.md`. This section covers the prose around
it.

- Before a code cell, state the question it answers in one sentence.
- After the output, say what it shows in one or two sentences, at the
  precision that survives re-execution (`CLAUDE.md`, "Experimental results in
  prose").
- One experiment per idea. If an experiment needs a paragraph of confounders
  before its result can be read, it is a research experiment, not a teaching
  one: simplify it or drop it.
- An experiment must show what the prose says it shows. If it does not (an
  MoE run with no gain, a "showcase" where both arms reach the maximum score,
  a headline A/B that holds only under one initialization), fix the
  experiment or remove it. Do not keep it and explain what it fails to show.
- State the strongest claim the evidence supports, once, without a stack of
  hedges. "Our five points do not identify any of the terms or show that
  this law caused the observed bend", followed twenty lines later by a
  causal claim (`scaling-laws.md:475` and `:499`), is worse than either
  statement alone.
- The PyTorch and JAX tabs teach the same lesson. When the prose under the
  two tabs draws opposite conclusions (linear attention slower than dense in
  one tab, several times faster in the other, `attention-at-scale.md:912`
  and `:930`), the result is not understood yet.
- Framework plumbing (JIT boundaries, padding, `no_grad`, allocator
  behavior) belongs in code comments or in the Computation and Performance
  chapters, not in the explanation of an algorithm.
- Code teaches; it does not draw (`CLAUDE.md`). An implementation exists to
  make the idea concrete. It is not a second explanation, and it does not
  replace one.

## 10. Sentences and Words

### 10.1 Plain Declarative Prose

- The grammatical subject is the object doing the work: the state, the
  gradient, the encoder, the reader. It is not "the picture", "the story",
  "the lesson", or "this move".
- One idea per sentence. A sentence that needs a semicolon and a dash to hold
  together is two sentences.
- Use ordinary verbs: *computes*, *stores*, *reduces*, *fails*. Avoid
  economic and theatrical metaphors: *buys*, *pays for*, *the price*,
  *earns its place*, *surgical*, *quietly*.
- Repeat a technical noun rather than vary it with synonyms.
- "We" is the author and reader working together ("we now compute").
  "You" is for instructions and exercises.
- Intensifiers carry no information. The current text uses *exactly* about
  1,000 times and *precisely* about 110 times. Keep them only where an
  approximation is being contrasted.

### 10.2 Patterns to Remove

These patterns mark text as machine-written. Remove them in revision.

| Pattern | Example from the current text | Write instead |
|---|---|---|
| Economic metaphor | "asked how much of that machinery is load-bearing" (`ssm.md:82`) | "asked which parts of the GRU are necessary" |
| Slogan | "The answer, as everywhere else in deep learning, is depth" (`ssm.md:137`) | "Stacking layers restores the lost expressiveness. Nonlinearities between layers do not block parallel evaluation." |
| Reflexive "not X; it is Y" | "The claim worth testing is not that coupled decay is too strong ... It is that the compensation depends on the learning rate" (`adamw.md:303`) | "The compensating value of $\lambda$ depends on the learning rate, so the two must be tuned together. Decoupling removes that dependence." |
| Drama | "its character is transformed" (`ssm.md:105`) | "the update becomes affine in $\mathbf{H}_{t-1}$" |
| Inflated certainty | "genuinely computes", "holds recall essentially perfect" (`recurrent-modern/index.md:35–39`) | the number, or "recall stays near 100%" |
| Stock nouns | *the lesson*, *the picture*, *the payoff*, *bookkeeping*, *scoreboard*, *spine* | the thing itself |
| Dash asides | a clause set off by — dashes inside an already long sentence (1,700 em dashes in scope) | a separate sentence, or commas |
| Colon punchline | "Formulas should be checked, not trusted:" (`kv-cache.md:477`); "Take the receipt." (`multi-gpu-practice.md:577`) | "We check the formula against the allocator." |
| Announcement | "Two consequences follow, and they organize the rest of this section." (`ssm.md:120`) | state the consequences |
| Stagecraft | "The prosecution calls two witnesses from earlier sections." (`mamba.md:135`); "Let's hear what 4.7 million parameters trained for a minute sound like:" (`gpt.md:474`); "Watch it lie:" (`performance-model.md:181`) | describe the evidence and what it shows |
| Extended metaphor | "the bf16 investment, correct all along but bought where the wall wasn't, will finally cash" (`fast-transformer.md:645`) | "Now that the step is limited by memory bandwidth, bf16 halves the bytes moved and shortens the step." |
| Anthropomorphism | "a neural network wants the opposite temperament" (`lr-scheduler.md:359`); "deserve a sentence of respect" (`adam.md:264`) | state the property |
| Rhetorical question answered by a slogan | "Why obsess over the size of a buffer that merely *sits* in memory? Because at generation time it does not sit — it moves." (`kv-cache.md:526`) | "At each generated token the whole cache is read from memory, so its size sets the speed of generation." |
| Hedge stack | "a descriptive diagnostic, not a certificate that the critic is accurate" (`actor-critic.md:352`) | say what the diagnostic measures |

### 10.3 Punctuation

- At most one dash pair per paragraph, and none in headings.
- Semicolons join two closely related clauses, not lists of claims.
- Parentheses hold asides the reader can skip. If the reader cannot skip it,
  it is not an aside.
- Italics mark a term at its definition, and occasionally a word that carries
  the contrast. Not more than once per paragraph.

### 10.4 Wit

Wit is welcome when it is brief, concrete, and true, and when it does not
carry the argument: MacKay's biologist, Strang's footnote on soldiers and
bridges ("One which I never really believed—but a bridge did crash this way
in 1831"), the sheep in *Learning with Kernels*. No jokes in headings and no
quips at the end of paragraphs.

## 11. Headings

Headings name the subject of the section, as noun phrases, the way a table of
contents in Cover and Thomas or Strang reads: "Entropy", "Diagonalization of
a Matrix", "Soft K-Means Clustering", "Support Vector Regression".

Not allowed:

- **Wh-clauses** ("What Weight Decay Is Actually Doing", "Why Compute
  Outruns Bandwidth", "When Compilation Hurts"). There are 44 in scope.
- **Sentences** ("Prefill Is Compute-Bound, Decode Is Memory-Bound",
  "Recurrence is Convolution").
- **Colon subtitles that make a claim** ("Discretization: the Step Size is a
  Gate", "Chunked Computation: Mostly Matmul, a Little Scan").
- **Participle tags and riddles** ("Decoupling, Demonstrated", "Same Logits,
  Measured", "A Grid, Twice", "The One That Kept the Normalizer", "Learning
  It, Without Cheating").
- **Questions**, except as the title of an exercise.

Sibling headings have parallel form. A worked example may be titled "Example:
…" when the book uses that form consistently.

| Current | Revised |
|---|---|
| What Attention Computes (`what-attention-computes.md:1`) | Attention Circuits |
| What Weight Decay Is Actually Doing (`adamw.md:429`) | Weight Decay in Large-Scale Training |
| Why Compute Outruns Bandwidth (`hardware.md:133`) | Compute and Memory Bandwidth |
| When Compilation Hurts (`compilation.md:377`) | Limits of Compilation |
| Where Gradients Vanish (`optimization-intro.md:78`) | Vanishing Gradients |
| Why Convexity Matters (`mdl-convexity.md:541`) | Consequences of Convexity |
| Where the Reward Comes From (`rl-sequences.md:191`) | Reward Models and Verifiers |
| What a Fixed State Cannot Do (`hybrids.md:44`) | Limits of a Fixed-Size State |
| Discretization: the Step Size is a Gate (`ssm.md:493`) | Discretization |
| Recurrence is Convolution (`ssm.md:653`) | The Convolutional Form |
| The One That Kept the Normalizer (`matrix-state.md:781`) | The mLSTM |
| Learning It, Without Cheating (`test-time-regression.md:113`) | Leave-One-Out Training |
| Decoupling, Demonstrated (`adamw.md:192`) | Coupled and Decoupled Decay Compared |
| Same Logits, Measured (`kv-cache.md:268`) | Checking the Cache |

## 12. Chapter and Section Templates

### 12.1 Chapter Introduction

At most 400 words, no tables, no experimental numbers, no lists of model
names.

1. **The problem** the chapter solves, with a concrete instance (one
   paragraph).
2. **Why it matters**, and what was unsatisfactory about the tools the
   reader already has.
3. **The route**, one sentence per section, phrased as the question that
   section answers.
4. **Scope**: what is left out, in one or two sentences, if a reader would
   otherwise expect it.

Defect: the State Space Models introduction (`recurrent-modern/index.md`)
runs to 1,840 words with two tables, a glossary of five meanings of "state",
a table of experimental confounders, a paragraph on what the chapter is not,
and more than a dozen model names.

Revision, as an illustration of length and register:

> A language model that generates text one token at a time has to remember
> what it has read. A transformer keeps everything: it stores a key and a
> value for every previous token, so the memory it needs, and the work per
> new token, grow with the length of the context. A recurrent network keeps
> a summary of fixed size instead, its *state*, and updates it once per
> token. Generating the millionth token then costs as much as generating the
> tenth.
>
> A fixed-size summary raises two problems. Classical recurrent networks
> update the state with a nonlinear function of the previous state, so
> training has to process a sequence one step at a time and leaves most of a
> GPU idle. And a state of fixed size must forget, which raises the question
> of what it should keep.
>
> :numref:`sec_lstm` shows how gates let a network decide what to keep.
> :numref:`sec_ssm` makes the state update linear, which allows training on
> all positions at once, and borrows from control theory a principled way to
> choose how fast the state forgets. :numref:`sec_mamba` lets each input
> decide how much to write and how much to forget. The next three sections
> ask how much a fixed state can store and how to write to it without
> destroying what is already there. The last section combines recurrent
> layers with a few attention layers, which is how most recent large models
> use them.

### 12.2 Section

1. **Opening** (150–300 words): problem, why, how, where (§3.1).
2. **Body**: two to five `##` subsections, each developing the section's one
   idea through the sequence of §5.
3. **Closing**: what the reader can now do, and the open question (§3.3).
4. **Summary**: three to six sentences that restate the answers to the four
   questions of §1. Not a list of everything mentioned.
5. **Exercises**: problems that are good and add value, chosen per the
   exercise rubric. Drop filler: drills, restatements of the text, sweeps
   without a question, API trivia, near-duplicates, and problems that depend
   on later chapters. Keep a good problem even if the section then exceeds
   the guide of about 30% of the body prose (`tools/exercise_budget.py`);
   the number shows balance, it is not a cap. A problem may have several
   parts and real depth if every part teaches something. Never add problems
   to fill space. Counts stay within one band across the book so that
   chapters are even: at least 5 problems per section; 5–8 in the main
   text, up to 10 for sections over about 3,000 words; 8–12 in the
   mathematics appendix; 5–8 in Tools. A section below the band needs more
   good problems, not filler; a section above it keeps the best ones.

Sections do not end with surveys, "In Practice" lists, or "Further Reading"
paragraphs. That material goes to the chapter's Further Reading notebook.

### 12.3 Further Reading Notebook

A chapter may end with one optional notebook, `further-reading.md`, titled
"Further Reading". It is the only place for material that fails the
teachable-core test (§4.1) but that some readers will want: related methods,
recent variants, systems used in practice, history, and pointers into the
literature. It is used sparingly.

- **At most one per chapter,** placed last, and never heavier than an
  ordinary section. Aim for 1,000–2,000 words; the section ceiling of 4,500
  words (§4.4) is a hard limit. Most chapters need less, and some need
  none.
- **Nothing depends on it.** No main section, exercise, or later chapter
  refers to its content. It may refer back to the main sections.
- **Short topics, each to the same standard.** Every topic opens with the
  problem it addresses and the one thing it changes relative to the method
  taught in the chapter, in a paragraph or two. A named method with no stated
  difference does not appear.
- **Little code.** A short demonstration is allowed when it shows the one
  difference. No experiments that need training runs.
- **Citations carry a clause** saying what each work adds.
- It replaces the "Resources and Further Reading" sections now found in
  chapter introductions, and the survey tails of individual sections.

## 13. Revision Checklist

Read the section top to bottom as a newcomer and check:

1. The opening answers what, why, how, and where before new notation appears.
2. The section's idea can be stated in one jargon-free sentence.
3. Every new concept is defined in words where it first appears, and only one
   arrives per paragraph.
4. The smallest case is worked completely, with numbers.
5. Each displayed equation is introduced and followed by a sentence that
   reads it.
6. Every use of a fact from another chapter restates it. No `:eqref:` points
   to another chapter as the only support for an argument.
7. No coined term from another section is used as if standard.
8. Each code cell has a stated question; each output is interpreted.
9. Limits are shown, not just mentioned, and the closing names the open
   question.
10. Headings are noun phrases; siblings are parallel.
11. No pattern from §10.2 remains; dashes are rare.
12. The section is within its length budget (§4.4). Anything cut is cut, not
    moved to a footnote.
13. Nothing from §4.5 remains: no draft history, paper audits, per-framework
    narration, or scope disclaimers.
14. Every symbol has one meaning in the chapter; no fact is derived twice.
15. Every exercise is good and adds value; no filler remains (§12.2).
16. Read the section aloud once. Sentences that do not parse ("This limit
    gives the value approached by the ratio ... converges to as we shrink $h$
    to zero", `calculus.md:95`) and references to questions that were
    deleted ("The pendulum question from this chapter's introduction",
    `hybrids.md:1008`) are found only this way.

## 14. Mechanical Checks

Some defects can be found by search. These commands are a minimum, not a
substitute for reading.

```bash
# Wh-clause headings
grep -nE '^#{1,4} (What|Why|How|When|Where|Which|Who)\b' chapter_*/*.md
# Stock phrases
grep -niE 'load-bearing|earns? its|the (lesson|payoff|picture|story)\b|scoreboard|spine|surgical|quietly|in plain sight|for free|genuinely' chapter_*/*.md
# Exercise volume against the ~30% guide (§12.2)
python tools/exercise_budget.py --over
# Dash density per file
for f in chapter_*/*.md; do printf '%5d %s\n' "$(grep -o '—' "$f" | wc -l)" "$f"; done | sort -rn | head
```

A future `tools/lint_source.py --prose` mode can report these as warnings.
