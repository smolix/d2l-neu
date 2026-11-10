# Writing Review of Part 1 (Basics): Overview

Date: 2026-11-09. Source: `main` at `201f2cb6`. Scope: chapters 1 to 8, all 58
sections plus the eight chapter introductions and the part introduction.
Question asked: how clear, well thought out, well taught and enjoyable is this
part, measured against Cover and Thomas, MacKay, and The Illustrated
Transformer? No source text was changed.

## Author decisions, 2026-11-09

Alex Smola read this overview and decided as follows. These override the
recommendations below and in the chapter files where they differ.

1. **Voice.** Give the prose a voice. The model is the first chapter of
   *Learning with Kernels* (Schölkopf and Smola, 2002). `docs/writing-guide.md`
   §10.2 now describes it.
2. **Repetition.** Remove it (guide §7.7, rewritten).
3. **Experiments wait.** The writing pass comes first. Findings that ask for
   a new or redesigned experiment, a changed code cell, or re-captured
   outputs (most of finding 1 below, and the items tagged pedagogy that
   propose a run) are deferred. Where a passage draws a conclusion its
   output does not show, the writing pass may change the prose to say what
   the output shows.
4. **Optimizers before chapter 9 are fine**, provided each first use has a
   forward reference. Finding 5 below reduces to checking those references.
5. **Chapter 2 scaffolding:** some of it may move to later sections.
6. **Chapter 5** may keep material that reads as framework documentation; the
   book is meant to be practical. Prune, do not halve.
7. **Chapter 7** may sit above the level of "Basics". Retelling papers is
   the problem, not the level.
8. **The Waldo figures stay.** They are a joke the reviewer missed (WHY-13 is
   withdrawn).

## Verdict

Part 1 is now a good textbook and not yet an excellent one. The sentences are
clear and correct; the reviewers found no mathematical errors in chapters 1, 2
and 6 after recomputing the worked numbers, and almost none of the mannerisms
that the October review counted in the thousands. Every chapter and nearly
every section opens on a question a reader can feel. The explanations, taken
one at a time, are at the level of a good textbook.

Three things separate it from the model texts.

1. **The experiments demonstrate less than the prose claims.** This is the
   largest gap, and it grows through the part. From chapter 4 on, sections
   repeatedly derive or quote a result, run a notebook that does not show it,
   and then explain what the run did not show. The model texts never ask the
   reader to take the central point of a section on trust.
2. **The prose has been cleaned until it is even.** It is tic-free, and it
   is also voiceless: one register, one sentence length (about 21 words in
   every chapter), one opening formula and one closing formula. Voice
   survives only in the examples. MacKay and Cover and Thomas are enjoyable
   because an author is visibly choosing what matters; here the writing guide
   is visibly choosing.
3. **It is long, and the length goes to repetition and software, not to
   ideas.** The part has about 245,000 words of prose outside code, roughly
   the length of the whole of Cover and Thomas. A reader of one framework
   sees less than that, but still meets most facts two to five times and a
   great deal of API narration.

None of this needs a rewrite. The best passages already show the method that
would close the gap, and it is the book's own: pose a concrete question,
predict a number from a short argument, run the code, and let the number
confirm or surprise.

## Method and limits

Ten reviewers each read one chapter, or half of chapters 5 and 7, in full:
every line of prose, mathematics, PyTorch code and exercises up to the slides
marker. They compared the prose with the stored PyTorch outputs, rendered the
figures, and verified every quoted anchor against the sources. I read the
part and chapter introductions, checked a sample of the more striking
findings against the sources, measured the part, and wrote the two split
chapter files and this overview. The review directory holds 73 files and
about 140,000 words; each finding has an ID, a severity, a verbatim anchor
and a direction for the fix.

Limits to keep in mind:

- Only the PyTorch outputs were checked. TensorFlow, JAX and MXNet tab prose
  was read but their outputs were not compared (except in 8.5).
- The rendered pages at d2l.smola.org were not inspected; the review is of
  the sources, the stored outputs and the figure files.
- Nothing was re-executed. Claims about run-to-run variation, and the new
  experiments the reviewers propose, are untested.
- Scores come from ten different readers and are coarse. Read them for the
  pattern across dimensions, not for differences of a few tenths between
  chapters.

## Scores

Mean over the sections of each chapter (chapter introductions excluded).
Scale: 5 = at home in Cover and Thomas or MacKay; 4 = good textbook prose with
local flaws; 3 = competent but ordinary, or good with a structural problem;
2 = a reader struggles.

| Chapter | Explanation | Thought | Pedagogy | Language | Enjoyment | Prose words |
|---|---|---|---|---|---|---|
| 1 Preliminaries | 4.1 | 4.0 | 3.9 | 4.0 | 3.6 | 34,500 |
| 2 Linear Regression | 3.9 | 3.4 | 3.6 | 4.0 | 3.3 | 31,200 |
| 3 Linear Classification | 4.0 | 3.7 | 3.4 | 4.0 | 3.7 | 25,300 |
| 4 Multilayer Perceptron | 3.9 | 3.7 | 3.3 | 3.9 | 3.4 | 27,300 |
| 5 Computation | 3.9 | 3.5 | 3.4 | 3.8 | 3.6 | 49,100 |
| 6 Convolutional Neural Networks | 4.0 | 3.7 | 3.3 | 4.0 | 3.2 | 18,000 |
| 7 Modern Convnets | 3.6 | 3.3 | 3.1 | 3.8 | 3.3 | 32,000 |
| 8 Sequence Models | 3.9 | 3.6 | 3.7 | 4.0 | 3.6 | 28,100 |

Prose words are counted in the source across all framework tabs, without code
and display equations; a reader of one framework sees roughly half to three
quarters of this, depending on the chapter.

The pattern is the finding. Explanation and language sit at 4 almost
everywhere. Pedagogy and enjoyment sit half a point to a point lower in every
chapter, and thought (one line of argument, claims that are supported) lies
between. Out of 58 sections, a 5 was awarded four times: explanation in 1.4
Calculus, thought in 1.5 Autograd and 4.1 MLP, pedagogy in 2.6 Generalization,
enjoyment in 3.6 Generalization in Classification. The lowest scores are 2.2
Object-Oriented Design (pedagogy 2, enjoyment 2) and 7.8 Design Spaces
(pedagogy 2, enjoyment 2).

Chapter 1 is the strongest chapter; chapter 7 is the weakest, mainly in its
second half.

## What has improved since October

- The language defects are gone from the body prose: em dashes, "exactly",
  "rather than", stagecraft, economic metaphor, wh-headings. Remaining
  em dashes are in the "Resources and Further Reading" lists.
- Section openings pose a problem instead of listing topics.
- Dependence on the mathematics appendix is light: about 17 pointers in
  chapter 1 against 34 in October, none carrying an argument.
- The history-tour structure of chapter 7 and the essay structure of the
  distribution-shift section have been replaced by sections with one idea.
- Most chapter-specific October findings are closed (all five in chapter 6,
  all four in chapter 3, all but one in chapter 4).

Still open from October: the plotting utilities inside 1.4 Calculus; the
meta-device subsection and fp8 in chapter 5; forward-defined language-model
vocabulary in chapter 5.

## Part-wide findings

### 1. Experiments that do not carry their claims

Every reviewer from chapter 2 onward reported this independently. It takes
four forms.

**The claim is asserted and never run.**

- 4.4 says a badly initialized network "does not learn" three times and never
  trains one; 5.3 repeats the pattern.
- 5.5 never trains with bf16 weights, never times mixed precision, and never
  shows loss scaling rescuing a run.
- 5.6 promises that training "resumes exactly" and does not compare with an
  uninterrupted run.
- 6.1, 6.6 and the chapter 6 introduction say three times that an MLP is
  indifferent to a fixed shuffling of pixels; no cell tests it.
- 7.4 quotes the degradation of deep plain networks from the paper and trains
  only ResNet-18.
- 8.5 describes gradient spikes with numbers that no cell prints.

**The run is a tie, and the prose reads a result into it.**

- 4.6: accuracy with and without dropout is the same; three interpretations
  of dropout's benefit follow.
- 7.5 to 7.8: on upsampled Fashion-MNIST the efficient network, the modern
  recipe, ConvNeXt and RegNet each land within two points of the baseline,
  some below it. Each is followed by 100 to 300 words on what the run does
  not show.
- 8.5: the trained RNN ties the Kneser-Ney trigram (2.11 against 2.09 bits
  per byte), so nothing shows a hidden state using context a trigram lacks.

**The run shows something else.**

- 2.7: by the text's own analysis the weight-decay experiment helps only by
  erasing the random initialization, and the regularized model is
  indistinguishable from predicting a constant.
- 7.2: the comparison table trains GoogLeNet at 96 by 96 and the others at
  224 by 224, then credits GoogLeNet with a third of AlexNet's arithmetic.
- 8.1: the rollout model damps its inputs, yet the text concludes that errors
  accumulate.
- 8.6: the only experiment plots $\rho^k$ for a symmetric matrix whose power
  has norm exactly $\rho^k$.

**The evidence is in an exercise.** Chapter 8 has at least nine sentences of
the form "an exercise shows"; in chapters 4, 6 and 7 the best exercises are
the experiments the section itself should have run (plain against residual
networks, the per-class confusions the MLP was meant to fix, the weight
templates of softmax regression).

Two causes recur. Fashion-MNIST saturates near 92 to 94 percent, so it cannot
separate any two designs from AlexNet onward. And sections were written around
a finished recipe: the notebook confirms the recipe works and has no failing
arm. The fix is the same in each case and the part already contains its
model: the paired run. LeNet with and without batch normalization (7.3), the
shift experiment in 6.5, the hand-built and then learned edge detector in
6.2, the doubling of the training set in 4.5. Each section should have one
comparison in which only the thing it teaches is varied and the outcome is
visible without commentary.

### 2. An even voice and visible templates

The writing guide's ban list has been applied and nothing was put in its
place. Reviewers of chapters 1, 2, 3, 4 and 7 used nearly the same words:
correct, clean, flat. The first half of chapter 7 has one light moment in
four sections.

The templates in the guide (§12.1, §12.2, §3.3) are now visible to a reader
who goes front to back:

- Every chapter introduction is: a scenario; a history paragraph, four of
  which begin "The question is older than ..." or "The question is old"; and
  a route paragraph of six to eight consecutive sentences "§x asks how ...".
  The part introduction uses the same frame. A route written as a run of
  identical sentences is a list in disguise, and the reader skips it.
- Sections open with a paragraph ending in a question and close with "We can
  now ..." (eight times in chapter 1, five of seven sections in chapter 3),
  followed by a Summary that says the same again. Chapter 4's reviewer calls
  this the double ending.
- The Fashion-MNIST garment opens the part, chapter 1, chapter 3, chapter 4
  and chapter 6. The Legendre and Gauss story is told three times in the
  first forty pages (part introduction, chapter 2 introduction, section 2.1).

The remedy is partly in the guide. It should present its templates as a
checklist of what an opening and closing must achieve, not as a form to fill;
allow a chapter route to be two or three sentences that group sections into
movements (chapter 5: build, number formats, run); and drop either the
closing paragraph or the summary. The rest is authorial: keep the memorable
examples that exist (the always-healthy classifier at 99 percent, the random
guessers who reach 70 percent by voting on labels they never saw, the
`Tripwire` pickle, $0.9^{40}$ against $1.1^{40}$) and write more like them.

The "Resources and Further Reading" lists are in a different, older voice
("a perfect warm-up", "what you actually do in 2026", "unusually good
figures") and carry the remaining em dashes. They need the same pass as the
body.

### 3. Repetition

The October review found repeated teaching; it is still the main cost in
length, and one cause is now the guide itself. §7.2 says to restate a fact
from more than one section back. In practice the restatement is a paragraph,
not a clause.

- Chapter 3: a table of ten facts told two to five times. The overflow
  threshold of 88 appears at least five times; shift invariance of softmax is
  derived three times; temperature scaling is explained three times and never
  applied.
- Chapter 2: the factor of 2 between the book's loss and the framework's is
  explained seven times. Better to remove the mismatch at its source.
- Chapter 4: "203,530" appears ten times and "std 0.01" about forty.
- Chapter 5: the $16N$ memory accounting five times; registration and buffers
  three times.
- Chapter 7: ResNet-18 rebuilt three times; the cosine schedule in four
  sections; He initialization three times (and three more in chapter 5).
- Chapter 1: "a small multiple of the cost, whatever the number of
  parameters" five times.

The scratch and concise pairs in chapters 2 and 3 are a structural source:
the same story is told in two sections and again in both summaries. The guide
should say that a restatement is one clause with a reference.

### 4. Software in the way of ideas

- **The scaffolding arrives before anything runs.** 2.2 Object-Oriented
  Design is the weakest section of the part by score. It defines `Module`,
  `DataModule` and `Trainer` with six stub methods, explains dictionary
  metrics, GPUs, clipping, schedules and `evaluate`, none of which chapter 2
  uses, and produces no result. The first training run comes about 9,000
  words after SGD is derived. The reviewer's proposed order: the model, the
  synthetic data, a bare training loop that runs, and only then the
  refactoring into classes, as an answer to code the reader has just written.
- **"From scratch" is not from scratch.** 2.4 shows a `fit_epoch` with
  clipping, scheduler, hooks and an averager; the four-operation loop never
  runs.
- **Hooks for later chapters are explained where they are defined.** Six
  instances in chapter 8, several in 2.2 and 5.7 (`Benchmark` is defined and
  unused). Explain a hook at its first use.
- **Framework tabs.** For a reader of one framework, tab notes are 22 percent
  of the prose in 4.2, about 30 percent in 5.1 and 5.2, about 900 words in
  chapter 6. Many restate one fact four times, or open "X has no ...". Some
  shared prose fits only one framework: the `ones(2, 3)` puzzle in 1.7 is no
  puzzle in PyTorch; the fp32 accumulation paragraph in 5.5 describes a stall
  the PyTorch output does not show; 5.1 promises a data-dependent loop that
  only the TensorFlow and JAX tabs contain. State the idea once in shared
  prose and keep tabs to code plus at most one sentence true of that
  framework alone.
- **Residue.** A boilerplate import cell (`math`, `nn`, `F`, `optax`, ...)
  opens notebooks that use none of it; about 90 lines of matplotlib utilities
  sit between the closing paragraph and the summary of 1.4; cells print a
  bare `True` or a shape; first-edition toy cells survive inside rewritten
  prose in 5.3.

### 5. The optimizers are used before they are taught

Momentum is first explained in a parenthesis in 7.5. Adam is named in 5.2 and
5.6, used as AdamW in 7.6, and twice called the real fix for a problem in
8.5. Warmup with a cosine schedule is defined inside the AlexNet experiment
and restated five times. Chapter 9 (Optimization) is in the Advanced part.
The part introduction says Basics "uses minibatch stochastic gradient
descent, momentum, and learning-rate schedules as tools", but no section
hands the reader those tools. This needs a decision: a short section, perhaps
at the end of chapter 4 or the start of chapter 7, that states momentum, Adam
and a schedule as recipes with one experiment each and points to chapter 9
for the reasons.

### 6. Order within chapters

- **Chapter 2:** scaffolding before the first run (above).
- **Chapter 3:** 3.3 mixes a base class with evaluation methodology on
  invented numbers, before any model is trained. Moving "Beyond Accuracy"
  after the concise implementation brings the first training run forward by
  about 2,000 words.
- **Chapter 1:** 1.6 Probability asks its coin question at the top and
  answers it 5,000 words later, with Bayes' theorem and the HIV test in
  between. It is two lessons interleaved.
- **Chapter 8:** in 8.3 smoothing is derived before perplexity exists, and
  Zipf's law follows the sparsity it explains. The implementation (8.5)
  precedes backpropagation through time (8.6), so clipping, truncation and
  detaching are each forward-referenced and taught twice, and 8.4 ends by
  pointing past the next section.
- **Chapter 7:** ends on a 2020 paper after a 2022 one, on an orphaned
  subsection, with no conclusion.
- **Chapter endings:** chapters 4, 5 and 7 stop on a section summary. A
  chapter should end by answering the question its introduction asked.

### 7. Questions posed and not answered

The openings are good enough that the reader remembers them, which makes it
noticeable when a section answers a neighbouring question instead.

- Chapter 2 opens by pricing a house; the house is never valued and all the
  chapter's data are synthetic.
- 4.2 asks whether a hidden layer fixes the shirt, pullover and coat
  confusions, then reports only total accuracy.
- 4.3 asks how one pass gives 203,530 derivatives and does not say.
- 5.7 asks "Will a given model fit?", answers no, and drops it.
- 5.8 asks whether a 0.3-point gain is real and never measures seed-to-seed
  spread.
- 8.3 promises learned representations ("cat" and "feline"); 8.5 finds the
  embeddings barely train.

### 8. Backpropagation and autograd

1.5 Autograd is one of the best sections of the part. 4.3 Backpropagation
then restarts from a scalar graph simpler than the one the reader already
traced, and stops at one hidden layer although the next section needs the
recursion through many. The two sections are consistent but the second does
not build on the first. It should take the graph, the local rules and the
cost argument as given and spend its words on what is new: the hidden-layer
gradient read as error sent backward, and the recursion over layers.

### 9. Notation, definitions and terms

- Symbols reused within a chapter: $\sigma$, $d$, $L$ (chapter 2);
  $\hat{y}$ as a probability vector and a class index (chapter 3); $K$ and
  $m$ (chapter 4); $\gamma$, $\beta$, $k$, $g$, $c_i$ (chapter 7); padding
  $p$ as a total in formulas and per side in code (chapter 6), which gives a
  reader the wrong shape.
- Gradients are $\partial J / \partial \mathbf{W}$ in one section and
  $\nabla_{\mathbf{h}} L$ in the next; the conditioning notation for
  sequences takes three forms in chapter 8.
- Missing definitions: the identity matrix and the matrix inverse (used in
  chapters 1 and 2); the two-standard-error rule and the p-value in 1.6.
- "Kernel" means a GPU routine, a weight matrix and a convolution filter
  within two chapters; "test accuracy" and "validation accuracy" name the
  same `val_acc` in chapter 7.

### 10. Figures

The mechanical figures are complete and in one style. The conceptual ones are
missing where they would do most.

- 6.1 derives convolution in four-index algebra and gives the one-dimensional
  weight-matrix figure (64, 15 and 3 weights) a single sentence. Running the
  derivation on that figure first would turn the proof into a picture: each
  column is an impulse response, and equivariance makes the matrix constant
  along its diagonals. This is what an "illustrated" treatment would do.
- The chapter has no picture of equivariance itself: an input with one
  distinctive patch and its response map, and the same pair after a shift.
  (An earlier version of this review called the Waldo photographs flawed.
  That finding is withdrawn: they are a deliberate joke and stay.)
- No figure in the part shows a learned feature: not the weight templates of
  softmax regression, not a hidden unit of the MLP, not a LeNet kernel.
- The inverted bottleneck is explained twice in chapter 7 without a figure.

### 11. Exercises

The exercise sets are a strength: well graded in chapters 1 and 5, and often
the sharpest thinking in a section. Three reservations. Too much evidence for
the text's claims has been moved into them (finding 1). Chapter 7's are
projects, with about 30 training runs in one section and 48 in a single
exercise. And chapter 3's nearly all start at medium difficulty; some
sections need one or two that a reader can do in five minutes to check
understanding.

### 12. Statements to correct or re-capture

These are places where the prose and the stored PyTorch output disagree, or a
claim is wrong as written. They are few, and worth fixing before anything
else because they are cheap.

- 1.6: the stored multinomial sample is [50, 50] directly after a sentence
  saying that outcome is rare (PRB-2).
- 2.6: "more than a thousand times larger" against a test error of 2.3e14
  (GEN-4). 2.7: the formula gives $\lambda = 0.05$ and the experiment uses 3
  without comment (WD-2).
- 3.5 and 3.7: validation loss "near 0.49" against 0.475; "few dozen
  effective examples" against 13; slope "between about 1 and 2" against just
  under 1.
- 5.5: the fp32 accumulation paragraph (NUM-12).
- 7.2: the GoogLeNet arithmetic comparison (BLK-9). 7.3: internal covariate
  shift restated as fact in the opening (BN-1).
- 8.3: the word-level baseline builds its vocabulary from held-out text, so
  the best row of the bits-per-byte table is disqualified by the next
  sentence (LM-4). 8.7: top-$k$ sampling is credited to the GPT-2 report; the
  usual credit is Fan et al. 2018, to be confirmed (DEC-11).
- 8.2 and 8.7 refer to `sec_machine_translation`, which now lives in the
  Attic.

## What is already at the bar

These passages should set the standard for the revision. What they share is
one small example carried all the way through, numbers the reader can check
by hand, and a notebook result predicted before it prints.

- **1.4 Calculus and 1.5 Autograd.** One line model carries the derivative,
  the gradient, a descent step and the chain rule; autograd then reproduces
  the same numbers.
- **2.6 Generalization.** One polynomial experiment carries risk, bias and
  variance, capacity and cross-validation, in prose written to survive
  re-execution.
- **3.1 Softmax Regression.** The logits 1.0, 2.2, 0.3 followed through loss,
  gradient and code.
- **3.6 Generalization in Classification, parts one and two.** The coin
  model and the voting guessers.
- **4.1 MLP.** XOR solved by proof, construction, code and picture, then
  reused as the tent map to explain depth.
- **4.5 Generalization in Deep Learning.** "The answer is not completely
  known, so this section measures."
- **5.3 Initialization of Residual Networks; 5.6 Saving and Loading.** The
  $2^{16}$ prediction; the pickle that runs code on load.
- **6.2, first half.** The edge detector solved by hand, tested against a
  negative control, then learned.
- **7.1 AlexNet; the opening of 7.6.** Checkable arithmetic; 76.1 to 80.4
  percent from the recipe alone.
- **8.6 Backpropagation through Time.** The scalar recurrence and the
  paragraph on why clipping cannot repair vanishing gradients.

## Decisions for the revision

Listed in the order I would take them. The first three change what the
per-chapter work looks like.

1. **Experiment policy.** Adopt the rule that each section has one paired
   comparison isolating its idea, with a failing arm, and that no central
   claim rests on an exercise or a quoted paper alone. Decide whether
   chapter 7 keeps Fashion-MNIST with redesigned experiments or moves to a
   harder small dataset. The reviewers' proposed experiments are untested and
   need to be confirmed in all four frameworks before prose is written around
   them.
2. **Writing guide.** Loosen the templates of §12 and §3.3 (route paragraph,
   "We can now", closing plus summary), restrict §7.2 restatements to a
   clause, and add a rule for tab prose. Otherwise the next revision will
   reproduce the same uniformity.
3. **Optimizers in Basics.** Decide where momentum, Adam and schedules are
   introduced before chapters 5, 7 and 8 use them.
4. **Chapter 2 order and scaffolding.** Run a bare loop first; introduce the
   classes as a refactoring; explain hooks at first use.
5. **Scope of chapters 5 and 7.** Cut the documentation side of chapter 5 (a
   quarter to a third). Decide whether ConvNeXt and design spaces stay in
   Basics, shrink, or move to Advanced.
6. **Then chapter by chapter**, starting with the cheap corrections of
   finding 12, then order (finding 6), then experiments, then cuts and
   repetition, and last the sentence-level items in the per-notebook files.

## Files

Each chapter file gives the verdict, a score table, the arc, the most
important findings with IDs, recurring patterns, cuts and gaps. The
per-notebook files in the directory of the same name hold the passage-level
findings.

| Chapter | Chapter file | Best section | Weakest section |
|---|---|---|---|
| 1 Preliminaries | [ch01-preliminaries.md](ch01-preliminaries.md) | Calculus | Documentation (lookup-api) |
| 2 Linear Regression | [ch02-linear-regression.md](ch02-linear-regression.md) | Generalization | Object-Oriented Design |
| 3 Linear Classification | [ch03-linear-classification.md](ch03-linear-classification.md) | Generalization in Classification | Measuring Classifiers |
| 4 Multilayer Perceptron | [ch04-multilayer-perceptrons.md](ch04-multilayer-perceptrons.md) | MLP | Implementation (4.2) |
| 5 Computation | [ch05-computation.md](ch05-computation.md) | Saving and Loading | Reproducibility and Inspection |
| 6 Convolutional Neural Networks | [ch06-cnn.md](ch06-cnn.md) | Convolutions for Images, first half | Pooling |
| 7 Modern Convnets | [ch07-modern-convnets.md](ch07-modern-convnets.md) | AlexNet; Training Recipes | Design Spaces |
| 8 Sequence Models | [ch08-sequence-models.md](ch08-sequence-models.md) | Backpropagation through Time | Implementing RNNs |
