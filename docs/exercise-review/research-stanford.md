# Stanford intro-ML/DL exercises: conceptual, pre-math, pre-code

Scope: exercises that ask students to frame a problem as ML (input/output/data/loss),
classify a scenario (supervised/unsupervised/RL, classification/regression), judge
when ML is/isn't appropriate, identify components, or reason about scale/history.
Excludes linear-algebra/probability warmups and coding assignments.

Key finding up front: **CS229's own problem sets contain zero conceptual exercises.**
Problem Set #1 (see.stanford.edu) is 100% derivation (Newton's method for least
squares, locally-weighted logistic regression, multivariate least squares, Naive
Bayes MLE, exponential family) from the first page on — no "what is ML" or
problem-framing question anywhere. The conceptual framing that CS229 lecture notes
give in prose (housing-price example, Tom Mitchell's T/P/E definition) never
resurfaces as a graded question in the CS229 problem sets themselves. The place
that framing *does* show up as a gradable exercise is Andrew Ng's public-facing
sibling course — the original Coursera "Machine Learning" (ml-class, same
instructor, same T/P/E definition, explicitly Stanford-branded) and its 2022
successor "Machine Learning Specialization" (Stanford Online + DeepLearning.AI).
I've treated those as the CS229-adjacent conceptual-exercise source per the task's
own suggestion to look at "anything Stanford-adjacent."

Similarly, CS231n Module 0 has no graded conceptual quiz — its problem-framing
content lives only in the prose of the "Image Classification" notes page. CS230's
own site does not publish separate intro problem sets; the graded conceptual quiz
is entirely on its Coursera companion (cross-listed as CS230). CS221 is the
strongest source of genuine *stand-alone* conceptual exercises, delivered as
live in-lecture clicker questions via its `cs221.stanford.edu/q` system.

---

## Archetypes observed

1. **T/P/E component identification** — given Tom Mitchell's "learn from
   experience E w.r.t. task T and performance measure P" definition and a
   scenario, identify which piece of the scenario is T vs. P vs. E.
2. **Classification-vs-regression judgment** — given a one-line prediction
   scenario, decide whether the output is continuous (regression) or
   discrete/categorical (classification).
3. **Supervised-vs-unsupervised classification (scenario, select-all)** — a
   list of short scenarios; mark which would be solved with labeled-data
   (supervised) methods vs. label-free/clustering (unsupervised) methods.
4. **Best-definition-of-ML judgment** — multiple choice among several
   candidate one-line definitions of "machine learning," where distractors
   are subtly too narrow ("learns from labeled data"), too broad ("robots
   acting intelligently"), or just restate "programming."
5. **Generate-your-own problem-framing example** — open-ended (not
   multiple-choice): student proposes their own input→output prediction task.
6. **Scale/magnitude estimation** — order-of-magnitude clicker question about
   the scale of modern ML systems (parameters, data), meant to recalibrate
   intuition before diving into math.
7. **Data/compute-scale-vs-performance reasoning** — reasoning about why
   deep learning "took off now" (more data, more compute) and reading a
   performance-vs-data-scale curve.
8. **Structured-vs-unstructured data judgment** — true/false calls on whether
   a given data type (tabular demographics vs. raw image pixels) counts as
   "structured" or "unstructured" data.
9. **Analogy/rhetorical-framing interpretation** — interpret a rhetorical
   device used to frame AI's importance (e.g. the "AI is the new electricity"
   line) and pick the reading that matches the intended economy-wide-impact
   claim rather than a literal or narrow one.
10. **ML-workflow / iteration-mental-model judgment** — true/false or MC about
    whether practitioners nail a good model on the first try, or how the
    idea→code→experiment loop works and what makes it faster.
11. **True-objective-of-ML / generalization judgment** — MC clicker question
    about what ML is actually optimizing for (training error vs. regularized
    training error vs. error on unseen future data).
12. **Ethical/social-impact scenario spotting** — given a deployment scenario,
    identify potential negative social impact or ethical issues, independent
    of any math — a "should we build this / what could go wrong" framing
    exercise rather than a technical one.
13. **(Negative pattern) Problem-framing-as-prose, not exercise** — some
    courses (CS231n module 0, CS229 lecture notes) present excellent
    conceptual framing (input/output, "why is this hard," data-driven
    rationale) but never turn it into a gradable question.

---

## Examples by archetype

### 1. T/P/E component identification
- **Source: Stanford "Machine Learning" (Andrew Ng), Coursera, Week 1 Quiz —
  Introduction** (CS229-adjacent; same instructor/definitions as CS229's intro
  lecture).
  Scenario: a learning algorithm is fed historical weather data to learn to
  predict weather. Students are asked, in separate questions, to pick which
  option is the *performance measure P* and which is the *task T* out of four
  paraphrases of the same scenario (a classic "don't confuse the task with the
  measure with the training process" trap).
  Mirror text: https://gist.github.com/mGalarnyk/3f3337294804c8729d26acbe06448e86
  (also: https://raw.githubusercontent.com/awsk1994/Machine-Learning-Coursera-Course/master/Week1/IntroToML_quiz.md)

### 2. Classification-vs-regression judgment
- Same Week 1 quiz, four back-to-back one-liners, each just asking
  "classification or regression?": predicting tomorrow's temperature in
  degrees (regression), predicting weather category Sunny/Cloudy/Rainy
  (classification), predicting a stock's dollar price tomorrow (regression),
  predicting whether a company will declare bankruptcy within 7 days
  (classification), predicting Microsoft's trading volume (regression).
  Source: https://gist.github.com/mGalarnyk/3f3337294804c8729d26acbe06448e86

### 3. Supervised-vs-unsupervised classification (select-all)
- Same quiz, "select all that apply" item: out of four scenarios, pick the
  ones suited to *supervised* learning. The two supervised ones are
  "classify a webpage as child-friendly or adult" and "predict next year's
  crop yield from 50 years of yield data"; the two unsupervised distractors
  are "discover categories of patient drug-response" and "discover clusters
  of heart-disease patients for tailored treatment" — both worded to *sound*
  like prediction tasks but actually lack a labeled target.
  Source: https://gist.github.com/mGalarnyk/3f3337294804c8729d26acbe06448e86
- **Machine Learning Specialization (2022, Stanford Online/DeepLearning.AI),
  Course 1 Week 1, "Practice quiz: Supervised vs unsupervised learning."**
  Same archetype modernized: spam-filtering as the supervised example,
  clustering/market-segmentation as the unsupervised example, plus a direct
  "which of these is a type of unsupervised learning?" item and "what are the
  two common types of supervised learning?" (answer: regression,
  classification).
  Source: https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%20-%20Regression%20and%20Classification/week1/Practice%20quiz%20-%20Supervised%20vs%20unsupervised%20learning/README.md
- **CS221 Lecture 2 slide "Types of unsupervised learning"** frames the same
  distinction structurally rather than as a quiz: supervised examples are
  (x,y) pairs; clustering and dimensionality reduction are shown as the two
  unsupervised archetypes, each with input/output diagrammed explicitly.
  Source: https://web.stanford.edu/class/archive/cs/cs221/cs221.1186/lectures/learning1.pdf (Lecture 2, "Machine learning I," Sadigh, Spring 2018)

### 4. Best-definition-of-ML judgment
- Same CS229-adjacent Week 1 quiz: pick the best one-line definition of
  machine learning among four candidates. The correct choice is the classic
  Arthur-Samuel-style phrasing about learning "without being explicitly
  programmed"; distractors equate ML with robots acting intelligently,
  with learning only from labeled data, or with programming in general.
  Source: https://gist.github.com/mGalarnyk/3f3337294804c8729d26acbe06448e86

### 5. Generate-your-own problem-framing example
- **CS221 Lecture 2, "Machine learning I" (Sadigh, Spring 2018), slide 11**,
  a live `cs221.stanford.edu/q` clicker slide:
  "Give an example of a prediction task (e.g., image ⇒ face/not face)."
  Open-ended — no options given; the point is to make students generate
  their own input/output pair rather than recognize one.
  Source: https://web.stanford.edu/class/archive/cs/cs221/cs221.1186/lectures/learning1.pdf (page/slide 11)

### 6. Scale/magnitude estimation
- **CS221 Lecture 2, slide 1** (opening clicker question of the ML unit):
  "How many parameters (real numbers) can be learned by machine learning
  algorithms using today's computers?" — options: thousands / millions /
  billions / trillions. Used to reset students' intuition about the scale of
  modern models before any math is introduced.
  Source: https://web.stanford.edu/class/archive/cs/cs221/cs221.1186/lectures/learning1.pdf (page 1)

### 7. Data/compute-scale-vs-performance reasoning
- **CS230 (Coursera Course 1, "Neural Networks and Deep Learning"), Week 1
  Quiz — "Introduction to deep learning."** Several items in this archetype:
  a multi-select on reasons deep learning "recently took off" (more compute,
  more data, algorithmic/application improvements — explicitly *not* "neural
  nets are brand new"); a question asking students to label the axes of the
  canonical performance-vs-amount-of-data plot; and true/false items on
  whether bigger data / bigger networks generally help performance.
  Source (mirror): https://raw.githubusercontent.com/Kulbear/deep-learning-coursera/master/Neural%20Networks%20and%20Deep%20Learning/Week%201%20Quiz%20-%20Introduction%20to%20deep%20learning.md

### 8. Structured-vs-unstructured data judgment
- Same CS230 Week 1 quiz: true/false items — is image data for cat
  recognition "structured" data? (false — it's unstructured); is tabular
  demographic/city data "unstructured"? (false — it's structured). This
  distinction is set up early because it's used all specialization-long to
  motivate why different architectures (CNNs, RNNs) suit different data.
  Source: https://raw.githubusercontent.com/Kulbear/deep-learning-coursera/master/Neural%20Networks%20and%20Deep%20Learning/Week%201%20Quiz%20-%20Introduction%20to%20deep%20learning.md

### 9. Analogy/rhetorical-framing interpretation
- CS230 Week 1 quiz, Question 1: asks what the "AI is the new electricity"
  analogy is meant to convey; correct choice is the one about AI
  transforming industries broadly the way electrification did roughly a
  century ago, as opposed to literal readings about power grids or devices.
  Source: https://raw.githubusercontent.com/Kulbear/deep-learning-coursera/master/Neural%20Networks%20and%20Deep%20Learning/Week%201%20Quiz%20-%20Introduction%20to%20deep%20learning.md

### 10. ML-workflow / iteration-mental-model judgment
- CS230 Week 1 quiz: true/false on whether experienced engineers can build a
  good model on the very first try without iterating, plus a multi-select on
  which statements about "iterating on ML ideas" are true (faster
  experimentation and faster compute both shorten the idea→code→experiment
  loop).
  Source: https://raw.githubusercontent.com/Kulbear/deep-learning-coursera/master/Neural%20Networks%20and%20Deep%20Learning/Week%201%20Quiz%20-%20Introduction%20to%20deep%20learning.md

### 11. True-objective-of-ML / generalization judgment
- **CS221 Lecture 4, "Machine learning III" (Sadigh, Spring 2018), opening
  clicker slide:** "What's the true objective of machine learning?" —
  options: minimize error on the training set / minimize training error with
  regularization / minimize error on unseen future examples / "learn about
  machines." This is posed *before* the lecture defines generalization, as a
  gut-check of the students' mental model.
  Source: https://web.stanford.edu/class/archive/cs/cs221/cs221.1186/lectures/learning3-6pp.pdf (Lecture 4, page 1)
- Later in the same deck, a "how can you reduce overfitting (select all that
  apply)" clicker question sits right at the boundary between conceptual and
  technical — included here because it's still framed as an intuition check
  rather than a derivation.

### 12. Ethical/social-impact scenario spotting
- **CS221, HW1 "Foundations," Problem 4: "Ethical Issue Spotting."**
  Instruction: for each scenario, write a "potential negative impacts
  statement" — determine whether the described algorithm/dataset/technique
  could have negative social impact or violate ethical norms. Four scenarios:
  (1) a loan-default model whose predictions show disparities by location,
  race, and gender; (2) stylometry technology that could be used to
  de-anonymize authors; (3) a facial-recognition dataset scraped without
  consent from copyrighted photos; (4) a plant-species model trained on
  consented iNaturalist photos (the "control" case, meant to show not
  everything is problematic). Notable because it's the one place in these
  four courses where "is this an appropriate/acceptable use of ML" is asked
  directly, with no math at all, in an assignment otherwise dominated by
  NumPy/einsum/gradient warmups.
  Source: https://stanford-cs221.github.io/autumn2025/assignments/hw1_foundations/index.html

### 13. Problem-framing-as-prose (present, but never becomes a graded exercise)
- **CS231n Module 0, "Image Classification" notes.** Frames the task
  explicitly as input (image, a 3D array of pixel values) → output (one
  label from a fixed category set), then lists the reasons this is hard for
  a computer despite being trivial for a human: viewpoint variation, scale
  variation, deformation, occlusion, illumination, background clutter, and
  intra-class variation — effectively the course's answer to "why do we need
  a learning/data-driven approach instead of hand-written rules," but stated
  as exposition, not as a question with a submittable answer.
  Source: https://cs231n.github.io/classification/
- **CS229 main lecture notes** similarly motivate supervised learning with
  the housing-price example and Mitchell's T/P/E definition in prose, but
  (confirmed by inspecting CS229 Problem Set #1 directly) none of that
  framing is echoed in a gradable question — the first problem set opens
  directly on Newton's method for least squares with no conceptual lead-in.
  Source: https://cs229.stanford.edu/main_notes.pdf ; https://see.stanford.edu/materials/aimlcs229/problemset1.pdf

---

## Per-course summary

- **CS229**: no conceptual exercises in its own problem sets (verified against
  PS#1, entirely derivation-based). Conceptual framing is prose-only in
  lecture notes. The functionally-equivalent public course for this archetype
  is Andrew Ng's Coursera "Machine Learning" (archetypes 1, 2, 3, 4) and its
  2022 successor "Machine Learning Specialization" (archetype 3).
- **CS230**: no separate intro problem set on the Stanford site; graded
  conceptual quiz lives entirely on the Coursera companion, Course 1 Week 1
  "Introduction to deep learning" (archetypes 7, 8, 9, 10).
- **CS231n**: Module 0 has framing prose (archetype 13) but no graded
  conceptual quiz; Lecture 1 slides cover history/applications overview but
  I found no embedded clicker-style question analogous to CS221's.
  (cs231n.github.io/classification/, cs231n.stanford.edu/slides/…)
- **CS221**: the richest source — genuine stand-alone conceptual exercises via
  live `cs221.stanford.edu/q` clicker slides embedded in Lecture 2-4
  (archetypes 3, 5, 6, 11), plus a distinctive ethics-scenario problem in
  HW1 "Foundations" (archetype 12) sitting alongside otherwise purely
  mathematical/programming content.
