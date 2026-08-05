# How classic ML/AI courses pose intro-chapter exercises

Research pass across AIMA, Murphy PML:AI, Bishop PRML, Goodfellow DL, Toronto
CSC311/321/413, NYU DL, and Google's ML Crash Course / Problem Framing / Rules
of ML. Read-only web research; all sources linked.

---

## Archetypes observed

1. **Boundary-drawing / "is this really AI?"** — Given a familiar system
   (bar-code scanner, spell-checker, search engine), decide whether it counts
   as AI/intelligence, and articulate what property (learning? reasoning?
   adaptivity?) makes the difference. Philosophical, not technical.

2. **Heuristic-vs-ML tradeoff** — Given a working (or proposed) heuristic,
   decide whether it's "good enough" or whether ML is justified, and
   articulate what the heuristic baseline even is before reaching for ML.
   This is the "don't be afraid to launch without ML" judgment call.

3. **Paradigm classification drill** — Given a one-paragraph business
   scenario, name the right ML paradigm/output type: supervised vs.
   unsupervised, regression vs. classification (binary/multiclass), or
   clustering/generative. Usually multiple-choice, self-checking.

4. **Problem-framing worksheet** — A structured, multi-step template (not a
   single question) that forces the student to write down, for their *own*
   chosen problem: the plain-language goal, the ideal outcome independent of
   the model, success/failure metrics, the output type and its shape, when
   and how the output is consumed (latency/serving constraints), the
   heuristic fallback, and finally the input data design (schema,
   provenance, which inputs are cheaply obtainable). This is the richest
   archetype for "input/output/data/label availability" framing.

5. **Label/feature/data-quality judgment** — Given a dataset description,
   judge whether it's fit to train on (representative? correct values?
   features with predictive power? not gathered from unpredictable sources?)
   or distinguish a label from a feature.

6. **Feasibility / state-of-the-art judgment** — Given a list of real-world
   tasks, judge which are currently solvable by AI/ML and which aren't, and
   predict when the gap will close. Tests judgment about current capability
   frontiers rather than problem structure per se.

7. **Negative finding: pure-math or no-exercise chapters** — Several
   flagship intro chapters either have zero exercises, or have exercises
   that are 100% mathematical derivations with no conceptual/framing content
   at all. Documented below since the task asked to note this explicitly.

---

## 1. Boundary-drawing / "is this really AI?"

**Source: Russell & Norvig, *AIMA*, Chapter 1 exercises** (all 20, official
digitized set)
https://github.com/aimacode/aima-exercises/blob/master/markdown/1-Introduction/README.md
(raw questions: `markdown/1-Introduction/exercises/ex_N/question.md`, N=1..20)

Verbatim highlights:

- **Ex. 9 / Ex. 10** — "To what extent are the following computer systems
  instances of artificial intelligence: Supermarket bar code scanners. Web
  search engines. Voice-activated telephone menus. Internet routing
  algorithms that respond dynamically to the state of the network." (Ex. 10
  swaps in "Spelling and grammar correction features in Microsoft Word.")
- **Ex. 4** — "Are reflex actions (such as flinching from a hot stove)
  rational? Are they intelligent?"
- **Ex. 15/16/17** — A chained trio of "surely X cannot be intelligent — it
  can only do what [its programmer / its genes / the laws of physics] tell
  it to." Is the premise true, and does it imply the conclusion? (computers
  → animals → all physical systems, forcing the student to see the
  determinism argument proves too much if valid at all)
- **Ex. 14** — "Is AI a science, or is it engineering? Or neither or both?
  Explain."
- **Ex. 1** — "Define in your own words: (a) intelligence, (b) artificial
  intelligence, (c) agent, (d) rationality, (e) logical reasoning."
- **Ex. 2/3** — Turing Test: which of Turing's objections still carry
  weight, and research the latest Loebner Prize winner.

These are essay/discussion prompts (no numeric answer), meant to be argued
in class — closer to philosophy-of-mind homework than problem-set exercises.

---

## 2. Heuristic-vs-ML tradeoff

**Source: Google, "Rules of Machine Learning"**
https://developers.google.com/machine-learning/guides/rules-of-ml

- **Rule #1**: "Don't be afraid to launch a product without machine
  learning." — simple heuristics (recency ranking, spam blocklists,
  alphabetical order) often capture most of the value before any model
  exists; wait until you've exhausted heuristics *and* have data.
- **Rule #3**: "Choose machine learning over a complex heuristic" — the
  threshold isn't simplicity, it's maintainability: a heuristic that's grown
  too many special cases is the actual signal to switch to ML.

**Source: Google, Problem Framing — "Understand the problem," Check-Your-Understanding**
https://developers.google.com/machine-learning/problem-framing/problem

- Q1 (verbatim): "Why is it important to have a non-ML solution or heuristic
  in place before analyzing an ML solution?" (Answer: it's the benchmark an
  ML solution must beat.)
- Q3, scenario-based: a call-center team wants to predict hold time using a
  heuristic ("`customers_on_hold / employees_answering × 10 min`") vs.
  training on columns like `number_of_callcenter_phones`, `user_issue`,
  `time_to_resolve`, `call_time`, `time_on_hold` — "determine if using ML is
  the best approach," with rationale required either way.

**Source: Google, Problem Framing — "Try It Yourself: Framing," Exercise 6**
https://developers.google.com/machine-learning/problem-framing/try-it/framing-exercise
(prompts live inside an embedded iframe; fetched directly)

- "Exercise 6: Your Heuristics — Write how you would solve the problem if
  you didn't use ML. For example, what heuristics you might use." Tip:
  "Think about a scenario where you need to deliver the product tomorrow,
  and you can only hardcode the business logic. What would you do?"

---

## 3. Paradigm classification drill

**Source: Google, "Test Your Understanding" (intro-to-ml)**
https://developers.google.com/machine-learning/intro-to-ml/understanding

Verbatim multiple-choice items:

- "If you wanted to understand the types of users that visit the site,
  would you use supervised or unsupervised learning?"
- "What type of ML would you use to predict kilowatt hours used per year
  for a newly constructed house?" (supervised vs. unsupervised)
- "If you wanted to predict the cost of an airplane ticket, would you use
  regression or classification?"
- "Could you train a classification model to classify airplane ticket cost
  as 'high,' 'average,' or 'low'?" (tests understanding that regression
  targets can be discretized into classification targets)
- "Which three features do you think are likely the greatest predictors for
  a car's price?" (feature relevance judgment, multiple-choice)

**Source: Google, Problem Framing — "Try It Yourself: Formulating a
solution," Exercise 7a**
https://developers.google.com/machine-learning/problem-framing/try-it/formulate-exercise

- Checkbox taxonomy the student must apply to their own problem: "Binary
  classification / Unidimensional regression / Multi-class single-label
  classification / Multi-class multi-label classification / Multidimensional
  regression / Clustering (unsupervised) / Generative AI / Other," followed
  by "which predicts or generates: ___". Exercise 7b then asks the student
  to re-cast their real problem as the *simplest* possible
  binary-classification-or-regression formulation.

---

## 4. Problem-framing worksheet (the richest archetype)

**Source: Google, Problem Framing — "Try It Yourself: Framing" (Exercises
1–6) and "Formulating a Solution" (Exercises 7–10)**
https://developers.google.com/machine-learning/problem-framing/try-it/framing-exercise
https://developers.google.com/machine-learning/problem-framing/try-it/formulate-exercise

Framed around a running example ("build a model to identify whether an
email in Gmail is 'important'") but designed for the student's own project.
Verbatim prompts (fill-in-the-blank, not multiple choice):

1. **Start Clearly and Simply** — "Write what you'd like the machine
   learned model to do." → "We want the machine learned model to: ___"
2. **Your Ideal Outcome** — "What is this outcome, independent of the model
   itself?" (explicitly *not* the model's own accuracy metric)
3. **Your Success Metrics** — success metrics *and* failure metrics,
   "phrased independently of evaluation metrics for the model... don't talk
   about precision, recall, or AUC."
4. **Your Output** — "The output from our ML model will be: ___", then pick
   one of: Unidimensional regression / Multidimensional regression / Binary
   classification / Multiclass classification / Generate text, image,
   audio, video, or multimodal.
5. **Using the Output** — when the output is needed and how it's consumed
   ("Will it be presented immediately to the user in a UI? ... What latency
   requirements do you have?"), plus the "Oracle Test: assume you always
   had the correct answer. How would you use that in your product?"
6. **Your Heuristics** — the non-ML fallback (see archetype 2 above).
7. **7a/7b** — paradigm classification + simplest-possible reformulation
   (see archetype 3 above).
8. **Design your Data for the Model** — fill in a blank input1/input2/
   input3/output table; tips stress "only include information available at
   the moment the prediction is made" and to flatten/avoid nested structure.
9. **Where the Data Comes From** — for each input, when it becomes
   available, and whether it's obtainable in the same format at serving
   time as at training time.
10. **Easily Obtained Inputs** — pick 1–3 cheap inputs likely to give "a
    reasonable, initial outcome," tying back to the heuristics from
    Exercise 6.

This is the single closest match in the wild to the "input/output/data/loss/
label availability" framing the task asked about — it's explicitly a
worksheet rather than a graded quiz, meant to be filled in per-project.

---

## 5. Label/feature/data-quality judgment

**Source: Google, Problem Framing — "Understand the problem"**
https://developers.google.com/machine-learning/problem-framing/problem

- Q2 (verbatim): "When analyzing your datasets, what are three key
  attributes you should look for?" Options: representative of the real
  world; contains correct values; features have predictive power for the
  label; small enough to load onto a local machine; gathered from a variety
  of unpredictable sources. (Correct: first three — the last two are
  distractors.)

**Source: Google, Problem Framing — "Framing an ML problem"**
https://developers.google.com/machine-learning/problem-framing/ml-framing

- Proxy-label judgment (verbatim scenario): a health/well-being app "wants
  to help people feel better" — must it use proxy labels? (Answer: yes,
  because happiness/well-being can't be measured directly and must be
  approximated via e.g. exercise hours or time with friends.)
- A fashion-firm scenario with two competing framings of ideal
  outcome/model goal/output/success-metric, where the student must spot
  which framing is coherent (Option A: predict `in_fashion` vs.
  `not_in_fashion` to decide manufacturing → coherent; Option B conflates
  "how much to manufacture" with a binary label → incoherent).

---

## 6. Feasibility / state-of-the-art judgment

**Source: Russell & Norvig, *AIMA*, Chapter 1, Exercises 18–19**
https://github.com/aimacode/aima-exercises/blob/master/markdown/1-Introduction/exercises/ex_18/question.md

- Ex. 18 (verbatim, abridged list): "Examine the AI literature to discover
  whether the following tasks can currently be solved by computers: Playing
  a decent game of table tennis... Driving in the center of Cairo, Egypt...
  Driving in Victorville, California... Buying a week's worth of groceries
  on the Web... Playing a decent game of bridge at a competitive level...
  Writing an intentionally funny story... Translating spoken English into
  spoken Swedish in real time... Performing a complex surgical operation."
- Ex. 19: "For the currently infeasible tasks, try to find out what the
  difficulties are and predict when, if ever, they will be overcome."

This is a distinct flavor from "problem framing" — it's judging *current
capability* against a fixed task list rather than decomposing a new
problem, but it's the clearest "when is AI/ML appropriate today" exercise
in AIMA.

---

## 7. Negative findings — no conceptual framing exercises here

- **Goodfellow, Bengio & Courville, *Deep Learning*** — confirmed via
  multiple secondary sources (book reviews, reader notes) that the book has
  **no end-of-chapter exercises at all**, intro chapter or otherwise. E.g.
  https://link.springer.com/content/pdf/10.1007/s10710-017-9314-z.pdf
  (book review noting the absence explicitly).

- **Kevin Murphy, *Probabilistic Machine Learning: An Introduction*,
  Chapter 1** — confirmed via the official table of contents
  (https://probml.github.io/pml-book/toc1.pdf) that Chapter 1 ("Introduction,"
  covering 1.1 What is machine learning? through 1.6.3 Caveats) has **no
  "Exercises" subsection** — it goes straight from §1.6 Discussion into Part
  I / Chapter 2. By contrast, Chapter 2 has "2.9 Exercises" (p.71) and
  Chapter 3 has "3.7 Exercises" (p.100). This is corroborated by the
  official partial solution manual
  (https://probml.github.io/pml-book/solns-public.pdf), whose first
  numbered solution is "2.2 Pairwise independence does not imply mutual
  independence" — i.e., solutions begin at Chapter 2, skipping Chapter 1
  entirely.

- **Bishop, *Pattern Recognition and Machine Learning*, Chapter 1** — has a
  full problem set (Exercises 1.1 through ~1.41, per
  https://github.com/thesstefan/bishop_prml and
  https://github.com/abhimanyu-jain/PRML_Solutions), but every exercise is a
  **mathematical derivation**: polynomial curve-fitting and regularized
  least squares (1.1–1.2), probability identities and Bayes' rule (1.3–1.6),
  Gaussian normalization/moments (1.7–1.16), volume of a sphere / Gamma
  function identities (1.17–1.23), decision theory — minimizing expected
  loss, the reject option, entropy of decision regions (1.24–1.28), and
  information theory / entropy / KL-divergence proofs (1.29–1.41). None of
  them touch problem framing, paradigm choice, or "is ML appropriate" —
  Bishop's Ch.1 exercises are purely about the math introduced in that
  chapter, not about ML-as-a-discipline judgment calls.

- **Toronto CSC311 (Grosse et al.), "Introduction to Machine Learning,"
  Homework 1** —
  https://www.cs.toronto.edu/~rgrosse/courses/csc311_f21/homework/hw1.pdf —
  checked directly: Q1 is a derivation of the curse-of-dimensionality
  (expectation/variance of squared distance in high-dim cubes) plus a
  simulation; Q2 is a coding exercise (`sklearn` `DecisionTreeClassifier` on
  real-vs-fake headlines, computing information gain); Q3 is a regularized
  linear regression derivation. **No conceptual "what is
  supervised/unsupervised learning" or "is ML appropriate" exercise appears
  in the graded homework** — CSC311/321/413 problem sets are uniformly
  math-derivation + coding, and framing-level discussion (if any) appears
  to live only in ungraded lecture material, not in citable assignment
  documents.

- **NYU Deep Learning (LeCun & Canziani), Week 1 practicum** —
  https://atcold.github.io/NYU-DLSP20/en/week01/01/ — the practicum content
  is entirely technical/hands-on (visualizing linear/non-linear
  transformations of data with PyTorch notebooks); no conceptual quiz or
  discussion question about ML history, learning paradigms, or motivation
  was found on this page.

---

## Notes on sourcing

- AIMA exercise text pulled directly from the aimacode markdown source
  (raw.githubusercontent.com), all 20 exercises fetched in full — quotes
  above are exact/verbatim.
- The Google Problem Framing "Try It Yourself" prompts are rendered inside
  an embedded `<iframe>` that a plain page fetch does not surface; the
  iframe source HTML was fetched directly to recover verbatim prompt text.
- The classic MLCC "Framing: Check Your Understanding" spam-classification
  quiz (labels vs. features, "unlabeled examples") that older write-ups
  reference appears to have been retired/merged when Google restructured
  the crash course into "Foundational courses" — the live
  `crash-course/framing/check-your-understanding` URL now serves the same
  car-price/quiz content as `intro-to-ml/understanding` (cited under
  Archetype 3). I was unable to retrieve the old spam-example quiz verbatim
  (web.archive.org is not fetchable from this environment); only the
  "Key ML Terminology" spam *example* (not a quiz) is confirmed still to
  exist, mirrored at
  https://github.com/litaotao/machine-learning-crash-course.
