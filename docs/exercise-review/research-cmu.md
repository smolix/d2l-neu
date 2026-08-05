# CMU intro-ML exercises + Mitchell (1997) Ch.1: conceptual, pre-math, pre-code

Scope: exercises/questions that ask students to frame a problem as ML (T/P/E),
classify a scenario (supervised/unsupervised/RL), judge whether ML is/isn't an
appropriate approach, or identify components of a learning system. Excludes
probability/linear-algebra/calculus prereq drills and programming assignments,
even when bundled into the same homework PDF.

Key finding up front: **at CMU, the T/P/E ("well-posed learning problem")
exercise lives almost entirely in the Lecture 1 in-class exercise, never in a
graded homework.** I checked three eras of 10-601 Homework 1 — Fall 2011
(Kolar/Bishop), Fall 2012 (O'Connor), and Fall 2025 (Gormley) — spanning 14
years and totally different staff, and every single one is 100% probability /
linear algebra / calculus / CS-foundations / decision-tree-math, with zero T/P/E
or supervised-vs-unsupervised questions. This is the same pattern the
Stanford-side research found for CS229 PS#1. The conceptual "what is a
well-posed learning problem" content is real and central at CMU — it's an
explicit, numbered Learning Objective of Lecture 1 — but it's assessed live,
in class, not on paper. Tom Mitchell's textbook (which CMU's own course grew
out of, and which is legally hosted on CMU's CS website by Mitchell himself)
is where the fully-specified, gradable version of this exercise actually
appears in writing.

---

## Archetypes observed

1. **T/P/E in-class exercise (live, ungraded)** — instructor presents a list
   of candidate real-world tasks; in small groups/individually, students pick
   one task T and must identify P and E for it, then report back verbally.
   No written submission, no rubric — a warm-up discussion exercise, not
   homework.
2. **T/P/E worked examples (textbook exposition)** — Mitchell's book doesn't
   ask a question here; it walks through 2-3 complete `<T, P, E>` triples
   (checkers, handwriting recognition, robot driving) as worked examples
   before asking students to do it themselves in the exercises.
3. **Generate-your-own T/P/E formulation (open-ended written exercise)** —
   pick an unfamiliar learning task, state T/P/E precisely, then also propose
   a target function and representation, and discuss the tradeoffs — a
   textbook end-of-chapter exercise, not a course homework.
4. **Appropriate-vs-inappropriate ML application judgment (open-ended)** —
   name three applications where ML is a good fit and three where it isn't,
   each with a one-sentence justification — no scenario is given to classify;
   students must generate their own examples in both directions.
5. **Design-choice tradeoff reasoning (component-level, non-mathematical)** —
   given several candidate strategies for *how a system should generate its
   own training experience* (e.g., random self-play vs. replaying unplayed
   branches of past games vs. a student-designed strategy), discuss the
   tradeoffs and predict which works best under a fixed data budget. This is
   reasoning about the "E" component design, not calculation.
6. **Supervised-vs-unsupervised / "what learning paradigm is this" judgment**
   — given a concrete scenario, name the learning paradigm it belongs to and
   justify why. At CMU this shows up as a graded exercise only late in the
   course (Homework 9, "Learning Paradigms," in a recommender-systems
   context), not as an intro-week exercise — the reverse of Stanford's CS229-
   adjacent Coursera course, where this is a Week 1 quiz item.
7. **(Negative pattern) "Background" homework is 100% math/CS prereqs** —
   across 2011, 2012, and 2025 versions of CMU 10-601's Homework 1/0, the
   entire graded written assignment is probability, linear algebra, calculus,
   Big-O/CS-foundations, and (in the older versions) decision-tree
   implementation math — never a single T/P/E or paradigm-classification
   item, despite those being explicit Lecture 1 learning objectives.
8. **Stated learning objective as an explicit conceptual competency** — the
   course syllabus/slides name "formulate a well-posed learning problem" and
   "describe common learning paradigms" as numbered, testable learning
   objectives of the very first lecture — i.e., CMU treats this as important
   enough to state as a competency goal, even though (per archetype 7) it
   isn't the subject of graded written homework.
9. **ML/DL/AI hierarchy + comparison-table framing (pre-quiz recitation)** —
   11-785 (Intro to Deep Learning) opens with "What is Deep Learning?": a
   nested Venn diagram (AI ⊃ ML ⊃ DL ⊃ Generative AI) and a side-by-side
   table contrasting ML vs. DL on data type, feature engineering, model
   complexity, and compute — framing exposition tied to a graded "Quiz 1,"
   whose actual questions are Canvas-gated and not publicly viewable.

---

## Examples by archetype

### 1. T/P/E in-class exercise (live, ungraded)
- **CMU 10-601/10-301 "Introduction to Machine Learning," Lecture 1 —
  "Well-Posed Learning Problems: In-Class Exercise"** (Matt Gormley, Fall
  2025 slide deck, slides 35-37 of 70). Verbatim instructions:
  > "1. Select a task, T
  > 2. Identify performance measure, P
  > 3. Identify experience, E
  > 4. Report ideas back to rest of class"

  Immediately followed by a menu of candidate tasks (explicitly credited
  "Examples from Roni Rosenfeld," a prior CMU instructor — i.e. this is a
  recurring, multi-instructor CMU tradition, not a one-off):
  > "Identify objects in an image · Translate from one human language to
  > another · Recognize speech · Assess risk (e.g. in loan application) ·
  > Make decisions (e.g. in loan application) · Assess potential (e.g. in
  > admission decisions) · Categorize a complex situation (e.g. medical
  > diagnosis) · Predict outcome (e.g. medical prognosis, stock prices,
  > inflation, temperature) · Predict events (default on loans, quitting
  > school, war) · Plan ahead under perfect knowledge (chess) · Plan ahead
  > under partial knowledge (poker, bridge)"

  Two follow-up slides just repeat the same three blank prompts
  ("task, T / performance measure, P / experience, E") for the class to
  fill in live.
  Source: https://www.cs.cmu.edu/~mgormley/courses/10601-f25/slides/lecture1-overview.pdf
  (slides 35-37; same course also hosted at http://mlcourse.org)

### 2. T/P/E worked examples (textbook exposition)
- **Tom M. Mitchell, *Machine Learning* (McGraw-Hill, 1997), Section 1.1
  "Well-Posed Learning Problems."** Formal definition, verbatim:
  > "A computer program is said to learn from experience E with respect to
  > some class of tasks T and performance measure P, if its performance at
  > tasks in T, as measured by P, improves with experience E."

  Followed by three fully-specified worked examples, verbatim:
  > "A checkers learning problem: Task T: playing checkers · Performance
  > measure P: percent of games won against opponents · Training experience
  > E: playing practice games against itself"
  >
  > "A handwriting recognition learning problem: Task T: recognizing and
  > classifying handwritten words within images · Performance measure P:
  > percent of words correctly classified · Training experience E: a
  > database of handwritten words with given classifications"
  >
  > "A robot driving learning problem: Task T: driving on public four-lane
  > highways using vision sensors · Performance measure P: average distance
  > traveled before an error (as judged by human overseer) · Training
  > experience E: a sequence of images and steering commands recorded while
  > observing a human driver"

  Source (official, hosted by the author on CMU's site):
  https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
  (Chapter 1, pp. 2-4 of the printed book)

### 3. Generate-your-own T/P/E formulation
- **Mitchell (1997), Chapter 1, Exercise 1.2**, verbatim:
  > "Pick some learning task not mentioned in this chapter. Describe it
  > informally in a paragraph in English. Now describe it by stating as
  > precisely as possible the task, performance measure, and training
  > experience. Finally, propose a target function to be learned and a
  > target representation. Discuss the main tradeoffs you considered in
  > formulating this learning task."
  Source: https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
  (Chapter 1 "EXERCISES," printed book p. 18)

### 4. Appropriate-vs-inappropriate ML application judgment
- **Mitchell (1997), Chapter 1, Exercise 1.1**, verbatim:
  > "Give three computer applications for which machine learning approaches
  > seem appropriate and three for which they seem inappropriate. Pick
  > applications that are not already mentioned in this chapter, and include
  > a one-sentence justification for each."
  Source: https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
  (Chapter 1 "EXERCISES," printed book p. 18)

### 5. Design-choice tradeoff reasoning (component-level, non-mathematical)
- **Mitchell (1997), Chapter 1, Exercise 1.4**, verbatim:
  > "Consider alternative strategies for the Experiment Generator module of
  > Figure 1.2. In particular, consider strategies in which the Experiment
  > Generator suggests new board positions by [i] Generating random legal
  > board positions [ii] Generating a position by picking a board state from
  > the previous game, then applying one of the moves that was not executed
  > [iii] A strategy of your own design. Discuss tradeoffs among these
  > strategies. Which do you feel would work best if the number of training
  > examples was held constant, given the performance measure of winning the
  > most games at the world championships?"
  Notable because it's reasoning about *how you'd design the source of
  experience E*, entirely in prose — no equations required, even though it
  sits in the same "EXERCISES" list as the LMS gradient-descent proof
  (Exercise 1.3, math, excluded from this survey) and the tic-tac-toe coding
  exercise (Exercise 1.5, coding, excluded).
  Source: https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
  (Chapter 1 "EXERCISES," printed book p. 18)

### 6. Supervised-vs-unsupervised / "what learning paradigm is this" judgment
- **CMU 10-301/10-601, Homework 9 Recitation, "Learning Paradigms"**
  (Gormley course, 12/1/2025), Section 4.2 "Content-Based Filtering,"
  Q1, verbatim:
  > "Suppose we are trying to recommend movies to a user. We are given a
  > feature vector for each movie with content information such as year of
  > release and genre, and for movies the user has watched, we are given
  > labels for whether or not they liked the movie. What learning paradigm
  > is suited for our recommendation task?"
  > Answer given: "Supervised learning. Train a model on the features and
  > labels and make predictions on unseen movies."

  Same document, Section 2 "K-Means," opens with a definitional framing of
  the same archetype in the other direction:
  > "Clustering is an example of unsupervised machine learning algorithm
  > because it serves to partition unlabeled data."

  This is the closest thing to Stanford's Week-1 "classify the paradigm"
  quiz item that I could find in CMU's 10-601 materials — but it's HW9 (the
  *last* homework, covering PCA/K-means/ensembles/recommenders), not an
  intro-week exercise. The paradigm-classification competency is named as a
  Lecture 1 learning objective (archetype 8) well before it's ever tested.
  Source: https://www.cs.cmu.edu/~mgormley/courses/10601-f25/handout/hw9_recitation_solution.pdf
  (pp. 5, 18-19 of 19)

### 7. (Negative pattern) "Background" homework is 100% math/CS prereqs
- **10-601, Fall 2011 Homework 1** ([Mladen Kolar / William Bishop], 60
  points total): Part 1 is pure probability review (biased-die win
  probability, conditional probability, expected value of an indicator
  function, entropy/mutual-information proofs); Part 2 is implementing and
  pruning a binary decision tree on the Wisconsin Breast Cancer dataset.
  Zero conceptual T/P/E or paradigm-classification content.
  Source: https://www.cs.cmu.edu/~aarti/Class/10601/homeworks/hw1.pdf
- **10-601, Fall 2012 Problem Set 1** (Brendan O'Connor, TA): probability
  review (Bayes' rule proof, contingency tables, chain rule, total
  probability), then a Titanic-survival decision-tree exercise (information
  gain, boolean-formula equivalence, model-complexity-vs-data-size
  reasoning), then MLE/MAP estimation of a Bernoulli parameter. Again zero
  T/P/E or paradigm-classification content — the closest it gets is asking
  *when* the simpler vs. more complex model is better, which is about
  overfitting, not problem framing.
  Source: https://www.cs.cmu.edu/~tom/10601_fall2012/hw/hw1.pdf
- **10-301/10-601, Fall 2025 "Homework 1: Background"** (Gormley course,
  68 written points): sections are LaTeX/template compliance, Course
  Policies (quiz on the syllabus itself), Probability and Statistics,
  Linear Algebra, Calculus, Geometry, and "CS Foundations" (Big-O,
  recursion vs. memoization, tree search) — entirely multiple-
  choice/select-all/fill-in math and course-logistics questions. No T/P/E,
  no supervised/unsupervised item anywhere in the 24-page handout.
  Source: https://www.cs.cmu.edu/~mgormley/courses/10601-f25/homework/hw1.zip
  (extracts to `hw1.pdf`)
- Corroborating data point: 10-315 ("Introduction to Machine Learning," the
  parallel/Qatar-linked course using much of the same staff and materials)
  titles its own intro assignment "Homework 0: Mathematical Background for
  Machine Learning" — the title alone confirms the same pattern holds there.
  Source: https://www.coursehero.com/file/49922539/10315HW0Solpdf/ (title
  only; full text paywalled)

### 8. Stated learning objective as an explicit conceptual competency
- **CMU 10-601/10-301, Lecture 1 slides, "Learning Objectives"** (final
  slide of the deck), verbatim, items 1-2:
  > "You should be able to…
  > 1. Formulate a well-posed learning problem for a real-world task by
  > identifying the task, performance measure, and training experience
  > 2. Describe common learning paradigms in terms of the type of data
  > available, when it's available, the form of prediction, and the
  > structure of the output prediction"
  Notable as the explicit textual link between archetype 1 (the live
  exercise) and archetype 6 (the paradigm-classification competency) — both
  are named as goals of lecture *one*, even though only the first is
  practiced that same day and the second isn't graded until homework nine.
  Source: https://www.cs.cmu.edu/~mgormley/courses/10601-f25/slides/lecture1-overview.pdf
  (final slide, slide 69 of 70)
- The equivalent framing survives, nearly word for word, in CMU 10-701
  ("Introduction to Machine Learning," PhD-level parallel course): its own
  syllabus opens with "how can we build adaptive algorithms that
  automatically improve their performance (on a given task) as they acquire
  more experience?" and lists "compare and contrast different paradigms for
  learning (supervised, unsupervised, etc.)" as a course learning outcome —
  the same T/P/E-plus-paradigms framing, one level up in the curriculum.
  Source: https://machinelearningcmu.github.io/F23-10701/ ;
  (an older instance of the same course: https://alex.smola.org/teaching/cmu2013-10-701/)

### 9. ML/DL/AI hierarchy + comparison-table framing (pre-quiz recitation)
- **11-785/685/485 "Introduction to Deep Learning," Lab 01 / Recitation 1**
  (Aug 30, 2024), slides titled "What is Deep Learning?": a nested Venn
  diagram (Artificial Intelligence ⊃ Machine Learning ⊃ Deep Learning ⊃
  Generative AI) and a definitional pair —
  > "Machine Learning is subset of AI that employs algorithms to analyze and
  > learn from data, enabling systems to make predictions or decisions
  > without being explicitly programmed for specific tasks... Deep Learning
  > is a subset of machine learning that uses neural networks with multiple
  > layers (deep neural networks) to model complex patterns in large
  > datasets."

  plus a "Key Differences" table (data type: structured vs.
  structured+unstructured; feature engineering: manual vs. automatic;
  model complexity: simpler/interpretable vs. complex/"black box"; compute:
  low vs. GPU/TPU-heavy) — structurally identical to Stanford CS230's
  Week-1 quiz distinctions (see Stanford research file, archetype 8), but
  presented here as recitation exposition rather than a gradable item.
  The same slide deck's announcements confirm a graded, Canvas-hosted "HW1
  Quiz" and a separate "Quiz 1" due that week, explicitly on "What is Deep
  Learning?" — but neither quiz's actual questions are publicly accessible
  (Canvas-gated), so I could not retrieve verbatim text for the graded
  version of this archetype at 11-785.
  Source: https://deeplearning.cs.cmu.edu/F24/document/recitation/Recitation_1_F24.pdf

---

## Per-course/source summary

- **10-601/10-301 (Gormley, current)**: the *only* CMU source with a
  genuine, dedicated T/P/E exercise (archetype 1), but it's a live,
  ungraded, in-class discussion exercise on day one, not homework. Homework
  1 itself (checked across 2011, 2012, 2025 instructors) is 100% math/CS
  prereqs (archetype 7). The supervised/unsupervised paradigm-classification
  competency is named as a Lecture 1 objective (archetype 8) but not graded
  until Homework 9, at the very end of the course (archetype 6).
- **10-315**: parallel intro course; its own "Homework 0" is explicitly
  titled a mathematical-background test, consistent with 10-601's pattern.
  Blank assignment PDFs for HW3/5/7/10 are public
  (cs.cmu.edu/~10315/assignments/) but none for HW0/1, and none I could
  access contain conceptual framing questions.
- **10-701 ("Introduction to Machine Learning," PhD-parallel, incl. an
  older Alex Smola-taught instance)**: same T/P/E-plus-paradigms framing
  appears in the syllabus prose (archetype 8's second example), but I found
  no evidence of a stand-alone graded conceptual exercise in its problem
  sets — consistent with the CS229 pattern on the Stanford side.
- **11-785 (Introduction to Deep Learning)**: has the right *topic* at the
  right *time* — a "Quiz 1: What is Deep Learning?" in week 1 — and public
  recitation slides show the intended conceptual content (archetype 9), but
  the graded quiz itself is Canvas-gated, so no verbatim question text was
  retrievable.
- **Tom Mitchell, *Machine Learning* (1997), Chapter 1**: the definitive,
  fully public, verbatim source for this entire task. Section 1.1 supplies
  the worked T/P/E examples (archetype 2); the chapter's own "EXERCISES"
  supply the generate-your-own-T/P/E exercise (archetype 3), the
  appropriate/inappropriate-application exercise (archetype 4), and the
  non-mathematical experience-design tradeoff exercise (archetype 5) — the
  three exercises in this chapter that are conceptual rather than
  mathematical/coding (Exercises 1.3 and 1.5 are LMS-gradient-descent proof
  and tic-tac-toe coding, respectively, and are out of scope here).
  Source (full text, hosted by the author): https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
