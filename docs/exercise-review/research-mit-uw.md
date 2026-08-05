# MIT & UW: How intro ML/DL courses pose pre-math conceptual exercises

Scope: MIT 6.036/6.390 (Intro to ML), MIT 6.S191 (Intro to Deep Learning), UW CSE 446/546
(Machine Learning), UW CSE 599G1 (Intro to Deep Learning). Focus on the "what is ML, how do
you frame a problem as ML" stage, before real math/code.

## Access note (important caveat)

MIT's actual graded exercises for 6.036/6.390 live on the **MIT Open Learning Library (OLL)**,
an edX-based platform. Every exercise block I could locate for Week 0/Week 1 — the
self-assessment, "Week 1 Exercises: Hyperplanes," and "Homework 1" — returned the same wall:
*"This assessment is only available to enrolled learners, please sign in or register, and enroll
in this course to view the content."* So the verbatim intro-week question text for 6.036/6.390 is
not publicly fetchable; only the (ungated) lecture notes and past final exams are. UW CSE
446/546 homework PDFs, by contrast, are fully public on `courses.cs.washington.edu`. 6.S191 has
no problem-set-style exercises at all (see Archetype D) — its "conceptual check" is embedded in
the lecture narrative itself, not a separate graded assessment.

---

## Archetypes observed

1. **Problem-class taxonomy exposition** — a chapter/lecture lays out the five-to-eight standard
   "kinds of ML problem" (supervised: regression/classification; unsupervised: clustering/density
   estimation/dim. reduction; sequence learning; reinforcement learning; semi-supervised; active;
   transfer) as vocabulary the student is expected to internalize and later apply. This is the
   substrate every other archetype below builds on, more exposition than exercise.

2. **Six-characteristics problem-framing checklist** — a named framework (MIT 6.390 calls it
   "problem class / assumptions / evaluation criteria / model type / model class / algorithm")
   that students are told to run through when analyzing *any* ML problem, i.e. an explicit
   input/output/data/loss decomposition ritual.

3. **Short-answer + True/or/False conceptual check, "answerable without external materials"** —
   a graded but low-stakes block (no math derivation, no code) asking students to explain a
   concept in their own words or judge a T/F claim about how learning behaves, explicitly framed
   as requiring no outside lookup. Recurs verbatim-in-spirit across many UW CSE446/546 quarters
   and echoed in MIT 6.036 exam T/F-with-explain items.

4. **Nested-definition / Venn framing (AI ⊃ ML ⊃ DL)** — a single slide that defines the field by
   containment rather than by exercise: "any technique that mimics human behavior" ⊃ "learn
   without being explicitly programmed" ⊃ "extract patterns from data using neural networks."
   Functions as the course's de facto "what is this thing" self-check even though it isn't a
   graded question.

5. **Scenario classification ("is this supervised/unsupervised/RL?")** — the canonical
   problem-framing exercise everyone expects to find in week 1: a list of real-world scenarios
   the student must sort into ML problem classes. Confirmed as a live archetype in ML pedagogy
   generally (concrete verbatim example below is from Imperial College London, not MIT/UW,
   because the MIT instance is login-gated and I could not find a public UW instance of this
   specific exercise shape). Flagging this gap explicitly rather than fabricating MIT/UW text.

6. **Scale/history reasoning via a visual timeline** — not a quiz question but a slide sequence
   that puts three technology-capability snapshots side by side (e.g., a blurry 2015 face vs. a
   photorealistic 2018 face vs. a 2023 GPT-4 "wow moment" tweet) and lets the visible jump do the
   argumentative work, implicitly asking "what changed, and why now?"

---

## Examples by archetype

### 1. Problem-class taxonomy exposition

**MIT 6.390, Chapter 1: Introduction**
Source: https://introml.mit.edu/_static/fall23/LectureNotes/chapter_Introduction.pdf
(mirrored at https://openlearninglibrary.mit.edu/assets/courseware/v1/2481f8f2964716032b134db99e369b81/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Introduction.pdf)

Verbatim opening framing (the line every 6.036/6.390 student sees first):

> "The main focus of machine learning (ML) is *making decisions or predictions based on data*.
> ... in economics and psychology, the goal is to discover underlying causal processes and in
> statistics it is to find a model that fits a data set well. In those fields, the end product is
> a model. In machine learning, we often fit models, but as a means to the end of making good
> predictions or decisions."

And on the human-in-the-loop framing step (this is the closest the notes get to naming
"problem framing" as a skill, margin-noted "and often undervalued"):

> "A human still has to *frame* the problem: acquire and organize data, design a space of possible
> solutions, select a learning algorithm and its parameters, apply the algorithm to the data,
> validate the resulting solution to decide whether it's good enough to use, try to understand
> the impact on the people who will be affected by its deployment, etc."

Then it enumerates the five problem classes verbatim: Supervised (Regression, Classification),
Unsupervised (Clustering, Density estimation, Dimensionality reduction), Sequence learning,
Reinforcement learning, and "Other settings" (semi-supervised, active, transfer learning) —
each defined in one paragraph with a worked-through data notation ({(x^(1),y^(1)),...}).

A margin note embedded in the reinforcement-learning section is the one place the text pauses
to self-quiz the reader rather than just tell them something — a rhetorical "check your
understanding" aside rather than a graded question:

> "This paragraph actually talks about both a random variable and a realization of it, can you
> spot that from the notation and do you feel the difference?"

---

### 2. Six-characteristics problem-framing checklist

**MIT 6.390, Chapter 1, section "We can describe problems and their solutions using six
characteristics"**
Source: same PDF as above, p.6 (https://introml.mit.edu/_static/fall23/LectureNotes/chapter_Introduction.pdf)

Verbatim:

> "1. Problem class: What is the nature of the training data and what kinds of queries will be
> made at testing time?
> 2. Assumptions: What do we know about the source of the data or the form of the solution?
> 3. Evaluation criteria: What is the goal of the prediction or estimation system? How will the
> answers to individual queries be evaluated? How will the overall performance of the system be
> measured?
> 4. Model type: Will an intermediate model be made? ...
> 5. Model class: What particular class of models will be used? ...
> 6. Algorithm: What computational process will be used to fit the model to the data and/or to
> make predictions?"

This checklist is explicitly the "how do you frame a problem as ML" tool of the course — every
later chapter/homework in 6.390 revisits it ("problem class / assumptions / evaluation criteria"
= input+output+data, "model type/class/algorithm" = the solution side). I could not retrieve the
graded exercise that has students apply this checklist to a novel scenario (it lives behind the
OLL enrollment wall — see Access note), but the checklist itself is the pedagogical device this
task is asking about.

---

### 3. Short-answer + True/or/False conceptual check ("answerable without external materials")

**UW CSE 446/546: Machine Learning, Homework #1, Section "Short Answer and 'True or False'
Conceptual questions" (A.0)**
Source: https://courses.cs.washington.edu/courses/cse446/21sp/assignments/hw1.pdf (Prof. Sewoong
Oh / Prof. Simon Du, Spring 2021 — same section shape recurs across quarters, e.g. 20wi)

Verbatim:

> "A.0 The answers to these questions should be answerable without referring to external
> materials. Briefly justify your answers with a few words.
> a. In your own words, describe what bias and variance are? What is bias-variance tradeoff?
> b. What typically happens to bias and variance when the model complexity increases/decreases?
> c. True or False: The bias of a model increases as the amount of training data available
> increases.
> d. True or False: The variance of a model decreases as the amount of training data available
> increases.
> e. True or False: A learning algorithm will always generalize better if we use fewer features to
> represent our data.
> f. To obtain superior performance on new unseen data, should we use the train set or the test
> set to tune our hyperparameters?
> g. True or False: The training error of a function on the training set provides an overestimate
> of the true error of that function."

Note: this section is about bias/variance/generalization judgment rather than "what is ML"
problem-framing per se — UW's HW0 (the actual week-0/1 assignment) is pure prerequisite math
review (Bayes' rule, covariance, hyperplanes, rank; verified across 20wi
https://courses.cs.washington.edu/courses/cse446/20wi/hw0/hw0.pdf and other quarters) with **no**
conceptual ML-framing questions at all. The A.0-style block above is UW's recurring answer to
"how do we quiz concepts without math," even though it shows up in HW1, not HW0, and targets
generalization judgment rather than supervised/unsupervised/RL framing.

**MIT 6.036 Final Exam, Spring 2019, Q2 (Decision Trees), parts (d)-(f)** — same T/F+Explain
shape, cross-institution convergence, though this is end-of-course, not week 1:
Source: https://introml.mit.edu/_static/spring23/final/review/final_spring2019.pdf

Verbatim:

> "(d) Decision trees built using our greedy algorithm are a good choice of classifiers for
> images. ○ T ○ F Explain.
> (e) For decision trees built using our greedy algorithm, standardizing feature values is
> important. ○ T ○ F Explain.
> (f) A disadvantage of using decision trees for classification is that they can only be used to
> classify data having two classes. ○ T ○ F Explain."

---

### 4. Nested-definition / Venn framing (AI ⊃ ML ⊃ DL)

**MIT 6.S191, Lecture 1 slides, "What is Deep Learning?"**
Source: https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf (this URL always
points at the current offering's deck; content captured here is the Jan 2026 IAP run, but the
same three-tier definition slide has anchored the opening lecture every year since the course's
early offerings)

Verbatim text from the slide (three nested boxes):

> "ARTIFICIAL INTELLIGENCE — Any technique that enables computers to mimic human behavior
> MACHINE LEARNING — Ability to learn without explicitly being programmed
> DEEP LEARNING — Extract patterns from data using neural networks
> Teaching computers how to learn a task directly from raw data"

This is the single conceptual "exercise" 6.S191 offers at the pre-math stage — 6.S191 has **no
problem sets**; assessment is entirely coding labs (music generation, computer-vision bias
mitigation, LLM fine-tuning) plus a final project pitch. So the "framing" work in this course
happens once, in prose/diagram form, rather than as a recurring quiz mechanism. Confirmed via
course lecture schedule slide (same deck): labs are "Software Lab 1/2/3," no separate homework
track exists.

---

### 5. Scenario classification ("is this supervised/unsupervised/RL?")

I was unable to retrieve a verbatim MIT or UW instance of this exact, widely-expected archetype:
MIT's version is behind the OLL enrollment wall (Week 1 Exercises / Homework 1 for 6.036, see
Access note), and no UW CSE446/546/599 homework I could fetch contains it (their HW0s are math
review; HW1s are bias-variance, as above). The nearest fully public verbatim analog, for shape
comparison only (NOT MIT/UW):

**Imperial College London, COMP60012/70050 Intro to Machine Learning, Module 1 quiz**
Source: https://intro2ml.pages.doc.ic.ac.uk/autumn2024/modules/module1/quiz

Paraphrase (per search-result excerpt): students are given real-world scenarios — e.g.
"identifying distinct groups of people arriving at hospital A&Es with COVID to understand
case-mix," or "a book distributor wants to build a system to automatically classify new products
using previously classified book categories" — and asked to classify each as supervised,
unsupervised, or reinforcement learning.

---

### 6. Scale/history reasoning via a visual timeline

**MIT 6.S191, Lecture 1 slides, "'Seeing' the progress of deep learning throughout the years"**
Source: https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf

Three side-by-side captioned images: a blurry 2015 generated face (Goodfellow et al.), a
photorealistic 2018 generated face (Karras/Laine/Aila, StyleGAN), and a 2020 deepfake-style
Obama clip captioned "Hi everybody, and welcome to MIT 6.S191" — followed by a second slide,
"Language has transformed the way that we interact with deep learning," contrasting 2022
ChatGPT/GPT-3 launch with the 2023 "GPT-4 wow moment." No explicit question is attached; the
pedagogical move is entirely the juxtaposition, functioning as an implicit "notice the scale
jump, and ask yourself why now" prompt rather than a graded item.

---

## Sources consulted (not all yielded usable content)

- https://courses.csail.mit.edu/6.036/ (course listing)
- https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week1/intro_ml/ (lecture notes page, ungated)
- https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/welcome/week0_exercises_v3/ (gated — enrolled learners only)
- https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week1/week1_exercises/ (gated)
- https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week1/week1_homework/ (gated)
- https://introml.mit.edu/_static/fall23/LectureNotes/chapter_Introduction.pdf (ungated, used above)
- https://introml.mit.edu/_static/spring23/final/review/final_spring2019.pdf and 6_036_Final_Spring_21.pdf (ungated finals, used above)
- https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf (ungated, used above)
- https://courses.cs.washington.edu/courses/cse446/20wi/hw0/hw0.pdf (ungated, used above — pure math review, no conceptual ML-framing content)
- https://courses.cs.washington.edu/courses/cse446/21sp/assignments/hw1.pdf (ungated, used above)
- https://courses.cs.washington.edu/courses/cse599g1/18au/ and hw0.pdf (login-gated via UW NetID/CSE NetID SSO — could not fetch)
- https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/ (course listing only; assignments page not navigable via fetch)
- https://intro2ml.pages.doc.ic.ac.uk/autumn2024/modules/module1/quiz (non-MIT/UW comparison point, used above)
