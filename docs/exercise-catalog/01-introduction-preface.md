# Chapter Overview — Preface & Introduction

Best external sources: Mitchell (1997) Ch.1 is the single most reusable
source in this whole survey — a fully public, verbatim T/P/E framework with
two directly adaptable exercises (1.1, 1.2) that map almost exactly onto the
book's own "four components" vocabulary. Berkeley CS189/289A's Fall 2025
Lecture 1 supplies the richest *structured triage* device (Engineering vs.
ML vs. Human problem) for judging when ML is appropriate at all. Google's
Problem-Framing worksheet is the richest *deliverable-shaped* archetype
(Ideal Outcome, Oracle Test, Heuristics) but is aimed at a student's own
project, so only individual steps transplant cleanly. Stanford CS221
contributes both a scale-estimation clicker item and the only fully public,
math-free ethics-scenario homework problem (HW1 "Ethical Issue Spotting").
Coverage gaps: none of the five research files found a discrete *exercise*
(vs. reading/blog material) built around feedback loops or distribution
shift — Berkeley's own researchers flag this gap explicitly. Nobody outside
this book has an exercise on dataset/memory/FLOPs growth tables or on
"where is end-to-end training not yet the default" — those stay original.
Preface-style onboarding actions (forum registration, environment install)
have *no* external exercise tradition at all — expected, since that's
book-logistics, not ML pedagogy. Introduction's existing 4 exercises are
already appropriately code-free and conceptual (matching the field-wide
pattern that intro material is argued in prose/discussion, not derived);
2 of 4 already clear the bar and are kept, 2 were think-about-it prompts
that needed a deliverable. Preface's 3 exercises are structurally sound
onboarding actions; only the "follow the links" item lacked a produced
artifact and was rewritten.

---

## chapter_introduction/index.md — Introduction

**Topic:** Framing an unfamiliar task as a machine-learning problem — the
four components (data, model, objective function, algorithm), the
supervised-learning taxonomy (regression/classification/tagging/search/
recsys/seq2seq), unsupervised and self-supervised learning, RL and
environment interaction, the field's history, and the decade-by-decade
data/memory/compute growth table.

**Current exercises:** 4; disposition: keep 2, rewrite 2, drop 0 — ex. 3
("relationships between algorithms, data, and computation") and ex. 4
("settings where end-to-end training isn't yet default") already name a
concrete deliverable and were not flagged by the style review; ex. 1 and
ex. 2 ("which code could be learned" / "which problems lack an explicit
procedure") were flagged as reflective think-about-it prompts with no
stated artifact, so both are rewritten with an explicit scenario set and
output to produce.

**External sources found:**
- Tom Mitchell, *Machine Learning* (McGraw-Hill, 1997), Ch. 1, Exercises 1.1
  and 1.2 — generate-your-own T/P/E-style task, and "three appropriate /
  three inappropriate" ML-applications judgment, each with a required
  one-sentence justification — https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
- CMU 10-601/10-301 (Gormley, Fall 2025), Lecture 1, "Well-Posed Learning
  Problems: In-Class Exercise" — students pick a task from a supplied menu
  and identify task/performance-measure/experience live in class —
  https://www.cs.cmu.edu/~mgormley/courses/10601-f25/slides/lecture1-overview.pdf
- Berkeley CS189/289A (Gonzalez/Norouzi, Fall 2025), Lecture 1, "Kinds of
  Problems" — triages a scenario as an Engineering problem (hand-written
  rules suffice), a Machine Learning problem (easy to evaluate, hard to
  implement directly), or a Human problem (needs judgment) —
  https://eecs189.org/fa25/lecture/lec01/
- Stanford CS221 (Sadigh, Spring 2018), Lecture 2, opening clicker slide —
  order-of-magnitude question on how many parameters modern ML systems learn,
  used to recalibrate scale intuition before any math —
  https://web.stanford.edu/class/archive/cs/cs221/cs221.1186/lectures/learning1.pdf
- Stanford CS221, HW1 "Foundations," Problem 4, "Ethical Issue Spotting"
  (Autumn 2025) — four scenarios (biased loan model, de-anonymizing
  stylometry, non-consensual face-recognition data, a consented control
  case), each requiring a written potential-negative-impact judgment —
  https://stanford-cs221.github.io/autumn2025/assignments/hw1_foundations/index.html
- Google, "Machine Learning Problem Framing," Try It Yourself worksheet —
  structured steps (Ideal Outcome, Success/Failure Metrics, Output type,
  Oracle Test, Heuristic fallback) for turning a real goal into an ML
  specification — https://developers.google.com/machine-learning/problem-framing/try-it/framing-exercise
- Russell & Norvig, *AIMA*, Ch. 1, Exercises 9/10 (boundary-drawing: "to
  what extent are bar-code scanners / spell-checkers instances of AI?") and
  18/19 (feasibility judgment: which listed tasks are solvable today, and
  when might the rest be) — https://github.com/aimacode/aima-exercises/blob/master/markdown/1-Introduction/README.md

**Proposed problem set** (8 problems):

1. [conceptual] **Four Components of a Learning Task.** Pick a task from
   your own work, or from the four bulleted problems that open this
   section, that resists rule-based coding. Write down, concretely: what
   data you would collect, the model family you would try, the objective
   function you would optimize, and the algorithm you would use to fit it.
   Deliverable: a short structured write-up covering all four components;
   success criterion is that a classmate could reconstruct your intended
   system from the write-up alone.
   *Provenance:* adapted from Mitchell (1997) Ex. 1.2 (overlap med — cite on
   adoption; re-mapped from T/P/E notation onto this section's own
   "data/model/objective/algorithm" vocabulary) and CMU 10-601's live
   in-class T/P/E exercise (overlap low, framing only).
1. [conceptual] **Rules, Learning, or Judgment.** For each of six short
   scenarios — a spam filter, an income-tax calculator, a coffee-shop
   route planner, an open-domain chatbot, an automated resume screener, and
   a chess move selector — decide whether it is best solved by hand-written
   rules, by a learned model, or by human judgment (more than one may
   apply), and justify each choice in one sentence.
   1. State which of your six justifications you are least confident in,
      and what evidence would change your answer.
   *Provenance:* adapted from Berkeley CS189/289A Lecture 1 "Kinds of
   Problems" triage (overlap high — cite on adoption); replaces the
   original ex. 1/ex. 2 pairing (which asked the reader to introspect on
   "code that could be learned" with no deliverable) with a scored,
   scenario-based version of the same judgment.
1. [conceptual] **Supervised, Unsupervised, or Reinforcement?** Given five
   scenarios (a wake-word detector trained on labeled audio clips; grouping
   news articles by topic with no topic labels; a program that plays chess
   against itself to improve; captioning images given (image, caption)
   pairs; and finding two clusters of similar users from browsing logs),
   label the learning paradigm each belongs to and justify in one or two
   sentences, referencing which of "data with labels," "data without
   labels," or "an environment and a reward" is present.
   *Provenance:* adapted from the Stanford CS229-adjacent Coursera ML Week 1
   quiz's supervised/unsupervised select-all item (overlap med) and Berkeley
   CS182/282A Discussion 1's paradigm-definitions setup (overlap low).
1. [conceptual] **Algorithms, Data, and Compute.** Describe the
   relationships between algorithms, data, and computation. How do the
   characteristics of the data and the currently available computational
   resources influence which algorithms are appropriate?
   *Provenance:* original (kept verbatim from the book's existing ex. 3 —
   the style review found no clarity defect and it already names a concrete
   comparative-reasoning task).
1. [conceptual] **End-to-End Training Frontiers.** Name some settings where
   end-to-end training is not currently the default approach but where it
   might be useful. For one of them, state what would have to change
   (data availability, compute, or tooling) for end-to-end training to
   become standard there.
   *Provenance:* original (kept from the book's existing ex. 4, with one
   added sub-question requiring a stated mechanism rather than only a
   named setting).
1. [conceptual] **Extending the Growth Table.** :numref:`tab_intro_decade`
   stops at 2020. Using public specifications for a system you have direct
   access to (your own laptop or phone, a cloud VM you can inspect, or a
   published training run for a well-known recent model), add one row for
   the 2020s: an order-of-magnitude dataset size, memory capacity, and
   floating-point throughput, each with its source or your estimation
   method stated explicitly.
   *Provenance:* inspired by Stanford CS221's scale/magnitude-estimation
   clicker question (overlap low — CS221 asks students to guess a single
   number from options; this problem asks for a sourced estimate extending
   the book's own table).
1. [conceptual] **Data Coverage and Bias.** This section's Data subsection
   describes a skin-cancer classifier that fails on darker skin tones due
   to underrepresented training data. Pick one application from this
   section (the wake-word detector, the mushroom classifier, or a resume
   screener) and describe: what demographic or contextual group could be
   underrepresented in its training data, one concrete way you would test
   for the resulting failure, and one mitigation.
   *Provenance:* adapted from Stanford CS221 HW1 "Ethical Issue Spotting"
   (overlap med — cite on adoption; narrowed from CS221's four generic
   scenarios to this section's own worked examples).
1. [extended] **Teardown of a Deployed System.** Choose a machine-learning
   product you use regularly (a voice assistant, a music or shopping
   recommender, an email spam filter). Write a one-page teardown that
   states, for that system: (i) the data it most likely trains on, (ii) which
   of this section's paradigms (supervised/unsupervised/self-supervised/RL)
   it uses, (iii) the objective it is plausibly optimizing, (iv) how you
   would know if it were serving you well without access to its internals
   (an "oracle test" of the sort described by Google's Problem Framing
   worksheet — assume you were told the perfect answer, then check what you
   would actually have wanted the system to do instead), and (v) one
   concrete failure mode or bias risk grounded in the Data subsection's
   discussion of coverage and feedback. Deliverable: a one-page write-up
   covering all five points.
   *Provenance:* inspired by Google's Machine Learning Problem Framing
   worksheet, specifically its "Ideal Outcome" and "Oracle Test" steps
   (overlap low — synthesized with this section's own paradigm taxonomy and
   bias discussion rather than following the worksheet's template directly).

---

## chapter_preface/index.md — Preface

**Topic:** Onboarding actions for a first-time reader — registering on the
book's discussion forum, installing a working Python environment, and
locating per-chapter discussion links.

**Current exercises:** 3; disposition: keep 2, rewrite 1, drop 0 — ex. 1
(register on the forum) and ex. 2 (install Python) already name a concrete,
checkable action and were not flagged by the style review, so both are
kept near-verbatim. Ex. 3 ("follow the links... where you will be able to
seek out help") was flagged as a navigation instruction rather than a task
with a deliverable, so it is rewritten to require a produced artifact (a
specific written question tied to a specific chapter's discussion thread)
rather than only "go look at this."

**External sources found:** No external exercise tradition found. None of
the five supplied research files (Stanford, Berkeley, CMU, MIT/UW,
classic-textbooks) surface a course or textbook that poses "register for
our forum" or "install the following software" as a graded or worksheet
exercise — every course's own environment-setup instructions live in
un-exercised installation docs, not in a numbered exercise list. This is
an expected finding rather than a gap: forum registration and environment
installation are administrative onboarding steps specific to this book's
own community-and-tooling workflow (:numref:`sec_code`, the forum links),
not ML-pedagogy content that other courses would have reason to exercise.
All three proposed problems below are therefore original.

**Proposed problem set** (3 problems):

1. [conceptual] **Forum Registration and First Post.** Register an account
   on the book's discussion forum, [d2l.discourse.group](https://d2l.discourse.group/).
   Then find the framework-specific Discussions link at the end of this
   section (for example, https://d2l.discourse.group/t/18 for MXNet,
   .../t/20 for PyTorch, .../t/186 for TensorFlow, or .../t/17963 for JAX)
   and post a one-sentence introduction or question there. Deliverable: a
   forum account and one visible post; success criterion is that both
   exist and the post is on-topic for that thread.
   *Provenance:* original (kept from the book's existing ex. 1, with the
   implicit "have an account" deliverable made explicit and a concrete
   posting action added).
1. [conceptual] **Environment Setup and Verification.** Install Python on
   your own computer or a cloud notebook environment. If you already know
   which framework you will use for this book (PyTorch, MXNet, TensorFlow,
   or JAX — see :numref:`sec_code`), install it as well. Confirm the
   installation by running `python --version` and importing your chosen
   framework without error. Deliverable: a version string and a
   successful import; success criterion is no error on either step.
   *Provenance:* original (kept from the book's existing ex. 2, with an
   explicit verification step added so the exercise has a checkable
   success criterion rather than only "install Python").
1. [conceptual] **Support Channels for Your Next Chapter.** Identify which
   chapter of the book (from the Content and Structure overview above) you
   are most interested in reading next. Find its framework-specific
   Discussions link, and write one specific question you would want
   answered — about a prerequisite, a tool, or a concept — before starting
   that chapter. Deliverable: one written question tied to a named chapter
   and its discussion thread.
   *Provenance:* rewrite of the book's existing ex. 3 (overlap high with
   the original — same underlying "find the forum links" action — but
   replaces the bare navigation instruction with a required, checkable
   artifact: a specific written question).
