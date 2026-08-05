# Berkeley intro-ML/DL "pre-math" exercise archetypes

Scope: CS189/289A, CS182/282A, CS188, Data 100, plus Data 104 (Human Contexts & Ethics of
Data) and CS294 Fairness in ML as adjacent "anything else Berkeley" hits. Focused on the very
first week(s) of material — before real math/code — where courses ask students to *frame*
problems as ML rather than *solve* them.

Overall finding: the richest, most concrete material by far is **CS189/289A Fall 2025 Lecture 1**
("Introduction + ML problem framing" — literally named that in the schedule) by Joseph Gonzalez
and Narges Norouzi. It is built almost entirely around live in-class poll questions (Slido) that
are pure conceptual/framing exercises. CS182/282A's Discussion 1 and CS188's exam bank supply the
"classify the paradigm" and "component identification" archetypes in worksheet/quiz form. Data 100
and Data 104 supply the ethics/social-framing archetype but in narrative/reading form rather than
discrete exercise form. Two of the requested archetypes — feedback loops and distribution shift —
are covered richly in *research and industry* framing (predictive-policing feedback-loop papers,
distribution-shift blog posts/benchmarks referenced from Berkeley labs) but I could not find a
discrete intro-level *exercise* built around them in these specific courses' early materials; see
the note at the bottom.

---

## Archetypes observed

1. **Kinds-of-problem triage** — Given a task, decide whether it's an "Engineering problem"
   (solvable with hand-written rules), a "Machine Learning problem" (easy to demonstrate/evaluate,
   hard to directly implement), or "a Human problem" (needs judgment, can't be well-specified).
   Often paired with a live poll ("select all that are an ML problem") and a worked example
   (spam, chatbots) walking through *why*.

2. **Paradigm classification** — Given a dataset/setup description, label it supervised,
   unsupervised, or reinforcement learning, and justify in 1-3 sentences.

3. **Problem-framing / component identification** — Given a real-world goal, name the concrete ML
   ingredients: the target/task (what to predict), the objective/loss (how to measure success),
   and the data (what's available, how it's represented, how it's split). Also includes
   "which features would let you solve this" style feature-identification questions.

4. **Validation/evaluation-methodology reasoning** — Why you must never tune on the test set, what
   the validation set is for, describe the train/val/test workflow in your own words. This is
   methodological rather than purely conceptual, but it's asked before any real math, as pure
   prose/discussion.

5. **History & scale reasoning** — Why did an old technique (neural nets, statistical learning)
   rise/fall/return; what changed (data availability, compute); open-ended "what defines the
   field today" discussion prompts.

6. **Ethics / social framing of data & classification systems** — Classification and
   data-collection choices encode assumptions and can reinforce historical patterns; who a
   dataset/model serves; presented as week-by-week reading topics and a course-intro ethics
   paragraph rather than as discrete graded questions.

---

## Examples by archetype

### 1. Kinds-of-problem triage

**CS189/289A, Fall 2025, Lecture 1 ("Introduction + ML problem framing")**
Instructors: Joseph E. Gonzalez, Narges Norouzi.
Source: https://eecs189.org/fa25/lecture/lec01/ (slides, Google Drive PDF export)

- Framework slide, "Kinds of Problems" (verbatim):
  > **Engineering Problem:** Can be solved with a direct, specifiable algorithm or a set of
  > hand-written rules.
  > **Machine Learning Problem:** For which it is easy to demonstrate or evaluate the solution but
  > difficult to directly implement.
  > **A Human Problem:** The problem cannot be well specified and/or human judgement is required.
  > → Often require *Engineering + ML + Humans* 🤝

- Live poll (Slido), verbatim, immediately preceded by "When should I use[d] Machine Learning?":
  > **What kinds of problems are Machine Learning Problems? Select all options that are a machine
  > learning problem.**
  (Answer options are shown live in the Slido app during lecture, not embedded in the exported
  slide PDF.)

- Worked example, "A Machine Learning Problem" slide, verbatim:
  > A problem for which it is **easy to demonstrate** or **evaluate the solution** but **difficult
  > to directly implement**.
  > *Machine Learning Solution:* The system **learns the desired behavior** (e.g., prediction,
  > representation, or a policy) through **demonstration** or **experience**.
  > *Example:* Determine if a text message is spam
  > Spam is **difficult to define** and **depends on the receiver** → Easier to **demonstrate
  > examples** and **learn a function** to detect [it]
  > (callout bubble) "How do you define Spam?"

- Second worked example, "Is Chatting a Machine Learning Problem?" slide, verbatim setup:
  > Example (ChatGPT): Engage a human in a productive conversation
  > "How do you program this?" → contrasted with ELIZA (1966), a rule-based conversational system,
  > with the framing payoff: "We can demonstrate good conversations." / "We can judge good
  > conversations." (i.e., hard to specify directly, easy to evaluate → ML problem)

### 2. Paradigm classification (supervised / unsupervised / reinforcement)

**CS182/282A, Spring 2021, Discussion 1** (Sergey Levine's course)
Source: https://cs182sp21.github.io/static/discussions/dis1.pdf

- Definitions given right before the exercise (context for what students are meant to apply):
  > In supervised learning, you are given a dataset D = {(x1,y1),...,(xn,yn)} containing input
  > vectors and labels... In unsupervised learning, your dataset is unlabeled... In reinforcement
  > learning, you do not have a fixed dataset, but instead interact with an environment...

- **Problem 1: Validation Potpourri** (verbatim; also relevant to archetype 4):
  > 1. Why should you never tune your hyperparameters on your test set?
  > 2. What should your validation set be used for?
  > 3. Describe a general ML workflow with datasets

**CS188, Summer 2024 Final Exam, Q1(c)** ("Potpourri" section — a warm-up conceptual question
before the technical ones)
Source: https://inst.eecs.berkeley.edu/~cs188/assets/exam/cs188-su24-final-solutions.pdf

- Verbatim:
  > Is the perceptron algorithm a form of supervised learning, or unsupervised learning? Explain
  > your answer choice in three sentences or fewer.
  (Official solution: "It's a form of supervised learning because we're training our algorithm
  using labeled data. In supervised learning, training is done using data that is already labelled
  for us, rather than data that isn't labelled.")

**CS188, Summer 2023, Lecture Note 20 ("Machine Learning")**
Source: https://inst.eecs.berkeley.edu/~cs188/su23/assets/notes/cs188-su23-note20.pdf

- Framing paragraph (not a graded exercise, but the canonical definition CS188 sections/exams draw
  the paradigm-classification questions from), verbatim:
  > There are many machine learning algorithms which deal with many different types of problems
  > and different types of data, classified according to the tasks they hope to accomplish and the
  > types of data that they work with. Two primary subgroups of machine learning algorithms are
  > **supervised learning algorithms** and **unsupervised learning algorithms**. Supervised
  > learning algorithms infer a relationship between input data and corresponding output data in
  > order to predict outputs for new, previously unseen input data. Unsupervised learning
  > algorithms, on the other hand, have input data that doesn't have any corresponding output data
  > and so deal with recognizing inherent structure between or within datapoints...
  Includes the "Training / Validation / Testing" 3-panel cartoon (robot teaching apple=fruit,
  car=vehicle; robot taking a "Practice Exam"; robot taking the "Final Exam").

**CS189/289A, Fall 2025, Lecture 1**
Source: https://eecs189.org/fa25/lecture/lec01/

- "Learning Settings" slide lists Supervised (Demonstration) / Unsupervised / Reinforcement
  (Reward), each annotated with what's observed: {(X,Y)} / {X} / X, reward(.) — then a following
  slide gives four concrete supervised examples side by side (Image Labeling: X=Image,
  Y={Hot Dog,...}; a text-generation example: X=Prompt, Y=Next Word; Stock Prediction: X=History,
  Y=Next Value; Image/Video Generation: X=Prompt+Noise, Y=Pixel Values) for students to map onto
  the X/Y framework themselves.

### 3. Problem-framing / component identification (target, objective/loss, data)

**CS189/289A, Fall 2025, Lecture 1 — "ML Lifecycle" diagram**
Source: https://eecs189.org/fa25/lecture/lec01/

- The lifecycle is presented as four stages (Learning Problem → Model Design → Optimization →
  Predict & Evaluate). The "Learning Problem" stage slide is the clearest verbatim problem-framing
  checklist found in this research:
  > This stage is about framing the real-world question into something a machine learning model
  > can answer.
  > **Target:** What do I want to **predict**? What is the machine learning task?
  > **Objective:** How would I **evaluate success**? What **loss** should I use?
  > **Data:** What **data** do I have? Data representation? Feature Engineering? Training/Test
  > split

- Companion diagram, "Machine Learning as Learned Function Approximation," verbatim labels:
  Input (X) → Function (Model) h_w → Output (Y), instantiated with the spam example (Text Message
  → Is it Spam? No(0)/Yes(1)) as the concrete input/output pair students are meant to generalize
  from.

**CS188, Spring 2024 Final Exam, Q7 ("ML: Spam Filter")** — a scenario-based feature/assumption
identification question, phrased with almost no math for its first parts
Source: https://inst.eecs.berkeley.edu/~cs188/assets/exam/cs188-sp24-final.pdf

- Setup (verbatim, abridged):
  > Pacman has hired you to work on his PacMail email service. You have been given the task of
  > designing a spam detector. You are given a dataset of emails X, each with labels Y of "spam"
  > or "ham." Here are some examples from the dataset: [4 sample emails, 2 spam/2 ham given
  > verbatim] ... Your job is to classify the emails in a second dataset, the test dataset, which
  > do not have labels.
- Q7(a), verbatim:
  > Considering only the examples given, which of the following features, in isolation, would be
  > sufficient to classify the examples correctly using a linear classifier?
  > □ The number of words in the email.
  > □ The number of times the exclamation point ("!") appears in the email.
  > □ The number of times "prize" appears in the email.
  > □ The number of capital letters in the email.
  > # None of the above.
- Q7(b) asks students to select true statements about applying Naive Bayes to this problem
  (conditional-independence assumption check) — a component/assumption-identification exercise.

**CS182/282A, Spring 2021, Discussion 1** — Problem 1.3 ("Describe a general ML workflow with
datasets") already listed above under archetype 2 doubles as a component-identification prompt.

**CS189/289A, Jonathan Shewchuk's "Introduction" lecture** (recurring across many semesters,
e.g. Spring 2020 and Spring 2017 versions; delivered as lecture but written as a sequence of
Socratic prompts students answer aloud in class)
Sources: https://people.eecs.berkeley.edu/~jrs/189s20/lec/01.pdf ,
https://people.eecs.berkeley.edu/~jrs/189s17/lec/01.pdf

- Walks through the credit-card-default classification example, then poses in-class questions
  such as (verbatim, from speaker notes in brackets in the source):
  > "How do we classify a new point?" [Draw a point in a third color.]
  > [One possibility: look at its nearest neighbor.] [Another possibility: draw a linear decision
  > boundary; label it.] [Those are two different *models* for the nature of this data.]
  and later, comparing a 1-NN classifier (zero training error) against a 15-NN classifier:
  > "The 1-nearest neighbor classifier at left has a big advantage: it classifies all the training
  > data correctly, whereas the 15-nearest neighbor classifier at right figure does not. But the
  > right figure has an advantage too. Somebody please tell me what."
  (This is the overfitting/generalization intuition delivered as a direct question to the room,
  before any formula for bias/variance is introduced.)

- The lecture's "Techniques" slide gives a four-cell taxonomy with one-line problem statements
  used as the running examples for the rest of the course, verbatim:
  > Supervised learning: Classification: is this email spam? / Regression: how likely does this
  > patient have cancer?
  > Unsupervised learning: Clustering: which DNA sequences are similar to each other? /
  > Dimensionality reduction: what are common features of faces? common differences?

### 4. Validation / evaluation-methodology reasoning

Already given above (CS182 Disc 1, Problem 1). Additional verbatim material from Shewchuk's
Introduction lecture on the same theme (train/validation/test, and the Kaggle public/private-set
framing used for the class's own homeworks):
> training set used to learn model weights
> validation set used to tune hyperparameters, choose among different models
> test set used as FINAL evaluation of model. Keep in a vault. Run ONCE, at the very end.
> [It's very bad when researchers in medicine or pharmaceuticals peek into the test set
> prematurely!]
> ... If your public results are a lot better than your private results, we will know that you
> overfitted.

### 5. History & scale reasoning

**CS189/289A, Fall 2025, Lecture 1 — "History of ML" section**
Source: https://eecs189.org/fa25/lecture/lec01/

- A timeline slide (1950s–60s Early Days → 1970s–80s Challenges/Advances → 1990s Rise of
  Statistical ML → 2000s Big Data Era → 2010s Deep Learning Revolution → Present) is interrupted
  by an open discussion poll, verbatim:
  > **What do you think defines AI today?**
  followed later by a slide breaking down "Today – GenAI" into three framing questions answered
  in bullet form: "What does it mean?" / "Why is it important?" / "Will we cover it?"

**CS189/289A, Jonathan Shewchuk's Introduction lecture** (Spring 2020 version) — delivered as
lecture narrative rather than a question, but functions as the "why does scale matter" argument
the course leads with, verbatim:
> The most important part of this is the data. Data drives everything else. You cannot learn much
> if you don't have enough data. You cannot learn much if your data sucks. But it's amazing what
> you can do if you have lots of good data. Machine learning has changed a lot in the last two
> decades because the internet has made truly vast quantities of data available... Some techniques
> that had fallen out of favor, like neural networks, have come back big in recent years because
> researchers found that they work so much better when you have vast quantities of data.

### 6. Ethics / social framing of data & classification systems

**Data 100, "Introduction" course notes**
Source: https://ds100.org/course-notes/introduction/

- Framing paragraph, close to verbatim:
  > Data science is fundamentally human-centered and facilitates decision-making by quantitatively
  > balancing tradeoffs.
  and the lifecycle's first two stages are framed as literal questions to ask before modeling:
  "What problems are we solving? What hypotheses should we test? What metrics define success?"
  (Ask a Question) and "is our data representative of the population we want to study?" (Obtain
  Data) — plus an explicit ethics caveat that data science "can also be used [to] obscure complex
  decisions and reinforce historical trends and biases."

**Data C104, "Human Contexts and Ethics of Data"** (a dedicated ethics-of-data course, not
ML-specific, but the closest Berkeley course to a dedicated feedback-loop/social-consequences
treatment at the intro level)
Source: http://data104.org/

- Week 2 topic, "Making Data": readings by Bowker & Star (classification as a human activity with
  consequences), Ruha Benjamin (how data practices encode social disparities), and Ian Hacking
  (how statistical categories can create new forms of identity) — i.e., the course's framing
  device is that *the act of building a classifier changes the world it classifies*, which is the
  conceptual seed of the feedback-loop archetype, taught via readings/discussion rather than a
  discrete quiz question.

---

## Gap note: feedback loops & distribution shift

I could not find a discrete, intro-level, pre-math **exercise** (quiz question, worksheet problem,
poll) in CS189/289A, CS182/282A, CS188, or Data 100's early materials that specifically asks
students to reason about feedback loops or distribution shift as a judgment exercise. What exists
at Berkeley on these topics lives either in upper-level/research material (e.g., BAIR's blog on
test-time distribution shift: https://bair.berkeley.edu/blog/2020/11/05/arm/) or in the
research literature on runaway feedback loops in predictive policing
(https://arxiv.org/abs/1706.09847, authored in part by a UC Berkeley-affiliated statistician,
Sharad Goel's collaborators) that fairness-adjacent Berkeley courses (CS294 Fairness in ML,
Data 104) draw on as readings rather than as a discrete framing exercise. If this archetype is
important for your purposes, the strongest adjacent Berkeley material is Data 104's week-2/3
readings (predictive policing, classification-has-consequences) rather than a CS189/182/188/Data100
exercise.

## Files fetched

Working PDFs saved locally during this research (for reference/reuse), under
`/private/tmp/claude-501/-Users-smola-Repositories-boson-easy-demos/e85eca00-c9ac-4621-94a9-143fd8325e73/scratchpad/exercise-review/pdfs/`:
cs182_dis1.pdf, cs182_dis1_sol.pdf, cs188_note20_su23.pdf, cs188_su24_final_sol.pdf,
cs188_sp24_final.pdf, cs188_fa25_lec24.pdf (LLMs lecture, not used), cs189_shewchuk_intro_s20.pdf,
cs189_shewchuk_intro_s17.pdf, cs189_fa25_lec01.pdf.
