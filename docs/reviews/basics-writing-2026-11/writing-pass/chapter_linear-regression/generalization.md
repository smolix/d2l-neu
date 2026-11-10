# Writing pass: `chapter_linear-regression/generalization.md` (section 2.6)

Prose words: 4,695 before, 4,683 after (checker: "OK: only prose changed";
`tools/lint_source.py` clean). This is the best section of the chapter, so the
hand was kept lighter than elsewhere: about twenty sentences changed, three
passages cut, and no code, label, figure or heading touched.

## Before and after

1. Before: "This section asks how we can tell, from a finite sample, whether a
   model will generalize."
   After: "How can we tell, from a finite sample, whether a model will
   generalize?"

2. Before: "This is the *IID assumption*. It often fails in practice."
   After: "This is the *IID assumption*. It is a bold one, and it often fails
   in practice."

3. Before: "We want a small generalization error, and the gap tells us how far
   the training error, the number we can compute, misleads us about it."
   After: "It is $R$ that we care about. The gap tells us how far the training
   error, the only one of the two that we can compute, misleads us about it."

4. Before: "When we trained the linear model of :numref:`sec_linear_scratch`,
   its loss on 1,000 held-out examples tracked its training loss closely: a
   model with three parameters fit to 1,000 examples has little room to
   memorize."
   After: "How optimistic? For the linear model of
   :numref:`sec_linear_scratch`, hardly at all: its loss on 1,000 held-out
   examples tracked its training loss closely, since a model with three
   parameters fit to 1,000 examples has little room to memorize."

5. Before: "We can compute both terms in our experiment. The least-squares
   weights ... are a linear function of the labels, and the noise has mean
   zero."
   After: "A decomposition is more convincing when its terms can be measured,
   and in our experiment they can. The key observation is that the
   least-squares weights ... are a linear function of the labels, and the
   noise has mean zero."

6. Before: "The degree of a polynomial measures complexity by counting
   parameters."
   After: "The degree of a polynomial measures complexity by counting
   parameters, which is convenient but rather crude."

7. Before: "In practice test sets are often reused across many rounds of
   experiments and many papers, which invites the same bias. In this book, the
   held-out data we report results on also serves to compare models, ..."
   After: "That is the ideal. In practice test sets are reused across many
   rounds of experiments and many papers, which invites the same bias. This
   book is no exception: the held-out data we report results on also serves to
   compare models, ..."

8. Before: "We can now measure how well a model generalizes, recognize
   underfitting and overfitting, and choose a model's flexibility with
   validation data. The polynomial degree is a coarse control, though."
   After: "Note, however, that the polynomial degree is a coarse control,
   however carefully we select it."

## Repetition removed

- The closing paragraph no longer recaps the section ("We can now measure
  ..."); the Summary is the only recap.
- The observation that degree 19 fits by enormous weights, with its pointer to
  weight decay, was told under "Model Complexity" and again in the closing
  paragraph. The closing now cites it in one clause ("the enormous weights we
  met at degree 19").
- The opening roadmap is one question plus the plan, instead of "This section
  asks ..." plus the plan.
- The wording "many orders of magnitude" and "enormous weights" (fixed today)
  is unchanged.

## Review findings

Applied:

- GEN-2. The first held-out examples are now introduced as "held out", with
  the sentence "The validation loss that our trainer has plotted since
  :numref:`sec_linear_scratch` is such an estimate", and one sentence saying
  that the experiments call them a test set and that the two names are told
  apart under "Model Selection". The code and plots still say "test", which is
  correct for this experiment.
- GEN-3. The paragraph on $\textrm{Var}(y)$ and the coefficient of
  determination is cut (neither is used again here or in the weight decay
  section). The one thing the later text needs, that $\sigma^2$ is the error
  no model can beat and that real data do not reveal it, is kept in two
  sentences.
- GEN-6. The rule "complexity should grow with the amount of data" is scoped
  to models whose capacity grows with the number of weights, with a forward
  reference to `sec_generalization_deep`.
- GEN-7. The three sentences on fixed versus random designs are cut; exercise
  4(c) raises the point itself.
- GEN-8. Reworded to "exactly when the variance is averaged over the $n$
  training inputs instead of the test inputs".
- GEN-9. The clause "it can be severe even for models whose complexity is
  tightly controlled" is dropped; the citation stays on the preceding claim.
- GEN-1, in part. The unused double integral is removed from the definition of
  the risk, and "we do not know $p(\mathbf{x}, y)$" becomes "we do not know
  $P$".

Already addressed in the file: GEN-4 (degree-19 wording). Left as it stands.

Deferred:

- GEN-1, the rest (two-argument loss, $R_\textrm{emp}(f)$): the slides use the
  same bracket notation and may not be edited, so the body keeps it.
- GEN-5 (panel titles, figure size): code.
- GEN-10 ($d$ for the degree): the code uses `d`, and the letter needs one
  decision for the chapter.
- Imports cell placed mid-section: moving a code cell.
- Suggested new exercise on Legendre or Chebyshev bases: exercises are out of
  scope.

## Factual claims added or changed

- "The validation loss that our trainer has plotted since
  :numref:`sec_linear_scratch` is such an estimate." (True: the scratch
  section evaluates on 1,000 examples the model never trains on.)
- The IID assumption is called "bold", and "We cannot do without it"
  (a judgment; the supporting sentence was already there).
- Counting parameters is "convenient but rather crude" (a judgment, supported
  by the two paragraphs that follow it).
- "surprisingly" added to the result of Zhang et al.
- "Cross-validation finds the bottom of the test error curve ..., though not
  every time" (consistent with the "four runs out of five" that follows).
- The dataset-size rule is now restricted to "models like these, whose
  capacity grows with the number of weights", and "Deep networks will
  complicate this rule".
- "This book is no exception" (to the reuse of held-out data); same content as
  before, said more plainly. "often" was dropped from "test sets are often
  reused".
- The claim that selection bias "can be severe even for models whose
  complexity is tightly controlled" is removed.
- No number that describes an output was changed.

## Slides that now differ from the body

- "Let the complexity grow **with** the data, not ahead of it." The body now
  scopes this rule and points to `sec_generalization_deep`; the slide states
  it flatly.
- The "Test set" bullet introduces held-out data as a test set at once; the
  body now says "held out" first and links it to the trainer's validation
  loss. Not a contradiction, but the slide lacks the link.
- The slides have no $R^2$ or fixed-design material, so the cuts leave nothing
  behind there.
