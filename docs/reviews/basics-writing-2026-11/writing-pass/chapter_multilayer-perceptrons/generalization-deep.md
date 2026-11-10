# Writing pass: `chapter_multilayer-perceptrons/generalization-deep.md` (section 4.5)

Prose words (checker, body plus exercises): 4631 before, 4630 after. The
checker ends "OK: only prose changed"; `tools/lint_source.py` is clean. The
checker's "em dash" warning is a false positive from the table separator
`|---|---|`, present before the pass.

## Before and after

1. Opening puzzle (GEN-1)
   - Before: "Yet it does not: its accuracy on test images, about 87%, is only a few points below its accuracy of about 89% on the training images. Why does a network that can memorize its training data generalize anyway?"
   - After: "After 30 epochs it had not: its accuracy of about 89% on the training images was only a few points above the 87% on test images. That run was short, however, and proves little. What happens if we train such a network until it fits every training label?"
2. Random-label result
   - Before: "With true labels, training accuracy passes 99% after 60 to 90 epochs, and test accuracy settles at about 80% to 83%."
   - After: the same sentence, followed by "A network trained until it fits its training labels still gets about four of five new images right."
3. The random-label argument (GEN-2)
   - Before: "Neither a smaller constant nor a sharper estimate of $v$ can rescue it: as exercise ... shows, random labels rule out every bound that depends only on the class, $n$, and $\delta$."
   - After: "Could a smaller constant or a sharper estimate of $v$ rescue it? Unfortunately not, and the random labels tell us why. A bound whose second term depends only on the class, $n$, and $\delta$ holds for every distribution of the data, including images with random labels. On our 1,000 images with such labels the network had training error 0, while its population error is 90%, so for this $n$ the second term is at least 0.9. The very same term applies when the labels are true, where it guarantees next to nothing (exercise ...)."
4. Double-descent setup (GEN-5)
   - Before: "Very wide networks trained with small steps stay close to a linear model of this kind, built from the gradients at their initialization; this is the *lazy* or *neural tangent* regime ... :numref:`sec_scaling` shows how width-dependent learning rates keep this feature learning from vanishing as networks grow wider."
   - After: "Real networks change their hidden features during training; we give this up for a model we can solve."
5. The peak
   - Before: "At the interpolation threshold $h = n = 1{,}000$, it peaks at about 90%, the error of random guessing."
   - After: "At the interpolation threshold $h = n = 1{,}000$ it peaks at about 90%. The model now reproduces every training label, and on new images it does no better than a ten-sided die."
6. Second descent (GEN-6)
   - Before: "at 10,000 hidden units it reaches 18% to 23%, below the best width before the threshold."
   - After: "At 10,000 hidden units it is back at about a fifth (18% to 23%), as low as at the best width before the threshold or lower."
7. From linear models back to the MLP
   - Before: "Both results concern linear models. For deep networks, statements this exact are rare: ... The next experiment shows a preference of gradient descent in our MLP that needs no such statement."
   - After: "Both results concern linear models, and they prove nothing about our MLP. For deep networks, statements this exact are rare. ... What we can do with the MLP is watch it."
8. Closing (GEN-9)
   - Before: "Gradient descent, not the parameter count, decides which of the many exact fits a network finds. From zero it finds the minimum-norm fit, on separable data it approaches the maximum-margin separator, and on noisy labels it fits the clean examples first. Left to run, it also memorizes the noise."
   - After: "How much of the opening question have we answered? For the two linear models we could say exactly which fit gradient descent finds. For the MLP we have seen one preference, clean examples before noisy ones, and no complete account of why its fits generalize exists yet."

Protected and unchanged: "The answer is not completely known, so this
section measures"; "Whatever lets the network generalize from the true
labels, it is not a lack of capacity to memorize"; the evaluated VC bound;
the doubling-the-training-set sentence (now followed by "More data can
hurt."); the conditioning paragraph; the proposition and its proof.

## Repetition removed

- Parameter count: the derivation $784 \times 256 + \dots = 203{,}530$ is
  gone from the opening (owned by `mlp-implementation.md`); the number is now
  cited twice (opening, VC bound) instead of three times.
- The polynomial study of chapter 2 was retold in the opening and again at
  the start of "Double Descent"; the second telling is one sentence.
- The definition of a too-flexible class from `sec_generalization_basics` is
  one clause with the reference.
- The 0.01 initialization: one clause, one reference (was two references to
  the same section in one sentence).
- The closing paragraph no longer lists the three results that the Summary
  lists; the Summary is the only recap.
- The minimum-norm figure paragraph no longer repeats what its caption says.

## Review findings

Applied: GEN-1 (puzzle posed as a question that the 1,000-image run
answers), GEN-2 (argument in the text; exercise 1 left as the check), GEN-3
(nearest neighbor introduced before the table; last row "not fully known;
the rest of this section"), GEN-5 (lazy and neural tangent regimes and the
`sec_scaling` forward reference cut), GEN-6, GEN-7 (expected outcome stated
before the setup), GEN-9, GEN-10 (distance $m/\|\mathbf{w}\|$). Structural:
the step from the random-features model and the linear models back to the
MLP now says what was observed and what was derived.

Partly applied: GEN-4. The filter is code, so it stays. Its paragraph is
shorter, says that it is a technicality unrelated to double descent, and the
forward reference "as described below" is now a parenthesis. GEN-8: the
grokking paragraph is shorter; the figure stays.

Deferred:
- GEN-4 proper (centre the images so the filter disappears): needs a changed
  cell and re-captured outputs.
- GEN-8 (remove the grokking figure): figure includes may not change.
- GEN-11 (reuse of $m$, $\mathbf{h}$, $\tau$): chapter-wide notation decision.
- Imports cell in the middle of "Random Labels": cells may not move.
- Exercises (turn exercise 1 into a check, fold grokking into a pointer):
  exercises were out of scope.

## Claims added or changed

- "That run was short, however, and proves little." (judgment on the
  30-epoch run of `sec_mlp-implementation`; the 89% figure is kept from the
  old text and is not reported in that section.)
- "still gets about four of five new images right": restates 80% to 83%.
- The GEN-2 argument: population error 90% under random labels, second term
  at least 0.9 "for this $n$" (the 1,000-image experiment, not 60,000).
- 1-nearest neighbor's bias in the table is now "nearby inputs have the same
  label" (was a restatement of the rule).
- "a few such nearly empty columns can make $\mathbf{H}$ lose rank": the
  detail about two units active on the same image was dropped.
- Training curve: the clause explaining why misclassification reaches zero
  before the squared error does was dropped; the fact is kept.
- 10,000 units: "below the best width" became "as low as ... or lower".
- "That is an observation. Our explanation by a nearly singular matrix holds
  for the random-features model only."
- "they prove nothing about our MLP" (of the two linear results).
- Early stopping: "The gain is modest, but so is the effort".
- Grokking: the sentence "stay near chance on validation data for many times
  as long" was dropped; the step counts are unchanged.
- Summary: "For linear models we know how ... For the MLP we know it only in
  part"; "ends training before most of the noise is memorized" (was "keeps
  the clean fit and discards the memorized noise").

## Slides that now disagree with the body

- "The problem" slide: "Counting parameters predicts that the MLP memorizes.
  It does not." The body now says the 30-epoch run proves little and poses
  the puzzle on the network trained to 99%.
- Double-descent slide: "$h = 10{,}000$: 18–23%, **below** the best
  $h < n$." The body says "as low as ... or lower".
- VC slide note: "Random labels rule out every bound that depends only on
  the class and $n$." Consistent, but the body now gives the argument and
  the 0.9.
- Inductive-bias slide table: nearest-neighbor cell and last row ("this
  deck") differ in wording from the body's table.
- Recap slide, "Implicit bias": states the two results without "for linear
  models"; the body now stresses that restriction.
