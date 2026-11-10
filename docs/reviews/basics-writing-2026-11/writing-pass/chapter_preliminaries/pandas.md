# Writing pass: Data Preprocessing (`chapter_preliminaries/pandas.md`)

Prose words (checker): 3030 before, 3029 after. Checker: "OK: only prose
changed". `tools/lint_source.py`: clean. No code cell, label, figure, exercise
or slide was touched.

## Before and after

1. Categorical features (PAN-1).
   - Before: "A model needs numbers, but numbering the three groups 0, 1, and
     2 would invent an order among them, and it would place a missing roof
     twice as far from slate as tile is. A column with only two values is
     different: a single 0/1 column represents it fully, because a weight of
     either sign can express any difference between the two. *One-hot
     encoding* avoids this."
   - After: "A model needs numbers, and the numbering 0, 1, and 2 is the first
     thing that comes to mind. Unfortunately, it does more than invent an
     order. It also places a missing roof twice as far from slate as tile is,
     as though not knowing the roof were an extreme case of tile. *One-hot
     encoding* avoids both defects."
2. Imputation.
   - Before: "The three missing room counts are now 3.29, the mean $23/7$ of
     the seven observed counts."
   - After: "The three missing room counts are now 3.29, the mean $23/7$ of
     the seven observed counts. No home has 3.29 rooms, of course. An imputed
     value is a placeholder, and we meet these three again when we rescale the
     column."
3. Standardization, motivation (PAN-4).
   - Before: "A model could compensate by giving the area a weight hundreds of
     times smaller than that of the number of rooms. However, the training
     methods of later chapters adjust all weights with a common step size, and
     a step size that suits one feature is then far too large or far too small
     for the other (:numref:`subsec_mdl-quadratic-model` analyzes this
     effect)."
   - After: "A feature on a much larger scale than the others also slows down
     training, for a reason that :numref:`sec_linear_scratch` gives."
4. Standardization, what it achieves.
   - Before: "An entry then says how many standard deviations it lies above or
     below the mean of its column, whatever the units of the column."
   - After: "... an entry says how many standard deviations it lies above or
     below the mean of its column. The column no longer remembers whether it
     was measured in square feet or in square meters."
5. Data types.
   - Before: "`NumRooms` holds whole numbers too, but pandas has made it a
     `float64` column, because `NaN` is a floating-point value and an integer
     column cannot hold it."
   - After: "`NumRooms` holds whole numbers too, and yet pandas has made it a
     `float64` column. The gaps are to blame: `NaN` is a floating-point value,
     and an integer column cannot hold it."
6. Leakage.
   - Before: "All statistics in this section, the means used for imputation
     and the means and standard deviations used for standardization, were
     computed from the whole table, with no homes held out."
   - After: "One shortcut of this section should not be copied. All
     statistics, the means used for imputation and the means and standard
     deviations used for standardization, were computed from the whole table,
     with no homes held out."
7. Closing.
   - Before: "We can now turn a table with missing entries and text columns
     into tensors, and we know the decisions this involves: how to fill gaps,
     how to encode categories, and how to bring features to a common scale.
     The pandas library holds a whole table in main memory ..."
   - After: "Ten homes are, admittedly, a toy. The pandas library holds a
     whole table in main memory ... The tensor `X` is a matrix with one row
     per example, the form in which models receive their data. What they do
     with such a matrix is the subject of the next section."
8. After the standardized output.
   - Before: "Imputation also shrank the standard deviation of `NumRooms` from
     1.11 ..."
   - After: "The placeholders have left a trace, however. Imputation shrank
     the standard deviation of `NumRooms` from 1.11 ..."

Protected verbatim: the "three square feet" sentence, "Selection by position
is short, but it breaks without an error", the 1.11 to 0.91 and 1.54 against
1.89 consequence.

## Repetition removed

- "4 of the 10 homes" was stated in the introduction and again as the result
  of the `dropna` cell. The introduction now says deletion "would throw away
  more than half of the table"; the number appears where the output shows it.
- "the number of rooms barely counts" appeared in the introduction and in
  Standardization. Kept in the introduction only.
- "Every column/entry of `inputs` now holds numbers" appeared three times
  (after one-hot encoding, opening Standardization, opening Conversion). Kept
  once, after the one-hot output.
- The column list (rooms, roof type, area, price) was spelled out in the
  introduction and again under CSV Files. The second is reduced to the column
  names with their units.
- The closing paragraph's recap of the three decisions ("We can now ...") is
  gone; the Summary is the only recap.
- "invent an order" remains in the introduction (motivation), once in the
  body, and in the Summary; the body no longer restates it as a new point.

## Review findings

Applied:

- PAN-1: two-valued remark moved after the one-hot output and stripped of
  "weight"; "avoids both defects" now follows the invented order directly.
- PAN-2: the indicator now has a stated cost (one more column per feature).
- PAN-3: the gloss of the standard deviation is separated from the formula,
  and the $n - 1$ points to `sec_prob`.
- PAN-4 and the structural note on step sizes: reduced to a one-clause
  forward pointer to `sec_linear_scratch`. The appendix pointer
  (`subsec_mdl-quadratic-model`) is dropped.
- PAN-6: "later in the book" became `:numref:`sec_rnn``, the first place in
  Basics where an embedding is defined.
- Chapter review, template closings: "We can now ..." replaced.

Deferred:

- Structural note "show what the encoding 0, 1, 2 does to distances": would
  need a new cell. The prose now carries the surprise in words.
- Exercise 6 "Scaling limits" (cut or shrink) and exercise 5's link:
  exercises are out of scope for this pass.
- PAN-5: no change needed.
- The opening still ends in a question (chapter-wide template); left for one
  decision across the chapter.

## Factual claims added or changed

- "Dropping every incomplete row would throw away more than half of the
  table" (6 of 10 rows; replaces "would leave 4 of the 10 homes").
- Indicator: "The cost is one more column per feature, which a small table
  can ill afford."
- "One of the three [one-hot columns] is therefore redundant, and for a
  column with only two values a single 0/1 column suffices." The old
  justification by a weight of either sign is gone.
- "A feature on a much larger scale than the others also slows down training,
  for a reason that :numref:`sec_linear_scratch` gives." That section says a
  feature scaled by $s$ lowers the largest stable learning rate to about
  $2/s^2$, "one reason why inputs are standardized (:numref:`sec_pandas`)".
  If the writer of that section removes or moves the sentence, this pointer
  loses its target.
- "The column no longer remembers whether it was measured in square feet or
  in square meters" (standardization is invariant to a change of unit).
- "Dividing by $n - 1$ instead of $n$ is a convention that :numref:`sec_prob`
  explains" (it does, in the paragraph on variance).
- Distances are "decided almost entirely by the area" (was "dominated by the
  area, and the number of rooms barely counts").
- Embeddings: "we meet them in :numref:`sec_rnn`".

## Slides against the body

No statement in the slides contradicts the revised body. The slide bullet
"Two values only: one 0/1 column suffices" matches the moved remark. The
slides do not use the step-size argument that the body dropped.
