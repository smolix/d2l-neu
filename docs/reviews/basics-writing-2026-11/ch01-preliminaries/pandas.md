# Data Preprocessing

Source: `chapter_preliminaries/pandas.md`. About 2,300 words of prose (3,250 in the file to the slides marker), plus 700 words of exercises.

## Verdict

A tidy, honest section that does one job: it takes a ten-row table with gaps and words and turns it into a tensor, and at each step it says what the decision costs. It is better than most treatments of this topic because the consequences are computed, not asserted (imputation shrinks the standard deviation from 1.11 to 0.91; the largest home scores 1.89 instead of 1.54). It scores 4 on most dimensions. The prose is plain to the point of flatness, and one paragraph in "Categorical Features" is out of order so that the sentence introducing one-hot encoding points at the wrong thing. That paragraph is the main fix.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 3 |

## What works

- The three decisions are stated in the second paragraph in the reader's words ("what to put in the gaps, how to turn words into numbers, and whether to bring the columns to a common scale") and the third paragraph shows why each one matters with a number or a consequence. The body then takes them in that order.
- The sharpest example in the section: "In raw units, two homes with the same area that differ by two rooms are closer to each other than two homes with the same number of rooms that differ by three square feet." It makes the need for scaling obvious in one sentence. Keep it verbatim.
- Every claim about the data checks out against the stored output: 4 complete rows, mean 23/7 = 3.29, area mean 1,654, standardized values 1.89 and 1.70.
- The paragraph after standardization follows a consequence all the way through: mean imputation puts three entries at the mean, the standard deviation falls from 1.11 to 0.91, "so every standardized room count moves farther from 0". This is the kind of second-order observation a good teacher makes and most texts skip.
- Leakage is introduced by confessing that the section itself has just leaked ("All statistics in this section ... were computed from the whole table") and saying what should be done instead. An exercise then makes the reader do it.
- "Selection by position is short, but it breaks without an error" is a real engineering lesson in one line.

## Structural findings

- The section is the right length and in the right place. It is the least mathematical section of the chapter and gives the reader a rest between tensors and linear algebra.
- The table has ten rows so that everything can be checked by eye, which is right, but the section never shows the reader anything unexpected. A single surprise would lift it: for instance, show what the "obvious" encoding 0, 1, 2 does to the distance between a slate roof and a missing roof before introducing one-hot encoding, rather than only asserting that it "would invent an order".
- Standardization is motivated partly by "the training methods of later chapters adjust all weights with a common step size". This is the real reason, but the reader has no picture of a step size yet, and the pointer goes to the appendix (`subsec_mdl-quadratic-model`). The distance argument that precedes it already suffices; the step-size sentence can become a one-clause forward pointer.

## Passage-level findings

**PAN-1** (moderate, thought). Under "Categorical Features".

> *One-hot encoding* avoids this.

"This" is meant to refer to the invented order of the encoding 0, 1, 2. But the sentence immediately before it is the aside about two-valued columns ("A column with only two values is different: a single 0/1 column represents it fully, because a weight of either sign can express any difference between the two"), so the reader takes "this" to mean the two-valued case, or has to back up. The aside also uses "weight" before the book has introduced weights. Move the two-valued remark to after the one-hot output, where it becomes a useful observation ("for a column with only two values, one 0/1 column suffices"), and let "avoids this" follow the sentence about the invented order directly.

**PAN-2** (minor, thought). Under "Missing Values".

> There are three common responses, and each has a cost.

The bullets give a cost for deletion (discards data, biases what remains) and for imputation (filled values are guesses, the column looks less spread out), but none for the indicator. Either state its cost (one more column per feature; with few rows, more parameters to fit) or write "the first two have a cost".

**PAN-3** (minor, explanation). Under "Standardization".

> For a column with entries $x_1, \ldots, x_n$, pandas computes the standard deviation, the typical distance of an entry from the mean, as $\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2}$.

A definition, a gloss and a formula with an unexplained $n - 1$, all in one sentence, in the middle of the paragraph about what standardization does. The $n - 1$ is explained only in the probability section, four sections later. Put the formula in a sentence of its own and add "(dividing by $n - 1$ instead of $n$ is a convention that :numref:`sec_prob` explains)".

**PAN-4** (minor, explanation). Under "Standardization".

> A model could compensate by giving the area a weight hundreds of times smaller than that of the number of rooms. However, the training methods of later chapters adjust all weights with a common step size, and a step size that suits one feature is then far too large or far too small for the other

Three unfamiliar notions (weight, training method, step size) carry the argument. See the structural note: shorten to a forward pointer, or move the sentence to the linear regression chapter where it can be shown.

**PAN-5** (minor, language). Under "Data Types and Summaries".

> `RoofType` has the type `str`, which pandas uses for text (versions before pandas 3.0 used `object`).

Correct against the stored output. Fine as is; noted only because version remarks date. Keep it in the parenthesis.

**PAN-6** (minor, pedagogy). Under "Categorical Features".

> Such columns are better represented by *embeddings*, short vectors of numbers that a model learns for each value; they appear later in the book.

Good one-sentence pointer. "Later in the book" should be a label reference so that the reader can find it.

## Exercises

Six exercises, well matched to the section. Exercise 2 (three imputation strategies and what each assumes) and exercise 4 (redo the pipeline with training-only statistics, then the constant-feature trap) are the strongest: each makes the reader discover something the text only states. Exercise 3's `ListingID` question is a nice test of understanding.

- Exercise 6 ("Scaling limits": write CSV files of up to $10^7$ rows, time `read_csv`, extrapolate) is a measurement chore whose answer depends on the reader's machine and teaches nothing about preprocessing. Cut it, or reduce it to one question about memory per row.
- Exercise 5 points to a specific Berkeley homework notebook. Keep the task; make sure the link is stable or describe the dataset so the exercise survives without it.

## Still open from October

Nothing specific to this section was raised in October.
