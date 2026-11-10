# Further Reading

Source: `chapter_mdl-probability-statistics/further-reading.md`. About 1,700
words, 15 topics in five groups, no code, no exercises. Every citation key
resolves in `d2l.bib` and every `:numref:`/`:exref:` target exists.

## Verdict

A useful and accurate guide, and better than the usual list of books. Each
entry names a limit of something the chapter taught, says in two or three
sentences what the next idea is and what it changes, and gives one or two
well-chosen sources. A reader can decide from the entry alone whether to
follow it. I found no false statement; three attributions or claims are
looser than they should be (FR-3, FR-4, FR-6). The main thing to fix is
orientation: the chapter has two reading lists with overlapping names, and
this one, despite its title, does not tell the reader what to read for the
material the chapter does teach. The second is evenness: fifteen entries of
the same size and shape, of which the four with a concrete example (the
Cantor digits, the red and blue bus, peeking at a running test, the uniform
rank behind conformal prediction) are far more readable than the rest.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The entry format.** "The $z$-test of `sec_mdl-statistics` is valid at a
  sample size fixed in advance. Checking a running comparison after every
  batch and stopping at the first $p<0.05$ makes a false alarm certain if
  the checks go on long enough". The reader gets the problem in terms of
  something they have already done, then the remedy, then its price ("wider
  ... only by a factor that grows like $\sqrt{\log\log n}$").
- **Sharp examples.** The base-3 number with digits 0 and 2 chosen by a fair
  coin explains a singular distribution in two sentences. The commuter with
  a car, a red bus and "an identical blue bus" shows what is wrong with the
  softmax as a choice model better than any definition of independence of
  irrelevant alternatives.
- **Connections back to the chapter.** "The argument is the one behind the
  permutation test: under exchangeability the rank of the test score among
  all $n+1$ scores is uniform." "The norm is the case $L=1$, which is why
  its fluctuation in the thin shell does not grow with $d$." These reward
  the reader who read the chapter.
- **Honest scope statements.** "The guarantee holds on average over inputs,
  not for each input separately." "Each such claim rests on an estimate of
  the tail index, which depends strongly on how much of the tail it uses."
  "Nothing else in the book depends on this section."
- **Order.** The groups follow the chapter, so a reader finishing a section
  can find its extensions.

## Structural findings

**FR-S1. Two reading lists, one name.** The chapter introduction ends with
"Resources and Further Reading", a list of ten books, courses and notes for
the material the chapter teaches. This section is titled "Further Reading"
and holds topics the chapter does not teach. Each points to the other in
one sentence, but a reader who clicks "Further Reading" in the table of
contents wanting a book on concentration inequalities finds PAC-Bayes and
no Vershynin recommendation. Rename one of them. The cleaner choice is to
call the list in the introduction "Resources" and this section "Beyond This
Chapter" or "Further Topics", since that is what it is. The same pair exists
in every chapter of the part (`chapter_mdl-*/further-reading.md`), so this
is a part-level decision.

**FR-S2. Coverage is uneven across the chapter.** Four entries extend the
two theory sections, three extend statistics, two extend Bayesian
computation, three extend the first three sections, and
`sec_mdl-maximum_likelihood`, the longest section of the chapter, has
none. That section already treats misspecification and the sandwich
covariance, so the gap may be deliberate. The candidate a deep learning
reader would look for is the use of Fisher information in training
(natural gradient, K-FAC); the term "natural gradient" does not occur in
that section. I did not read the section in full, so treat the suggestion
as a candidate; the absence of any entry is certain.

**FR-S3. Three things a deep learning reader would look for and not
find.** (i) Under "Posteriors of Neural Networks": deep ensembles and
dropout as approximate posteriors, which are what practitioners use far
more often than SGLD or Laplace; "ensemble" in this sense appears nowhere
in the chapter. (ii) Proper scoring rules (log loss and the Brier score as
the reason a model trained by likelihood can be calibrated at all); neither
term appears in the chapter, and the calibration entry is the natural
place. (iii) Under "Classifiers from Counts": the generative versus
discriminative comparison of Ng and Jordan, which is the natural sequel to
`sec_mdl-naive_bayes` and is at present cited only inside its last
exercise, and one sentence on graphical models, which the figure of that
section gestures at. The two entries now in that group (calibration maps,
semi-supervised naive Bayes) are the least necessary of the fifteen; the
second could give way.

**FR-S4. Fifteen entries of one shape.** Each is 90 to 130 words: the
chapter's limit, the new idea in italics, a consequence, "X develop the
theory, and Y give a tutorial". Read front to back it is even to the point
of monotony, which matters less in a section meant for lookup. The remedy
is not variety for its own sake but an example in the entries that lack
one, and a judgement in a few: which of these a practitioner should read
first (conformal prediction, I would say), which is of mainly theoretical
interest (VC dimension), and which is unsettled (heavy-tailed gradient
noise; the text hints at this and could say it).

## Passage-level findings

**FR-1** · minor · language · opening paragraph
> "Each topic states the problem it"

"Each topic states the problem it addresses and what it changes relative to
the chapter" describes the template instead of helping the reader. The
preceding sentence already lists the topics. Cut it, and use the space to
say which entries matter most (FR-S4).

**FR-2** · minor · thought · "Distributions Without Atoms or Densities"
> "footing, in which the chapter's results hold without the regularity"

The reader cannot tell which results or which assumptions are meant, so the
sentence promises something unspecified. Name one: for example, that
expectations, conditioning and Bayes' rule take a single form for discrete,
continuous and singular variables, so the separate sum and integral
versions in `sec_mdl-random_variables` become one statement.

**FR-3** · minor · correctness (sure of the mathematics, the fix is one word) · "Heavy Tails and Stable Laws"
> "Gaussian. The Cauchy is the stable law with $\alpha=1$ and the Gaussian the"

For $\alpha=2$ the stable law is unique up to location and scale; for
$\alpha=1$ it is not, since skewed stable laws with $\alpha=1$ exist. The
Cauchy is the symmetric one. Write "The Cauchy is the symmetric stable law
with $\alpha=1$". The convergence statement before it also needs the two
tails to be balanced in a definite proportion (regular variation), which
"falls like $x^{-\alpha}$" may be read to include; that is acceptable at
this level.

**FR-4** · minor · correctness (attribution; fairly sure) · "Choice Models and Irrelevant Alternatives"
> "noise and stated this property, the *independence of irrelevant"

Independence of irrelevant alternatives is Luce's choice axiom (1959), and
the two-bus objection goes back to Debreu's review of Luce (1960), in a
version with recordings of music. McFadden derived the conditional logit
from extreme-value noise and discussed the restriction with the red and
blue bus; he did not originate the property. "Stated" can be read either
way. Write "and discussed this property, Luce's *independence of irrelevant
alternatives*, as its main restriction".

**FR-5** · minor · explanation · "Better Bootstrap Intervals"
> "how the standard error changes with $\theta$. For a one-sided interval both"

This is the only entry with no reason for a deep learning reader to care.
One clause would supply it: the chapter's own bootstrap intervals are for
accuracies near 70% on 10,000 examples, where the percentile interval is
fine; the corrections matter for small evaluation sets and for skewed
statistics such as a ratio of two error rates or a maximum over
checkpoints.

**FR-6** · minor · thought (fairly sure) · "Posteriors of Neural Networks"
> "predictive uncertainty for trained networks at little extra cost."

Stronger than the cited paper, which shows that these approximations are
competitive with more expensive methods on calibration and
out-of-distribution benchmarks. "Calibrated" without qualification is a
claim the previous section of this chapter has just taught the reader to
distrust. Write "give predictive uncertainty competitive with ensembles at
a fraction of their cost", which also brings in the missing comparison
(FR-S3).

**FR-7** · minor · explanation · "Algorithmic Stability"
> "The random-label experiment shows that the class a network can represent is"

A reader who arrives by lookup does not know which experiment. It is
`subsec_random_labels` in the main text, discussed in
`sec_mdl-generalization-bounds`. Add the reference.

**FR-8** · minor · explanation · "PAC-Bayes Bounds" and "VC Dimension and Covering Numbers"
> "posterior that stays close to the prior is cheap even if the class is huge."

These two entries and "Algorithmic Stability" are the abstract ones: no
number, no picture. One number each would do. For PAC-Bayes the cited paper
supplies it (a bound of roughly 0.16 to 0.2 on the error of a network for
which the VC bound exceeds 1; check the figure before quoting). For VC
dimension: halfspaces in $d$ dimensions have dimension $d+1$, so the 785
weights of one output of softmax regression cost about $\sqrt{785\log n/n}$.

**FR-9** · minor · thought · "Calibration Beyond One Temperature"
> "more held-out data, and for modern networks a single temperature is often as"

Fine as far as it goes. The entry answers "how do I fix the scale of the
scores" and not the question the naive Bayes section leaves open, which is
why likelihood training gives calibrated probabilities for softmax
regression and not for the counted model. One sentence on proper scoring
rules would answer it (FR-S3).

## Exercises

None, correctly.

## Still open from October

Not covered by the October review.

## For the chapter review

- The section is accurate and worth keeping at its present length. It
  follows the order of the chapter and depends on labels in every section,
  all of which resolve.
- The naming clash with "Resources and Further Reading" in `index.md` is a
  part-level matter: every chapter of the part has the same pair.
- No entry extends `sec_mdl-maximum_likelihood`; the last group depends on
  the naive Bayes section staying in the chapter.
- The entries with a concrete example show the standard the others should
  meet. If the chapter sections themselves read as evenly as the
  example-free entries here, the "voiceless" finding of the Basics review
  carries over.
