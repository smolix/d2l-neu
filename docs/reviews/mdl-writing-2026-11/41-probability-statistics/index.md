# Probability and Statistical Learning (chapter introduction)

Source: `chapter_mdl-probability-statistics/index.md`. About 400 words of prose before the table of contents, plus an annotated resource list of about 380 words.

## Verdict

A good chapter introduction with one paragraph that fails. The first two paragraphs are what the writing guide asks for and better than most of the Basics introductions: they start from a number the reader has already trusted (a test accuracy), turn it into a coin that lands heads 9 times in 13, and name four objects "that are easy to confuse". The coin then really is the running example of the chapter (it returns in the distributions, exponential-family, statistics and maximum-likelihood sections), so the promise is kept. The third paragraph is the route, and it is the defect that §12.1 of the guide describes by name: one sentence per section, six of them built on "asks how / asks which / asks why". The most important fix is to rewrite that paragraph as three movements.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 3 |
| Language | 3 |
| Ease and enjoyment | 4 |

## What works

- The opening turns a habit into a question: "We read such a number as a fact about the model. It is also a fact about the particular examples that happened to be drawn." This is the reader's own practice, questioned in two sentences. Keep it word for word.
- "A test accuracy is the same coin, flipped once per test question" is the sentence a reader will remember, and it justifies the whole chapter to a deep learning reader.
- The four objects (distribution, estimator, posterior, generalization bound), each with a one-clause job, give the reader a map that is about ideas and not about section numbers.
- The three uses named at the end of the second paragraph (losses are negative log-likelihoods, curvature sets the error bar, a validation score degrades under selection) are concrete and true. They are the right advertisement.
- The scope sentence is short and honest.

## Structural findings

- **The route is a list of nine sections, not a description of movements.** The chapter has four movements: the language (random variables, distributions, the exponential family), inference from a sample (statistics, maximum likelihood, Bayesian computation), guarantees (concentration, generalization bounds), and one worked classifier (naive Bayes). The second paragraph already hints at this with its four objects. The route should follow the four objects instead of the table of contents, which sits directly below it and says the same thing.
- **No reference to the main text by label.** The second paragraph says "The same objects explain tools of the main text" and names cross-entropy, squared error, the error bar and model selection, without one `:numref:`. A reader who starts in the appendix cannot follow the pointer back, and a reader who arrives from the main text is not told which section sent them. Add the labels of the sections that introduce cross-entropy and squared error, and of the model-selection section.
- **The prerequisite sentence undersells the overlap.** "We assume the probability of :numref:`sec_prob` and calculus" is accurate, but the first two sections then re-derive a good part of `sec_prob` (mean, variance, covariance, correlation, the Bernoulli variance). Say in one clause that the first section repeats those results with proofs and for densities, so that a reader who knows `sec_prob` knows what to skim.
- **Scope claims and the sections disagree in one place.** "Measure theory ... left out" is right for the text, but exercise 5(c) of the random-variables section introduces sets of measure zero and asks for a proof about them. Either soften the exercise or leave the scope sentence as it is and accept the exception.

## Passage-level findings

**IDX-1** (moderate, language). Under the chapter title, third paragraph.
Anchor: `quantities that vary continuously. :numref:`sec_mdl-distributions` asks`
Six sentences in a row have the shape "§x asks how/which/why ...". The guide calls this out as the pattern readers skip. The paragraph also carries no information that the table of contents below it lacks. Rewrite as three or four sentences that group the sections, for example: "The first three sections build the language: densities for quantities that vary continuously, the few distributions that model most labels, and the one algebraic form that makes their losses convex. The next three ask what a sample says about the parameters: how much an estimate varies (:numref:`sec_mdl-statistics`), why maximizing likelihood produces the losses of the main text, and how to average over a posterior that has no closed form. Two sections then ask how far a sample average can stray from its mean, and what is left of that guarantee once the model was chosen after seeing the data. The chapter ends by building one classifier from these parts."

**IDX-2** (minor, thought). Second paragraph.
Anchor: `Answering them needs four objects that are easy to confuse:`
"Easy to confuse" is a strong and interesting claim that is then dropped: the paragraph defines the four but never says which pairs get confused or why (a posterior and a sampling distribution are the pair that readers actually mix up). Either add the one clause that pays off the claim ("the spread of an estimator across samples and the spread of a posterior for one sample are different things that produce similar-looking error bars") or drop "that are easy to confuse".

**IDX-3** (minor, explanation). Second paragraph.
Anchor: `negative log-likelihoods, the curvature of a loss sets the error bar of the`
Three claims about the main text with no labels. Add `:numref:` to the sections of the main text that introduce the two losses and model selection, and to the appendix sections that prove each claim.

**IDX-4** (minor, language). First paragraph.
Anchor: `a fair coin does this often enough that 13 flips cannot rule it out. A`
Good sentence, but "often enough" could carry its number at no cost: nine or more heads has probability about 13% (the distributions section computes $1093/8192$). "A fair coin does at least this well about one time in eight" is sharper and prepares the later sections. The guide bans experimental numbers in introductions, but this is arithmetic, not an experiment.

## Resources list

The list is annotated and every entry points to a section by label, which is what §12.3 asks. Two small points: Stat 110 appears twice (course page and its YouTube page) with near-identical annotations and could be one entry; and there is no entry for the exponential-family and Bayesian-computation material at the level of a tutorial (the books cover it), which is acceptable.

## Still open from October

The October review asked for Statistics to precede maximum likelihood. The route paragraph and the table of contents now have that order, so it is closed at the level of the introduction.

## For the chapter review

- The introduction promises a chapter held together by one coin (9 heads in 13 flips) and four objects. The first two sections honour the coin; the chapter review should check that the later sections honour the four objects and say which one each section is about.
- The route paragraph is the only place where the introduction falls to the template the Basics review found. The fix is local.
- Two patterns seen in the first two sections probably hold for the chapter: the main text is named but not referenced by label, and each section ends with a paragraph that begins "We can now ...".
