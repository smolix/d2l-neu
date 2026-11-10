# Information Theory and Divergences (chapter introduction)

Source: `chapter_mdl-information-theory/index.md`. About 380 words of introduction, followed by an annotated resource list of about 530 words.

## Verdict

A good chapter introduction, close to the template of guide section 12.1 and better than most. It starts from what the reader has done ("Almost every model in this book was trained by minimizing a cross-entropy"), admits what the book skipped ("We never asked what that number means"), gives Shannon's question in two sentences, and states scope. Two things keep it from being excellent. The route paragraph is five sentences of one shape, the defect the guide names. And the introduction raises a concrete question ("a loss of 1.2 nats per character, is that good?") that the chapter never answers with that number, although the material to answer it is in the second section. The most important fix is to make the chapter honour that question.

## Scores

| Dimension | Score |
|:--|:--|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- The first paragraph is the right opening for this reader: four questions a practitioner has had while looking at a loss curve, none of them rhetorical. Keep it.
- "His measure of unpredictability, *entropy*, is the average surprise of an outcome. It is also the floor below which no model's cross-entropy can fall." Two sentences give the chapter's main result before any formula. This is the Cover and Thomas habit of meaning before formula.
- The scope sentence ("We leave out the construction of practical error-correcting codes, Kolmogorov complexity and information geometry") is honest and short.
- The resource list is annotated and every entry points to a section by label. Cover and Thomas and MacKay are both named with what they accompany.

## Structural findings

1. **The prerequisite sentence hides a large overlap.** "We assume the probability of :numref:`chap_mdl-probability-statistics`" is true, but chapter 41 does more than supply probability. It defines entropy as average surprise, defines KL divergence and cross-entropy, proves Gibbs' inequality (in `mdl-maximum-likelihood.md`, "Cross-Entropy and the Entropy Floor", by the tangent line) and proves a general maximum-entropy proposition with the uniform, Gaussian and geometric distributions as corollaries (`mdl-exponential-family.md`, "The Maximum-Entropy Proposition"). The first section of this chapter then defines and proves all of these again. The introduction should say which chapter is the home. See INF-S1 in `mdl-information-theory.md`.
2. **The introduction promises examples that are not the ones the reader meets first.** It names "a pair of distributions on three outcomes, and a pair of correlated Gaussian variables". In the first two sections the examples that carry the argument are the dyadic source $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ and the coin with $P(\texttt{a}) = 0.9$; the correlated Gaussian pair does not appear until the fourth section, and the three-outcome pair does not appear in the last two sections at all.

## Passage-level findings

**IDX-1** (moderate, pedagogy). Anchor, first paragraph: "1.2 nats per character, is that good? How low could it go? And when we"

The question is concrete and a reader will remember it. The chapter answers "how low could it go" in general (the entropy rate) but never returns to 1.2 nats per character. The answer is available in `mdl-coding-compression.md`: 1.2 nats is 1.73 bits per character, the order-3 counting model of that section needs 2.55, and Shannon's estimate for English is about 1. Fix: either answer it in one sentence in the entropy-rate subsection (see COD-13) or change the opening number to one the chapter does compute.

**IDX-2** (moderate, language). Anchor, route paragraph: ":numref:`sec_mdl-information_theory` asks what a log-loss measures and how"

Five consecutive sentences have the form "section X asks ...". Guide section 12.1 names this construction as the one readers skip. Fix: group the five sections into two or three movements. A draft: "The first two sections stay with one distribution and one model: what a log-loss measures, how low it can go, and why it is the length of a file (:numref:`sec_mdl-information_theory`, :numref:`sec_mdl-coding-compression`). :numref:`sec_mdl-divergences-distances` then asks what to do when the KL divergence is infinite or uninformative. The last two sections turn from two distributions to two variables: how much one tells about the other, and how much of that can be measured from samples."

**IDX-3** (minor, thought). Anchor, third paragraph: "small examples recur: a pair of distributions on three outcomes, and a pair"

See structural finding 2. Name the examples the reader actually meets: the dyadic source and the 0.9 coin in the first two sections, the three-outcome pair through the third, the Gaussian pair from the fourth.

**IDX-4** (minor, thought). Anchor, last paragraph before the table of contents: "We assume the probability of :numref:`chap_mdl-probability-statistics` and"

Say what is assumed more exactly, and say that entropy and Gibbs' inequality, which chapter 41 uses, are developed here from the beginning. A lookup reader then knows this chapter can be read first.

## For the chapter review

- The introduction is sound and needs only the route paragraph and the two loose promises fixed.
- The part introduction (`chapter_mathematics-for-deep-learning/index.md`) describes this chapter as "entropy, cross-entropy, and the KL divergence that most losses descend from". That undersells it: codes, divergences beyond KL, mutual information and its estimation are four of the five sections.
- The main text (`chapter_linear-classification/softmax-regression.md`, "Entropy and Cross-Entropy") says Gibbs' inequality is proved in `sec_mdl-information_theory` "together with the interpretation of entropy as the length of the shortest code". The code interpretation now lives in `sec_mdl-coding-compression`. The pointer is stale.
- Bits-back coding, which reads the VAE objective as a code length, is not in the chapter or its Further Reading notebook. A deep learning reader who has just learned that a log-loss is a file size will look for it.
