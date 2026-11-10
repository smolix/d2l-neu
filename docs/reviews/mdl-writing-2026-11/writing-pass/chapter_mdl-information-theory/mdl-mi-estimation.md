# Writing pass log: Estimating Mutual Information

File: `chapter_mdl-information-theory/mdl-mi-estimation.md`. Not split (no
markers). No code cell, cell ID, figure include, label or slide changed.
Checker: "OK: only prose changed", no em-dash warning; `lint_source.py` clean.

## Words

Checker count (body, captions, exercises): 7,099 before, 7,185 after (+86).
The section is longer by the sentences the brief exempts: the four numbered
steps of the distribution-free proof (EST-5, about 50 words), the hypothesis
in Donsker and Varadhan (EST-9), the marginal-law sentence in InfoNCE Step 3
(EST-11, 25 words), and the definition of the histogram estimator that makes
exercise 1 self-contained (35 words). Everything else nets to a cut. One
display equation (the re-derived $\log k - L$) became inline with a
reference.

Numeric checks were run with `python3` and NumPy only (no SciPy, no
PyTorch): KSG was reimplemented with a hand-written digamma and a NumPy
Chebyshev distance. Script:
`/private/tmp/claude-501/-Users-smola-Repositories-github-d2l-neu/3e1abec1-b823-4073-adea-e57cf85abc34/scratchpad/est/chk.py`.

## Correctness changes

| Finding | Before | Now | Verified |
|---|---|---|---|
| EST-17 | Summary: "can be bounded from below but not measured" | "can be measured when it is small and the dimension is low; beyond that, every method reports at most the logarithm of something it can afford" | KSG 0.79 to 0.81 against 0.83 at $d=1$ (store and my runs). |
| EST-16 | Summary: "A learned critic reproduces the exact one ... so the gap ... belongs to the bound" | "On the Gaussian pair a learned critic reproduces ..., so the gap there belongs to the bound"; body adds that a scalar pair is an easy job and that in high dimension the two errors cannot be separated without a known truth | Stored table: 0.57/0.57, 1.40/1.42, 1.77/1.79, 1.88/1.91. |
| EST-14 | InfoNCE applied to (features, label): "only with $N \ge e^{6.36} \approx 581$ candidates, ... more than a thousand" | Fano gives a scale (6.36 nats for 5% error on 1,000 classes); a loss with 128 candidates reports at most 4.85, and $N=1024$ comes close only at about 7 nats shared; classification quality "cannot be read off this number" | 6.36 is the other section's stored output; exact-critic InfoNCE at $N=1024$, $I=7$: 6.31 (60 batches). The 581 is gone ($e^{6.36}=578$). |
| EST-9 | "Proposition (Donsker–Varadhan representation)", hypothesis $\mathbb{E}_Q[e^T]<\infty$ | "Proposition (Donsker–Varadhan bound)", adds "$\mathbb{E}_P[T]$ is defined"; the supremum clause is named the representation. After NWJ: "For any other critic it is smaller than the DV bound, which is what the tangent inequality says." | $\log z \le z/e$ with equality only at $z=e$. |
| EST-11 | Step 3 divided by $p(v)$ without saying it is the bag's marginal | "The marginal distribution of the bag $V$ is $p(v)$, because the positive is marginally a draw from $P_Y$ and is independent of the negatives." | By construction of the bag. |
| EST-5 | the key step compressed into "The two events overlap" | four numbered steps; step 2 applies the guarantee to the uniform distribution on $n^2$ symbols; $B$ is stated to be a function of the type before the steps | $n(n-1)/(2n^2) < 1/2$; $1-\delta > 1/2$. |
| EST-6 | "all it can rely on is which values repeat" (unqualified) | "an estimator that assumes nothing about the coordinates ... can rely on [only] which values repeat"; KSG "assum[es] ... that distances in the given coordinates mean something and the density is smooth in them" | wording. |
| EST-12 | "which is one reason CPC, SimCLR and CLIP use large batches" | "which is the reason usually given for the large batches of ..." | softened. |
| EST-13 | curve readings 3.7, 4.6, 6.3 stated flat | "about 3.7" and "about 6.3"; the 4.6 reading dropped here (the next table prints it) | recomputed 3.68 and 6.31. |
| EST-4 / structural 1 | "it does as well in two (the cell does not print that case)", "about what it reports at $d=8$", counts at $d=8$ | $d=2$ and $d=8$ claims dropped; the box-side argument now precedes the cell as the prediction; counts 70, 31, 26 kept, flagged as not printed and attributed to exercise 3 | my runs: counts 72, 31, 26; KSG 2.52 and 2.79. |
| EST-2 | digamma "to remove the bias of small counts", unexplained | adds: the mass $p$ of the box reaching the $m$-th of $n-1$ neighbors has $\mathbb{E}[\log p]=\psi(m)-\psi(n)$, "we do not derive the correction" | that mass is Beta$(m, n-m)$, whose mean log is $\psi(m)-\psi(n)$. I did not use the reviewer's wording (a count in a fixed region), which is not exact. |
| Structural 3 ("certify") | three meanings | defined once at the head of "Variational Lower Bounds" as the population inequality; KSG now "reports"; reconciliation with the distribution-free limit added at the end of "Exact Critics Compared" | see "claims added". In the distribution-free subsection "certify" keeps its high-confidence sense (the protected sentence about 177 and 30 nats). |

No finding disputed.

## Numbers no cell prints, kept after recomputation

- Strip counts about 70, 31, 26 at $d = 1, 4, 16$ (72, 31, 26).
- Box sides 0.05 and 0.83 ($0.0025^{1/2}$, $0.0025^{1/32}$).
- $\psi(2000)-\psi(5)=6.09$, $\log(n/m)=5.99$.
- Curve readings about 3.7 ($N=128$, $I=4$) and about 6.3 ($N=1024$, $I=7$).
- At $I=6$, $N=128$: more than half of the batches have a mean of
  $e^{\textrm{pmi}}$ below 0.1 (61%); about one in four holds a pair with
  $e^{\textrm{pmi}}\ge N$ (26%); standard deviation 36 nats (35.7).
- $4.3\times10^{-66}$, 177 and 27.6 nats.

Dropped: KSG at $d=2$ and $d=8$; "median at most a few hundredths"; 581;
the classifier example $\ln 10 - 0.3 = 2.0$ (it belongs to the certificate's
owner); the recap numbers 1.78 against 1.96 of the histogram.

## Six before/after pairs

1. Before: "How large can the estimate be? Each summand of
   :eqref:`eq_mdl-ksg` involves only counts, so counting alone bounds it."
   After: "Each summand of :eqref:`eq_mdl-ksg` involves only counts, and
   counting alone bounds how large the estimate can be."
2. Before: "Does the estimator keep its accuracy in more dimensions?"
   After: "In more dimensions the estimator fails well below its ceiling,
   and for a different reason. ... The estimate should therefore fall behind
   the truth as $d$ grows. By how much?"
3. Before: "With the exact critic $T = \textrm{pmi}$, how does the estimate
   depend on the true information and on $N$?" After: "One might guess that
   with the exact critic $T = \textrm{pmi}$ the estimate follows the truth up
   to $\log N$ and stops there." (and, after the cell, "The guess is wrong.")
4. Before: "The rare pairs described in the NWJ subsection explain these
   numbers." (after the table) After: "The rare pairs of the NWJ subsection
   let us predict the outcome." (before the cell; after it, "At $I = 6$ the
   table is the one predicted.")
5. Before: "Does the learned critic reproduce the exact one? The evaluation
   cell cuts the evaluation pool ..." After: "The comparison that answers
   the question puts both critics on the *same* batches."
6. Before: "The term $e^T$ has a different problem." After: "Unbiasedness
   has a price, and it is paid in the term $e^T$."

## Repetition and ownership

- Loss certificate (owner: mutual information). The first Barber and Agakov
  instance now cites :eqref:`eq_mdl-info-certificate` and keeps only "no
  10-class classifier certifies more than $\ln 10$. InfoNCE will have this
  shape." The display and the 0.3 example are gone.
- Histogram estimation (owner: this file). The recap of the other section's
  numbers is a clause. The subsection now states the complete account in
  prose: coarse bins hide dependence, fine bins invent it (about
  $(b-1)^2/(2n)$ for an independent pair), the estimate is not invariant
  under monotone maps, and $b^d$ cells in high dimension. The cells remain
  in `mdl-mutual-information.md` (deferred).
- Distribution-free proof: the body owns it. Exercise 6(a) to (c) replaced
  by two parts the body does not answer (the constant
  $M \ge n^2/(2(1-\delta))$; the sample size at which $2\log n$ reaches 177
  nats); old (d) kept.
- "The previous section": seven uses down to three; the others are labels.
- The NWJ paragraph is three (EST-10); the rare-pairs explanation is told
  once, before the comparison cell.
- One ending: the "We can now name ..." sentence is deleted; "Two questions
  remain open ..." ends the last subsection. The Summary is shorter and
  qualified.
- The temperature paragraph no longer points to the distillation
  temperature (the entropy section is cutting that material).

## Ranking, demotions, labels

No Proposition demoted (all four are cited or carry the section). The
opening now names the two results that matter and says the nearest-neighbor
estimator can be skipped. No label added or removed.

## Structure

"The Compression-Phase Debate" is now `###` (label kept), the last
subsection of "InfoNCE", as assigned. It is topically a nonparametric
subsection; the review's other option (after the histogram subsection)
would file it better and no cell would move. Left for the author.

## Cells to cut or redesign on the development machine

- `mdl-mi-estimation-estimating-mutual-information-2` (no finding ID; rule
  on confirming cells): the two prints confirm identities; the cell must
  stay for its helpers. Prose reduced to one sentence.
- `mdl-mi-estimation-ksg-in-dimension` (EST-4): add $d = 2, 8$ and the mean
  strip count.
- `mutual-information-log-n-ceiling` (EST-13): print the quoted values; fix
  the legend.
- `mutual-information-perfect-critic-bounds` (EST-15): drop the eleven-line
  plot in favour of the table; print the fraction of batches with a large
  pair.

## Other deferred findings

- Structural 4: move the three histogram cells here from
  `mdl-mutual-information.md` (code move); exercise 1 then finds its
  function.
- EST-3: caption shortened (edge-case parenthesis cut); the figure and the
  InfoNCE figure's "nearer candidates score higher" are unchanged.
- Exercise 9 still cites `subsec_negative-sampling` and `sec_word2vec_data`
  (Attic chapter; both labels resolve today). Repoint or move the exercise
  when the new word-embedding section exists.
- $\delta$ is the failure probability here and the flip probability in the
  mutual-information section (not changed, shared symbol).

## Claims added or changed

- $\mathbb{E}[\log p]=\psi(m)-\psi(n)$ for the mass of the $m$-th-neighbor
  box (Beta$(m,n-m)$).
- Histogram: mean estimate about $(b-1)^2/(2n)$ for an independent pair
  ($2n\hat I$ is the $G$-statistic, mean $(b-1)^2$; exercise 1 shows where
  the approximation fails); not invariant under monotone maps.
- "the truth quadrupled between the last two rows" (3.32 to 13.29); KSG at
  $d=16$ is "less than half of its 6.1-nat ceiling" (2.8).
- NWJ is smaller than DV for every critic but the optimal one.
- "InfoNCE reports at most $\log N$ with $N$ at most the sample size, so it
  respects the [distribution-free] limit by construction, while DV and NWJ
  can report more, and their batch values are then no longer guarantees."
- "While $I$ is small, independent pairs that look like partners are common
  and the three bounds should agree" (stored table up to $I=3$).
- Exercise 6(a): any integer $M\ge n^2/(2(1-\delta))$ gives
  $B(\mathcal T_0)\le\log M$ (then $n(n-1)/(2M) < 1-\delta$, so the two
  events overlap); at $\delta=0.05$ the gain is $\log 1.9 = 0.64$ nats.
  6(b): $2\log n\ge 256\ln 2$ needs $n\ge 2^{128}$.

## Slides that now disagree with the body

- "The strips hold about 26 of 2,000 points at $d = 8$ and at ..." (the
  body no longer quotes $d=8$).
- "A 10-class classifier at loss 0.3 certifies $\ln 10 - 0.3 = 2.0$" (the
  example is no longer in this body).
- The cover kicker and any slide treating the compression debate as its own
  top-level part.
