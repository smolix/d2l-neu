# Writing pass: chapter 42 introduction and further reading

Files: `chapter_mdl-information-theory/index.md`,
`chapter_mdl-information-theory/further-reading.md`. No code cells, no slides
sections; no label, toc line or entry of the "Resources and Further Reading"
list touched. Both pass `check_mdl_prose.py` ("OK: only prose changed", no
em-dash warning) and `tools/lint_source.py` (clean).

## Words

| File | `wc -w` before | after | checker before | after |
|---|---|---|---|---|
| `index.md`, whole file | 916 | 928 | 1023 | 1049 |
| `index.md`, introduction only (title to toc) | 383 | 395 | | |
| `further-reading.md` | 1584 | 1609 | 1654 | 1683 |

Both counts rise only by reference tokens. The introduction has 13 `:numref:`
against 7 (six new, each one `wc` word), so its prose is 382 words against
376. Further reading has 27 label references against 18, two new citations
and two display equations reformatted to `$$` on their own lines (four extra
tokens); without these 15 tokens it is 10 words longer than before, which I
could not remove without dropping a warning sentence the review asked for.

## Introduction (`index.md`)

Kept verbatim: the first paragraph (the review asks for it; its "1.2 nats per
character" is now answered in the coding section, "The Entropy Rate"), and
the Shannon paragraph up to "It is also the floor below which no model's
cross-entropy can fall."

- **IDX-2.** The five "asks ..." sentences are gone. One paragraph ranks
  three results (Gibbs' inequality as the floor; arithmetic coding, log-loss
  plus at most two bits; a critic trained on samples bounds a divergence from
  below, with the $\log N$ ceiling of a contrastive bound). One paragraph
  gives the route in three parts.
- **Survives the split.** The second part is named "from
  :numref:`sec_mdl-divergences-distances` on" and described by content
  ("other functions of the density ratio, test functions and transport, and
  scores"). No section number or count appears; "first part", "second",
  "third" count movements, not sections.
- **What can be skipped.** "It needs from the second only data processing
  and the critic bound, so transport and scores can wait until
  :numref:`chap_gans` or :numref:`chap_diffusion` sends you back." Verified
  by grep: the two mutual-information sections cite the divergences section
  only for :eqref:`eq_mdl-f-div-dpi` and for the f-GAN bound.
- **Main-text uses, by label** (all resolve): `subsec_info_theory_basics`
  (softmax regression), `subsec_perplexity` (language models),
  `sec_gan_objectives`, `chap_gans`, `chap_diffusion`.
- **IDX-4 and the agreed division with chapter 41.** "We assume the convexity
  of ... and the probability of :numref:`chap_mdl-probability-statistics`,
  where :numref:`sec_mdl-exponential-family` proves Gibbs' inequality for its
  own use; entropy and the KL divergence are developed here from the
  beginning." This matches the entropy section's opening ("There the
  inequality was a tool. Here it is the subject.").
- **Appendix as additional reading.** "The main text needs little of this
  (:numref:`subsec_info_theory_basics`); the chapter is for readers who want
  the proofs."
- **IDX-3.** The sentence on the two recurring examples is dropped, for
  length. It was inaccurate (the first two sections run on the dyadic source
  and the 0.9 coin as much as on the three-outcome pair), and each section now
  introduces its own running example.
- **Cut from the Shannon paragraph** to pay for the ranking: "A handful of
  related ideas follow from it ... which way to move." (the ranking paragraph
  says the same with results instead of topics).

Judgments that may be too strong: "The main text needs little of this"; the
choice of three results, which leaves out data processing, Fano and the loss
certificate (the mutual-information section ranks those itself); "a critic
trained on samples bounds a divergence from below" is the population
statement (a minibatch value can overshoot, as the estimation section shows).

## Further reading: correctness

- **FR-6.** Was: "Every estimator of :numref:`sec_mdl-mi-estimation` is a
  lower bound". Now: "Every variational estimator of ... is a lower bound".
  Verified against the section (KSG is not a bound; the Summary there now
  says the same).
- **SVGD, letters of the divergences section** ($P$ sample or data, $Q$ the
  distribution known through its score). Was: target $P$ with $s_P$,
  particles $Q$, $\mathbb{E}_{x\sim Q}[k\,s_P + \nabla k]$,
  $D_{\textrm{KL}}(Q\|P)$. Now: "how far a sample from $P$ is from a
  distribution $Q$ known only through its score $s_Q$";
  $\phi^\star(\cdot) = \mathbb{E}_{x \sim P}[k(x,\cdot)\,s_Q(x) + \nabla_x k(x,\cdot)]$;
  decreases $D_{\textrm{KL}}(P\|Q)$; "Its norm is the kernel Stein
  discrepancy, the square root of :eqref:`eq_mdl-ksd-def`". Checked term by
  term against Liu and Wang's formula (their $q$ is the particles, $p$ the
  target) and against $\mathrm{KSD}^2(P,Q) = \mathbb{E}_{x,x'\sim P}[k_Q]$:
  $\|\mathbb{E}_P[\mathcal{A}_Q k(x,\cdot)]\|^2$ is that double expectation.
  Note for the author: in SVGD the "target" is $Q$, although the section's
  convention sentence reads "$P$ is the data or the target"; the entry avoids
  the word.
- **Information geometry (FR-3).** $\mathbf{F}$ and $d\boldsymbol\theta$ are
  gone; the entry writes $\mathbf{I}(\boldsymbol\theta)$ and cites
  :eqref:`eq_mdl-f-div-local` for the quadratic. The uniqueness statement is
  now named (Chentsov's theorem) and cited to `Amari.2016`, since `d2l.bib`
  has no entry for Chentsov and I may not edit it.
- **CLUB (FR-6, second part).** "replaces the fixed $r$ by the batch" is now
  "needs no $r$: it averages $\log p(z\mid x)$ over the matched pairs of a
  batch and subtracts its average over the mismatched pairs. It is an upper
  bound in expectation when the density is the encoder's own; a learned
  approximation needs an extra condition." (Cheng et al. prove the bound for
  the known conditional and give a sufficient condition for the variational
  one; from memory of the paper, not re-read.) The variational bound and its
  gap are no longer restated: the mutual-information section now derives
  them, and the entry cites `sec_mdl-information-bottleneck`.
- **Polar codes (FR-9).** "the first family proved to achieve the capacity"
  is now "provably achieve it on symmetric binary-input channels"; the
  priority claim is dropped.
- **Goldfeld et al.** Was: "the measured compression depends on the noise
  scale, and it tracks how the representations of each class cluster". I
  could not confirm the noise-scale statement. Now: "It can then fall during
  training, and they trace the fall to the clustering of the representations
  of each class, which, they argue, is also what the binned estimates on
  deterministic networks measured" (the paper's abstract, from memory).
- **Blier and Ollivier (FR-10).** Now: "code the training labels of deep
  networks in this way and find the code far shorter than two-part and
  variational codes: a large network does compress its labels." The
  consequence the review asked for is stated without "more parameters than
  data points", which I did not check.
- **Calibration (FR-1).** Added: "one constant cannot help a model that is
  overconfident on some classes and underconfident on others." I did not use
  the review's "cannot change the ranking of examples": with more than two
  classes one temperature can reorder examples by confidence.

Not re-verified, left as the reviewer checked them: Lempel and Ziv
universality, the scaling-law entry (kept verbatim; Kaplan et al. fit pure
power laws, i.e. $L_\infty = 0$, which the stated form includes), Rényi
monotonicity and additivity, Ballé et al., Song and Ermon, Poole et al.

## Further reading: opening, ranking, repetition

- **Opening.** The template paragraph (and "Each topic states the problem it
  addresses ...") is replaced. The new one ranks: "Read first the three that
  bear on practice: calibration, the caveats on scaling laws, and the bias
  and variance of mutual-information estimators. The entries on capacity and
  on information geometry are for the curious."
- **Against the resource list.** "Books and papers for what the chapter
  teaches are listed in its introduction ...; these entries are for what it
  does not." No entry repeats an item of the list.
- **Fisher information as the local quadratic**: one clause with the
  equation reference; the entry keeps Chentsov, the natural gradient, K-FAC
  (`Martens.Grosse.2015`) and the KL trust region of `sec_ppo`.
- **Rényi**: opens with the privacy question; "not an f-divergence but an
  increasing function of one" is replaced by a reference to exercise
  `ex_mdl-divergences-distances-recover-kl-reverse-kl` (d). The display
  stays, since additivity is read off it.
- **Sinkhorn**: the definition, the bias and the two limits (Wasserstein and
  MMD) belong to `subsec_mdl-sinkhorn` and its exercise, which already has
  both limits. The entry keeps only the theorem of Feydy et al. and the
  sliced distance, under one common question (FR-5), with a use
  (`Deshpande.Zhang.Schwing.2018`).
- **Lipschitz critics**: pointer to `sec_gan-further-reading` and
  `sec_gan_convergence`, plus the one remark on unit slope along transport
  lines.
- **Reliability diagrams**: cited as `subsec_beyond-accuracy` (the
  Calibration subsection there has no label of its own).
- **Estimators**: the result of `subsec_mdl-exact-critics` is one sentence;
  the binning argument of the compression debate is one clause.

## Before and after

1. Before: ":numref:`sec_mdl-information_theory` asks what a log-loss measures
   and how low it can go." After: "The first part stays with one distribution
   and one model: what a log-loss measures and why it is a file length".
2. Before: ":numref:`sec_mdl-mi-estimation` asks what samples alone can tell
   us about dependence." After: "The third turns to two variables: how much
   one tells about the other ..., and how much of that samples can show".
3. Before: "The chapter builds these ideas from one starting point, the
   surprise of a single outcome, and ties each to a loss or a model from
   earlier chapters." After: "The main text needs little of this ...; the
   chapter is for readers who want the proofs. Three results matter most."
4. Before: "This section collects, in the order of the chapter, topics they
   had no room for. Each topic states the problem it addresses and what it
   changes relative to the chapter." After: "The chapter kept to what can be
   proved in a page and checked on a small example, and each entry below
   starts where a section stopped."
5. Before: "Every estimator of :numref:`sec_mdl-mi-estimation` is a lower
   bound, so none can certify ..." After: "Every variational estimator of
   :numref:`sec_mdl-mi-estimation` is a lower bound, so none can certify ..."
6. Before: "The entropic transport cost of :numref:`subsec_mdl-sinkhorn` is
   not zero for two copies of one distribution, and in many dimensions exact
   transport needs a linear program over $n^2$ variables." After: "How is a
   transport distance used as a loss on minibatches in many dimensions?"

## Claims added or changed

Introduction: arithmetic coding costs the log-loss plus at most two bits
(:eqref:`eq_mdl-arith-bound`); a contrastive bound with $N$ candidates never
exceeds $\log N$; the third part needs from the second only data processing
and the critic bound. Further reading: the $\ln 100$ "was a change of
reference"; the KL penalty of a variational autoencoder does not depend on
the coordinates of the latent variable; the Sinkhorn divergence is zero only
for equal distributions (Feydy et al., "positive definite"); image generators
are trained on the sliced distance; belief propagation decodes low-density
parity-check codes; the KL trust region is, to second order, a ball in the
Fisher metric; and the items under "correctness" above.

## Deferred

- **FR-7, bits-back coding.** Not added: the file may not grow and `d2l.bib`
  has no entry for it. It belongs after "Learned Lossy Compression". The
  second half of FR-7 (FID and KID) was done by the divergences writer in the
  body.
- **Chentsov** needs a bibliography entry; then replace `Amari.2016` at the
  theorem.
- **After the split of the divergences section:** the heading "Divergences"
  here holds five entries; Rényi and information geometry go with the
  f-divergence part, Sinkhorn and Lipschitz critics with transport, SVGD with
  scores. The references used (`sec_mdl-f-divergences`,
  `subsec_mdl-sinkhorn`, `sec_mdl-optimal-transport`,
  `sec_mdl-stein-discrepancy`, two equation labels, two exercise labels) are
  all subsection-level and survive it.
- The Resources paragraph of the introduction still lists "Rényi
  divergences" among topics the chapter does not teach; the chapter now has
  them in an exercise and one clause. Left, as the list is not to be edited.
- The part introduction's description of this chapter and the stale pointer
  in `softmax-regression.md` (chapter review) are in files I may not edit.
- The chapter's scope sentence still says information geometry is left out,
  while the divergences section has the Fisher quadratic and this file has
  an entry; I read "left out" as "not taught" and kept it.
