# Further Reading

Source: `chapter_mdl-information-theory/further-reading.md`. About 1,550 words, 15 topics under five headings that follow the chapter's sections, two displayed formulas, no code, no figures, no exercises.

## Verdict

This is a good instance of the house form (writing guide §12.3): each topic opens with the problem it addresses, says what it changes relative to the chapter, and gives one or two citations with a clause on what each adds. It is within the length budget, nothing depends on it, and it does not repeat the resource list of the chapter introduction. I checked the technical statements against what I know of the cited work and against the sections they refer to, and found one inaccurate sentence (FR-6: "Every estimator ... is a lower bound", which the KSG estimator of the same section contradicts) and nothing else wrong. As a reading guide it is accurate and economical but uneven in usefulness. The best entries tell a deep learning reader something to do or to watch for (temperature scaling, tokenizers and scaling laws, the gradient penalty). Three entries duplicate material that already exists elsewhere in the book (FR-2, FR-3, FR-4), and the selection leaves out the two connections a reader of this chapter is most likely to look for (FR-7). The most important fix is to replace the duplicated entries by pointers and use the space for those two.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 3 | 4 | 3 |

Pedagogy is judged as a guide: does the reader know what to read next, and why? Mostly yes, but each entry has the same shape and weight, so the reader cannot tell which of the fifteen matter for deep learning practice and which are for the curious.

## What works

- **Problem first, every time.** "How much does a randomized algorithm reveal about one record when it is run many times?" "Can the held-out loss of a language model be predicted before training it?" These openings make a reader want the answer.
- **Each entry states its one difference from the chapter.** "Every coder of :numref:`sec_mdl-coding-compression` is driven by a model that predicts the next symbol. The Lempel–Ziv algorithms need none." "A *prequential* code sends no parameter".
- **Links back to computed results.** "`zlib`, which combines LZ77 with a Huffman code, loses on *The Time Machine* to the small adaptive context model that opens that section." The claim matches the coding section (2.84 against 2.55 bits per character).
- **Sentences that warn.** "The fitted $L_\infty$ is a regression constant, not a measured entropy rate of language, and since the loss is per token ..., fits with different tokenizers cannot be compared directly." "This is the temperature of distillation used for a different purpose." "It gives no global guarantee, but it targets the property of the optimal function, which has unit slope along the lines that transport moves mass on." These are the sentences with a point of view, and the section would be better with one in every entry.
- **The Rényi entry** explains in three clauses why the divergence is used for privacy accounting (the logarithm outside the sum, hence additivity, hence "the bounds of repeated runs simply add"). This is the model for an entry.
- **The closing entry** on noisy networks finishes the compression-phase story of `subsec_mdl-compression-debate` honestly: the question becomes well posed, and the answer depends on the noise scale.

## Structural findings

- **Three entries repeat other parts of the book.** "Lipschitz Critics in Wasserstein GANs" is nearly the same paragraph as "Three ways to a Lipschitz critic" in `chapter_generative-adversarial-networks/further-reading.md` (same three methods, same citations, same contrast between a global loose bound and a local tight one), and `convergence.md` of that chapter treats the gradient penalty in the text. "Information Geometry" restates the Fisher-information paragraph of `sec_mdl-f-divergences` with another symbol. "Rényi Divergences and Privacy" repeats the definition and the "increasing function of an f-divergence" remark of exercise 1(d) in `mdl-divergences-distances.md`, and "Sinkhorn Divergences" repeats exercise 11(b) there. See FR-2 to FR-5.
- **The heaviest heading is "Divergences"** with five entries and about 480 words, a third of the section, for a part of the chapter that is itself overlong. If the kernel Stein discrepancy moves here from the main section, as the review of that section proposes, it should merge with the SVGD entry, not add a sixth.
- **Even weight.** Fifteen entries of 70 to 130 words each. Calibration, which every practitioner meets, gets the same space as polar codes, which the chapter introduction says the chapter leaves out and which no deep learning reader needs. A sentence at the top that ranks ("If you read three of these: calibration, the scaling-law caveats, and the limits of mutual-information estimators") would use the author's judgment where the reader has none.
- **The introduction is accurate but flat.** "Each topic states the problem it addresses and what it changes relative to the chapter" describes the template to the reader. The reader does not need the template named; cut that sentence.
- **Is anything here better placed in the main text?** No. Calibration is the one topic with enough practical weight to raise the question, and the Basics part already treats reliability diagrams in the classification chapter; a labelled reference to that section would help (FR-1).

## Passage-level findings

**FR-1** (minor, pedagogy). Anchor: "systematically overconfident. They measure the gap with a reliability", under "Calibration after Training".
The entry introduces the reliability diagram as if new, but the main text computes one for a trained classifier in the linear-classification chapter. Refer to that section by label and spend the saved clause on what temperature scaling cannot repair (it rescales confidence uniformly and cannot change the ranking of examples), which is the practical limit a reader needs.

**FR-2** (moderate, thought). Anchor: "Weight clipping :cite:`Arjovsky.Chintala.Bottou.2017` and spectral", under "Lipschitz Critics in Wasserstein GANs".
Duplicate of the GAN chapter's further-reading paragraph "Three ways to a Lipschitz critic", almost sentence for sentence ("bound the product of the layers' Lipschitz constants, a global bound that is often loose" here; "bound the product of the layers' Lipschitz constants, a bound that holds everywhere but is often loose" there). The GAN chapter is the right home, since it also explains why zero-centered penalties replaced the one-centered one. Replace this entry by one sentence with a labelled reference (`sec_gan_convergence` and that chapter's further reading), and keep only the remark that is specific to the appendix: the optimal Kantorovich function has unit slope along transport lines, which is what a gradient penalty asks for.

**FR-3** (moderate, thought). Anchor: "Near equality every smooth f-divergence is the same quadratic", under "Information Geometry".
The first three sentences repeat the closing paragraph of "The Local Quadratic" in `mdl-divergences-distances.md`, with $\mathbf{F}$ for the Fisher information where that section and `sec_mdl-maximum_likelihood` write $\mathbf{I}$, and with $d\boldsymbol\theta$ where that section writes $\mathbf{h}$. The entry's own contribution is one sentence: the Fisher metric is, up to scale, the only one invariant under sufficient statistics. That sentence has no citation (the result is Chentsov's). Cut the restatement to a clause, use the main section's symbols, cite the uniqueness theorem, and add the one fact a deep learning reader would act on: where natural-gradient methods appear in practice (K-FAC and its relatives, or the trust regions of policy optimization), with a reference into the main text if one exists.

**FR-4** (minor, thought). Anchor: "has its logarithm *outside* the sum. It is therefore not an f-divergence but", under "Rényi Divergences and Privacy".
The formula and the relation to f-divergences are exercise 1(d) of the divergences section. The guide's rule that nothing depends on this notebook is kept (the exercise is self-contained), but the reader meets the same display twice. Start the entry from the privacy question, as it already does, and cite the exercise for the definition's relation to $D_{f_\alpha}$. The mathematical statements are correct: additivity over independent components, monotonicity in $\alpha$, the limit at $\alpha \to 1$.

**FR-5** (minor, explanation). Anchor: "The entropic transport cost of :numref:`subsec_mdl-sinkhorn` is not zero for", under "Sinkhorn Divergences and Sliced Wasserstein".
Two unrelated remedies share an entry, joined by an opening sentence that states two unrelated problems ("is not zero for two copies of one distribution, and in many dimensions exact transport needs a linear program over $n^2$ variables"). The second problem is the one entropic regularization itself was introduced to solve, so the sentence reads as if Sinkhorn had not helped. Split into two short entries, or lead with the common question ("How is a transport distance used as a loss on minibatches in high dimension?") and give the two answers. The first half overlaps exercise 11(b) of the divergences section. For the sliced distance, name a deep learning use (it is a common training loss and evaluation metric for generative models) since "each direction costs a sort" says why it is cheap but not why one would want it.

**FR-6** (moderate, correctness; fairly sure). Anchor: "Every estimator of :numref:`sec_mdl-mi-estimation` is a lower bound, so none", under "Upper Bounds".
The section referred to begins with nonparametric estimation and the KSG estimator, which is not a bound in either direction (it has a ceiling of $\psi(n) - \psi(m)$ but can overestimate). The sentence is true of the variational estimators (Barber–Agakov, Donsker–Varadhan, NWJ, InfoNCE). Write "Every variational estimator of ...". The rest of the entry is correct: the variational upper bound is loose by exactly $D_{\textrm{KL}}(p(z)\,\|\,r(z))$, and CLUB contrasts matched with mismatched pairs. "CLUB ... replaces the fixed $r$ by the batch" is a loose description; say that it replaces the marginal term by an average of $\log p(z \mid x)$ over mismatched pairs, which is what the next clause states anyway.

**FR-7** (moderate, pedagogy). Anchor: "topics they had no room for. Each topic states the problem it addresses and", introduction.
Two topics that a reader of this chapter is likely to look for are absent. (1) **Bits-back coding and latent-variable models as compressors.** The chapter builds an arithmetic coder driven by a language model and treats the ELBO elsewhere in the appendix; the fact that a VAE's negative ELBO is an achievable code length, and how bits-back coding achieves it, is the natural continuation of "Learned Lossy Compression" on the lossless side. No section of the book mentions it. (2) **Evaluation metrics built from this chapter's distances.** FID is the squared $W_2$ between Gaussians and KID is an unbiased $\textrm{MMD}^2$; both are taught in `dcgan.md`, so a one-line pointer by label is enough, either here or in the divergences section. A third candidate, density-ratio estimation by classification, is covered in the GAN chapter and needs at most a pointer. The space is available from FR-2 and FR-3.

**FR-8** (minor, explanation). Anchor: "chosen units. :citet:`Jaynes.1968` obtains the form as the limit of", under "Entropy Relative to a Reference".
"The ordinary $h$ is this quantity with $m = 1$ in the chosen units" is the key sentence and is easy to miss; it explains why differential entropy changed under rescaling (the reference changed with the units). Put it right after the definition and add the consequence in the terms of the section it refers to: with metres and centimetres the two references differ by the factor $100$, which is the $\ln 100$. The entry does not say why a deep learning reader should care; one clause would do (the KL term of a VAE or of any maximum-entropy regularizer is such a relative entropy, which is why it, unlike a differential entropy, has a meaning independent of the parameterization).

**FR-9** (minor, language). Anchor: "Polar codes :cite:`Arikan.2009` were the first family proved to achieve", under "Codes That Approach Capacity".
Accurate, and the least connected entry. The chapter introduction says "We leave out the construction of practical error-correcting codes"; this entry then describes two constructions without a link to learning. Either add the link that exists (belief propagation on the code's graph is the same message passing as on other graphical models, and decoding is a standard test problem for learned decoders) or reduce the entry to one sentence.

**FR-10** (minor, explanation). Anchor: "of deep networks and find it far shorter than the lengths that two-part and", under "Description Length beyond Two-Part Codes".
"find it far shorter" needs its consequence to be worth reading: the short prequential code length is evidence that a network with many more parameters than data points still compresses its training labels, that is, it is consistent with generalization in the sense of `subsec_mdl-mdl`. Add that clause. The last sentence, on Kolmogorov complexity, is a second topic appended to the entry; it is fine at this length.

**FR-11** (minor, language). Anchor: "information, while InfoNCE is steady but capped at $\log N$. :citet:`Poole.Oord.Alemi.ea.2019` interpolate between the two", under "Trading Bias for Variance".
A long unbroken source line and a sentence that restates the result of `subsec_mdl-exact-critics` in full before saying anything new. Per §7.7, restate in one clause. The Song and Ermon summary is accurate (exponential variance, failed self-consistency checks, clipping at the price of bias).

## Exercises

None, which is right for this notebook.

## Still open from October

The October review asked to cut Rényi and Stein methods from the chapter. They were moved here instead, which is the appropriate treatment under §12.3. Nothing else from October concerns this file.

## For the chapter review

- The notebook follows the chapter's section order and its headings map one to one onto the five sections; if `mdl-divergences-distances.md` is split, the "Divergences" heading should split with it or shrink (two of its five entries are proposed for replacement by pointers, and the kernel Stein discrepancy would arrive from the main text).
- It keeps the guide's contract with the chapter introduction: the introduction lists resources for what the chapter teaches, this notebook covers what it does not, and neither repeats the other. The introduction's opening sentence points here, as required.
- Cross-chapter duplication is the pattern to check in the other further-reading notebooks of the part: this one shares a paragraph with the GAN chapter's further reading, and the appendix and main text were evidently written without comparing their closing notebooks.
- Notation should be aligned with the main sections in the chapter pass ($\mathbf{F}$ against $\mathbf{I}$ for the Fisher information; $N$ against $n$ for sample or batch size).
- Could not check: the bibliographic details of the citation keys (years and attributions were checked only against my own knowledge of the papers, not against the `.bib` file or the papers themselves).
