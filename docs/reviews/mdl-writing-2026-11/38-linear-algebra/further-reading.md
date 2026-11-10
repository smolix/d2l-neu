# Further Reading

Source: `chapter_mdl-linear-algebra/further-reading.md`. About 1,800 words, 15 topics under three `##` headings, no code, no figures, no exercises.

## Verdict

A useful and accurate reading guide, and a model of what writing guide 12.3 asks for. It is inside the length budget, nothing in the chapter depends on it, every topic starts from something the chapter taught and says what changes, and nearly every citation carries a clause saying what the work adds. I checked the technical statements against what I know of the sources (Johnson-Lindenstrauss, the conjugate-gradient bound, the Kreiss matrix theorem with the constant $e\,n$, Cheeger, the hardness of CP rank, Candès-Recht, dynamical isometry, and others) and found one imprecise sentence (FR-3) and one mismatch of symbols (FR-4). The most important thing to fix is small: the section reads as fifteen entries of equal weight, and a reader would be helped by being told which three or four to read first.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 3 |

## What works

- **Each topic is anchored in the chapter by a concrete object.** "The nearest-mean rule of :numref:`subsec_mdl-nearest-mean` compares an image with two class means through one dot product ... Kernel methods keep the rule and change the inner product". "A $10^6\times10^6$ matrix with five nonzeros per row is cheap to store, yet its inverse is typically dense, with $10^{12}$ entries." "A ratings matrix is observed in a few percent of its entries. Eckart--Young ... needs the whole matrix, so it does not apply."
- **Statements are quantitative where they can be.** The target dimension $\varepsilon^{-2}\log n$, the bound $2\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^k$, the bracket $K(\mathbf A)\le\sup_k\|\mathbf A^k\|_2\le e\,n\,K(\mathbf A)$, the cost $n^3$ against $2n^2$ for the two contraction orders.
- **Candour about evidence.** On the spectra of trained weights: "using the exponent to rank models by quality without test data rests on an empirical correlation, not a theorem." This is the judgment a reading guide exists to give.
- **The tensor paragraph says what breaks.** "Eckart--Young has no analogue: computing the CP rank is NP-hard, and a best approximation of a given CP rank need not exist."
- **It points back into the main text** ("Low Rank in Attention", "Steepest Descent under Other Norms"), so a reader who starts in the appendix learns where the book uses the mathematics.
- No overlap with the "Resources and Further Reading" list of the chapter introduction: that list has books and courses for what the chapter teaches, this has topics beyond it, and each points to the other.

## Structural findings

- **Flat (minor).** Fifteen topics of 90 to 150 words each, none marked as more important than another. For a deep learning reader the ones that matter most are, I would say, random projections, iterative solvers, random matrices and trainable depth, and sketching; the tensor-train paragraph and the Kreiss theorem are for few. One sentence in the opening paragraph that says so would give the section the voice it lacks.
- **One topic partly repeats the chapter (minor).** "Elimination, Determinants, and Rank in Practice" restates what :numref:`sec_mdl-linear-maps` already teaches in "Solving Linear Systems in Practice" and after the determinant ("Libraries factor the matrix into triangular ...") and what :numref:`subsec_mdl-svd-subspaces` says about numerical rank. What is new is two citations (why partial pivoting is stable in practice; Higham). Shorten to those two sentences or merge with "Sparse and Iterative Solvers" under "Solving Linear Systems at Scale".
- **Depends on a decision elsewhere.** "Pseudospectra and the Kreiss Matrix Theorem" opens with ":numref:`subsec_mdl-pseudospectra` showed". If that subsection is cut from the eigendecomposition section, as my review of it recommends (EIG-16), this topic becomes the home of the definition and needs two more sentences: the definition of the $\varepsilon$-pseudospectrum and the number 200.3 against 10 for the matrix $\mathbf M$.
- **Candidates to add (optional, and only if something goes).** Two topics a deep learning reader may look for here and not find: (i) differentiating through `eigh` and `svd`, whose gradients contain factors $1/(\lambda_i-\lambda_j)$ and blow up for close eigenvalues, a practical trap that follows directly from the chapter's perturbation results; (ii) the graph Laplacian, whose eigenvectors underlie spectral clustering and graph networks, the natural further reading for the spectral theorem and the Rayleigh quotient. I would not add more.

## Passage-level findings

**FR-1** (minor, language). Opening paragraph.

> Each topic states the problem it addresses and what it changes relative to

This sentence is the instruction of writing guide 12.3 addressed to the author, left in the text addressed to the reader. Delete it; the topics show it. The first sentence ("teach the geometry of one inner product, maps read through their columns, the eigenvectors of one square matrix") is a compressed table of contents that only someone who has read the chapter can decode; "the chapter" would do.

**FR-2** (minor, language, pattern: 4 instances). 

> one difference from matrices is that Eckart--Young has no analogue:

Also "The one change is to minimize the *nuclear norm*", "PiSSA ... changes the initialization", "DoRA ... changes the parameterization instead". The guide's phrase "the one thing it changes" has become the wording of the topics. Vary it; in the two quoted cases the sentence is better without the frame ("Eckart--Young has no analogue for tensors").

**FR-3** (moderate, correctness: imprecise, fairly sure). "Adapter Variants".

> $\mathbf{B}\mathbf{A}$ at zero, so early fine-tuning moves along random

LoRA starts $\mathbf B$ at zero and $\mathbf A$ at random. The update is zero at the start, and in the first steps only $\mathbf B$ receives a gradient, so the update lies in the row space of a random $\mathbf A$. "Moves along random directions" suggests random steps. Suggested: "LoRA starts with $\mathbf B=\mathbf 0$ and a random $\mathbf A$, so the first updates are confined to a random $k$-dimensional subspace of the inputs, unrelated to the pretrained weight."

**FR-4** (minor, correctness: symbols, confident). "Mixing Times and the Spectral Gap".

> smallest stationary probability. Cheeger's inequality bounds the gap

The paragraph defines $\lambda_\star$ as the largest modulus among eigenvalues other than 1 and bounds the mixing time by $1/(1-\lambda_\star)$, then says that Cheeger bounds "the gap $1-\lambda_2$". These are different quantities when the most negative eigenvalue is larger in modulus than $\lambda_2$: a nearly periodic chain mixes slowly however good its conductance is. Either say "for a lazy chain, where the two gaps coincide" or define the gap once and say which one Cheeger controls. Also "at rate $1/4$, its second eigenvalue" at the start: the chapter says "shrinks by a factor of 4 per click", which is clearer than "rate".

**FR-5** (minor, explanation). "Other Inner Products: Kernel Methods".

> space, and the rule becomes a Parzen-window classifier whose boundary can

"Parzen-window classifier" is a term the reader has not met and the sentence does not explain. Either drop the name or add what it means (each training point votes with weight $k(\mathbf x,\mathbf x_i)$). The closing sentence, that Schölkopf and Smola "start their book from exactly this classifier", is a nice link to the voice model of the writing guide and is true.

**FR-6** (minor, thought). "Spectra of Trained Weights".

> describes the spectrum of a matrix of independent noise, and the weights of

"The weights of an untrained network follow it" holds for the usual independent initializations and not for orthogonal initialization, which the previous topic has just recommended. Add "with independent random initialization".

## Exercises

None, as intended for this notebook.

## Still open from October

Not covered in October (the notebook did not exist in this form).

## For the chapter review

- The notebook does its job and needs only light edits. It is the best evidence that the chapter's authors know what is central: the judgment missing from the main sections (what matters, what is settled, what is only empirical) is present here.
- It could absorb two things from the main sections with no loss to them: the definition of pseudospectra (from the eigendecomposition section) and the description of randomized SVD (from "Measuring and Controlling the Spectrum" in the SVD section, where it has no cell and is repeated by an exercise and by the sketching topic here).
- The matrix-completion topic and exercise 12 of the SVD section open with nearly the same sentence; if the exercise stays, the topic can be shortened to the recovery theorem and its incoherence condition.
- The opening-paragraph leak of an authoring instruction (FR-1) recurs in the Further Reading notebooks of all six chapters of the part: `grep -l "Each topic states" chapter_mdl-*/further-reading.md` lists every one.
