# Chapter 1, Preliminaries: writing review (November 2026)

Sources: `chapter_preliminaries/{index,ndarray,pandas,linear-algebra,calculus,autograd,probability,lookup-api}.md`, read in full to the slides marker, with the stored PyTorch outputs and all figures rendered and checked. Per-notebook reviews:

- [index](ch01-preliminaries/index.md)
- [ndarray: Data Manipulation](ch01-preliminaries/ndarray.md)
- [pandas: Data Preprocessing](ch01-preliminaries/pandas.md)
- [linear-algebra: Linear Algebra](ch01-preliminaries/linear-algebra.md)
- [calculus: Calculus](ch01-preliminaries/calculus.md)
- [autograd: Automatic Differentiation](ch01-preliminaries/autograd.md)
- [probability: Probability and Statistics](ch01-preliminaries/probability.md)
- [lookup-api: Documentation](ch01-preliminaries/lookup-api.md)

## Verdict

This is a good chapter, clearly better than the October review found it, and in its two central sections (calculus and automatic differentiation) it is close to the standard of the model texts. Its method is right: each section opens with a question the reader can feel, works a small example with numbers that can be checked by hand, and closes by naming what is still missing. The numbers are trustworthy. I recomputed every worked figure in the chapter (the loss values and gradients of the line model, the eigen-iterates, the operation counts, the Bayes posteriors, the standard errors) and compared the prose with every stored output, and found no mathematical error and only two places where the prose and the page disagree in spirit (PRB-2, CAL-4). The explanations are precise and the language is plain, correct and free of the mannerisms the writing guide lists.

It is not yet at the bar for three reasons. First, it is long and in places overloaded: about 27,000 words of prose in one framework's view, with the linear algebra and probability sections each carrying two or three lessons. Second, a few structural defects remain that a reader feels directly: the plotting code at the end of the calculus section, the Bayes lesson inserted in the middle of the coin lesson, the documentation section's puzzle that is not a puzzle in PyTorch. Third, the voice. The prose has been cleaned so thoroughly that it is even in tone from first page to last. It is never wrong and rarely memorable. The model texts are economical too, but they let an author show: MacKay's biologist, Strang's four cents. Here the moments of that kind can be counted (Archimedes and the polygons; two homes that differ by "three square feet"; shoe size and reading level; the defense lawyer in an exercise). The chapter would be more enjoyable with a few more, and with less uniformity in how sections begin and end.

## Scores

1 to 5; 5 means the page would sit comfortably in Cover and Thomas or MacKay.

| Notebook | Explanation | Thought | Pedagogy | Language | Enjoyment | Prose words (PyTorch view) |
|---|:-:|:-:|:-:|:-:|:-:|--:|
| index | 4 | 4 | 4 | 4 | 3 | 400 |
| ndarray | 4 | 4 | 4 | 4 | 3 | 3,550 |
| pandas | 4 | 4 | 4 | 4 | 3 | 2,300 |
| linear-algebra | 4 | 4 | 4 | 4 | 4 | 5,350 |
| calculus | 5 | 4 | 4 | 4 | 4 | 4,900 |
| autograd | 4 | 5 | 4 | 4 | 4 | 3,250 |
| probability | 4 | 4 | 4 | 4 | 4 | 5,650 |
| lookup-api | 4 | 3 | 3 | 4 | 3 | 1,300 |

Best section: calculus, for the running example that carries eight ideas on numbers a reader can check, with autograd a close second for its line of thought. Weakest: lookup-api, because its one investigation has no surprise for the default reader and its other thread is dropped; it is also the least consequential.

## Arc and ordering

The order is right and the sections hand over to one another well. The closings are the best structural feature of the chapter: "The numbers in this section were made up" (tensors to tables), "What we cannot do yet is improve the weights" (linear algebra to calculus), "What remains is the labor" (calculus to autograd), and "Whether a lower loss reflects a pattern that will hold on new data, or only noise ... is a question of probability and statistics" (autograd to probability). A reader always knows why the next section exists.

The middle three sections form one continuous story: a weighted sum, its derivative, the machine that computes the derivative. The line model of the calculus section is reused in autograd down to the same gradient $(2, 4)$ and the same step to $(0.8, -0.4)$, which is exactly how a running example should work. The chapter has four running examples in all (the image minibatch, the house table, the line model, the coin), each confined to one or two sections. That is not too many, but two seams show: linear algebra promises to find out how the error changes "when $\mathbf{W}$ changes" and calculus then starts from a different, smaller model (see LA structural findings); and the norms part of linear algebra borrows a regression-style error vector that its own classifier does not produce.

Probability sits apart from the other sections, as it must, and the transition into it is well made. Internally it is the one section whose order should change (PRB structural findings). The documentation section is a coda and reads as one.

## The chapter-specific question: length, level, and reliance on the appendix

**Level.** Right for the stated reader in five of seven sections. The tensor, preprocessing, calculus and probability sections start from nothing beyond school mathematics and Python, define each term in words, and work small cases. A rusty reader can follow them. Two sections climb above "preliminaries" in their last third: linear algebra (spectral norm, power iteration with its provisos, FLOPs per byte of memory traffic) and automatic differentiation (forward and reverse sweeps, vector-Jacobian and Jacobian-vector products through `torch.func`, Hessian-vector products). These parts are correct and well written, and they are what a reader needs before chapters 9 to 13, not before chapter 2. The chapter does not tell the reader this.

**Length.** About 27,000 words of prose plus 5,500 words of exercises in one framework's view, something like a hundred printed pages with code and figures. That is long for a chapter whose title promises preliminaries, and three sections exceed the guide's 4,500-word ceiling (linear algebra, calculus counting the plotting block, probability). The length is mostly content, not padding: I found almost no sentence that says nothing. The excess is in topics. The cuts listed below remove roughly 3,000 words and 100 lines of code without touching any idea that chapters 2 to 8 use. After them the chapter is still substantial, and I think that is right: this chapter is where the book teaches the gradient, the backward pass and the standard error, and those deserve their space. What the chapter should add is a sentence of reading advice in its introduction and "can be skipped on a first reading" marks on the advanced subsections.

**Reliance on the appendix.** Light and healthy. There are about 17 references into `chapter_mdl-*` in the prose and exercises (October counted 34): three in the introduction, three in linear algebra, four in calculus, one in autograd, five in probability, one in preprocessing. Every one of them points to a proof or to further depth ("for a proof, see", "states the condition and proves the approximation", "derives both modes in detail"). None carries a step of an argument. I could follow every section without opening the appendix, and where the text leans on an unproved fact it says precisely what is unproved (the best instance is in calculus: "the approximation needs partial derivatives that change continuously"). This is the policy of the writing guide, section 7.6, carried out well. If anything the risk now runs the other way: the chapter proves Cauchy-Schwarz, develops power iteration and explains both autodiff modes, all of which the appendix presumably also does. I did not read the appendix chapters, so I cannot say how much is duplicated; it is worth a check by whoever reviews them, since material taught twice could be taught once here and cited there.

## The most important findings

1. **Plotting utilities end the calculus section** (CAL-9, major). Ninety lines of matplotlib plumbing between the closing paragraph and the summary of the best section in the chapter. Move to Tools. Open since October.
2. **Probability is two lessons, interleaved** (PRB structural findings; PRB-10). The coin question is asked at the top and answered 5,000 words later, with the whole of conditional probability and the diagnostic test in between. Split into two sections or reorder, and put the first test directly after Bayes' theorem.
3. **Linear algebra is overloaded** (LA structural findings; LA-5, LA-7, LA-11). Thin it: remove the bytes-per-FLOP argument, the unused tensor notation and the batched-product aside; give the linear-map paragraph room; derive the iterates of $\mathbf{S}^k\mathbf{e}_1$ from the eigen-decomposition before stating the general formula.
4. **The documentation section's puzzle does not work in PyTorch** (DOC-1, major for that section; DOC-2). `torch.ones(2, 3)` does what it looks like. Carry one real ambiguity (the gamma sampler's rate against scale, or a standard deviation's divisor) through all four steps.
5. **Missing definitions that later text uses**: the identity matrix (used in calculus, CAL-8, and never defined), the matrix inverse (needed on the first pages of chapter 2), and the two-standard-error rule with a one-sentence mention of the normal approximation (PRB structural findings, PRB-11). Each needs two or three sentences.
6. **The advanced third of autograd** (AG structural findings; AG-7, AG-8). Mark as optional or shorten to the cost argument; drop the Jacobian-filling cell and the higher-order subsection from the main line.
7. **Notation in probability** (PRB-4). $\mathcal{A}$ an event, $A$ a set of values, then $A$ a random variable, with $X$ and $Y$ used before and after; $p$ a probability and $p(x)$ a density. Rename once.
8. **Promised demonstrations that are not read out** (NDA-4, NDA-9; PRB-2). The view created by `reshape` is shown in an output and not mentioned; the stored multinomial sample is exactly 50 and 50 under a sentence that says this is rare.
9. **Motivations borrowed from far ahead** (NDA-2, PAN-4, AG-4, LA-7, LA-9). Symmetry breaking in the tensor section, step sizes in preprocessing, reinforcement-learning targets for `detach`, memory bandwidth in the matrix product. Each is true and none can be followed where it stands.
10. **One misplaced paragraph in preprocessing** (PAN-1). "One-hot encoding avoids this" points at the wrong antecedent.

## Recurring patterns

- **Template openings and closings.** Every section but one opens with a paragraph that ends in a question, and six of seven close with a paragraph beginning "We can now ..." (eight occurrences of the phrase in the chapter). The roadmap in the introduction uses "asks how" seven times (IDX-1), and the part introduction and the chapter 2 introduction use the same frame. Each instance is good. In sequence they become a visible template. Vary two or three of them. This is very likely a book-wide pattern, an artifact of applying the guide's section template uniformly.
- **Unused imports.** The calculus and probability sections begin with a boilerplate imports cell (`math`, `nn`, `F`; `gluon`, `optax`, `nnx` in other tabs) of which the section uses almost nothing, plus a second imports cell (CAL-1, PRB-1). The other sections import only what they use. Likely book-wide wherever `d2l` is imported.
- **The cost claim, five times.** "a small multiple of the cost of computing the loss, whatever the number of parameters" appears in the calculus closing, the autograd opening, twice in "Forward and Reverse Mode" and in the autograd summary. Also repeated: minibatches of 256 images of $28 \times 28$ pixels are introduced, with the arithmetic, in the tensor section, the linear algebra opening and the linear algebra tensor subsection; the garment classifier is introduced in the part introduction, the chapter introduction and the linear algebra opening.
- **Terminology for the shape of a tensor.** Order, `ndim`, rank, dimension, dimensionality, size, length (NDA-1, LA-2). The text is careful, and the care itself costs a paragraph. Settle on order (or `ndim`), size and shape; use "dimension" only informally.
- **Names for the backward computation.** Backward pass, reverse sweep, reverse mode, vector-Jacobian product, backpropagation. Each is tied to the previous one when introduced, which is right; "sweep" can go.
- **Variable names rebound in code.** `x`, `X`, `Y`, `a`, `b` change meaning several times in the tensor section (NDA-5); `A` is $2 \times 3$ in code while $\mathbf{A}$ is square in the eigenvalue text; `X`, `y`, `w` are NumPy arrays in calculus and tensors in autograd (deliberately, and stated).
- **Framework tabs.** Handled about as well as four frameworks allow. In the PyTorch view the prose is continuous. The costs: many one-line tab notes of the form "In TensorFlow, this function is called ..."; three places where the shared prose shrinks to a sentence and the tab carries the explanation (writing elements, in-place updates, most of autograd); and one section whose shared story fits only one framework (documentation). Where a tab note explains a surprise in an output, it should come after the cell, not before (autograd, gradient accumulation).
- **Dependence on other chapters.** Backward: none beyond the introduction. Forward: about a dozen pointers to later chapters (numerics, numerical stability, backpropagation, the Kaggle house-price section, linear regression), all of the promise kind. Appendix: see above. The chapter stands on its own.
- **Not found.** Em dashes in prose (none), stock phrases from the guide's list (none), "exactly" as an intensifier (five uses, all in probability and all literal), sentences that do not parse (one, in an exercise: NDA exercises, JAX tab).

## What to cut, and what is missing

Cut or move (about 3,000 words and 100 lines of code):

- Calculus: "Plotting Utilities" in full, to Tools (CAL-9). The unused notation $\partial_{x_i} f$ and the Frobenius-norm gradient.
- Linear algebra: the FLOPs-per-byte half of the cost paragraph (LA-7); general tensor notation and the channel-order parenthesis (LA-3); the batched `@` remark (LA-6); the terminology paragraph reduced to a sentence (LA-2); the spectral norm reduced to one mention.
- Autograd: the Jacobian-filling cell and its tab notes (AG-8); "Higher-Order Derivatives" reduced to two sentences (AG-7); the recap paragraph in the opening (AG-1).
- Probability: the covariance matrix (PRB-10); explaining away and the chain, or at least the chain (PRB-7); the general-distribution sentences of "Random Variables" (PRB-3); the classifier analogy (PRB-8).
- Tensors: the symmetry-breaking argument (NDA-2); the NumPy write-through cell; one of the two identity-and-storage exercises.
- Preprocessing: the "Scaling limits" exercise.

Missing:

- The identity matrix and the inverse, in linear algebra (two or three sentences and one exercise).
- The normal approximation and the two-standard-error rule, and the name "p-value", in probability (two or three sentences).
- Reading advice in the chapter introduction, and "optional on first reading" marks.
- One sentence in the tensor section that says what a view is, at its first occurrence.
- A worked decomposition of $\mathbf{e}_1$ into the eigenvectors of $\mathbf{S}$ (LA-11).

## What is best, and should set the model for the rest

- **The line model in calculus and autograd.** One model with three data points; every quantity computed by hand, by finite differences, by formula, by the chain rule, and by the framework, all giving $(2, 4)$; one gradient step computed by hand and then by the framework, both reaching loss $0.4$. Later chapters should copy this: a case small enough to check, carried across sections.
- **Derived need, not asserted need.** The learning rate comes out of the $3h^2$ gap between tangent and parabola; `keepdims` comes out of a broadcast that fails; the standard error comes out of the coin question; reverse mode comes out of counting sweeps.
- **Limits shown.** Accuracy has zero derivative almost everywhere; finite differences lose half their digits; autograd returns "a number with no meaning" at a jump; 56 heads are no evidence of fairness either; zero covariance does not imply independence; $p < 1$ breaks the triangle inequality. Each claim in the chapter is fenced the way Cover and Thomas fence theirs.
- **Promises kept across distance.** The HIV simulation is "looser than two million patients might suggest", and the reason arrives with the standard error. The $n - 1$ of the preprocessing section is explained in probability. The opening question of calculus is answered by name.
- **Closings that open the next section.** Quoted above. Every chapter should end its sections this way.
- **Exercises.** With few exceptions they test understanding and are graded. The standouts: mean and median (calculus), programs versus functions (autograd), comparing two classifiers and the defense lawyer (probability), the pairwise-obtuse triple and the square-minibatch trap (linear algebra), tests that can fail (documentation), shifted slices (tensors), training-only statistics (preprocessing).

## Status of the October findings

Closed: the unparseable sentence in calculus; eight notations for a partial derivative; variance used before definition; steepest descent resting on appendix facts; the linear algebra opening as a contents list; the cut list for probability except as noted; the appendix pointer count (34 then, about 17 now, none load-bearing). Still open: plotting utilities in the calculus section (CAL-9). The October "story it should tell" (the tools needed to fit a model to a table and judge the fit, each introduced by the step that needs it) is now largely the story the chapter tells.

## Not checked

- The three non-PyTorch tabs were read as prose but their stored outputs were not compared with the text, except where a tab note makes a specific claim that the source itself shows.
- The appendix chapters were not read; duplication between this chapter and `chapter_mdl-*` is a guess to be verified.
- Whether chapter 2 builds on the gradient-descent paragraph of the calculus section or introduces gradient descent afresh.
- External links in the resource list and in exercise attributions.
- The rendered HTML pages; figures and output plots were rendered from the SVG sources.
