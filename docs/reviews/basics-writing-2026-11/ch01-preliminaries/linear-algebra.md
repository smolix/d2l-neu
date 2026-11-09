# Linear Algebra

Source: `chapter_preliminaries/linear-algebra.md`. About 5,350 words of prose in the PyTorch view (7,900 in the file to the slides marker), plus 1,050 words of exercises. The longest section of the chapter after probability, and above the writing guide's 4,500-word ceiling.

## Verdict

The strongest conception in the chapter. The whole section is organized around one expression, $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ for 256 images and ten categories, shown in the opening and then "built piece by piece": dot product for one image and one category, matrix-vector product for one category and all images, matrix-matrix product for everything. The eigenvalue part opens with a question the reader can feel ($1.1^{50} \approx 117$, $0.9^{50} \approx 0.005$) and closes by answering it in code. This is Strang's method and it works. What keeps the section at 4 is overload. It tries to carry notation, reductions, three products, cost, five norms, cosine similarity, a proof, eigenvectors, power iteration and the spectral norm, and in several places two or three of these are packed into one paragraph. The most important fix is to thin the section: take the bytes-per-FLOP argument out, give the linear-map paragraph room, and work the eigen-decomposition of $\mathbf{e}_1$ explicitly instead of asserting the general formula.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The opening tells the reader what linear algebra is for in this book: "It lets us write the computation of a model on a whole minibatch as one expression, and it tells us, before we run anything, what shape the result has and how much work it takes." Then it shows the expression and promises to build it.
- The ladder dot product, matrix-vector, matrix-matrix is motivated at each rung by the classifier: "Now score one category for all 256 images", "Scoring all ten categories takes one matrix--vector product per category". The reader never meets a product without a use.
- "Each column of $\mathbf{W}$ acts as a template of 784 pixel weights, and the classifier scores how well each image matches each template." A memorable sentence that will pay off again in the classification chapter.
- The derivation of $\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ from the law of cosines, followed by "we read :eqref:`eq_dot_cos` backward and use it to define the angle", followed by the question whether that definition makes sense, followed by a five-line proof of Cauchy-Schwarz. This is a genuine line of thought, each step raising the next. The proof is short and elegant and belongs here.
- Norms are compared with a number: in $(1, -1, 1, -1, 10)$ the large entry is "96% of the squared $\ell_2$ norm but only ... 71% of the $\ell_1$ norm". The condition $p \geq 1$ is fenced by a counterexample ($(1 + 1)^2 = 4 > 2$), in the manner of Cover and Thomas.
- The eigenvalue section: scalar case first, then "in a few special directions it reduces to the scalar case", then a $2 \times 2$ example whose eigenvectors can be found by inspection, then iterates $(2, 1)$, $(5, 4)$, $(14, 13)$ the reader can check, then the 50-layer experiment that returns 117 and 0.005. All stored outputs match the prose.
- The closing names what is still missing: "What we cannot do yet is improve the weights."
- `keepdims` is motivated by a failure the reader can predict from the broadcasting rule, and the square-matrix trap ("raises no error and subtracts the wrong numbers") is turned into an exercise.

## Structural findings

- **Too much for one section.** Five `##` parts, each of which would be a lesson. A reader who is rusty will need two or three sittings. The parts that could go with least loss: the FLOPs-per-byte paragraph (LA-7), the general tensor notation $\mathsf{X}$ (LA-3), the batched `@` remark (LA-6), the spectral norm (two mentions, neither used later in the Basics part as far as this section says), and the provisos of power iteration. Removing them brings the section near 4,500 words without touching its line of argument.
- **Missing: the identity matrix and the inverse.** The section never defines $\mathbf{I}$, yet the calculus section uses "with $\mathbf{A} = \mathbf{I}$" two sections later and exercise 9 here has to define it in passing. Chapter 2 needs the inverse on its first pages ($(\mathbf{X}^\top\mathbf{X})^{-1}$) and introduces it inline there. Two or three sentences after the matrix product would close both gaps: the identity as the matrix that changes nothing, the inverse as the matrix that undoes a map when one exists, and a pointer to the appendix for when it exists. This matters more for later chapters than the spectral norm does.
- **The running example slips in "Norms".** Norms are motivated by "the errors $\hat{\mathbf{y}} - \mathbf{y} \in \mathbb{R}^{256}$ of the predictions", but the classifier of this section outputs 2,560 scores and its labels are categories, for which a difference is not defined. The error vector belongs to regression. Say so in one clause ("for a model that predicts one number per example, such as a price") or use the weights $\mathbf{W}$ as the first thing to measure.
- **Seam to the next section.** The closing promises "how the error changes when $\mathbf{W}$ changes", and the calculus section then starts with a one-parameter model and never returns to $\mathbf{W}$. That is the right pedagogical choice there, but the hand-over sentence over-promises. "when a parameter changes" is enough.

## Passage-level findings

**LA-1** (minor, pedagogy). Section opening, third paragraph.

> Stack the images as the rows of a matrix $\mathbf{X}$, the weights of each category as the columns of a matrix $\mathbf{W}$, and the ten offsets (also called biases) in a vector $\mathbf{b}$.

The opening runs to four paragraphs, an equation and a figure before the first heading, and uses "matrix" and the product $\mathbf{X}\mathbf{W}$ before either is defined. This works because the next paragraph says the section "builds that expression piece by piece", and the figure helps. Keep the device. Trim the second paragraph (the reader has just read about minibatches of 256 in two earlier places) so that the expression arrives sooner.

**LA-2** (moderate, language). Under "Vectors".

> We keep "dimensionality" for the number of elements, so a vector in $\mathbb{R}^{784}$ has order 1 and dimensionality 784. (Libraries call the order `ndim`, and some call it the rank, which is unrelated to the rank of a matrix.)

A paragraph of terminology hygiene that introduces five words (dimension, order, dimensionality, `ndim`, rank) to rule out a confusion the reader does not yet have. The section then uses "dimension" informally anyway ("In 784 dimensions we cannot draw the angle", "one per dimension", "in any number of dimensions"). Reduce to one sentence: "We call the number of elements of a vector its dimensionality; the number of axes of a tensor is its order (`ndim` in code)."

**LA-3** (minor, thought). Under "Tensors".

> We denote general tensors by capital letters in a special font, such as $\mathsf{X}$, and index them like matrices, as in $x_{ijk}$.

The notation is introduced and not used again in the section or the chapter. The parenthesis about color images and channel order in the same paragraph is also a distraction at this point. Cut both; the notation chapter holds them.

**LA-4** (minor, language). Heading.

> ### Reduced Axes Kept for Broadcasting

An awkward noun phrase, evidently forced by the rule against sentence headings. "Keeping the Reduced Axis" or simply "`keepdims`" reads better and is still a name, not a claim.

**LA-5** (moderate, explanation). Under "Matrix--Vector Products", last paragraph.

> Linearity means that the map is determined by what it does to the *coordinate vectors* $\mathbf{e}_j$, which have a 1 in position $j$ and 0 elsewhere.

This paragraph is the column view of a matrix, the idea the eigenvalue section rests on, and it is compressed into about 170 words that introduce linear maps, coordinate vectors, the columns as images of $\mathbf{e}_j$, the unit square and its parallelogram, the rotation by $90^\circ$, and a one-sentence description of a neural network layer. A first-time reader loses the thread at "any $\mathbf{x} = \sum_j x_j \mathbf{e}_j$ is mapped to $\sum_j x_j \mathbf{A}\mathbf{e}_j$". Give it two paragraphs and one number: with the $2 \times 3$ matrix `A` already on the page, $\mathbf{A}\mathbf{e}_2 = (1, 4)$ is its second column, and $\mathbf{A}(0, 1, 2) = 1 \cdot (1, 4) + 2 \cdot (2, 5) = (5, 14)$, the output the reader has just seen. Then the plane and the figure. Move the neural-network sentence to the end of "Matrix--Matrix Products".

**LA-6** (minor, thought). Under "Matrix--Matrix Products".

> For tensors with more than two axes, `@` treats the leading axes as a batch and multiplies the matrices formed by the last two axes pair by pair: shapes $(5, 2, 3)$ and $(5, 3, 4)$ give shape $(5, 2, 4)$.

Dropped between the small example and the classifier, with no cell and no use in this chapter. It interrupts "A small example first ... Now the classifier itself". Move it to the section that first needs batched products (attention), or to an exercise.

**LA-7** (moderate, pedagogy). Under "Matrix--Matrix Products", cost paragraph.

> so it performs only about 5 FLOPs per byte that it moves

The first half of this paragraph (a product of an $m \times k$ and a $k \times n$ matrix costs about $2mkn$ operations; here about 4 million) is exactly what the opening promised. The second half introduces a different idea, arithmetic per byte of memory traffic, and concludes that "Small products such as $\mathbf{X}\mathbf{W}$ are limited by moving data rather than by arithmetic". The reader has no model of memory bandwidth that would make "FLOPs per byte" meaningful; this is the roofline argument of the Computational Performance chapter arriving twelve chapters early. The accounting is also uneven (the small product counts only the bytes of $\mathbf{X}$, the square one counts all three matrices). Keep the operation count and the scaling ($2n^3$ work for $3n^2$ numbers), cut the rest, and let chapter 13 make the hardware argument.

**LA-8** (minor, language). Under "Norms", opening paragraph.

> Training minimizes a *loss*, a single number that measures how wrong the predictions are, and a norm of the errors, or its square, is a common choice.

The paragraph says the same thing twice: its first sentence already reads "training must measure how wrong they are". The loss was defined in the chapter introduction and again in the index of the part. One statement suffices.

**LA-9** (minor, thought). Under "Other Vector Norms".

> As penalties on the weights, the squared $\ell_2$ norm shrinks all weights toward zero, while the $\ell_1$ norm tends to set many of them to zero.

An unexplained and unreferenced claim. It is true and interesting, and the reader cannot verify or even picture it here. Add a label reference to the weight decay section, or cut.

**LA-10** (minor, thought). Under "Matrix Norms".

> A matrix is also a map, which raises a second question: by how much can it stretch a vector?

This begins a new topic (the spectral norm) in the middle of a paragraph that started with weight decay and ends with a list of where norms appear. Give the spectral norm its own short paragraph or defer it entirely to "Powers of a Matrix", where it is derived for symmetric matrices.

**LA-11** (moderate, explanation). Under "Powers of a Matrix".

> the perpendicular eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ of length 1 let us take any vector apart, as $\mathbf{x} = \sum_i c_i \mathbf{v}_i$ with $c_i = \mathbf{v}_i^\top \mathbf{x}$

The central step of the section is asserted in a subordinate clause. Why can any vector be taken apart this way, and why are the coefficients dot products? One sentence answers both (take the dot product of both sides with $\mathbf{v}_j$; all terms but one vanish because the eigenvectors are perpendicular). More important, the worked example is not used: for $\mathbf{S}$, $\mathbf{e}_1 = \tfrac12 (1, 1) - \tfrac12 (-1, 1)$, so $\mathbf{S}^k \mathbf{e}_1 = \tfrac12 3^k (1, 1) - \tfrac12 (-1, 1)$, which is the formula $\left((3^k + 1)/2, (3^k - 1)/2\right)$ that the text then states "in general" without derivation. Showing these two lines before the general equation :eqref:`eq_matrix_power` would make the general case a restatement of something the reader has computed.

**LA-12** (minor, thought). Under "Powers of a Matrix", after the power-iteration cell.

> The same decomposition gives the spectral norm of a symmetric matrix: it equals the largest $|\lambda_i|$, because no vector is stretched more than the eigenvector of that eigenvalue

The "because" restates the claim. The actual reason is one line from the decomposition ($\|\mathbf{A}\mathbf{x}\|^2 = \sum_i c_i^2 \lambda_i^2 \leq \max_i \lambda_i^2 \sum_i c_i^2$). Either give it or drop the "because" and leave the pointer to the appendix. This paragraph also carries the definition of power iteration and both of its provisos; it is the densest paragraph in the section and comes just before the payoff. Split it.

**LA-13** (minor, explanation). Under "Matrices".

> Their eigenvalues, which we meet at the end of this section, are particularly well behaved.

A forward use of an undefined term to justify introducing symmetric matrices. Harmless, but "symmetric" could simply be defined where $\mathbf{S}$ is first needed, in "Eigenvectors", which would shorten the notation part.

## Exercises

Nine exercises, the best set in the chapter: every one asks for an argument or a prediction, and they are graded from "prove $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$" to power iteration with a shift. "A pairwise-obtuse triple" ends with a consequence for "ten class templates that are all equally far apart", which ties a pure geometry problem back to the classifier. "Pairwise distances" teaches the standard trick and its numerical pitfall. "Shape prediction" part 3 (the square minibatch) is a model of how to teach a silent bug.

- Exercise 6 (order of a triple product) is the classic and fine. Its sizes ($2^{10} \times 2^{16}$ and so on) could be made friendlier.
- If the identity matrix and the inverse are added to the text, add one short exercise on them (for instance, the inverse of the scaling and shear maps of the figure, and why the matrix of ones has none).

## Still open from October

October: "The linear-algebra opening is a contents list". Closed; the opening is now the best in the chapter. October also noted that $\mathbf{A}$ was reused for a different matrix; in this section $\mathbf{A}$ is the $2 \times 3$ example in code and a general matrix in the definitions, which is tolerable, but in "Eigenvectors" $\mathbf{A}$ becomes a square matrix while the code variable `A` is still $2 \times 3$. Minor.
