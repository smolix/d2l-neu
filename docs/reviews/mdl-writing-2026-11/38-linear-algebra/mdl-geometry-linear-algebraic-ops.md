# Vectors, Angles, and Projections

Source: `chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md`, read to the slides marker. About 7,000 words of prose and mathematics, 12 distinct code cells, 8 exercises (about 1,900 words).

## Verdict

This is good textbook writing, and in places better than good. Every proposition is preceded by a worked number that already has the shape of the result ($10 = 5 + 5$ before Pythagoras, the three-point line fit before the projection theorem, $(3,4)$ and $(1,0)$ before Gram–Schmidt), the proofs are the short ones, and the second half does something the model texts do: it derives a reference value ($1/\sqrt{784}$), runs the check, and then lets real data contradict the reference ("They do not."). I found no mathematical error after recomputing every worked number and every stored output. What keeps it from the top mark is structure. The section opens on Fashion-MNIST and promises to check each result on the images, then does not show an image for its first half; it re-teaches a page of the Basics section it cites four times; and it tells the reader three times that a linear layer is a stack of hyperplanes. The single most important fix is to make the first half and the second half one argument: either keep the promise of the opening or change it, and remove the overlap with :numref:`sec_linear-algebra` so that the proof by projection is the first new thing the reader meets.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

A 4 on enjoyment is earned by the second half (from "Similarity in High Dimensions" on). The first half is a clean 4 on explanation and a 3 on enjoyment.

## What works

- **Number first, statement second, everywhere.** "Projection onto a Line" tries $t = 1, 0.5, 1.5$, notices the right angle and $10 = 5 + 5$, and only then states the proposition with "The general statement keeps this shape." The line fit through $(6, 0, 0)$ is solved by hand before the projection theorem and reused for the normal equations, the code cell, and the figure. Protect all three worked examples.
- **The Cauchy–Schwarz proof is the picture.** "The inequality says that the shadow is never longer than the vector, and that it is as long only when nothing is left over." The equality case comes free. This is the proof worth having here, and it is different from the one in the Basics.
- **The reader's objection is asked and answered.** "An objection: is this defined angle the one we would measure?" followed by coordinates in the plane of the two vectors. Most texts define the angle by $\arccos$ and move on. This paragraph is the reason the definition is believable.
- **The near-orthogonality proposition and its proof** (symmetry gives the mean, $\sum_i v_i^2 = 1$ gives the variance in one line). Short, elegant, and used three times afterwards (the 784-dimensional check, attention scaling, the maximum over $n$ keys).
- **The data contradict the reference.** "They do not. Raw images have cosines around $0.6$ ..." and the conclusion "A cosine therefore counts as large only when it exceeds the spread of unrelated pairs from the same data." This is the best paragraph of the section: a prediction, a measurement, and a judgement.
- **The nearest-mean classifier** is derived, not asserted ("So 'nearest mean' is one hyperplane: its normal points from one mean to the other, and it passes through their midpoint"), tested, and then shown to fail on harder pairs ("It does not."), with the reason and the remedy named. The stored outputs support every number: 0.9155 (169 of 2,000), 101 and 68 errors, spreads 2.36 and 1.66, 0.786 and 0.716; I recomputed $\|\mathbf{w}\|/2 = 3.23$ and the class means along $\mathbf{w}$, $-3.28$ and $+3.15$. These counts are deterministic, so quoting them exactly is safe.
- **Sentences with a voice**: "In $784$ dimensions there is no protractor."; "a fact known to Archimedes"; "This is the smallest fitting problem there is."; "Nothing about the threshold is chosen by hand."
- **The attention paragraph** gives the variance computation and then the geometric reading ("each vector has length about $\sqrt{d}$, and the cosine between unrelated directions has size about $1/\sqrt{d}$"), and the softmax weights $0.9997/0.0003$ against $0.73/0.27$ make the consequence concrete. Correct (checked).
- **Candour about what is not proved**: "(a theorem we use without proof)" for the invariance of dimension; "to leading order" for the maximum of $n$ cosines.
- **Questions before cells.** Most cells are introduced by the question they answer ("Is the standard deviation of the cosine really $1/\sqrt{784}$?", "Do two unrelated Fashion-MNIST images look orthogonal?", "Does it survive on pairs of classes whose means are close?"). It works here. It is used nine times, which is near the point where a device becomes a template; the reviser should not add more.

## Structural findings

1. **The opening promises the images and the first half does not show one.** The opening says "We prove each result once, with a picture drawn in the plane, and check it on the images." Projection, Pythagoras, Cauchy–Schwarz, the angle, least squares and Gram–Schmidt are all checked on $(1,3)$, $(2,1)$ and a three-point line fit. The images return 570 lines later. A reader who came for "Do the pictures still mean anything there?" reads 3,300 words of plane and $\mathbb{R}^3$ geometry before the question is taken up again. See GEO-1.

2. **Overlap with the Basics is acknowledged but not removed.** :numref:`sec_linear-algebra` already gives the law of cosines, the definition of the angle by reading the identity backward, the legitimacy argument, a proof of Cauchy–Schwarz with its equality case, cosine similarity, and an image at a different brightness. This section cites that section four times and then repeats the law of cosines, the turn-around, the legitimacy argument, cosine similarity and the brightness example. The new content (the proof by projection, the objection paragraph, the floating-point clip, the triangle inequality) is diluted. About 350 words can go. See GEO-4 and GEO-12.

3. **The section is three topics under a title that names one.** "Vectors, Angles, and Projections" also holds the subspace vocabulary of the whole chapter (span, independence, basis, dimension), least squares with the normal equations, Gram–Schmidt and QR, a concentration result, the attention scale, hyperplanes, margins and a classifier. A reader arriving by lookup for "basis", "least squares" or "hyperplane" would not guess they are here. At 7,000 words it is well over the guide's ceiling. I do not think it must be split: the line of argument (dot product, projection, closest point, then the same tools in 784 dimensions) holds. But if a split is wanted, the clean cut is before "Similarity in High Dimensions": the first part is "Projections and Least Squares" (about 3,300 words), the second "Similarity and Hyperplanes in High Dimensions" (about 3,300 words), and the second depends on the first only through Cauchy–Schwarz and the signed length. Without a split, the title or the opening should name subspaces, least squares and hyperplanes.

4. **The projection theorem leans on something built after it.** The proposition assumes an orthonormal basis of $S$; that one exists is shown only in the next subsection. See GEO-8.

5. **What could go, with no loss:** the law-of-cosines paragraph and :numref:`fig_mdl-la-angle` (GEO-4); the orphan paragraph "Least squares leaves the target unchanged ..." (GEO-9); the first paragraph of "The Nearest-Mean Classifier" and one of the three statements that a layer is a set of hyperplanes (GEO-15); the closing recap (GEO-18). Together about 600 words.

6. **Nothing here belongs in the main text instead**, and the level is consistently above the Basics. Missing for a deep learning reader: nothing essential. The attention scale, cosine similarity for retrieval and contrastive learning, the last layer as hyperplanes, and (in the exercises) the margin, Fisher's discriminant and superposition are all named.

## Passage-level findings

**GEO-1** (moderate, pedagogy). Opening, third paragraph.
Anchor: `is a ratio of dot products, and the angle between two vectors is defined from`
The next sentence, "We prove each result once, with a picture drawn in the plane, and check it on the images", is not what the section does: only the cosine results and the hyperplane are checked on the images. Fix in the writing pass: say what is done ("We prove each result with a picture drawn in the plane. The second half of the section then takes the results to the images, where no picture is possible"). A later pass with code could keep the promise cheaply: after the projection proposition, project one test t-shirt onto the mean t-shirt and report the coefficient, the residual norm and the Pythagoras check in $\mathbb{R}^{784}$; that one cell would tie the halves together (deferred: needs a new cell).

**GEO-2** (minor, thought). Opening, second paragraph.
Anchor: `draw in the plane did not carry over to $784$ dimensions, a rule as simple as`
The sentence argues from the success of the classifier to the validity of the geometry. It is the wrong way round: the formulas hold in every dimension by algebra, whether or not the classifier works, and the classifier's 92% says something about the data (the two classes are far apart relative to their spread), as the section itself shows later when the same rule drops to 72%. Fix: "That so simple a rule works says that the two clouds of points are well separated, and that 'closer' in $\mathbb{R}^{784}$ can be computed although it cannot be seen."

**GEO-3** (minor, language). "Points and Directions".
Anchor: `we also write it with a dot, $\mathbf{u}\cdot\mathbf{v}$`
Two notations for one operation are introduced in the same paragraph, and the section then uses the dot in the geometry and the transpose in the proofs with $\mathbf{Q}$, while the Basics use the transpose only. This is defensible, but say why once ("the dot when we think of two arrows, the transpose when the product is part of a matrix expression").

**GEO-4** (moderate, thought: repetition). "Cauchy–Schwarz and the Angle".
Anchor: `We turn to the angle. In the plane, the law of cosines applied to the triangle with`
This paragraph, the figure after it (two arrows and a $\theta$, carrying nothing), and the sentence "so we turn the planar identity around and use it as a definition" repeat :numref:`sec_linear-algebra` almost step for step ("we read :eqref:`eq_dot_cos` backward and use it to define the angle"). The same subsection has already said "proved it algebraically, from $\|\mathbf{u} - \mathbf{v}\|^2 \ge 0$ for unit vectors". Fix: one clause with the reference ("In the plane $\mathbf{v}\cdot\mathbf{w} = \|\mathbf{v}\|\|\mathbf{w}\|\cos\theta$ (:numref:`sec_linear-algebra`). In $\mathbb{R}^d$ there is no triangle to measure, so the identity becomes the definition."), drop :numref:`fig_mdl-la-angle`, and go straight to the objection paragraph, which is the new material. The opening paragraph of the section already lists what the Basics did; the later restatements at the anchors `proved it algebraically, from` and `called the ratio in` can each shrink to a reference.

**GEO-5** (minor, explanation). "Projection onto a Line", end.
Anchor: `The length of the shadow, with a sign that says whether it points along`
The signed length is the quantity the hyperplane subsection is built on, and the text says so ("returns at the end of the section"), but its display has no label. When it returns, the text cites the wrong equation: under "Hyperplanes and Signed Distance", "By :eqref:`eq_mdl-dot_geom`, every point $\mathbf{x}$ on it has the same signed shadow on $\mathbf{w}$" points to the cosine identity. Fix: label the signed-length display and cite it there.

**GEO-6** (minor, pedagogy). "Cauchy–Schwarz and the Angle", code.
Anchor: `Does the formula return $\pi/4$ on the worked pair, and what does it give on`
The second pair, $(0,1,2)$ and $(2,3,4)$, teaches nothing ("the two vectors point roughly the same way"). The interesting fact is in the prose after the cell: for $(10^8+1, 3, 7)$ and three times that vector the ratio is $1 + 2.2\times10^{-16}$ and `arccos` returns `nan`. I verified the claim, but the stored output does not show it. Deferred (code): replace the $\mathbb{R}^3$ pair by the collinear pair, with and without the clip, so the reader sees the `nan`.

**GEO-7** (minor, language). "Span, Bases, and Subspaces".
Anchor: `Every basis of a given`
A paragraph break falls in the middle of the thought: the paragraph on bases ends with the slanted pair, and "Every basis of a given subspace has the same number of elements" starts a new paragraph with a line broken after four words. Join it to the previous paragraph. More generally this subsection is the densest run of definitions in the section (linear combination, span, subspace, independence, basis, dimension in about 600 words). It works because every definition is applied at once to $\mathbf{1}$ and $\mathbf{t}$ of the line fit; keep that. Term formatting is mixed (bold for span, subspace, basis, dimension; italic for residual, projection, normal equations, column space); choose one.

**GEO-8** (moderate, correctness: order of the argument, sure). "Projection onto a Subspace".
Anchor: `orthonormal columns spanning $S$, and set $\mathbf{P} =`
The proposition is stated for a subspace that has an orthonormal basis. That every subspace has one is established only by Gram–Schmidt in the next subsection, but the text applies the proposition at once to the column space of an arbitrary $\mathbf{A}$ ("by the proposition this is the point whose residual is orthogonal to every column"), including the case of dependent columns, which Gram–Schmidt as described ("Independence is needed") does not handle. The mathematics is true; the order leaves a hole. Fix, cheapest first: (a) add at the definition of an orthonormal basis: "Every subspace has one; :numref:`subsec_mdl-gram-schmidt` builds it from any basis"; and note in the Gram–Schmidt subsection that a dependent vector is skipped, so the process works for any spanning list. (b) Or move "Gram–Schmidt and QR" before "Projection onto a Subspace"; it needs only projection onto a line, and the three-point fit can then be solved with $\mathbf{Q}$ in hand.

**GEO-9** (minor, explanation). "Projection onto a Subspace", after the normal equations.
Anchor: `Least squares leaves the target unchanged. It finds the point of the column`
"Leaves the target unchanged" is obscure (what would change it?), and the rest of the paragraph repeats the paragraph two above ("the point of $\operatorname{col}(\mathbf{A})$ closest to $\mathbf{y}$ ... residual is orthogonal to every column"). Cut the paragraph.

**GEO-10** (minor, explanation). "Projection onto a Subspace", last paragraph.
Anchor: `the sensitivity of the problem to rounding errors when the columns are nearly`
"Squares the sensitivity" refers to the condition number, which is defined two sections later. Add the reference (:numref:`subsec_mdl-condition-number`) so that the remark is a pointer and not an unexplained assertion. The same point is made again at the end of "Gram–Schmidt and QR" ("loses orthogonality") and in the next section; one home is enough.

**GEO-11** (minor, enjoyment). "Gram–Schmidt and QR".
Anchor: `returns the negatives of both, with $\mathbf{R}$'s diagonal $-5$ and $-0.8$`
Good: the library disagrees with the hand computation, and the text has prepared the reader for it. No change; noted so that the reviser keeps the sign discussion.

**GEO-12** (moderate, thought: repetition). "Cosine Similarity".
Anchor: `An image and a copy of it dimmed to $10\%$ brightness are`
The Basics section has the same example at twice the brightness, with the same conclusion. The October review asked to keep the dimmed image, and it should stay in one of the two places. Here the subsection's own contribution is the second paragraph (retrieval, SimCLR, and the contrast with attention's unnormalized score). Fix: reduce the first paragraph to the definition by reference and one clause on invariance, and lead with the uses.

**GEO-13** (minor, thought). "Near-Orthogonality of Random Directions", low dimensions.
Anchor: `is about $0.29$, against about $0.06$ for $|\cos\theta| < 0.1$. In $d = 3$ the`
The paragraph is a pleasure to read and correct (I recomputed $0.287$ and $0.064$). But it gives the answers to the exercise on the cosine in two and three dimensions (both probabilities, and "From $d = 4$ on, the density has its peak at $0$", which is the last question of that exercise). Keep the text; change the exercise (see Exercises).

**GEO-14** (moderate, enjoyment). "Near-Orthogonality of Random Directions", after the image cell.
Anchor: `the images vary along far fewer effective directions than $784$`
The section has just proved that isotropic directions in $d$ dimensions have cosine standard deviation $1/\sqrt{d}$, and then does not use it on its own measurement. A standard deviation of $0.34$ corresponds to $1/0.34^2 \approx 9$. Say it: "By the proposition, $0.34$ is the spread of random directions in about nine dimensions. Measured by their cosines, the centered images behave like a cloud in nine dimensions, not $784$." (I recomputed 0.340 and 8.6 from the data.) This is the sentence a reader would remember from the section, and it prepares the low-rank theme of :numref:`sec_mdl-svd-low-rank`. Also in the same paragraph, "every pair has a positive dot product" should be "non-negative" (sure; it is positive for these images because every image has ink in the centre, which is the point made later about the two means).

**GEO-15** (moderate, thought: repetition). "The Nearest-Mean Classifier", first paragraph.
Anchor: `A linear classifier computes $\mathbf{w}\cdot\mathbf{x} + b$, as the`
That a linear layer is a set of hyperplanes is said at the head of "Hyperplanes and Decision Boundaries" ("the form used by the layers of"), here, again in "Layers as Stacked Hyperplanes" ("A softmax layer for the ten Fashion-MNIST classes holds ten normals"), and a fourth time in the first sentence of the next section. This paragraph also interrupts: the reader has just met the margin and is about to be asked "Which hyperplane should we choose?". And "predicts by the sign" is not what the layer of :numref:`sec_softmax` does (it takes the largest of $q$ scores; the sign rule is the two-class case). Fix: move the content of this paragraph (pairwise boundaries where scores tie, the last layer of a deep classifier) into "Layers as Stacked Hyperplanes", where it is the subject, and open this subsection with "Which hyperplane should we choose?".

**GEO-16** (minor, correctness: loose statement, sure). "Hyperplanes and Signed Distance".
Anchor: `$b$ its *offset*. It has dimension $d - 1$: a line in the plane, a plane in`
Dimension has been defined only for subspaces, and a hyperplane with $b \neq 0$ is not one (the text says so two sentences later). The count $d - 1$ is also not justified here; it is rank–nullity for the $1 \times d$ matrix $\mathbf{w}^\top$. Fix: "It is a shifted copy of the subspace $\mathbf{w}\cdot\mathbf{x} = 0$, which has dimension $d - 1$ (:numref:`subsec_mdl-rank`)."

**GEO-17** (moderate, thought: claim not shown by the output). "The Nearest-Mean Classifier", after the last cell.
Anchor: `The means of these pairs are close, while the`
The cell prints three accuracies. That the means of t-shirt and shirt, or pullover and coat, are close, and that the classes spread widely along $\mathbf{w}$, is asserted as the explanation but not computed, although the previous cell computed exactly these quantities for the first pair ($\|\mathbf{w}\|/2 \approx 3.2$ against spreads of 2.4 and 1.7). Writing-pass fix: "We would expect the means of these pairs to be closer relative to the spread of each class; :exref:... measures it." Deferred fix (code): print $\|\mathbf{w}\|/2$ and the two spreads in the loop, so that the three rows read as "distance between means, spread, accuracy" and the explanation is in the output.

**GEO-18** (minor, language). "Layers as Stacked Hyperplanes", closing.
Anchor: `We can now measure an angle in any dimension, drop a vector onto a line or a`
A recap sentence in the "We can now ..." formula, directly before a Summary that says the same. The next section closes the same way ("We can now read a matrix off its columns ..."). The guide (section 12.2) asks for a closing that belongs to the argument. Keep only the turn to what is still missing, which is good: "The row view shows a layer as a stack of hyperplane tests, but it does not tell us which inputs the map destroys, or whether its action can be undone."

**GEO-19** (minor, pedagogy: code). Imports cell.
Anchor: `all the code is plain NumPy, the same in every framework`
True, and then the PyTorch tab imports `math`, `torch`, `nn` and `F`, none of which is used before the slides marker (the other tabs likewise). A reader who is told the code is plain NumPy and sees four framework imports wonders what they are for. Deferred (code): reduce to `numpy` and the `d2l` import that loads the data. The four tabs then differ only in the data-loading cell, which is as it should be.

## Exercises

Eight problems, within the band for the appendix, and a strong set: most have a result worth knowing (the projection matrix from the normal equations, the Welch bound, classical against modified Gram–Schmidt, Fisher's direction with shrinkage). The line-fit problem is credited to Strang, as it should be. Changes:

- **The cosine in two and three dimensions**: the text already states both probabilities for $d = 2$, the uniformity for $d = 3$ and the answer "$d = 4$" to the last part (GEO-13). Keep the derivations (density for $d = 2$, Archimedes for $d = 3$, the variance check) and drop the questions whose answers are printed in the text, or replace the last one by "at which $d$ does $|\cos\theta| < 0.1$ first become more likely than $|\cos\theta| > 0.9$?".
- **Cosines between random vectors**, second part: the text states the answer "about $0.3$ for $n = 10^5$ keys in $d = 256$". Either remove the number from the text or change the exercise's $d$.
- **Tilting the boundary** uses a matrix inverse, covariance matrices, positive definiteness, the trace and the condition number. All are defined in later sections of the chapter (the inverse and the trace in :numref:`sec_mdl-linear-maps`, the others after that). It is a good problem in the wrong place for a front-to-back reader. Move it to :numref:`sec_mdl-svd-low-rank` (it is a conditioning problem) or add a line that says which later sections it needs.
- **Packing nearly orthogonal directions** is excellent and long (four parts, a training loop). Keep; it is marked as extended.
- Difficulty is graded, with markers only on the last three. Nothing is busywork.

## Still open from October

The October review listed the dimmed image as a passage to keep; it is still there (see GEO-12 on where it should live). October made no other finding on this section, which was then 9,800 words and has since been cut to about 7,000.

## For the chapter review

- This section supplies the vocabulary the rest of the chapter stands on (orthogonality, projection, span, independence, basis, dimension, orthonormal $\mathbf{Q}$, least squares, QR), and its last paragraph hands over to :numref:`sec_mdl-linear-maps` with a real question (which inputs does the map destroy, can it be undone). The hand-over is told twice: at the end of this section and in the first paragraph of the next.
- Overlap with neighbours: $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ "because entry $(i,j)$ is $\mathbf{q}_i\cdot\mathbf{q}_j$" is explained here and again under "Orthogonal Matrices" in the next section; least squares and the instability of the normal equations return in the SVD section (check that the SVD section cites and does not re-derive).
- Overlap with the Basics (:numref:`sec_linear-algebra`) is the main repetition problem of this section; the Basics section also has its own Cauchy–Schwarz proof, cosine similarity and a short eigenvalue part. A chapter-level decision is needed on which of the two places teaches the angle.
- Patterns I suspect hold across the chapter: (i) the closing formula "We can now ..." (both sections I read); (ii) a worked $2 \times 2$ number before every statement, which is a strength; (iii) questions placed before code cells, a strength that is close to becoming a template; (iv) no mathematical errors, but arguments whose pieces arrive in an order that leaves a temporary hole (GEO-8 here, several in the next section).
- The running matrix makes its first appearance in the last subsection ("Layers as Stacked Hyperplanes"), read by rows, with the same input $(2,-1)$ that the next section reads by columns. That link is well made.
