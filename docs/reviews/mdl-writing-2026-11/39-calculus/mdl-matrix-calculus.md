# Matrix Calculus

Source: `chapter_mdl-calculus/mdl-matrix-calculus.md` (lines 1 to 1270, up to the slides marker). About 5,600 words of prose and mathematics outside code, plus 1,600 words of exercises; 6 code cells per framework, 3 figures.

## Verdict

This is a good section, close to the bar in its best parts. It has one question, one small network that is worked by hand three times (by nudging, by the chain rule, by the backward rules) and gives the same three numbers each time, and one method, $df=\langle\nabla f,d\mathbf x\rangle$, that replaces a table of identities. The mathematics is correct: I recomputed every worked number (singular values, the gaps $0.061$ and $0.015$, the operation counts, the backward trace, the softmax values and eigenvalues) and found no error in a derivation or a proof. The definitions are careful where care matters (the Jacobian is defined by the linear approximation, with the partials as its entries and a counterexample for partials without a Jacobian). What keeps it from the model texts is not a flaw of explanation but of economy and of surprise. The section says several things two to four times, it hands part of its material to the next section, which then teaches it again, and four of its five output-producing cells confirm an identity whose truth was never in doubt, on shapes for which the mistake the section warns against could not occur silently. The single most important fix is to settle, together with the next section, where the order of multiplication and the layer rules live, and to state each once.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening network.** The 2-2-1 ReLU network at $\mathbf x=(1,1)$ is small enough to hold in the head and is used to the end: the row $(2,4)$ and the matrix $\left(\begin{smallmatrix}2&2\\0&0\end{smallmatrix}\right)$ are found by nudging in the opening, the row is recovered as $\mathbf w_2^\top\operatorname{diag}(\mathbf m)\mathbf W_1$ under "The Jacobian of a Network", and the matrix is recovered by the three-step backward trace under "Backward Rules of a Layer". This is the method of the best Basics passages, and a reviser should not touch the numbers.
- **The stated danger.** "A hand-derived gradient with a transposed square factor raises no error, and training goes wrong." This is a real problem the reader can feel, and it is quotable.
- **The definition before the formula.** "The partial derivatives supply the entries of $\mathbf J$, but the linear approximation is what defines it", followed at once by the function $2xy/(x^2+y^2)$ whose partials exist and whose Jacobian does not. Few deep learning texts take this care, and it costs one paragraph.
- **The first code cell.** The ratio error$/\|\boldsymbol\delta\|$ falls tenfold per tenfold step, and the prose reads more out of it than it promised: "The definition promises only the first property; the second holds because $\mathbf f$ has continuous second derivatives." This is the one cell in the section whose output teaches something the reader could not have predicted from the previous line.
- **The shape rule.** "One rule checks every formula in this section: a gradient has the shape of its variable. If a derived gradient does not, a transpose is missing." Also the plain statement of the row and column conventions ("this book uses both, each for one job").
- **The chain-rule proof.** Short, by substitution of one linear model into the other, and honest about its one real step ("this is the one step that is not routine").
- **The count.** Three million against two billion multiplications for one chain of three factors, with the figure showing the shapes. The reader sees why the order matters before any talk of modes.
- **The $10^{12}$ entries** that motivate the backward rule, and the reading of the result: "The inactive unit shows up three times ... A training step on this input leaves all four of the unit's parameters unchanged."
- **The variance argument** for the softmax Jacobian ($\mathbf v^\top\mathbf J\mathbf v$ is a variance, so log-sum-exp is convex) and the remark that $\mathbf p-\mathbf y$ holds for any target that sums to one, with the corrected formula when it does not.
- **The exercises** (see below): most of them extend the section instead of drilling it.

## Structural findings

1. **Overlap with the next section, in both directions.** "The Order of Multiplication" gives the cost count, names reverse and forward mode, and states that a scalar loss makes the left-to-right order cheap. The next section's "Two Orders of One Product" then restates the product (under a second equation label), the two orders, and a second count with other numbers. "Backward Rules of a Layer" derives $\bar{\mathbf W}=\bar{\mathbf y}\mathbf x^\top$; the next section's "Parameter Gradients" asks "Where are they?" about the weights and derives the same outer product again. A reader going front to back meets each idea twice within twenty pages; a reader arriving by reference does not know which of the two is the home. Proposal: this section is the home of the product, the count, and the layer rules. The next section cites them in a clause each and keeps only what is new there (seeds as programs, the tape, the measured cost, memory). See AD-3 and AD-10 in the review of the next section.
2. **Three definitions of the same objects across the chapter.** The adjoint $\bar a$ and the backward recursion were defined in the multivariable section (`eq_mdl-adjoint`), are introduced here as if new ("call it the *adjoint*"), and are defined a third time in the next section. The Hessian of log-sum-exp, $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$, is derived in the multivariable section (`eq_mdl-lse-hessian`), here (`eq_mdl-softmax-jacobian`), and in the next section (`eq_mdl-ad-lse-hessian`). This section handles the second with a good parenthesis that points back; the adjoint needs the same treatment.
3. **No link to the main text in the body.** The body contains no reference to a main-text section. The only one is in an exercise (`sec_softmax_scratch`). Yet the section derives exactly what `sec_backprop` uses (the three local rules) and what `subsec_softmax_and_derivatives` states ($\mathbf p-\mathbf y$), and the fused loss is the subject of `subsec_softmax-implementation-revisited`. A reader who came from those sections for the derivation is not told that they have arrived, and a reader who starts here is not told where the result is used. One clause with a label at each of the three places is enough.
4. **Nine uses of "the previous section" and none of its label.** For a reader who lands here from a reference, "the previous section" is not an address. See MC-4.
5. **A misplaced paragraph.** The vanishing and exploding product $\mathbf W^K$ sits at the end of "The Order of Multiplication", a subsection about cost. It is about the size of the product, not its order. See MC-7.
6. **Length.** The section is of reasonable length and is not two sections. About 500 words can go with no loss: the index derivation of the quadratic form (MC-9), the repeated motivations (MC-1, MC-8), the recap paragraph before the summary (MC-19), and the restated argument after the last cell (MC-18).
7. **One gap a deep learning reader will notice.** Every real layer adds a bias to a minibatch, which broadcasts on the way forward and sums on the way back. The body gives the bias rule only for one example ($\bar{\mathbf b}=\bar{\mathbf y}$); the minibatch rule appears only as a hint in the exercise on softmax regression. One sentence after the bias rule ("for a minibatch, $\mathbf Z=\mathbf X\mathbf W+\mathbf 1\mathbf b^\top$ and $\bar{\mathbf b}=\bar{\mathbf Z}^\top\mathbf 1$, by the matrix-product rule") closes it.

## Passage-level findings

**MC-1** · minor · language (repetition)
Anchors: "is not written in scalars. Each layer maps a vector to a vector, and its" (opening); "function $\mathbf f:\mathbb R^n\to\mathbb R^m$ with many outputs, and this part" (The Jacobian); "every later layer, and each of those maps a vector to a vector. We start with" (The Best Linear Approximation).
The fact that a layer maps a vector to a vector, so that a slope or a gradient is not enough, is stated four times in the first ninety lines (the fourth is "maps a vector to a vector. Neither a slope nor a gradient describes its derivative" in the opening). The opening makes the point well. The two-sentence introduction under "The Jacobian" and the first sentence under "The Best Linear Approximation" make it again.
Fix: delete the two-sentence paragraph under "## The Jacobian" and begin "### The Best Linear Approximation" at "We start with the smallest such map that is not linear".

**MC-2** · minor · pedagogy
Anchor: "the smallest such map that is not linear, the one this section and the next" (The Best Linear Approximation).
The opening ends on the hidden layer of the network ("the output reaches $\mathbf W_1$ only through that layer"), and the next page studies a different map, $(x+\sin y,\ y+x^2/2)$, with no word on why the network was dropped. The reason is good and worth a sentence: the ReLU layer is piecewise linear, so its linear approximation is exact near the point and there would be no error to look at.
Fix: add one sentence, for example: "The hidden layer of our network is linear near $\mathbf x=(1,1)$, so it would show no approximation error at all. We therefore take a curved map first and return to the network once the definition is in place."

**MC-3** · minor · thought
Anchor: "the matrix is invertible: near $\mathbf x_0$ the map does not fold the plane" (The Best Linear Approximation).
The determinant and the remark on folding arrive before the matrix has been said to be the derivative, and invertibility is not used again in the body (only in the second exercise). The sentence interrupts the step from "four numbers in a matrix" to "this matrix is the derivative".
Fix: move the determinant remark after the figure, where the ellipse with two nonzero semi-axes shows it, or leave it to the exercise.

**MC-4** · moderate · explanation (lookup use)
Anchors: "$f(x,y)=2xy/(x^2+y^2)$ with $f(0,0)=0$, met in the previous section, vanishes on"; "The gradient and the Hessian of the previous section are both built from"; "the single-variable section. A network of depth $K$ is a composition".
"The previous section" occurs nine times in the body and its label never. The appendix is read by lookup, and house rule is that references use labels. Three of the nine carry a specific fact the reader may want to look up (the counterexample, Clairaut–Schwarz, the convention for ReLU at $0$).
Fix: at the first occurrence and at each of these three, use the label (`sec_mdl-multivariable_calculus`, `sec_mdl-single_variable_calculus`, or the subsection label). The remaining occurrences can stay as they are.

**MC-5** · minor · language
Anchors: "finite difference along each axis reproduces these entries without error for"; "steps small enough to keep the mask fixed; the first exercise makes this precise."
A hardcoded "the first exercise" will be wrong after the next reordering of the exercises. The same holds for "(see the exercises)" under "The Order of Multiplication", which is vague in the other direction.
Fix: give the two exercises `:exlabel:` labels and refer with `:exref:`, as the next section does for its Hessian exercise.

**MC-6** · minor · thought
Anchor: "limit that matters for training: it is the derivative with respect to the" (The Jacobian of a Network).
This paragraph is one of the best transitions in the section: the product is a derivative with respect to the input, and the weights "appear in it as factors, never as variables". Keep it. It is the reason the same question, asked again in the next section's "Parameter Gradients", reads as a repeat there.
Fix: none here; noted so that the reviser keeps this copy and cuts the other.

**MC-7** · moderate · thought, with a small correctness point
Anchor: "The values of the factors matter as well as their shapes. When every factor is" (The Order of Multiplication).
Two problems. (a) Placement: the subsection is about the cost of the product, and this paragraph is about its magnitude. It reads as an afterthought, and it carries the section's only statement about vanishing and exploding gradients. (b) Precision: "$\mathbf W^K$ ... grows or shrinks like $|\lambda_{\max}|^K$" is the asymptotic rate. For a non-normal $\mathbf W$ the norm of $\mathbf W^K$ can grow for many steps before it decays, and the size of a product of different factors is governed by singular values, not eigenvalues, which is what the cited `subsec_mdl-jacobian-products` treats. Confidence that the statement needs the qualifier "eventually" or "for large $K$": high. The statement is not false for the diagonalizable case as written, since the factor $\kappa(\mathbf V)$ is a constant, but a reader will take it as a bound.
Fix: move the paragraph to the end of "The Jacobian of a Network", after the product formula, and write "for large $K$ the entries of $\mathbf W^K$ grow or shrink like $|\lambda_{\max}|^K$". The next section's closing paragraph repeats the statement and can then cite it in a clause.

**MC-8** · minor · language (repetition)
Anchors: "Reference tables of matrix derivatives run to hundreds of entries, and they" (opening); "its input, its weights, its bias. Tables of such derivatives are long, and a" (Derivatives with Respect to Vectors and Matrices).
The warning about tables and the silent wrong transpose is given in the opening and again, in nearly the same words, at the head of the third part.
Fix: keep the opening. At the head of the third part write only what is new: "This part derives every gradient the section needs with one method, which produces each result in the shape of its variable."

**MC-9** · moderate · pedagogy (length)
Anchor: "Indices give a second route that settles any doubt about a transpose, at the" (The Differential Method).
The gradient of $\mathbf x^\top\mathbf A\mathbf x$ is obtained four times: by the differential, by indices with Einstein summation and the Kronecker delta, by the $1\times1$ check with its counterexample, and by automatic differentiation at a random point. The index route introduces two pieces of notation that the body never uses again, and it arrives at the result the reader already has. The claim that it "settles any doubt about a transpose" is true, but the counterexample two paragraphs later settles the same doubt in three lines and is far more memorable.
Fix: reduce the index paragraph to two sentences ("Writing the same computation in indices, $\partial(x_iA_{ij}x_j)/\partial x_k=A_{kj}x_j+x_iA_{ik}$, gives the same result and is the route to take when a transpose is in doubt") and leave the full index derivation to the exercise on the matrix product, which already asks for it. The main text's `sec_backprop` derives the matrix-product rule by indices, so the two routes are both in the book.

**MC-10** · moderate · explanation
Anchor: "Four rules compute $df$. The differential is linear. It obeys the product rule" (The Differential Method).
The list is not the list the section uses. The fourth rule, the move rule, does not compute $df$; it rearranges $df$ once computed, which is the second half of the recipe. And two rules that the section does use are missing: the differential of an elementwise function, $d\varphi(\mathbf X)=\varphi'(\mathbf X)\odot d\mathbf X$, which appears without comment under "Backward Rules of a Layer" ("has $d\mathbf Y=\varphi'(\mathbf X)\odot d\mathbf X$, where $\odot$ multiplies"), and the companion of the move rule for $\odot$, $\langle\mathbf A,\mathbf B\odot\mathbf C\rangle=\langle\mathbf A\odot\mathbf B,\mathbf C\rangle$, which is what turns that differential into $\bar{\mathbf X}=\varphi'(\mathbf X)\odot\bar{\mathbf Y}$. A reader who tries the softmax or LayerNorm exercise with "four rules" lacks the one for functions.
Fix: say "Three rules compute $df$" (linearity, product, transpose), add the scalar chain rule applied entrywise as a fourth, and present the move rule separately as the step that brings $df$ into the form $\langle\mathbf M,d\mathbf x\rangle$, with its elementwise companion in the same sentence.

**MC-11** · moderate · correctness (a false statement about a heuristic; confidence high)
Anchor: "since $\frac{d}{dx}(ax^2)=2ax=(a+a)x$. This *$1\times1$ check* catches wrong" (The Differential Method).
"This $1\times1$ check catches wrong shapes" says the opposite of what the check can do. Once every matrix is $1\times1$ there are no shapes left to be wrong, no order of factors, and no transposes. What the check catches is a wrong scalar skeleton: a missing factor of $2$, a dropped term, a wrong sign or power. Wrong shapes are caught by the shape rule of the previous subsection. The two checks are complementary, and the interesting fact, which the text nearly states, is that neither of them catches a wrong transpose of a square factor. That is exactly the case the opening warned about, and it is the reason a numerical check is needed at all.
Fix, as a draft: "This *$1\times1$ check* catches a wrong coefficient or a missing term. It is blind to transposes and to the order of factors, because $1\times1$ matrices commute and equal their transposes. The shape rule catches most of those, but not when the factor is square. For that case only a numerical check helps." Then the counterexample as it stands. The slide title that repeats the claim needs the same change.

**MC-12** · minor · explanation
Anchor: "respect to $\mathbf Y$ and call it the *adjoint* of $\mathbf Y$; it has the" (Backward Rules of a Layer).
The word "adjoint" is introduced as new, although the multivariable section defined $\bar a$ with the same bar and the same meaning, and it is never said why the quantity has this name. The Proposition that follows is the reason: $\langle\bar{\mathbf X},d\mathbf X\rangle=\langle\bar{\mathbf Y},d\mathbf Y\rangle$ says that the backward rule is the transpose (the adjoint, in the language of linear maps) of the layer's linearization. One sentence to that effect would explain the name, tie the Proposition to $\bar{\mathbf x}=\mathbf J^\top\bar{\mathbf y}$, and prepare the next section's unexplained phrase "obtains the VJP by transposing the JVP" (AD-9).
Fix: "We write $\bar{\mathbf Y}=\nabla_{\mathbf Y}\ell$, the adjoint of `eq_mdl-adjoint`, now for a matrix." After the Proposition: "In words, the backward rule of a layer is the transpose of its linearization, which is where the name adjoint comes from."

**MC-13** · minor · language
Anchor: "order of the previous subsection applied to one factor. A row vector times a" (Backward Rules of a Layer).
The left-to-right order was introduced under "The Order of Multiplication", which is not the previous subsection ("The Differential Method" lies between) and is in the previous part.
Fix: "the left-to-right order of `eq_mdl-jacobian-product`" or name the subsection.

**MC-14** · moderate · explanation (notation)
Anchor: "the weights $\mathbf W\in\mathbb R^{d\times m}$ to give $\mathbf Y=\mathbf X\mathbf W$." (Backward Rules of a Layer).
The section changes the side on which the weights multiply three times. The opening network and least squares use a column, $\mathbf W\mathbf x$. The matrix-product rule uses a minibatch with one example per row, $\mathbf X\mathbf W$, and a different letter for the output width ($m$, which was the number of outputs of $\mathbf f$ and is about to be the mask $\mathbf m$). The backward trace returns to columns, and the next section's "Parameter Gradients" derives the column rule again and then explains that the other is "transposed because there the weights multiply from the right". Each switch is announced, but the reader carries two sets of transposes through the most error-prone formulas of the section, in a section whose subject is not getting transposes wrong.
Fix: state the two conventions once, side by side, where the matrix-product rule is derived: one example, $\mathbf y=\mathbf W\mathbf x$, gives $\bar{\mathbf W}=\bar{\mathbf y}\mathbf x^\top$ and $\bar{\mathbf x}=\mathbf W^\top\bar{\mathbf y}$; a minibatch with examples as rows, $\mathbf Y=\mathbf X\mathbf W$, gives $\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$ and $\bar{\mathbf X}=\bar{\mathbf Y}\mathbf W^\top$; the second is the sum of the first over the examples, transposed. The sentence beginning "For a single column" already says half of this; move it up and complete it. Use a letter other than $m$ for the output width.

**MC-15** · minor · thought
Anchor: "the variable on the way forward. Every rule also uses forward values:" (after the random-point cell).
"Every rule" is too strong: the bias rule $\bar{\mathbf b}=\bar{\mathbf y}$, derived a page earlier, uses none, and neither does a sum. The distinction matters for the memory count that this sentence sets up, since additions cost the tape nothing.
Fix: "The product and the activation also use forward values ... the bias rule does not."

**MC-16** · moderate · pedagogy and enjoyment (pattern, 4 of 5 cells)
Anchors: "Both lines print $(2,4)$. The product :eqref:`eq_mdl-jacobian-product` has a"; "Each hand formula matches the framework entry for entry. The inactive unit"; "All three lines print `True`, for any random draw."; "The formula matches, $\mathbf J\mathbf 1$ vanishes, and the gradient of the".
Four of the five cells that print something ask "does the framework agree with the formula?" and answer yes. The reader learns to skip them, since the formula has just been derived (twice, in the case of the network) and the answer is known. The cells do verify something the prose discusses, so they pass the house rule, but against the bar they are the weakest part of the section: MacKay's and the first cell's kind of experiment produces a number the reader has to think about. The random-point cell has a second weakness. The danger the section names is a wrong transpose of a square factor. The cell tests the matrix-product rule with $\mathbf X$ of shape $3\times4$ and $\mathbf W$ of shape $4\times2$, where a wrong transpose would raise a shape error, so the test cannot fail in the way the section fears. Only the quadratic form, with a random nonsymmetric $3\times3$ matrix, is a real test, and the prose does not say so.
Fix within the writing pass: say in one sentence what makes the quadratic-form line a real test (a nonsymmetric square $\mathbf A$, for which $2\mathbf A\mathbf x$ would print `False`), and shorten the framing of the others to a clause each. Fix that needs a code change (deferred category): make the cell print the wrong guess as well, `grad x^T A x == 2 A x : False`, and use square $\mathbf X$ and $\mathbf W$ with a transposed rule printing `False`. A check that can fail teaches the reader how to check.

**MC-17** · minor · language
Anchor: "$p_j$ at rate $p_ip_j$. Two checks against intuition confirm the formula." (Softmax and Cross-Entropy).
The first item is a check ($\mathbf J\mathbf 1=\mathbf 0$ because of shift invariance). The second is not a check of the formula; it is a new and important fact (the Jacobian is the Hessian of log-sum-exp, hence symmetric and positive semidefinite, hence log-sum-exp is convex). Announcing it as a check undersells it, and the same sentence opens the same pair of remarks in the next section.
Fix: "One check: ..." for the first, and start the second paragraph with the fact itself: "$\mathbf J$ is symmetric, which a Jacobian need not be. The reason is that it is a Hessian."

**MC-18** · minor · language (repetition)
Anchors: "Libraries fuse softmax and cross-entropy into one primitive, for a numerical"; "checks $\nabla\operatorname{LSE}=\mathbf p$ at this $\mathbf z$. The identity".
(a) The three sentences on the fused primitive restate what the main text teaches in `subsec_softmax-implementation-revisited` and the appendix in `subsec_mdl-stable-softmax` (the October review counted stable log-sum-exp three times in chapters 25 and 26). (b) The paragraph after the cell repeats the argument, made a page earlier, that the softmax Jacobian is the Hessian of log-sum-exp.
Fix: (a) one clause with a reference: "Libraries fuse the two into one primitive for numerical reasons (`subsec_mdl-stable-softmax`)". (b) cut the last two sentences of the paragraph after the cell.

**MC-19** · moderate · language (closing formula)
Anchor: "We can now write the Jacobian of any layer as its best linear approximation," (end of the body).
The body ends with a recap paragraph that begins "We can now ...", followed directly by a Summary that restates the same results. The writing guide (12.2, item 3) rules out both the separate recap and this opening formula, and the next section closes its body with the same words. The second half of the paragraph is good and is not a recap: the product was multiplied by hand for a chain, and a program must do it for a graph, keep the forward values, and stay cheap.
Fix: delete the first two sentences. Begin the closing at "The product $\mathbf J_K\cdots\mathbf J_1$ was multiplied by hand here, for a chain of three factors."

**MC-20** · minor · language (pattern, 3 instances across this section and the next)
Anchors: "this is the one step that is not routine. The coefficient of"; "The step that is not routine is the second equality: $d\mathbf x^\top\mathbf A\mathbf x$".
Telling the reader which step of a derivation is the real one is what the voice guidance asks for, and in the chain-rule proof it is exactly right. As a fixed phrase used three times in two sections (the third is in the next section, on a step that is routine) it becomes a formula.
Fix: keep it in the chain-rule proof. For the quadratic form say what the step is: "The second equality transposes a $1\times1$ matrix".

## Exercises

Ten problems, about 1,600 words, well above the usual standard of the book. They are graded (the first four stay close to the text, the last three are real matrix calculus), each has a deep learning reason stated in its first sentence (normalizing flows for the log-determinant, spectral normalization for $\sigma_1$, attention for the softmax vector–Jacobian product), and several return to the section's figures and numbers with a new question. I checked the answers that could be checked cheaply: the largest $r$ in exercise 1 is $1/2$; the area ratio of exercise 2 is $1.012$ and $1.003$, above $1$ and approaching it like $r^2$; in exercise 3 both end orders cost $3\times10^7$ and splitting at the two narrow layers costs about $10^7$; the LayerNorm Jacobian, the two hints, and the rank $n-2$ are correct; the hyperparameter, linear-solve and spectral-norm formulas are correct.

- **Cut or trim.** Exercise 4(a) asks the reader to "derive `eq_mdl-matmul-vjp` by the differential method", which the text has just done; keep only the index half. The count of forward against backward multiplications is asked in 4(b) and again in 5(c), and a third time in the next section's exercise on a tensor tape; keep one.
- **Fix.** The self-reference in the body to "the first exercise" and "the exercises" should use labels (MC-5).
- **Keep as they are.** Exercises 1, 2, 3, 6, 7 are the best: each takes an object of the section and asks something the section did not answer. Exercises 8 to 10 form a set of three classical derivatives (log-determinant, inverse, spectral norm); if the count must fall, 9 and 10 are both adapted from the same MIT problem set and one of them could move to the Further Reading notebook's pointers, but neither is filler.
- **Could not check.** The attributions give "Problem 1" for the linear-solve exercise here and also for the hyperparameter exercise of the next section, and "Problem 3" for the spectral norm. I did not open the problem set.

## Still open from October

The October review's note for this chapter, "backprop is explained three times", is still open: the multivariable section's "Backpropagation", this section's "Backward Rules of a Layer", and the next section's "The Tape" and "Parameter Gradients" each define the adjoint and derive a backward pass. The three are at rising levels of generality, which is defensible, but each presents itself as the first.

## For the chapter review

- **Order and fit.** The section is in the right place and is the hinge of the chapter: it turns the scalar chain rule of `sec_mdl-multivariable_calculus` into matrix products and hands the product to the next section. It assumes the gradient, the Hessian, Clairaut–Schwarz and the sum over paths from the previous section, and the SVD and the trace from the linear algebra chapter.
- **Boundary with the next section.** The cut between "what the product is" and "how a program evaluates it" is currently blurred from both sides: this section names the modes and counts their cost, and the next re-derives the product, the count, and the weight gradient. Fix the homes (proposal under Structural findings 1) before either section is revised.
- **Chapter-wide repetition to check in the other sections.** The log-sum-exp Hessian is derived in three sections, the adjoint is defined in three, the product $\mathbf J_K\cdots\mathbf J_1$ carries two labels, and the $10^8$-parameter example opens the chapter introduction, the next section, and `sec_autograd` in the main text.
- **A wrong address in the main text.** `sec_backprop` says the appendix "derives them for general tensor operations in `sec_mdl-matrix-calculus-autodiff`"; the layer rules are derived here, in `subsec_mdl-layer-vjp`. The reference should point to this section.
- **Suspected pattern for the part.** Every code cell is introduced by a question of the form "Does the framework agree with the formula?", and most are answered "yes". The cells that work best are those where the output says more than the prediction. Other reviewers may want to count the two kinds in their sections. A second suspected pattern: bodies that end on a "We can now ..." paragraph directly before the Summary.
