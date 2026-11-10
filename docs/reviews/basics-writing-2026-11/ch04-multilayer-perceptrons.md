# Chapter 4, Multilayer Perceptron: writing review

Sources: the eight `.md` files of `chapter_multilayer-perceptrons/`, read in full up to the slides marker, in `_quarto.yml` order. Stored PyTorch outputs and all generated figures were checked against the prose. About 23,000 words of prose (including the resources list of the introduction) and 6,600 words of exercises.

Per-notebook reviews:
[index](ch04-multilayer-perceptrons/index.md) ·
[mlp](ch04-multilayer-perceptrons/mlp.md) ·
[mlp-implementation](ch04-multilayer-perceptrons/mlp-implementation.md) ·
[backprop](ch04-multilayer-perceptrons/backprop.md) ·
[numerical-stability-and-init](ch04-multilayer-perceptrons/numerical-stability-and-init.md) ·
[generalization-deep](ch04-multilayer-perceptrons/generalization-deep.md) ·
[dropout](ch04-multilayer-perceptrons/dropout.md) ·
[kaggle-house-price](ch04-multilayer-perceptrons/kaggle-house-price.md)

## Verdict

This is a good chapter with a real architecture, and after the revision since October it is consistently good textbook prose: plain, correct, almost free of the mannerisms the writing guide lists, with numbers that match the stored outputs everywhere I checked. Its plan is the right one. `mlp.md` asks what a hidden layer can represent; its "Limits of the Theorem" paragraph leaves three questions open (width, training, generalization); the closing of `mlp-implementation.md` restates them as three things the experiment left unexplained; and the next three sections answer them in turn. Each section ends by handing the next its question. Two sections, `mlp.md` and `generalization-deep.md`, have passages I would be happy to find in MacKay.

It is not yet at the bar, for four reasons that recur. (1) The chapter explains mechanisms better than it shows them working: on real data the hidden layer gains three points, dropout gains nothing, the MLP beats a linear model on house prices by a hundredth, and the claim that a deep network with small weights "does not learn" is never run. (2) Three sections pose a sharp question in their first paragraph and answer a neighbouring one. (3) Backpropagation is taught as a self-contained topic beside the autograd section of chapter 1 instead of on top of it. (4) The prose is clean to the point of austerity: a uniform declarative register, almost no voice, and a great deal of restatement of the same few numbers. A reader will understand the chapter; fewer will enjoy it than its content deserves.

## Scores

1 to 5; 5 means I would be happy to find the page in Cover and Thomas or MacKay.

| Notebook | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| index | 4 | 4 | 4 | 4 | 3 |
| mlp | 4 | 5 | 4 | 4 | 4 |
| mlp-implementation | 4 | 3 | 3 | 3 | 3 |
| backprop | 4 | 3 | 3 | 4 | 3 |
| numerical-stability-and-init | 3 | 4 | 3 | 4 | 3 |
| generalization-deep | 4 | 4 | 3 | 4 | 4 |
| dropout | 4 | 3 | 3 | 4 | 3 |
| kaggle-house-price | 4 | 4 | 4 | 4 | 4 |

## The chapter question: understanding, or a list of facts?

**Why hidden layers help.** Understanding, at the level of the toy; a gap at the level of data. The XOR thread in `mlp.md` (no line separates four points; two ReLU units fold the input so that one line does; the same fold composed with itself doubles the pieces) is a single idea carried from a proof to a picture to depth, and a reader can retell it. But the chapter never shows a learned feature. The hand-built XOR network is the only hidden representation the reader ever sees. `mlp-implementation.md` opens by asking whether the hidden layer fixes the shirt, pullover and coat confusions of softmax regression and reports only total accuracy (IMPL-1). So the reader understands why a hidden layer *can* help and has 84% against 87% as the evidence that it *does*.

**How backpropagation works.** The reader comes away able to do it, which is more than most texts achieve: the three local rules are derived from entries, checked by shapes, run by hand on a network with numbers, and verified against autograd. What is missing is the layer of meaning above the mechanics. The section's own opening question, how one pass yields 203,530 derivatives, is answered in chapter 1 and in two exercises but not in its text (BP-2). The hidden-layer gradient is never read as the output error sent back through the outgoing weights (BP-5). The recursion over many layers is not written, though the next section starts from it (BP-7).

**Fit with `chapter_preliminaries/autograd.md`.** The two are consistent and do not contradict each other: same terms (forward pass, backward pass, local rule), same figure convention (values in black, gradients in red), and autograd's promise that `sec_backprop` "opens it for a network with one hidden layer" is kept. But they overlap more than they build. Autograd already walks a four-operation matrix graph, states the $\mathbf{X}^\top$ rule for a matrix-vector product, explains that saved forward values are "why training needs more memory than prediction", introduces vector-Jacobian products, compares forward and reverse mode, and gives the cost argument. Backprop then restarts from a three-scalar graph that is simpler than what the reader has already traced, presents the memory point as a new question, and refers to the earlier section in one clause. The division of labour should be made explicit and the backprop section rebuilt on it: chapter 1 owns the general mechanism and its cost; this section owns what is new for a network (gradients with respect to weight matrices, the gradient passed through a layer, the activation rule, the meaning of the hidden gradient, depth, and the memory count for a real model). See BP-1.

**Why deep networks generalize.** Closer to understanding than a list, and honestly framed: "The answer is not completely known, so this section measures." The reader gets one thread (the class can fit anything, so the training procedure decides) and experiments that make each half of it concrete. The thread thins where it returns to the actual MLP: double descent and minimum norm are shown on a frozen-feature model, maximum margin on a linear one, and the closing sentence claims a little more for networks than was shown (GEN-9). Then `dropout.md` weakens the story: its experiment shows a regularizer that does not change accuracy, and it reads a robustness measurement as proof of co-adaptation (DO-1, DO-2).

## Arc and order

The order is right and should not change: representation, a first experiment, gradients, the scale of gradients, generalization, a regularizer, a capstone. Implementation before backpropagation is the correct choice: the reader has trained the model and wants to know what the framework did.

- The hand-offs between sections are the chapter's structural strength. Each section's last paragraph poses the next section's question in one or two sentences ("A deep network multiplies many such factors, and the next section asks what a long product of them does to the size of the gradient."). Preserve them.
- The initialization thread is well planted: 0.01 is used in `mlp-implementation`, observed to slow the start, explained in `numerical-stability-and-init`, and replaced by He initialization from `generalization-deep` on. What it lacks is its payoff: no network is ever trained with the bad and the good scale side by side (NS-1).
- Sections are nearly equal in length (2,500 to 3,600 words). Length is therefore not proportionate to importance. Backpropagation and generalization carry the chapter's hardest ideas; dropout, a method the text says later models use sparingly or not at all, gets as many words as backpropagation. Dropout should lose about a third (DO-3, DO-4, DO-5).
- The chapter has no ending. The last paragraph of the Kaggle section, about trees and tabular data, is the last paragraph of the chapter. Three or four sentences that return to the three problems of the introduction and say what chapter 5 takes up would close it.

## Most important findings

1. **Rebuild the first third of backpropagation on the autograd section, answer its opening question in the text, and say what the hidden gradient means.** BP-1, BP-2, BP-5; also BP-7 (write the recursion for $K$ layers, with a short loop, since the next section depends on it).
2. **Answer the question `mlp-implementation.md` opens with.** Per-class comparison of softmax regression and the MLP in the text, not in an exercise; ideally also a picture of what a few hidden units respond to. IMPL-1 and the structural finding in `mlp.md`.
3. **Show a deep network failing and then training.** One experiment with the three initializations pays off a claim made three times and never run. NS-1.
4. **Derive the per-layer gradient factor once, in the right place.** It is the central formula of the initialization section and is currently a one-sentence sketch with a forward pointer, while the place it points to says "the same computation shows". Reorder to measurement, derivation, cure, measurement. NS-2, NS-3, NS-5.
5. **Replace the dropout experiment, and stop reading it as evidence of co-adaptation.** The chapter's own noisy-label setting from the previous section is the obvious candidate. DO-1, DO-2.
6. **Tighten the frame of the generalization section.** Pose the puzzle on a network that has actually fitted its data (GEN-1), put the two-sentence random-label argument in the text (GEN-2), remove the scaffolding from the double-descent experiment (GEN-4, GEN-5), and end with what is not known (GEN-9).
7. **Cut framework plumbing from the prose of `mlp-implementation.md`.** Fifteen tab notes, 540 words, more than a fifth of the body, including a subsection on the derivative of `maximum` at zero in four frameworks. IMPL-2, IMPL-3.
8. **Remove the catalogue paragraphs that survive.** GELU approximations, SiLU and SwiGLU (MLP-6); lazy and neural tangent regimes (GEN-5); the He-paper reconciliation (NS-8); "Use in Practice" with five forward references (DO-5); the grokking figure (GEN-8).
9. **Add one plot to the Kaggle section.** The text describes a U-shaped curve from five printed lines. KH-1.
10. **Say each thing once.** See the repetition counts below. IDX-1, MLP-1, IMPL-6.

## Recurring patterns

**The experiments under-deliver, and the text reports it honestly.** Hidden layer: 84% to 87% for 26 times the parameters and three times the epochs. Dropout: no change in accuracy. MLP against linear on house prices: 0.146 to 0.136. Early stopping: one to five points. Honest reporting is a virtue, and I would not trade it for staged wins. But a chapter whose purpose is to make the reader want hidden layers should contain at least one experiment where the new idea clearly matters. The candidates are already implied: a properly initialized MLP against softmax regression; the deep network that trains only with He initialization; dropout in a setting that overfits. This is worth checking in other chapters: experiments that were made rigorous in the last revision may also have been made inconclusive.

**Questions posed and not answered.** `mlp-implementation` (which errors does the hidden layer remove), `backprop` (how can one pass compute so many derivatives), `numerical-stability-and-init` (does a deeper network train as easily; the section measures random matrices). The writing guide's rule that every opening state a question has been followed; the check that the body answers *that* question has not.

**Every section ends twice.** A closing paragraph ("We can now ...", four of seven sections begin it with those words) is followed by a `## Summary` that repeats it. In `mlp-implementation.md` the main result appears in the comparison text, the table, the closing paragraph and the summary: "about 87%" five times in one file. Across the chapter "203,530" appears ten times and the standard deviation 0.01 about forty times. This is the writing guide's §3.3 (closing) plus §12.2 (summary) applied together, and here the guide pushes the prose the wrong way. Suggest: the closing paragraph carries only what is still open; the summary carries what was established; neither repeats numbers already in a table.

**Every section opens with a recap of the previous one, with its numbers.** The guide's "start from the reader" has become a formula: paragraph one restates the last section's result (often the same 203,530, 83%, 0.01), paragraph two is a roadmap. Read in sequence, with the closing and summary just before, the reader meets each result three times in a page. The index and `mlp.md` open with the same paragraph (IDX-1, MLP-1).

**The route paragraph formula.** Seven consecutive "X asks ..." sentences in the introduction (IDX-3). Likely book-wide, since it is the guide's template.

**Framework tabs fragment the prose.** Thirty tab-note blocks in the chapter: 15 in `mlp-implementation`, 9 in `dropout`, 4 in `numerical-stability-and-init`, 2 in `kaggle-house-price`. Most restate one fact four times with different identifiers. Two places have prose that fits only some frameworks (BP-8: a remark about printed $-0$ that the PyTorch output does not show; BP-10: "by calling `backward`"). Rule of thumb for the revision: a tab note is justified only when the frameworks behave differently in a way that affects the result.

**Voice.** The prose has been cleaned of dashes, slogans and stagecraft, and it is better for it. What remains is one register: declarative sentences of similar length, subject first, usually "The X does Y". There is no humour, one human-scale example (body temperature), and history only in the introduction. The best sentences in the chapter are the ones where an author's judgement shows: "The answer is not completely known, so this section measures"; "A baseline is only useful if it is trained as carefully as the models it is compared with"; "`=` where `+=` belongs". The Kaggle section, with its pool that 1,453 houses do not have, shows that concrete detail is enough to give the prose life without any mannerism. The writing guide permits wit that is "brief, concrete, and true"; this chapter has not used the permission. I would not add jokes. I would ask each section for one more example of the PoolQC kind.

**Small tics.** "whatever" as a universal quantifier, 13 times (6 in `mlp.md`, 5 in the initialization section). Explanatory colons mid-sentence are frequent (about 90 in the chapter, 20 in `mlp.md` and 19 in `generalization-deep.md`); most are fine, but in places three consecutive sentences use one. Experimental results are quoted as ranges ("about 84% to 88%", "110 to 150 epochs", "tens to thousands of times"), which follows the project rule on re-execution noise; in two places a conclusion is drawn from overlapping ranges (GEN-6, and the dropout comparison).

**Notation drift.**
- $K$: number of layers (`mlp`, `backprop`, initialization), number of folds and of ensemble members (`kaggle`). `dropout` exercise 2 counts hidden layers with $L$, which is the loss everywhere else (KH-3, DO-8).
- Gradients: $\partial J / \partial \mathbf{W}$ with batch rows in `backprop`; $\nabla_{\mathbf{h}} L$ with per-example columns and a new matrix $\mathbf{M}^{(k)}$ in the next section; $\mathbf{G}$ for both a generic incoming gradient and a code variable (NS-3, BP-11).
- $J$ against $L$: the objective is $J = L + s$ in `backprop` and $L$ elsewhere.
- $m$: number of compositions (`mlp`), mean square $m^{(k)}$ (initialization), margin (`generalization-deep`), mask (`dropout`). $\tau$: weight standard deviation, and a tolerance in an exercise. $\mathbf{h}$: hidden activations, and a row of the feature matrix in the minimum-norm figure. $s$: the sum $x_1 + x_2$, the penalty, a standard deviation, singular values.
- Done well: $\sigma$ is reserved for the sigmoid throughout; the weight layout (rows are inputs) is the same in every section, which closes an October finding.

**Dependence on other chapters.** Heavy but well handled backward: `sec_softmax_scratch` (the 83% and the confusions), `sec_generalization_basics` (polynomial fits, U curve, cross-validation), `sec_weight_decay`, the VC bound, `sec_autograd`, `sec_calculus`. In almost every case the fact is restated before the reference, as the guide asks. Forward references are the problem: `sec_transformer-block` (MLP-6), `sec_scaling` (GEN-5), `sec_scheduler` (DO-3), and `sec_alexnet`, `sec_batch_norm`, `sec_resnet`, `sec_training_recipes` in one paragraph (DO-5). The appendix is cited for depth only, never as a prerequisite, which is correct.

## What to cut

About 1,800 words in all, none of it argument:

- `mlp.md`: tanh approximation of the GELU, SiLU and SwiGLU; the duplicated opening paragraph.
- `mlp-implementation.md`: the ReLU-at-zero tab notes; three quarters of the registration and initializer tab notes; the result restated in the closing paragraph.
- `backprop.md`: the scalar graph as it stands (replace with a recall of the autograd graph and, if wanted, a scalar graph with a reused value); the paragraph "The forward pass and the backward pass depend on each other"; the $-0$ remark.
- `numerical-stability-and-init.md`: the reconciliation with the He paper's count; the dropout aside; the construction of the weighted-sum loss.
- `generalization-deep.md`: the rare-unit filter and its paragraph (by changing the features); the lazy and neural-tangent paragraph; the grokking figure.
- `dropout.md`: the additive-noise derivation; the learning-rate schedule paragraph; "Use in Practice" down to two sentences; one of the three 20-epoch training runs.

## What is missing

- A learned hidden feature, shown (weights of a few trained units as images, or per-class recall before and after).
- The sentence that says why reverse mode is cheap, in the backpropagation section itself.
- The meaning of $\partial J / \partial \mathbf{H}$ as back-propagated error.
- Backpropagation through $K$ layers, as an equation and as a loop.
- A deep network trained with poor and with good initialization.
- An experiment in which dropout changes the result.
- A plot in the Kaggle section.
- A closing paragraph for the chapter.
- One sentence of prerequisites in the introduction.

## What is best and should set the model

- **`mlp.md`, XOR to depth.** One example, solved by proof, by construction, by code and by picture, then reused to explain the next idea ("It is the XOR network read along the diagonal"). This is the writing guide's §5 sequence done as intended.
- **`mlp-implementation.md`, "Checks before Training".** Predictions stated with reasons, then tested in a fraction of a second, then what each failure would mean.
- **`backprop.md`, the matrix-product rule.** Entries, then shapes, then outer products: three readings of one identity.
- **`numerical-stability-and-init.md`, prediction against measurement.** A formula gives $0.0128^9$; the cell prints `9.1e-18`.
- **`generalization-deep.md`, "More data can hurt" and the minimum-norm proposition.** Two printed lines that overturn an intuition, explained by a singular value; and a proposition with proof, picture and numerical check in half a page.
- **`dropout.md`, the two-unit counterexample.** Four lines that fence a claim.
- **`kaggle-house-price.md`, columns as teachers.** `PoolQC`, `MSSubClass`, code 150: each preprocessing rule arrives attached to a real column that needs it, and the code is shaped so that leakage cannot happen by accident.

## Notes on the writing guide

Three places where following the guide has cost the reader something in this chapter: the closing paragraph plus summary (double endings); the route template (seven "asks"); and the opening formula, which has turned "start from the reader" into a recap of numbers. A fourth is subtler: the guide's ban list has been applied thoroughly, and nothing was asked for in exchange. The model texts are plain *and* have a voice. A revision pass should be allowed to add a sharp example or a short human remark where it helps, and the guide should say so more strongly than §10.4 does.

## Could not check

- Rendered HTML pages (I read sources, stored PyTorch outputs and rasterized figures, not the site).
- TensorFlow, JAX and MXNet outputs. Claims that depend on a specific tab (the derivative of `maximum` at zero, which outputs print $-0$, default initializers) were not verified.
- Claims about run-to-run variation ("in every run", "changes from run to run"): one stored run per framework.
- The proposed experiments (NS-1, DO-1, GEN-4) are untested; each fix says what to verify.
- "this book sets the initialization of its models explicitly" (NS-5), outside this chapter.
