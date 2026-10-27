# Standalone Motivation Review: Mathematics for Deep Learning

## Scope and review standard

This review covers the Mathematics for Deep Learning part index and every notebook listed under it in `_quarto.yml`. It evaluates the authoritative Markdown sources rather than generated Quarto files. The central test is whether a reader arriving directly at a section can answer, before substantial formal development begins:

1. What problem is this section trying to solve?
2. Why do the preceding tools fail or become inconvenient?
3. What new idea is being introduced, and what does it buy?
4. What prior knowledge is assumed, without requiring a recap of the previous notebook?

The priorities follow the standalone-motivation rubric. **High** means that a direct reader cannot reliably identify the problem before meeting substantial technical detail. **Medium** means that the progression is sound but the opening is method-first, depends too heavily on a predecessor, or presents a catalogue without a governing problem. **Low** means that a short bridge, qualification, or local reorder would materially improve orientation. Line links point to the current source and may need refreshing after edits.

## Executive findings

The Mathematics part is generally much stronger on standalone motivation than the weak examples that prompted this review. Its chapter indexes often distinguish neighboring concepts explicitly, and many notebooks begin with a concrete failure mode or computational question. The strongest material should not be expanded merely for consistency: in particular, the current ODE-solvers notebook is no longer brief. It opens with a stiff scalar equation, demonstrates Euler instability, and distinguishes the exact flow from a numerical approximation immediately. The rendered page cited as an example may be stale relative to the current source.

The principal cross-cutting issue is not missing exposition but **method-first ordering**. Several notebooks first name a family of abstractions, algorithms, or applications and only later reveal the concrete problem that makes the family necessary. In most cases the needed motivation already exists in the notebook and should be promoted, not newly invented.

The highest-priority revision is:

- `mdl-score-matching-diffusion-flow.md`: its opening assumes the preceding SDE and Fokker--Planck narrative and inventories diffusion, score matching, probability-flow ODEs, flow matching, and related methods before stating the standalone generative-modeling problem. The notebook already contains the right motivating obstacle--an energy model with an intractable normalizer, followed by the unknown data-score problem--but only after the opening.

The recurring medium-priority patterns are:

- decomposition notebooks that open with definitions or use lists rather than a matrix problem (`mdl-eigendecomposition.md`, `mdl-svd-low-rank.md`);
- optimization notebooks that open with abstractions or recipe names before a failed training step (`mdl-gradient-based-optimization.md`, `mdl-adaptive-stochastic-methods.md`);
- survey-scale notebooks whose unifying dependency is stated too late (`mdl-matrix-calculus-autodiff.md`, `mdl-concentration-generalization.md`);
- information-theoretic methods named before the representation-learning question or estimation barrier (`mdl-mutual-information.md`).

Most other issues are low priority and can be fixed with one concrete sentence, a small reorder, or a short comparison bridge. A global editing pass should resist adding repeated historical background. The preferred pattern is: one concrete problem or failure, one sentence locating the section in its chapter, then the existing development.

## Mathematics for Deep Learning part index

### Intended progression

The part moves from representations and transformations, through rates of change and optimization, to uncertainty and information, and finally to continuous-time dynamics. That progression is coherent: linear algebra supplies the objects, calculus supplies local change, optimization turns local change into algorithms, probability and information quantify uncertainty and fit, and dynamics describes evolving states and distributions.

### Notebook assessment

#### `chapter_mathematics-for-deep-learning/index.md` -- low priority

The index gives a compact inventory of the six chapters and connects each to machine learning. It is adequate for a reader entering the part directly. However, the opening is almost entirely declarative: it says which mathematics is ubiquitous without giving a single model-building situation in which several chapters become necessary. Some claims are also broader than the text needs, such as decompositions being behind "every layer" and KL divergence being what "most losses descend from" ([index.md:3](../../chapter_mathematics-for-deep-learning/index.md#L3), [index.md:14](../../chapter_mathematics-for-deep-learning/index.md#L14)). Such absolutes distract from an otherwise useful map.

**Revision instruction.** Add a two- or three-sentence running scenario before the inventory--for example, fitting a probabilistic model requires representing data, differentiating a loss, choosing a stable update, and quantifying uncertainty. Then retain the existing chapter map, qualifying universal claims rather than lengthening every entry.

**Preserve.** Keep the one-paragraph-per-chapter scale and the explicit transition from static mathematics to dynamics ([index.md:5](../../chapter_mathematics-for-deep-learning/index.md#L5), [index.md:16](../../chapter_mathematics-for-deep-learning/index.md#L16)). The index should remain a map, not become a tutorial.

### Dependency-ordered implementation sequence

1. Qualify the two overbroad claims without changing the chapter ordering.
2. Add one compact end-to-end modeling scenario ahead of the inventory.
3. Re-read the final paragraph only for continuity into linear algebra; do not add per-notebook summaries here.

## Linear Algebra

### Intended progression

The chapter index establishes a sensible sequence: geometry gives meaning to vectors and linear maps; eigendecomposition analyzes invariant directions of square operators; SVD extends spectral reasoning to rectangular matrices and supplies optimal low-rank approximation. A direct reader should understand not just that these are decompositions, but which matrix question makes each decomposition the right one.

### Notebook assessment

| Notebook | Standalone assessment | Priority |
|---|---|---|
| `index.md` | Strong question-based map that distinguishes eigenvectors from singular vectors and square from rectangular matrices. | Preserve |
| `mdl-geometry-linear-algebraic-ops.md` | Broad opening depends slightly on prior exposure, but concrete geometric and ML examples quickly supply context. | Low |
| `mdl-eigendecomposition.md` | Starts with a definition and a use list; the dynamic-system problem that best motivates invariant directions appears much later. | Medium |
| `mdl-svd-low-rank.md` | Starts by comparing decompositions and listing applications; the rectangular-operator and compression problems should lead. | Medium |

### Chapter index: preserve the question map

The chapter index asks what matrices do geometrically, which directions a square matrix preserves, and what replaces eigendecomposition for rectangular maps ([index.md:4](../../chapter_mdl-linear-algebra/index.md#L4), [index.md:12](../../chapter_mdl-linear-algebra/index.md#L12)). This is exactly the level of standalone context needed. Its explicit eigendecomposition/SVD distinction prevents the two later notebooks from seeming redundant.

**Revision instruction.** No substantive change. If chapter titles or scope change, update the table, but do not insert a generic history of linear algebra.

### Geometry and linear algebraic operations: make the first use concrete

The opening announces geometric interpretations, similarity, tensors, and computation, but its first paragraph leans on what has already been learned and reads as a roadmap ([mdl-geometry-linear-algebraic-ops.md:4](../../chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md#L4)). A direct reader encounters the concrete point-cloud interpretation later ([mdl-geometry-linear-algebraic-ops.md:61](../../chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md#L61)). This is a low-priority issue because the notebook becomes concrete quickly and repeatedly.

**Revision instruction.** Add one opening question using an object already developed in the notebook: how can the same vector operations express displacement in a point cloud, similarity between an image and a dimmed copy, and a classifier score? Do not preview every subsection. One example is sufficient to establish that the goal is to connect notation to behavior.

**Preserve.** The dimmed-image cosine-similarity example ([mdl-geometry-linear-algebraic-ops.md:505](../../chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md#L505)), the Fashion-MNIST classifier interpretation ([mdl-geometry-linear-algebraic-ops.md:687](../../chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md#L687)), the matrix-as-map development ([mdl-geometry-linear-algebraic-ops.md:889](../../chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md#L889)), and the tensor/einsum motivation ([mdl-geometry-linear-algebraic-ops.md:1614](../../chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md#L1614)) all give abstractions a visible purpose. They should not be compressed into a definition catalogue.

### Eigendecomposition: lead with repeated action, not a use list

The notebook begins by saying that eigenvalues and eigenvectors reveal invariant directions and then lists applications ([mdl-eigendecomposition.md:4](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L4)). That is correct, but it does not let a new reader feel why invariant directions are a solution to a problem. The first diagonal example gives computational meaning later ([mdl-eigendecomposition.md:45](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L45)), while the clearest motivating question--what repeated multiplication does to a state--appears in the power-iteration discussion ([mdl-eigendecomposition.md:1128](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L1128)).

**Revision instruction.** Open with a two-dimensional recurrence such as (x_{t+1}=Ax_t): ask which components persist, grow, decay, or oscillate after many applications. State that invariant directions turn repeated matrix action into scalar multiplication, then introduce the eigenvalue equation. This should be conceptual rather than a second worked example; the existing example can carry the computation.

There is a second, smaller ordering issue. Early computational remarks distinguish factorization from matrix-vector work ([mdl-eigendecomposition.md:175](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L175)), but the later "Localizing and Computing" material does not immediately restate which computational question each method answers ([mdl-eigendecomposition.md:1030](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L1030)).

**Revision instruction.** Add a short bridge at that later heading: Gershgorin-style bounds cheaply localize the whole spectrum, whereas power iteration estimates a dominant eigenpair using only matrix-vector products. This prevents a list of spectral tools from looking like unrelated techniques.

**Preserve.** The non-normal/transient-growth discussion is unusually well motivated: it explains why eigenvalues alone can miss short-term amplification before presenting the example ([mdl-eigendecomposition.md:574](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L574), [mdl-eigendecomposition.md:630](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L630)). The random-matrix and initialization connection also states its neural-network consequence before giving asymptotic facts ([mdl-eigendecomposition.md:1267](../../chapter_mdl-linear-algebra/mdl-eigendecomposition.md#L1267)). Keep both.

### Singular value decomposition: put the rectangular and low-rank questions first

The opening compares SVD with eigendecomposition and lists geometry, compression, PCA, least squares, and modern deep-learning uses ([mdl-svd-low-rank.md:4](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L4)). This is a useful preview for a returning reader but remains method-first for a direct reader. The geometric need appears more sharply later: rectangular maps have input and output directions in different spaces ([mdl-svd-low-rank.md:57](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L57)). The low-rank theorem begins formally ([mdl-svd-low-rank.md:387](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L387)); the concrete image-compression question does not appear until much later ([mdl-svd-low-rank.md:540](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L540)).

**Revision instruction.** Reframe the opening around two questions that cover almost the whole notebook:

1. For a rectangular matrix, which input direction is stretched most, and where does it go?
2. If only (k) directions can be retained, which approximation loses the least information?

Then introduce SVD as the common answer and use the existing eigendecomposition comparison as a boundary statement. Before the low-rank theorem, preview the image or model-compression task in words so that the theorem answers an already-visible problem.

The final survey of modern uses moves through LoRA, update preconditioning, spectral constraints, and diagnostics ([mdl-svd-low-rank.md:785](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L785)). Each item is relevant, but the section lacks a single organizing axis.

**Revision instruction.** Organize those uses by the problem they solve--parameter-efficient adaptation, optimization geometry, stability control, and diagnostics--using a short lead sentence or comparison table. Avoid adding more named methods.

**Preserve.** The PCA connection ([mdl-svd-low-rank.md:556](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L556)), least-squares connection ([mdl-svd-low-rank.md:624](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L624)), and conditioning discussion ([mdl-svd-low-rank.md:702](../../chapter_mdl-linear-algebra/mdl-svd-low-rank.md#L702)) each state a recognizable task before deriving consequences.

### Dependency-ordered implementation sequence

1. Revise the eigendecomposition opening around repeated matrix action; it establishes the contrast that SVD relies on.
2. Add the computational-task bridge before eigenvalue localization and iteration.
3. Revise the SVD opening around rectangular stretch and rank-(k) approximation.
4. Add the image/model-compression preview before the low-rank theorem.
5. Organize the modern SVD applications by problem rather than method name.
6. Add only the one-sentence concrete hook to the geometry notebook after the decomposition openings are stable.

## Calculus

### Intended progression

The chapter first connects one-variable derivatives to optimization, extends derivatives and chain rules to many variables, turns those rules into scalable matrix calculus and automatic differentiation, and then treats integration as the machinery for continuous probability and expectation. The index correctly signals that integration is a parallel branch rather than a prerequisite for differentiation.

### Notebook assessment

| Notebook | Standalone assessment | Priority |
|---|---|---|
| `index.md` | Clear split between differentiation and integration, with explicit dependency guidance. | Preserve |
| `mdl-single-variable-calculus.md` | The motivating training-loss problem is present immediately after a brief topic list. | Low |
| `mdl-multivariable-calculus.md` | Opens with the exact scalar-loss/many-parameter problem and sustains it through gradients, chain rules, and curvature. | Preserve |
| `mdl-matrix-calculus-autodiff.md` | Opens with terminology and a capability list before stating the decisive computational asymmetry of reverse mode. | Medium |
| `mdl-integral-calculus.md` | Coherent development, but the continuous-probability need that makes integration essential arrives late. | Low |

### Chapter index: preserve the two-path dependency statement

The index distinguishes derivatives, multivariable calculus, matrix/autodiff machinery, and integration, and explicitly says that the integration notebook can be read independently of the differentiation sequence ([index.md:4](../../chapter_mdl-calculus/index.md#L4)). That is excellent navigation for direct entry.

**Revision instruction.** No motivation change is needed. Preserve its brevity and dependency statement.

### Single-variable calculus: reverse the first two moves

The opening first lists limits, derivatives, Taylor approximation, and optimization ([mdl-single-variable-calculus.md:4](../../chapter_mdl-calculus/mdl-single-variable-calculus.md#L4)). The next paragraph supplies the real problem: training changes parameters to reduce a scalar loss, and derivatives predict the effect of a small change ([mdl-single-variable-calculus.md:12](../../chapter_mdl-calculus/mdl-single-variable-calculus.md#L12)).

**Revision instruction.** Put the training-loss question first, then use the existing topic list as the answer path. This is mostly a paragraph reorder; avoid adding an extended neural-network example before limits are defined.

**Preserve.** The gradient-descent development makes the local-linear approximation and step-size limitation explicit ([mdl-single-variable-calculus.md:293](../../chapter_mdl-calculus/mdl-single-variable-calculus.md#L293)), and the nondifferentiability section explains why practical optimization can still proceed ([mdl-single-variable-calculus.md:672](../../chapter_mdl-calculus/mdl-single-variable-calculus.md#L672)).

### Multivariable calculus: already standalone

The notebook begins with the central machine-learning mismatch--a scalar loss depending on many parameters--and explains why partial derivatives alone are insufficient ([mdl-multivariable-calculus.md:4](../../chapter_mdl-calculus/mdl-multivariable-calculus.md#L4)). The gradient is tied to directional change ([mdl-multivariable-calculus.md:16](../../chapter_mdl-calculus/mdl-multivariable-calculus.md#L16)), steepest descent to a norm-dependent optimization question ([mdl-multivariable-calculus.md:200](../../chapter_mdl-calculus/mdl-multivariable-calculus.md#L200)), the chain rule to composed computation ([mdl-multivariable-calculus.md:397](../../chapter_mdl-calculus/mdl-multivariable-calculus.md#L397)), and the Hessian to the limitations of first-order information ([mdl-multivariable-calculus.md:683](../../chapter_mdl-calculus/mdl-multivariable-calculus.md#L683)).

**Revision instruction.** Preserve the opening and local transitions. Do not add a redundant recap of single-variable calculus.

### Matrix calculus and automatic differentiation: expose the scaling problem first

The opening introduces Jacobian-vector products, vector-Jacobian products, forward mode, reverse mode, and implicit differentiation before the reader has seen the computational problem that decides among them ([mdl-matrix-calculus-autodiff.md:4](../../chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md#L4)). Much later the notebook states the decisive asymmetry: for one scalar loss and (n) parameters, reverse mode obtains the gradient in one reverse traversal, while forward mode would need roughly (n) directional passes ([mdl-matrix-calculus-autodiff.md:697](../../chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md#L697)).

**Revision instruction.** Move a nontechnical version of that asymmetry into the opening. Start with a network having millions of parameters and one scalar loss; ask how to compute all sensitivities without building a full Jacobian or rerunning once per parameter. Then name JVPs and VJPs as products that avoid materializing the Jacobian, and introduce forward/reverse modes as choices dictated by input and output dimensions. Keep the formal explanation in its current location.

**Preserve.** The matrix-derivative identities are introduced as reusable rules rather than an unexplained formula dump ([mdl-matrix-calculus-autodiff.md:266](../../chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md#L266)). The implicit-differentiation section also starts from an actual solver or argmin whose internal iterations should not all be differentiated ([mdl-matrix-calculus-autodiff.md:1054](../../chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md#L1054)).

### Integral calculus: surface the probability use earlier

The opening explains accumulation, antiderivatives, numerical quadrature, and improper integrals ([mdl-integral-calculus.md:4](../../chapter_mdl-calculus/mdl-integral-calculus.md#L4)). It is accurate, but a machine-learning reader does not see the unavoidable modeling need until the probability section explains that continuous probabilities and expectations are integrals ([mdl-integral-calculus.md:651](../../chapter_mdl-calculus/mdl-integral-calculus.md#L651)).

**Revision instruction.** Add one concrete probability question to the opening: a density assigns zero probability to any exact point, so how is the probability of an interval or an expected loss computed? Use that question only as motivation; leave probability definitions in the later section. This will also clarify why numerical integration and Monte Carlo belong in the notebook.

**Preserve.** The treatment of improper integrals motivates limiting definitions ([mdl-integral-calculus.md:241](../../chapter_mdl-calculus/mdl-integral-calculus.md#L241)); the Gaussian example shows why exact normalization matters ([mdl-integral-calculus.md:549](../../chapter_mdl-calculus/mdl-integral-calculus.md#L549)); and the differentiation-under-the-integral discussion states the conditions and failure risk rather than presenting a symbolic trick ([mdl-integral-calculus.md:766](../../chapter_mdl-calculus/mdl-integral-calculus.md#L766)).

### Dependency-ordered implementation sequence

1. Reorder the single-variable opening so the scalar training-loss problem precedes the topic inventory.
2. Revise the matrix-calculus/autodiff opening around the scalar-output, many-parameter scaling problem.
3. Check that the revised autodiff opening does not duplicate the multivariable notebook; it should emphasize computation, not redefine gradients.
4. Add the continuous-probability question to the integral-calculus opening.
5. Leave the chapter index and multivariable notebook substantively unchanged.

## Optimization

### Intended progression

The chapter should answer a sequence of increasingly realistic questions. First, how does local derivative information determine a useful direction and step? Second, what changes when gradients are noisy, coordinates have different scales, and models are large? Third, when does convex structure turn local reasoning into global guarantees? Fourth, how are constraints handled? Finally, how do finite-precision arithmetic and conditioning alter otherwise correct algorithms? The index already communicates most of this dependency structure.

### Notebook assessment

| Notebook | Standalone assessment | Priority |
|---|---|---|
| `index.md` | Strong problem-oriented roadmap and dependency statement. | Preserve |
| `mdl-gradient-based-optimization.md` | Opens with direction-versus-distance abstractions before showing a step that fails because the learning rate is wrong. | Medium |
| `mdl-adaptive-stochastic-methods.md` | Opens by naming the modern training recipe; the nonconvexity, noise, scale, and noise-floor problems come later. | Medium |
| `mdl-convexity.md` | Clearly states what convexity buys and motivates the advanced tools locally. | Low |
| `mdl-constrained-optimization-duality.md` | The initial constraint problem is clear; the late sequence of worked duals needs a small governing bridge. | Low |
| `mdl-numerical-stability-conditioning.md` | Excellent failure-first opening and sustained connection from formulas to floating-point behavior. | Preserve |

### Chapter index: retain the question-driven route

The index frames gradient methods, adaptive stochastic methods, convexity, constraints, and numerical stability as responses to distinct practical questions ([index.md:4](../../chapter_mdl-optimization/index.md#L4)). It also distinguishes the main path from optional structural material ([index.md:13](../../chapter_mdl-optimization/index.md#L13)).

**Revision instruction.** No substantive change. If cross-links are edited, preserve the current distinction between algorithmic progression and theory-focused branches.

### Gradient-based optimization: demonstrate the failed step before abstracting it

The notebook begins by separating direction selection from distance traveled and previewing several algorithms ([mdl-gradient-based-optimization.md:4](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L4)). The distinction is important, but it is presented before a reader sees why the negative gradient alone does not specify a safe update. The initial formal development ([mdl-gradient-based-optimization.md:48](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L48)) introduces a quadratic around [mdl-gradient-based-optimization.md:96](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L96), after the conceptual claim has already been made.

**Revision instruction.** Open with one step on an elongated quadratic or a one-dimensional smooth objective. Show in words or one equation that the negative gradient points downhill locally, yet an oversized step can increase the objective or oscillate. Then introduce direction versus distance as the two choices every first-order method must make. This creates a visible need for smoothness bounds and line search without lengthening their derivations.

The transition from Newton and quasi-Newton methods to modern named optimizers briefly becomes a catalogue ([mdl-gradient-based-optimization.md:1059](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L1059)).

**Revision instruction.** Reduce that bridge to the structural distinction relevant here--full curvature, low-memory curvature approximation, and structured/adaptive approximations--and defer detailed named methods to the adaptive notebook.

**Preserve.** Smoothness is introduced as the condition that turns a local gradient into a guaranteed decrease statement ([mdl-gradient-based-optimization.md:137](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L137)). The full-gradient cost motivates stochastic gradients ([mdl-gradient-based-optimization.md:789](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L789)); Newton's computational limits are explicit ([mdl-gradient-based-optimization.md:999](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L999)); and the trust-region section explains why a local model needs a region of credibility ([mdl-gradient-based-optimization.md:1151](../../chapter_mdl-optimization/mdl-gradient-based-optimization.md#L1151)).

### Adaptive and stochastic optimization: organize the recipe by its failure modes

The opening names AdamW, warmup, decay schedules, clipping, and other components of a modern training recipe ([mdl-adaptive-stochastic-methods.md:4](../../chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md#L4)). A reader entering here can tell what will be covered, but not why these parts coexist. The actual combined problem--nonconvex objectives and noisy minibatch gradients--is stated later ([mdl-adaptive-stochastic-methods.md:50](../../chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md#L50)), and the failure of one global step size under unequal coordinate scales is clearer later still ([mdl-adaptive-stochastic-methods.md:204](../../chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md#L204)).

**Revision instruction.** Recast the opening around three observed training failures:

- minibatch noise prevents exact descent and creates a late-stage noise floor;
- different coordinates or layers need different effective scales;
- a step size appropriate late in training may be unstable at the start or wasteful near the end.

Then state that the notebook dissects the modern recipe component by component. Name AdamW and schedule families only after their respective problem is visible.

**Preserve.** The AdaGrad--RMSProp--Adam sequence is organized as successive corrections to per-coordinate scaling and stale accumulation ([mdl-adaptive-stochastic-methods.md:199](../../chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md#L199)). The weight-decay discussion distinguishes shrinkage from loss regularization ([mdl-adaptive-stochastic-methods.md:531](../../chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md#L531)); the schedule survey acknowledges tradeoffs rather than declaring a universal winner ([mdl-adaptive-stochastic-methods.md:632](../../chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md#L632)); variance reduction begins from the cost/variance tension ([mdl-adaptive-stochastic-methods.md:750](../../chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md#L750)); and structured preconditioners form a meaningful expressiveness/cost ladder. Keep those local motivations.

### Convexity: add one signpost before the advanced tail

The opening states the practical benefit of convexity--local information can certify global behavior--and acknowledges that deep-network objectives are usually nonconvex ([mdl-convexity.md:4](../../chapter_mdl-optimization/mdl-convexity.md#L4)). The later sections motivate recognizing convexity and its limits ([mdl-convexity.md:754](../../chapter_mdl-optimization/mdl-convexity.md#L754)), conjugates ([mdl-convexity.md:873](../../chapter_mdl-optimization/mdl-convexity.md#L873)), proximal operators ([mdl-convexity.md:966](../../chapter_mdl-optimization/mdl-convexity.md#L966)), and coordinate descent ([mdl-convexity.md:1065](../../chapter_mdl-optimization/mdl-convexity.md#L1065)) locally. The only risk is that the final three tools can feel appended to a reader who expected only definitions and guarantees.

**Revision instruction.** Add one sentence at the start of the advanced tail distinguishing the jobs: conjugates reformulate or dualize objectives, proximal operators handle nonsmooth regularizers, and coordinate descent exploits cheap blockwise updates. No new examples are needed.

**Preserve.** Keep the existing opening and the explicit discussion of where convex reasoning stops applying.

### Constrained optimization and duality: state why the worked examples differ

The notebook clearly motivates constraints as requirements that cannot be represented safely by an unconstrained objective and develops equality and inequality conditions from that need ([mdl-constrained-optimization-duality.md:4](../../chapter_mdl-optimization/mdl-constrained-optimization-duality.md#L4), [mdl-constrained-optimization-duality.md:59](../../chapter_mdl-optimization/mdl-constrained-optimization-duality.md#L59)). The SVM example also introduces its margin problem before the derivation ([mdl-constrained-optimization-duality.md:895](../../chapter_mdl-optimization/mdl-constrained-optimization-duality.md#L895)). However, the broad "Worked Duals" sequence beginning near [mdl-constrained-optimization-duality.md:890](../../chapter_mdl-optimization/mdl-constrained-optimization-duality.md#L890) can look like a set of exercises rather than a progression.

**Revision instruction.** Add a two-sentence bridge explaining what the examples are selected to demonstrate: SVM exposes support-vector sparsity through complementary slackness; water filling shows resource allocation through a shared multiplier; the nonconvex example shows why a dual gap may remain. Keep each derivation in place.

**Preserve.** The opening, KKT motivation, and explicit warnings about constraint qualification and strong duality are appropriately problem-led.

### Numerical stability: protect the failure-first structure

The notebook opens with the discrepancy between mathematically equivalent formulas and finite-precision computations, then previews overflow, underflow, cancellation, accumulation error, conditioning, and gradient pathologies ([mdl-numerical-stability-conditioning.md:4](../../chapter_mdl-optimization/mdl-numerical-stability-conditioning.md#L4)). It makes the problem concrete before giving remedies. Stable softmax/log-sum-exp is developed from an actual overflow failure ([mdl-numerical-stability-conditioning.md:273](../../chapter_mdl-optimization/mdl-numerical-stability-conditioning.md#L273)), and the conditioning section distinguishes an ill-conditioned problem from an unstable algorithm ([mdl-numerical-stability-conditioning.md:673](../../chapter_mdl-optimization/mdl-numerical-stability-conditioning.md#L673)).

**Revision instruction.** No motivation change. Avoid adding a generic floating-point preamble before the current examples.

### Dependency-ordered implementation sequence

1. Revise the gradient-optimization opening around a failed oversized step.
2. Simplify its late catalogue so the adaptive notebook owns the optimizer taxonomy.
3. Revise the adaptive/stochastic opening around noise, heterogeneous scale, and training phase.
4. Add the one-sentence advanced-tool signpost in convexity.
5. Add the purpose bridge before the worked duals.
6. Leave the index and numerical-stability notebook substantively unchanged.
## Probability and Statistics

### Intended progression

The chapter moves from the meaning of random variables to reusable distribution families, then to fitting probabilistic models by likelihood and to Bayesian computation when uncertainty must be retained. It subsequently develops statistical summaries and tests, concentration and generalization guarantees, and a compact classification case study in naive Bayes. The chapter index usefully distinguishes objects, models, inferential principles, computation, and guarantees.

### Notebook assessment

| Notebook | Standalone assessment | Priority |
|---|---|---|
| `index.md` | Excellent distinctions among probability, statistics, likelihood, Bayesian computation, and generalization. | Preserve |
| `mdl-random-variables.md` | Slightly predecessor-dependent opening; a concrete dartboard problem soon supplies the missing intuition. | Low |
| `mdl-distributions.md` | Long reference-style gallery, but a family tree and support/modeling table give it a coherent selection principle. | Low |
| `mdl-maximum-likelihood.md` | States the estimation principle before the concrete coin problem; mostly a local ordering issue. | Low |
| `mdl-bayesian-computation.md` | Exemplary standalone opening centered on posterior prediction and an intractable integral. | Preserve |
| `mdl-statistics.md` | Clear purpose and locally motivated progression from estimation to uncertainty and testing. | Preserve |
| `mdl-concentration-generalization.md` | Opens with a theorem roadmap; the coin-estimation gap and the fixed-function/data-selected-function dependency arrive later. | Medium |
| `mdl-naive-bayes.md` | Method-first opening; the exponential input-pattern problem that justifies the independence assumption arrives later. | Medium |

### Chapter index: preserve its distinctions

The index asks separate questions about randomness, model families, parameter fitting, posterior computation, empirical evidence, and generalization ([index.md:4](../../chapter_mdl-probability-statistics/index.md#L4)). It also provides reading paths rather than assuming every notebook must be read in order ([index.md:17](../../chapter_mdl-probability-statistics/index.md#L17)).

**Revision instruction.** No substantive change. Maintain the present separation between maximum likelihood as a principle and statistics as a broader set of estimators and tests.

### Random variables: promote the geometric question

The opening refers to preceding probability ideas and previews discrete and continuous random variables ([mdl-random-variables.md:4](../../chapter_mdl-probability-statistics/mdl-random-variables.md#L4)). It does mention the initially surprising fact that a continuous variable assigns zero probability to an exact value, but the concrete dartboard construction appears later ([mdl-random-variables.md:58](../../chapter_mdl-probability-statistics/mdl-random-variables.md#L58)).

**Revision instruction.** Put the dartboard question, or a one-sentence version of it, in the opening: if hitting one exact point has probability zero, how can regions have positive probability? Then state that random variables, densities, transformations, and expectations supply the language needed to answer that question and analogous ML questions.

**Preserve.** The geometric construction and subsequent discrete/continuous contrast should remain the primary intuition rather than being replaced by a formal measure-theory preface.

### Distributions: explicitly declare the lookup workflow

The notebook begins with a broad catalogue but quickly organizes distributions by construction and supplies both a family tree and a table matching support/modeling assumptions to candidate families ([mdl-distributions.md:4](../../chapter_mdl-probability-statistics/mdl-distributions.md#L4), [mdl-distributions.md:26](../../chapter_mdl-probability-statistics/mdl-distributions.md#L26)). This is a legitimate reference-notebook design, not an unmotivated list.

**Revision instruction.** Add one explicit sentence telling the reader how to use the notebook: first identify the support and data-generating story, then choose a family, and only then consult its formulas. Keep the existing table early. Do not add a separate motivating example for all eleven distributions.

**Preserve.** The repeated "where it arises" pattern and relationships among families are precisely what prevent the gallery from becoming a formula sheet.

### Maximum likelihood: let the coin pose the estimation problem

The opening introduces likelihood, log-likelihood, asymptotics, MAP, and latent-variable methods before grounding model fitting in a small data problem ([mdl-maximum-likelihood.md:4](../../chapter_mdl-probability-statistics/mdl-maximum-likelihood.md#L4)). The coin example later supplies a transparent question with a closed-form answer ([mdl-maximum-likelihood.md:78](../../chapter_mdl-probability-statistics/mdl-maximum-likelihood.md#L78)).

**Revision instruction.** Begin with the coin question: after observing heads and tails, which bias makes the observed data least surprising? Then name maximum likelihood as the general principle and retain the current roadmap. This should be a short reorder, not an expanded probability lesson.

**Preserve.** The notebook broadens the principle responsibly: asymptotic behavior explains what repeated data buys, MAP distinguishes a prior penalty from pure likelihood, and the latent-variable/EM material states why direct optimization becomes difficult before introducing alternating updates.

### Bayesian computation: preserve as a model opening

The notebook begins with the limitation of a point estimate, defines the posterior predictive quantity of interest, displays the integral that is often intractable, and asks how it can be approximated ([mdl-bayesian-computation.md:4](../../chapter_mdl-probability-statistics/mdl-bayesian-computation.md#L4), [mdl-bayesian-computation.md:23](../../chapter_mdl-probability-statistics/mdl-bayesian-computation.md#L23)). It then previews four approximation families and uses a shared logistic-regression reference problem ([mdl-bayesian-computation.md:27](../../chapter_mdl-probability-statistics/mdl-bayesian-computation.md#L27)). This balances standalone context with minimal repetition unusually well.

**Revision instruction.** No motivation change. Preserve the common reference problem; it makes algorithmic differences interpretable.

### Statistics: preserve the inference-oriented progression

The opening distinguishes describing a sample from making claims about a population and sets out estimation, uncertainty, testing, and dependence as linked tasks ([mdl-statistics.md:4](../../chapter_mdl-probability-statistics/mdl-statistics.md#L4)). That is sufficient standalone motivation, and the later sections introduce procedures in response to inferential questions rather than merely naming tests.

**Revision instruction.** No motivation change. Avoid turning the opening into a glossary of every estimator covered.

### Concentration and generalization: show why fixed-estimate control is not enough

The opening previews Markov, Chebyshev, exponential-tail bounds, high-dimensional concentration, and learning-theory guarantees ([mdl-concentration-generalization.md:4](../../chapter_mdl-probability-statistics/mdl-concentration-generalization.md#L4)). This describes coverage but does not yet reveal the central dependency. A concrete coin/sample-mean comparison later shows how a variance-only bound can be far too weak ([mdl-concentration-generalization.md:56](../../chapter_mdl-probability-statistics/mdl-concentration-generalization.md#L56)). Much later, the generalization section states the second key obstacle: a bound for a fixed predictor does not automatically apply to a predictor selected using the same sample ([mdl-concentration-generalization.md:542](../../chapter_mdl-probability-statistics/mdl-concentration-generalization.md#L542)).

**Revision instruction.** Rebuild the opening around this two-stage problem:

1. A sample average differs from its expectation; how unlikely is a large deviation, and what assumptions sharpen the answer?
2. In learning, the predictor is chosen after seeing the sample; how can a fixed-predictor bound be made uniform over the choices an algorithm might make?

Then map Markov/Chebyshev/exponential bounds to stage one and union bounds, complexity, stability, or related tools to stage two. This explains why concentration and generalization share a notebook rather than reading as two adjacent surveys.

**Preserve.** The high-dimensional section makes the geometric surprise explicit before deriving it ([mdl-concentration-generalization.md:351](../../chapter_mdl-probability-statistics/mdl-concentration-generalization.md#L351)), and the fixed-versus-data-selected distinction is the right conceptual hinge. Promote it; do not dilute it with a general statement that training and test data differ.

### Naive Bayes: motivate the assumption through sample complexity

The opening names Bayes' rule, conditional independence, smoothing, and calibration and then introduces the classifier ([mdl-naive-bayes.md:4](../../chapter_mdl-probability-statistics/mdl-naive-bayes.md#L4)). The concrete obstacle appears later: a direct table over hundreds of binary input features would have exponentially many patterns and could not be estimated from realistic data ([mdl-naive-bayes.md:52](../../chapter_mdl-probability-statistics/mdl-naive-bayes.md#L52), [mdl-naive-bayes.md:89](../../chapter_mdl-probability-statistics/mdl-naive-bayes.md#L89)).

**Revision instruction.** Open with the high-dimensional count. For example, 784 binary pixels imply $2^{784}$ possible input patterns, so a separate class-conditional probability for every pattern is impossible to estimate. Introduce conditional independence as the drastic factorization that turns that impossible table into featurewise counts, then immediately acknowledge why the method is called "naive." This gives the assumption a computational and statistical purpose before deriving the rule.

**Preserve.** Keep smoothing tied to unseen events and keep the distinction between useful classification and poorly calibrated probabilities; these prevent the notebook from overselling the independence model.

### Dependency-ordered implementation sequence

1. Revise the concentration/generalization opening first, because it clarifies the chapter's path from probability bounds to learning guarantees.
2. Revise the naive Bayes opening around the exponential table and factorization.
3. Move the coin-estimation problem ahead of the maximum-likelihood roadmap.
4. Promote the dartboard question in the random-variables opening.
5. Add the lookup-workflow sentence to the distributions notebook.
6. Preserve the index, Bayesian-computation opening, and statistics opening.
## Information Theory

### Intended progression

The chapter should begin with uncertainty in one variable, move to discrepancy between distributions, and then quantify dependence between variables. This gives a clean chain: entropy measures uncertainty, cross-entropy and divergence compare beliefs, and mutual information asks how much observing one variable changes uncertainty about another. The index communicates this chain well.

### Notebook assessment

| Notebook | Standalone assessment | Priority |
|---|---|---|
| `index.md` | Strong quantity-by-question table and clear dependency path. | Preserve |
| `mdl-information-theory.md` | Connects familiar losses to coding quickly, but the opening remains a taxonomy and one late applications heading is internally inconsistent. | Low |
| `mdl-divergences-distances.md` | Excellent example-led opening and organization by the decision or objective each discrepancy supports. | Preserve |
| `mdl-mutual-information.md` | Names modern methods and a variational bound before posing a concrete dependence or representation-learning problem. | Medium |

### Chapter index: preserve the quantity-by-question table

The index maps entropy, cross-entropy/KL, other discrepancies, and mutual information to distinct questions ([index.md:4](../../chapter_mdl-information-theory/index.md#L4)). It explicitly separates uncertainty in one variable from discrepancy between distributions and dependence between variables ([index.md:12](../../chapter_mdl-information-theory/index.md#L12)). This is strong standalone navigation.

**Revision instruction.** No substantive change. Retain the table even if notebook titles change; it carries more motivational value than a longer prose overview would.

### Entropy and information theory: pose a prediction or coding question first

The notebook opens by noting that cross-entropy is a common loss and that entropy, KL divergence, mutual information, and coding are related ([mdl-information-theory.md:4](../../chapter_mdl-information-theory/mdl-information-theory.md#L4)). That establishes relevance but remains a list of quantities. A simple card-report example later makes the operational question visible: how many binary answers are needed to identify an outcome ([mdl-information-theory.md:57](../../chapter_mdl-information-theory/mdl-information-theory.md#L57))?

**Revision instruction.** Begin with either the card-report question or an equivalent classification question: how much information is needed to communicate an uncertain label, and how much extra cost results from predicting with the wrong probabilities? Then introduce entropy and cross-entropy as the two answers. This naturally leads to KL divergence without listing the entire chapter in the first paragraph.

When connecting entropy to code length, qualify the relationship as ideal or optimal expected code length, including the usual integer-code caveat where appropriate. The current coding interpretation is otherwise strong ([mdl-information-theory.md:624](../../chapter_mdl-information-theory/mdl-information-theory.md#L624)).

The late "Modern Uses" material says it will discuss two techniques but ranges across minimum description length, label smoothing, and distillation ([mdl-information-theory.md:1043](../../chapter_mdl-information-theory/mdl-information-theory.md#L1043)). The individual subsections are motivated--MDL as a complexity/fit tradeoff ([mdl-information-theory.md:1051](../../chapter_mdl-information-theory/mdl-information-theory.md#L1051)), label smoothing as overconfidence control ([mdl-information-theory.md:1080](../../chapter_mdl-information-theory/mdl-information-theory.md#L1080)), and distillation as transfer from a soft target--but the heading-level promise makes the section feel appended.

**Revision instruction.** Rename and organize the applications by the question each addresses: choosing a compact explanation, preventing overconfident targets, and transferring predictive information. Remove the incorrect count rather than adding another method.

**Preserve.** Keep the coding view and the derivation of cross-entropy/KL from mismatched codes. It gives the quantities an operational meaning beyond their use as loss functions.

### Divergences and distances: preserve the problem-led taxonomy

The opening explains why one scalar discrepancy cannot serve every purpose and immediately distinguishes likelihood fitting, sample-based comparison, geometric transport, adversarial objectives, and local score comparison ([mdl-divergences-distances.md:4](../../chapter_mdl-information-theory/mdl-divergences-distances.md#L4)). Later sections continue this pattern: density-ratio estimation motivates dual forms ([mdl-divergences-distances.md:329](../../chapter_mdl-information-theory/mdl-divergences-distances.md#L329)), unavailable normalized densities motivate score-based discrepancies ([mdl-divergences-distances.md:1023](../../chapter_mdl-information-theory/mdl-divergences-distances.md#L1023)), and the final objective table maps mathematical properties to training settings ([mdl-divergences-distances.md:1294](../../chapter_mdl-information-theory/mdl-divergences-distances.md#L1294)).

**Revision instruction.** No motivation change. Preserve the fact that the notebook first asks what information is available--densities, samples, geometry, or scores--and only then recommends a discrepancy.

### Mutual information: lead with dependence, then expose the estimation barrier

The opening names SimCLR, CPC, CLIP, InfoNCE, the information bottleneck, and variational bounds ([mdl-mutual-information.md:4](../../chapter_mdl-information-theory/mdl-mutual-information.md#L4)). These names signal modern relevance, but a direct reader has not yet seen what representation problem mutual information solves or why the quantity is hard to estimate. The estimation barrier is stated much later: high-dimensional joint and marginal densities are unknown, and naive plug-in estimation is unreliable ([mdl-mutual-information.md:577](../../chapter_mdl-information-theory/mdl-mutual-information.md#L577)). The route from a variational classification problem to InfoNCE becomes clear only around [mdl-mutual-information.md:733](../../chapter_mdl-information-theory/mdl-mutual-information.md#L733).

**Revision instruction.** Open with a paired-data problem. Given two augmented views of an image, or an image and its caption, ask what a representation should retain about their shared content while discarding view-specific noise. Explain that mutual information quantifies dependence and becomes zero under independence. Then state the computational obstacle: in modern representation spaces neither the joint density nor the density ratio is known. Only after that should the notebook name variational bounds and contrastive objectives as estimable surrogates.

This revised opening should distinguish three questions that the notebook later answers:

1. What does mutual information mean and how is it related to entropy and KL divergence?
2. How can it be bounded or estimated from samples?
3. How do contrastive learning and the information bottleneck use such bounds without literally computing exact mutual information?

**Preserve.** The information-bottleneck section states its compression-versus-prediction question before giving the Lagrangian ([mdl-mutual-information.md:1213](../../chapter_mdl-information-theory/mdl-mutual-information.md#L1213)). Preserve this local motivation and the cautions around estimators; do not imply that optimizing InfoNCE always recovers exact mutual information.

### Dependency-ordered implementation sequence

1. Revise the mutual-information opening around paired views, dependence, and the unknown-density barrier.
2. Verify that the revised opening relies only on the chapter index's entropy/KL concepts and does not require the divergences notebook's derivations.
3. Revise the entropy opening around a code-length or predictive-cost question.
4. Reorganize and accurately label the entropy notebook's modern applications.
5. Leave the chapter index and divergences/distances notebook substantively unchanged.
## Dynamics

### Intended progression

The chapter begins with deterministic trajectories and their numerical approximation, adds stochastic forcing, moves from random sample paths to the evolution of entire probability densities, and then uses scores or velocity fields to build modern generative models. This is the clearest dependency chain in the Mathematics part. A standalone section should locate itself on that chain without retelling all preceding derivations.

### Notebook assessment

| Notebook | Standalone assessment | Priority |
|---|---|---|
| `index.md` | Exemplary error-based map of trajectory simulation, stochastic paths, density evolution, and learned transport. | Preserve |
| `mdl-odes-solvers.md` | Current source has a strong concrete instability example and a detailed, well-motivated progression; it is not brief. | Preserve |
| `mdl-sdes.md` | Opens with a concrete reason for stochastic dynamics and the essential square-root time scaling. | Preserve |
| `mdl-fokker-planck-probability-flow.md` | Clearly motivates replacing many random paths by deterministic density evolution and distinguishes forward and reverse constructions. | Preserve |
| `mdl-score-matching-diffusion-flow.md` | Assumes the predecessor narrative and inventories methods before stating the standalone generative problem and its intractable quantities. | High |

### Chapter index: protect the error decomposition and dependency map

The index distinguishes modeling error, discretization error, sampling error, and learning error before mapping them to notebooks ([index.md:4](../../chapter_mdl-dynamics/index.md#L4)). It also states the sequence from ODEs to SDEs, density PDEs, and learned reverse dynamics ([index.md:13](../../chapter_mdl-dynamics/index.md#L13)). This gives a direct reader both a purpose and a route.

**Revision instruction.** No motivation change. Preserve the three-error decomposition even if notebook scope changes; it gives the chapter a governing scientific question rather than a list of differential equations.

### ODE solvers: current source already fixes the cited weakness

The current notebook does not match the characterization of a brief, under-motivated section. It opens with the scalar equation $dx/dt = -50x$, observes that the exact solution decays, and shows that explicit Euler with a seemingly modest step oscillates and diverges ([mdl-odes-solvers.md:4](../../chapter_mdl-dynamics/mdl-odes-solvers.md#L4)). It then distinguishes an exact flow map from a numerical approximation and previews accuracy, stability, stiffness, adaptive steps, neural ODEs, adjoints, and continuous normalizing flows ([mdl-odes-solvers.md:12](../../chapter_mdl-dynamics/mdl-odes-solvers.md#L12)). That is a strong standalone opening.

The internal progression is likewise problem-led. Higher-order solvers are introduced as ways to control local truncation error ([mdl-odes-solvers.md:464](../../chapter_mdl-dynamics/mdl-odes-solvers.md#L464)); stiffness is defined through conflicting stability and accuracy step requirements ([mdl-odes-solvers.md:649](../../chapter_mdl-dynamics/mdl-odes-solvers.md#L649)); neural ODEs begin from continuous-depth modeling and the need to train through a solver ([mdl-odes-solvers.md:809](../../chapter_mdl-dynamics/mdl-odes-solvers.md#L809)); and continuous normalizing flows begin from the need to track density under an invertible flow ([mdl-odes-solvers.md:1188](../../chapter_mdl-dynamics/mdl-odes-solvers.md#L1188)).

**Revision instruction.** Do not lengthen the opening. First verify that the published page is built from the current source and that deployment is current. If a shorter version is still rendered, fix the source/build mismatch rather than adding a second motivational layer to this notebook.

**Preserve.** The opening instability calculation, exact-flow-versus-solver distinction, and transitions from classical numerical analysis to neural ODEs and CNFs should remain.

### Stochastic differential equations: preserve the noise-scaling problem

The notebook starts from systems whose unresolved fluctuations matter and asks how random forcing should scale as time steps shrink ([mdl-sdes.md:4](../../chapter_mdl-dynamics/mdl-sdes.md#L4)). It highlights the essential square-root time scaling rather than treating an SDE as an ODE with a noise term. The Brownian-motion section then makes the modeling need concrete: one deterministic noising path cannot describe a distribution of corrupted states ([mdl-sdes.md:63](../../chapter_mdl-dynamics/mdl-sdes.md#L63)).

**Revision instruction.** No motivation change. Keep the square-root scaling in the opening; it gives readers a reason to learn stochastic calculus conventions instead of assuming ordinary calculus applies unchanged.

### Fokker--Planck equations: preserve the path-versus-density contrast

The opening asks how the law of an SDE evolves when simulating many paths is expensive or obscures the distribution-level behavior ([mdl-fokker-planck-probability-flow.md:4](../../chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md#L4)). It clearly contrasts random paths with a deterministic PDE for density and previews both forward evolution and reverse-time construction ([mdl-fokker-planck-probability-flow.md:16](../../chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md#L16)). The vector-calculus identities are introduced because the derivation needs conservation of probability, not as a detached prerequisite list ([mdl-fokker-planck-probability-flow.md:100](../../chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md#L100)).

**Revision instruction.** No motivation change. Avoid adding a recap of all SDE definitions; the current opening supplies exactly the local bridge needed.

### Score matching, diffusion, and flow matching: rebuild the opening around the learnable field

This is the only high-priority standalone-motivation gap in the chapter. The opening starts from what prior notebooks established, then names score matching, denoising score matching, diffusion models, probability-flow ODEs, flow matching, stochastic interpolants, and optimal transport ([mdl-score-matching-diffusion-flow.md:4](../../chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md#L4)). A reader knows that the notebook surveys important methods but cannot yet state the generative task, which quantity is unavailable, or why these methods are learnable from samples.

The right problem already appears shortly afterward. An energy-based density has an intractable normalizing constant; taking a score removes that constant, but the data score is unknown because only samples are observed ([mdl-score-matching-diffusion-flow.md:78](../../chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md#L78)). Integration by parts and denoising then turn an inaccessible target into an estimable regression problem. This is the conceptual hinge and should govern the opening.

**Revision instruction.** Replace the current catalogue-first opening with this sequence:

1. State the generative task: learn to transform a simple reference distribution into the unknown data distribution using only data samples.
2. Identify the obstacle: the density, its normalizer, or its transport field is not directly known.
3. Introduce two field-based routes. Score-based models learn a density-gradient field that can define reverse stochastic or probability-flow dynamics; flow matching learns a velocity field that transports probability along a chosen path.
4. State the shared statistical trick: construct conditional targets from perturbed or paired samples, then use conditional expectation to recover the required marginal field.
5. Only then name diffusion models, denoising score matching, flow matching, stochastic interpolants, and optimal-transport variants as instances or design choices.

After this standalone hook, include a compact prerequisite locator: the reader needs the meaning of an SDE, density evolution, and an ODE flow, with links to the prior notebooks. Do not recap their derivations.

The transition into flow matching is already strong: it explains why one may prefer to learn a deterministic velocity field directly and what training target is available ([mdl-score-matching-diffusion-flow.md:875](../../chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md#L875)). Preserve that section and make the opening promise the same score-field/velocity-field comparison.

A late paragraph names consistency models, latent diffusion, and discrete diffusion without developing their distinct problem settings ([mdl-score-matching-diffusion-flow.md:1813](../../chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md#L1813)). In a notebook already carrying several complete derivations, this final catalogue weakens the endpoint.

**Revision instruction.** Move undeveloped variants to a clearly labeled further-reading note or remove their names from the main progression. End on the existing comparison of sampling routes and model families ([mdl-score-matching-diffusion-flow.md:1715](../../chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md#L1715), [mdl-score-matching-diffusion-flow.md:1820](../../chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md#L1820)), which gives readers a decision framework.

**Preserve.** Keep the normalizer-to-score-to-denoising derivation, the locally motivated flow-matching transition, and the final sampling/model-family comparisons. The needed repair is primarily reordering and framing, not more technical content.

### Dependency-ordered implementation sequence

1. Confirm whether the published ODE-solvers page is stale relative to the current source; do not edit the notebook merely to match an outdated diagnosis.
2. Rewrite the score/diffusion/flow opening around the generative task, inaccessible field, and conditional-regression solution.
3. Add only a compact prerequisites locator after that opening.
4. Check that the score-learning derivation now fulfills the opening's first route before introducing reverse dynamics.
5. Check that the existing flow-matching transition fulfills the opening's second route.
6. Demote undeveloped late variants to further reading and retain the final comparison as the conclusion.
7. Leave the dynamics index, SDE opening, Fokker--Planck opening, and ODE-solvers opening substantively unchanged.

## Cross-chapter implementation order

Some revisions should precede others because later notebooks depend on the distinction introduced earlier:

1. **Field-based generative modeling:** repair mdl-score-matching-diffusion-flow.md first. It is the only high-priority gap and depends on otherwise strong dynamics prerequisites.
2. **Matrix questions:** repair eigendecomposition before SVD so invariant square directions and rectangular singular directions remain distinct.
3. **Optimization failures:** repair the basic gradient notebook before the adaptive/stochastic notebook so the latter can assume direction and step-size concepts while motivating noise and scaling.
4. **Derivative computation:** repair the matrix-calculus/autodiff opening after preserving the multivariable gradient narrative; its unique problem is computational scaling.
5. **Probability guarantees:** repair concentration/generalization before naive Bayes, then make the smaller maximum-likelihood and random-variable reorders.
6. **Information estimation:** repair mutual information before polishing the entropy applications, because the former is a substantive entry-point issue and the latter is mainly local organization.
7. **Low-priority bridges:** finish with the part index, geometry, integral calculus, convexity, constrained examples, and distribution lookup sentence.

Across all passes, prefer moving an existing concrete problem earlier over writing a new preamble. A successful revision should usually add no more than one compact paragraph to an opening and should often reduce a method catalogue by the same amount.
