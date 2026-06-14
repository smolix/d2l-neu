# §22 Linear Algebra: slide-deck content plan (depth pass)

Planning doc for regenerating the reveal.js decks of the three §22 sections in
`chapter_mdl-linear-algebra/`. This is a **content-architecture plan**, not a
regeneration. Scope:

- `mdl-geometry-linear-algebraic-ops.md`
- `mdl-eigendecomposition.md`
- `mdl-svd-low-rank.md`

The bar is the §2 math reference decks (`chapter_preliminaries/linear-algebra.md`,
`calculus.md`) read against `docs/slides/north-star.html`, the §3 conceptual
rules and §8 acceptance checklist of `docs/slides-northstar-design.md`, and the
visual-vocabulary + overflow rules of `docs/slides.md`.

No em-dashes in any prose authored from this plan (Alex's standing rule); the
existing blocks use `---` em-dashes inside several captions and callouts, and
**removing them is itself a required edit** in the regeneration (see the
cross-deck notes).

---

## 0. The most important finding: these decks already exist and are NOT a blank slate

All three sections already carry a **full north-star `<!-- slides -->` block**
(committed in `277ce80 "Generate north-star slide decks for chapters 3-5 and
22-24"`, on top of the `6a3ee3b` quality pass). They already have the cover →
why/what → numbered dividers → recap bookends, `.kicker` eyebrows, `.cols .vc`
two-column figure pairings, `.d2l-note` / `.d2l-note .rule` / `.d2l-note .warn`
callouts, theorem statements staged with `. . .` fragments, and computed
outputs injected by cell id. Counts today: geometry **25 slides**, eigen **24
slides**, svd **27 slides**.

So Alex's verdict ("visually decent but technically shallow") is **only half
right for these three**: the visuals are north-star, and the *math* is, in
absolute terms, already deeper than most university decks. The chapter prose
itself (the `.md` bodies) is at a genuine 5/5/5 bar: full proofs of
Cauchy–Schwarz, the spectral theorem in three parts, Eckart–Young in two norms,
the min-norm least-squares characterization, condition-number tightness, etc.
**The decks under-exploit that prose.** The regeneration's job is therefore
*not* "add depth from nothing" but **"close the specific gaps where the slide
flattened a result the chapter proves, and lift the highest-value theorems from
a one-line callout to a staged proof-sketch."** That is a smaller, sharper task,
and this plan is written as an edit spec against the current blocks rather than a
ground-up outline.

### What is already good (keep, do not churn)
- Every lead figure exists and resolves. 18 `img/mdl-la-*.svg` figures are
  committed; the eigen/svd decks reference 8 of them as `@fig:` (resolving to
  `img/auto/<id>.svg`, all 8 present) and the rest as plain `![](../img/...)`.
  **No new figure is strictly required** for any of the three decks (one
  *optional* figure is flagged in §3).
- The computed outputs are real and load-bearing (verified against
  `outputs/pytorch/...`): determinant `tensor(5.)`; einsum tuple whose 2nd entry
  is `[0, -5]`; eig returns `λ=1,4`; PSD-gram `[1.1459 7.8541]` vs `[0. 30.]`;
  power-iteration `2.4532329634 == max|eigenvalue|`; defective-shear singular
  values `[1.618034 0.618034]` (golden ratio, product 1.0); numerical-rank
  `[4.35 0.91 0 0] -> rank 2`; PCA `[8.0873 0.5705]` with ratio `[0.9341
  0.0659]`; least-squares `cond(A)=7.469`, `cond(AᵀA)=55.782=cond²`;
  weight-spectrum `rank 18 / LoRA 10.5%`; svd-verify reconstruction error `0.0`.
  These are the right cells; keep them.
- The downstream cross-links are already present in prose (PCA, Hessian,
  conditioning, attention, LoRA/Muon/spectral-norm). The gap is that several
  live only in a final bullet wall instead of *on the slide that earns them*.

### The framework-scoping question, settled
There are **no PyTorch-only cells** in any of the three decks. Every code cell
is either tagged for all four frameworks (`#@tab mxnet/pytorch/tensorflow/jax`)
or untagged = all frameworks. So the prompt's instruction to scope pytorch-only
cells `only="pytorch"` **does not apply here**, there is nothing to scope.

The one genuine per-framework wrinkle is **MXNet in the eigen deck**: MXNet
arrays have no complex dtype, so the chapter routes all MXNet eigen computations
through plain NumPy. Concretely `eigendecomposition-an-example` prints a
real-valued `array([1., 4.])` under MXNet vs `tensor([1.+0.j, 4.+0.j])` under
torch/jax/tf, and `complex-rotation` is untagged (plain NumPy, identical across
decks). This does **not** require an `only=`/`except=` split: the slide framing
("the library returns the same λ = 1, 4, normalized to unit length, parallel to
ours") is true in every framework. Keep one shared slide. Just do not write a
caption that promises a *complex* dtype, since the MXNet deck shows reals. This
is the only cross-framework framing note for all three decks; everything else is
a pure code/output swap (exactly like the §2.3 linear-algebra reference deck,
which needed zero scoped slides).

---

## Deck 1: `mdl-geometry-linear-algebraic-ops.md`

### Depth verdict: **good** (the strongest of the three already), with 3 targeted gaps

The deck leads with geometry, derives the dot-product/angle identity, stages
Cauchy–Schwarz as a `.rule` callout with a fragment proof-sketch, gives
projection + orthogonality, the hyperplane-as-decision-boundary with the
zero-training Fashion-MNIST classifier (a genuinely great "code earns its place"
moment: 92% accuracy, nothing trained), the grid/linear-map picture, orthogonal
matrices, the determinant as signed area, the unifying equivalence (det = 0 ⇔
dependent ⇔ singular), and einsum. This is already a top-tier deck.

The gaps are not "shallow," they are "three places where a proved result is
asserted rather than staged," plus one structural risk.

**Gap 1 (the headline one): the dot-product = projection + cos θ derivation is
told, not shown.** The §22 depth anchor explicitly wants the dot product
*derived* as projection and angle. The current "Dot products and the angle"
slide states `v·w = ‖v‖‖w‖cosθ` and says "the law of cosines proves it," then
jumps to code. The chapter has the actual two-line argument (expand
`‖v − w‖²` two ways, equate, cancel). Lift it onto the slide as a 2-fragment
proof-sketch beside `img/mdl-la-angle.svg`. This is the single most valuable
deepening in the deck.

**Gap 2: the high-dimensional near-orthogonality result (a genuine highlight of
the chapter) is under-weighted.** The "Random vectors are nearly orthogonal"
slide states `E[cosθ]=0, Var=1/d` as a `.rule` but omits the one-line *why*
(rotate so `u=e₁`, then `cosθ=v₁`; symmetry kills the mean; `Σvᵢ²=1` gives
`Var=1/d`). It is the deck's best "concentration of measure" payload and a
direct motivation for why cosine similarity / attention work. Add the
proof-sketch as a fragment; the `mdl-la-cosine-highd.svg` histogram figure (d=2
arcsine vs d=1000 spike) is already there to anchor it.

**Gap 3: the determinant-as-signed-area slide shows the number but not the
"why 5."** It prints `det = 5` and shows `mdl-la-determinant.svg` (the three
regimes). Fine, but the chapter's beautiful payoff, multiplicativity
`det(AB)=det A · det B` *because area scales compose*, and hence
`det Q = ±1` for orthogonal Q, is dropped entirely from the deck. At least the
multiplicativity one-liner deserves a fragment on the determinant slide or the
unifying-equivalence slide, because it is what ties orthogonal matrices
(rigid, `det=±1`) back to the decompositions to come. Low cost, high
through-line value.

**Structural risk (flag, do not over-fix): this deck is long and covers a lot.**
25 slides spanning vectors → similarity → hyperplanes → maps → einsum is at the
upper edge. The chapter is genuinely 5 sections, so a long deck is defensible,
but two slides are candidates to *merge or cut* to make room for the three
deepenings without ballooning: the standalone "Span, basis, dimension" bullet
slide (it is a definitions slide with a figure but no computed payoff) could
fold its `column space / null space` callout forward into the linear-maps
section where it pays off; and "Cosine similarity" + "nearly orthogonal" could
be tightened. Net target: keep ~24–25 slides.

### Core spine (the through-line to make explicit)
1. **A vector is a picture** (point or arrow); the **dot product turns algebra
   into geometry** (length, angle, projection).
2. **Cauchy–Schwarz is the one inequality** that makes the angle well-defined,
   and in high dimensions it forces unrelated vectors to be nearly orthogonal
   (why cosine similarity / attention work).
3. **A hyperplane is the decision boundary of every linear classifier** (one
   normal, one bias, a signed-distance margin), provable with one unit of
   geometry, demonstrable with a 92%-accurate untrained classifier.
4. **A matrix is a rigid-or-not motion of the whole grid**; the determinant is
   its volume scale, zero exactly when it collapses space, and these facts set
   up the two decompositions that follow.

### Must-show theorem + proof-sketch (the one to stage with fragments)
**Geometric dot product (the deck's anchor).**
Statement: `v·w = ‖v‖‖w‖cosθ`, valid in every dimension because v and w span a
plane and θ lives in it.
Stage in two fragments:
- Fragment 1: expand `‖v − w‖² = ‖v‖² − 2 v·w + ‖w‖²` (pure algebra).
- Fragment 2: the planar law of cosines gives `‖v − w‖² = ‖v‖² + ‖w‖² −
  2‖v‖‖w‖cosθ`; equate and cancel `‖v‖² + ‖w‖²` ⇒ the identity; solve for
  `θ = arccos(v·w / ‖v‖‖w‖)`.
Then the existing `.rule` Cauchy–Schwarz callout reads as the *guarantee* that
the arccos argument never leaves `[−1, 1]` (keep its current 1-fragment
"squared length ≥ 0 ⇒ parabola ⇒ discriminant ≤ 0" sketch, it is already good).

### Lead diagram(s): all exist, reuse as-is
- `img/mdl-la-vectors.svg` (point vs direction): "Two readings of a vector"
- `img/mdl-la-angle.svg` (the angle θ): pair with the derivation above
- `img/mdl-la-projection.svg` (projection + right-angle residual): projection slide
- `img/mdl-la-cosine-highd.svg` (d=2/10/1000 histograms): near-orthogonality
- `img/mdl-la-hyperplane.svg` (normal, half-space, signed distance): cover + hyperplane slide
- `img/mdl-la-span.svg`: span/basis slide (candidate to demote, see Gap)
- `img/mdl-la-linear-map.svg` (grid carried by the columns): linear-map slide
- `img/mdl-la-determinant.svg` (3 regimes: +, −, 0): determinant slide
- `img/mdl-la-vector-add.svg` (tip-to-tail): available; currently unused; could
  enrich the "two readings" slide as a fragment if room allows. **Optional**, not required.

No new figure needed for this deck.

### Code that earns its place (cell id → what it computes; all 4-framework, no scoping)
- `geometry-linear-algebraic-ops-dot-products-and-angles` → `angle(...)` returns
  `tensor(0.4190)` rad: the angle formula, executed. Keep on the dot-product slide.
- `geometry-linear-algebraic-ops-hyperplanes-2` → `@!` (output only) the two
  class-mean images. Keep (figure-as-output is legitimate here: it is a computed
  result, the blurry mean t-shirt/trousers).
- `geometry-linear-algebraic-ops-hyperplanes-4` → `tensor(0.9155)`: the
  zero-training classifier's 92% accuracy. This is the deck's best code moment;
  keep, and keep the "nothing was trained" fragment.
- `geometry-linear-algebraic-ops-projection-histogram` → `@!` the two-hump
  histogram with threshold. Keep ("the whole story in one projection").
- `geometry-linear-algebraic-ops-determinant` → `tensor(5.)`: confirms det = 5.
  Keep; add the multiplicativity fragment (Gap 3).
- `geometry-linear-algebraic-ops-invertibility` → identity matrix: confirms the
  2×2 inverse formula. Keep on the unifying-equivalence slide with the existing
  `linalg.solve` over `inv(A)@b` `.warn` callout.
- `geometry-linear-algebraic-ops-expressing-in-code-2` → tuple whose 2nd entry
  is `tensor([0., -5.])`, tying einsum `Av` back to the worked grid example.
  Keep on the einsum slide.
- Note `geometry-linear-algebraic-ops-geometry-of-vectors` (`v = [1,7,0,1]`) is
  trivial; it is fine as a tiny illustrative cell on "two readings" but carries
  no output. Keep or drop; not load-bearing.

### Cross-links / forward-points to surface ON the relevant slide
- Dot product / cosine slide → **attention scoring** and **embedding retrieval**
  (already named in the cosine slide; good).
- Projection slide → "this is a 1-D least-squares; the SVD scales it up"
  (forward to §22.3). Add as a one-liner; currently only in prose.
- Orthogonal-matrices slide → "the building blocks of *both* decompositions:
  `QΛQᵀ` (spectral) and `UΣVᵀ` (SVD)." Already present; keep.
- Determinant slide → `det A = ∏ λᵢ` forward-point to §22.2 (eigen). Add one line.

### Slide arc (target ~24–25; cover → recap). Bold = change vs current block.
1. Cover: geometry under the algebra (keep).
2. Why a geometric view (keep; 4-bullet + hyperplane figure + forward `.d2l-note`).
3. Divider 01: Vectors and their geometry (keep).
4. Two readings of a vector (keep; optional `vector-add.svg` fragment).
5. **Dot products and the angle: now with the 2-fragment derivation** (Gap 1).
6. Why the angle is always defined: Cauchy–Schwarz `.rule` + discriminant sketch (keep).
7. Projection and orthogonality (keep; **add the "1-D least squares → SVD" forward line**).
8. Span, basis, dimension (keep, or **fold the column/null-space callout forward**).
9. Divider 02: Similarity in high dimensions (keep).
10. Cosine similarity (keep; tighten).
11. **Random vectors are nearly orthogonal: add the 3-step proof-sketch fragment** (Gap 2).
12. Divider 03: Hyperplanes and decision boundaries (keep).
13. A hyperplane as a decision boundary (keep).
14. A classifier with nothing trained: class means (keep).
15. One hyperplane, ~92% correct (keep).
16. The whole story in one projection (keep).
17. Divider 04: Matrices as linear maps (keep).
18. A matrix moves the whole grid (keep).
19. Orthogonal matrices: the rigid motions (keep; keep the `det Q = ±1` line if added).
20. The determinant is signed area (keep; **add multiplicativity + `det A=∏λᵢ` fragment**, Gap 3).
21. One equivalence ties it together (keep; `inv` vs `solve` `.warn`).
22. Divider 05: Einstein summation (keep).
23. Sum over the repeated index (keep).
24. Recap (keep; rewrite to carry the 4-point spine, **strip em-dashes**).

---

## Deck 2: `mdl-eigendecomposition.md`

### Depth verdict: **good**, with 2 targeted gaps + 1 must-stage theorem

This is a deep deck: eigenpair definition with the diag(2,−1) geometry, the
circle→ellipse picture, the characteristic equation worked by hand, `eig`
verification, the decomposition with telescoping powers, det/trace from the
spectrum, the multiplicity/diagonalizability counting, the spectral theorem (as
a `.rule`), PSD via eigenvalue signs with the bowl/trough/saddle figure, the
Rayleigh quotient with the condition-number tie-in, Gershgorin discs, power
iteration with the convergence figure and the 10-decimal numerical match,
complex-eigenvalues-as-rotations, and the spectral-radius → RNN/init payoff.

The gap is **the single most important theorem in the section, the spectral
theorem, is currently a bare callout** with no proof-sketch, exactly the
"key theorem stated without intuition/proof-sketch" failure the rubric calls
out. Everything else is staged well; this one is not.

**Gap 1 (headline): stage the spectral theorem.** The chapter proves it in
three short, instructive parts (real eigenvalues; orthogonality of eigenvectors
for distinct eigenvalues; full basis by induction on the orthogonal
complement). The slide just states `A = WΛWᵀ` and reads off the geometry. Lift
**at least parts (i) and (ii)** as fragments, they are two-liners and they are
the soul of the section:
- (i) real eigenvalues: `λ‖v‖² = v*Av = (Av)*v = λ̄‖v‖²` ⇒ `λ = λ̄`.
- (ii) orthogonality: `λ⟨u,v⟩ = ⟨Au,v⟩ = ⟨u,Av⟩ = μ⟨u,v⟩` ⇒ `(λ−μ)⟨u,v⟩ = 0`.
Part (iii) (induction on `w₁^⊥`, which is A-invariant) can stay as a one-line
"and a clean induction on the orthogonal complement fills out the basis." This
turns the deck's centerpiece from an assertion into a proof.

**Gap 2: the "rotate → scale → rotate-back" action of a symmetric matrix is
stated but not given its own beat.** The §22 anchor explicitly wants the action
shown as rotate → scale → rotate-back. The spectral-theorem slide mentions it in
one sentence ("Wᵀ rotates the eigenvectors onto the axes, Λ stretches,
W rotates back"). This is the *same* mechanism the SVD generalizes, so it is
worth making visually explicit. The `mdl-la-eig-ellipse.svg` figure shows the
result but not the three-stage motion. **Option A (cheap, preferred):** add the
three stages as a fragment list under the spectral slide, pointing at the
ellipse figure. **Option B (only if Alex wants a new figure):** a 3-panel
`circle → axis-aligned ellipse → rotated ellipse` schematic, but note the SVD
deck's `mdl-la-svd-action.svg` already does exactly this for the general case,
so a symmetric-case twin is arguably redundant. Recommend Option A; flag B as
optional.

**Already-good depth to preserve:** the power-iteration expansion
`Aᵏv₀ = λ₁ᵏ(c₁w₁ + Σ cᵢ(λᵢ/λ₁)ᵏ wᵢ)` is on the slide with the decay argument , 
keep it, it is exactly right. The Rayleigh "weighted average of eigenvalues"
intuition is on the slide, keep. The complex-eigenvalue-as-rotation cell and
the `ρ ~ √n` circular-law init story are present and correct, keep.

### Core spine
1. **Eigenvectors are the invariant directions** a matrix only stretches; the
   circle becomes an ellipse with axes along them (for symmetric A).
2. **The eigendecomposition `A = WΛW⁻¹` turns matrix questions into scalar
   ones**, powers, det, trace, rank, *when* an eigenbasis exists (counting:
   geometric = algebraic multiplicity; defective shear shows the failure).
3. **Symmetric ⇒ the spectral theorem guarantees an orthonormal eigenbasis**
   (`A = WΛWᵀ`), which is the workhorse behind PSD, Rayleigh/conditioning, PCA,
   the Hessian test, and (applied to `AᵀA`) the SVD itself.
4. **The spectral radius `ρ = max|λᵢ|` governs iterated maps**: power
   iteration finds it, and keeping it near 1 is the principle behind weight init
   and exploding/vanishing gradients.

### Must-show theorem + proof-sketch
**Spectral Theorem** (the deck's required centerpiece, see Gap 1 for the staged
fragments). State: every real symmetric A has a full orthonormal eigenbasis with
real eigenvalues, `A = WΛWᵀ`, `WᵀW = I`. Stage parts (i) and (ii) as fragments;
summarize (iii). Then the existing "applied to `AᵀA` it builds the SVD" `.rule`
callout becomes the bridge to deck 3 (keep it).

Secondary theorem worth keeping staged (already is): **distinct eigenvalues ⇒
diagonalizable** is implicitly covered by the multiplicity slide; the explicit
"shortest dependence relation" proof from the chapter is too long for a slide , 
the current "n distinct eigenvalues guarantee it" statement is the right
altitude. Do not expand it.

### Lead diagram(s): all exist (`@fig:` → `img/auto/`), reuse
- `@fig:mdl-la-eig-ellipse`: cover + "circle becomes an ellipse" + spectral slide.
- `@fig:mdl-la-psd`: bowl / trough / saddle for the PSD sign story.
- `@fig:mdl-la-gershgorin`: the four discs with eigenvalue crosses.
- `@fig:mdl-la-power-iter`: iterates swinging onto w₁; norm ratio → λ₁.
No new figure required (optional symmetric-action 3-panel is flagged in Gap 2,
recommended *against*).

### Code that earns its place (all 4-framework; MXNet eig → real via NumPy, see §0)
- `eigendecomposition-an-example` → `eig` returns `λ = 1, 4` with unit
  eigenvectors. Keep; caption must not promise a complex dtype (MXNet shows
  reals). Keep the "QR on a Hessenberg reduction, never the char. poly" `.d2l-note`.
- `eigendecomposition-psd-gram` (untagged, all frameworks) → `[1.1459 7.8541]`
  (PD) vs `[0. 30.]` (PSD). Keep; this is the Hessian "is it a minimum?" test
  made concrete.
- `eigendecomposition-gershgorin-circle-theorem` → eigenvalues ≈ 0.99, 2.97,
  4.95, 9.08 inside the four ranges. Keep with the disc figure.
- `eigendecomposition-power-iteration` → `stabilized norm ratio = 2.4532329634
  == max|eigenvalue|`. Keep; the 10-decimal match is the payoff.
- `eigendecomposition-complex-rotation` (untagged) → eigenvalues `0.866 ± 0.5j`,
  moduli `1`. Keep on the complex-rotation slide.

### Cross-links / forward-points (surface on-slide)
- Spectral-theorem slide → PCA, covariance, Hessian (named); **+ "applied to AᵀA
  ⇒ SVD"** bridge to §22.3 (keep the `.rule`).
- Rayleigh slide → `κ = λmax/λmin` is the **condition number** that slows
  gradient descent (already present; keep) and reappears as σ₁ in the SVD deck.
- Spectral-radius slide → RNN backprop-through-time `ρᵀ`, orthogonal init,
  gradient clipping, LSTM/GRU gating (already present; keep). Also the PageRank /
  Perron–Frobenius aside is in the chapter but **not on a slide**, optional add
  as a single `.d2l-note` ("power iteration also *computes* PageRank: the
  dominant eigenvector of a stochastic matrix"), since it is a memorable
  real-world anchor for "the dominant eigenvector is often the whole point."

### Slide arc (target ~24; cover → recap). Bold = change.
1. Cover (keep).
2. Why eigenvalues? (keep; ellipse figure + PCA/Hessian/PageRank `.d2l-note`).
3. Divider 01: Eigenvalues & eigenvectors (keep).
4. The defining equation `Av = λv` + diag(2,−1) geometry (keep; λ=0 `.rule`).
5. The circle becomes an ellipse (keep).
6. Finding eigenvalues: char. equation worked by hand (keep).
7. Check it with `eig` (keep; **fix caption re MXNet reals**).
8. The eigendecomposition + telescoping powers (keep).
9. Determinant & trace from the spectrum (keep).
10. When does an eigenbasis exist?: multiplicity + defective shear (keep).
11. Divider 02: Symmetric matrices (keep).
12. **The spectral theorem: now with parts (i) & (ii) staged as fragments** (Gap 1).
13. **(optional) The symmetric action: rotate → scale → rotate-back** as a fragment
    list on slide 12, or its own slide (Gap 2, Option A).
14. The sign of λ is the shape: PSD bowl/trough/saddle (keep).
15. Positive definiteness, in code: Gram PD vs PSD (keep).
16. Eigenvalues as extreme stretches: Rayleigh + condition number (keep).
17. Divider 03: Locating & iterating (keep).
18. Gershgorin discs (keep).
19. Gershgorin: tight enough to use (keep; diagonal-dominance `.rule`).
20. Power iteration: the eigenbasis expansion (keep).
21. Power iteration, watched: figure + 10-decimal match (keep).
22. Complex eigenvalues are rotations (keep). **(optional PageRank `.d2l-note`.)**
23. Spectral radius & deep networks (keep; the ρ payoff).
24. Recap (keep; carry the 4-point spine; **strip em-dashes**).

---

## Deck 3: `mdl-svd-low-rank.md`

### Depth verdict: **good but the longest and most application-front-loaded**; 2 gaps + 1 must-stage theorem

This deck is rich: rotate–scale–rotate with the `svd-action` figure, existence
via `AᵀA` with the "orthonormal output frame for free" line, the defective shear
finally decomposed (golden ratio), svd-verify, the four fundamental subspaces,
numerical rank, Eckart–Young as a `.rule`, the image-compression visual proof,
PCA = Eckart–Young on centered data, pseudoinverse / min-norm least squares,
the condition number with the `κ(AᵀA)=κ²` warning and the gradient-descent bowl,
and a full "SVD in modern deep learning" section (LoRA, Muon, spectral norm,
effective-rank diagnostic).

The two gaps mirror the eigen deck: **the section's signature theorem
(Eckart–Young) is stated but its beautiful proof is not staged**, and the
**existence construction is told rather than shown** even though its crux is a
single elegant line.

**Gap 1 (headline): stage the Eckart–Young lower bound.** The chapter's
dimension-counting proof is one of the most elegant arguments in the whole
appendix and it is *slide-sized*. The current slide states
`‖A − A_k‖₂ = σ_{k+1}` as a `.rule` and stops. Stage the lower-bound idea in
2–3 fragments:
- Setup: any rank-k B has `dim ker B ≥ n − k`; the top-(k+1) right-singular
  span `V = span{v₁..v_{k+1}}` has dim k+1.
- Collision: `(n−k) + (k+1) > n` ⇒ they intersect in a unit vector x with
  `Bx = 0`.
- Punchline: on that x, `‖(A−B)x‖ = ‖Ax‖ = √(Σ cᵢ²σᵢ²) ≥ σ_{k+1}`, so
  `‖A−B‖₂ ≥ σ_{k+1}`, with equality at the truncation. "B is blind on a
  direction where A still stretches by σ_{k+1}."
This converts the deck's most important result from an assertion to a proof and
is the single highest-value change in the deck.

**Gap 2: existence-via-`AᵀA` is the deck's structural backbone and its one
elegant step deserves a fragment, not a buried `.rule`.** The slide already has
the orthonormality-for-free identity in a `.rule` callout, good, but the
*construction* (diagonalize AᵀA, `σᵢ = √λᵢ`, `uᵢ = Avᵢ/σᵢ`) and the
"this is the spectral theorem in disguise; Gram matrices are never defective ⇒
the SVD never fails" punchline are split awkwardly. Tighten into one staged
slide: (1) AᵀA is symmetric PSD ⇒ spectral theorem applies; (2) set
`σᵢ=√λᵢ`, `uᵢ=Avᵢ/σᵢ`; (3) the uᵢ are orthonormal for free (the existing
identity). Keep "never fails." Mostly a re-stage of existing content, low cost.

**Structural note (the real risk for this deck): it is 27 slides and the modern-DL
section is heavy.** LoRA / Muon / spectral-norm are great motivation, but they
are three dense application slides plus an effective-rank cell, and they can
crowd out the *mathematics* the deck is supposed to teach. Recommendation:
**keep the LoRA + spectral-norm beats (they pay off Eckart–Young and σ₁=‖A‖₂
directly), compress Muon to a single fragment or a `.d2l-note`** (the polar
decomposition + Newton–Schulz story is lovely but is the least central to a
linear-algebra deck and the most likely to overflow 720px). This buys the room
for the two proof-sketches above without growing the deck.

**Already-good depth to preserve:** the polar decomposition `A = QP` as the
"matrix `z = re^{iθ}`" analogy (keep, it justifies the rotate–scale–rotate
slogan); the `σ ≠ |λ|` warning via the scaled rotation (keep, it is the right
gotcha); the four-fundamental-subspaces bijection with `svd-subspaces.svg`
(keep); `κ(AᵀA)=κ(A)²` warning (keep); image-compression visual proof (keep , 
this is the deck's "code/figure earns its place" centerpiece).

### Core spine
1. **The SVD is rotate–scale–rotate for *every* matrix**: `A = UΣVᵀ`,
   `Avᵢ = σᵢuᵢ`, the eigen-picture generalized to two different orthonormal
   frames, which is exactly what lets it handle rectangular and defective
   matrices the eigendecomposition could not.
2. **It exists because it is the spectral theorem applied to `AᵀA`** (symmetric
   PSD, never defective), with `σᵢ = √λᵢ(AᵀA)` and the output frame orthonormal
   for free; equivalently `σ₁ = max‖Ax‖ = ‖A‖₂`.
3. **Eckart–Young: the top-k truncation is the *provably best* low-rank
   approximation** (`σ_{k+1}` in spectral, `Σ_{i>k}σᵢ²` in Frobenius). This is
   the theorem behind image compression, PCA, denoising, and the LoRA ceiling.
4. **Two scalars run the rest:** `σ₁ = ‖A‖₂` (Lipschitz / spectral norm) and
   `κ = σ₁/σ_r` (error amplification *and* gradient-descent zig-zag); `κ(AᵀA)=κ²`
   is why the normal equations are worse.

### Must-show theorem + proof-sketch
**Eckart–Young–Mirsky** (the deck's required centerpiece, staged per Gap 1).
State the spectral-norm case `min_{rank B ≤ k} ‖A − B‖₂ = σ_{k+1}`, attained at
`A_k = Σ_{i≤k} σᵢuᵢvᵢᵀ`. Stage the dimension-counting lower bound in fragments;
the "truncation achieves σ_{k+1}" direction is one line (the error is itself a
sub-SVD with top singular value σ_{k+1}). Mention the Frobenius case
`Σ_{i>k}σᵢ²` as the "PCA relies on this" variant without proving it.

Supporting result to keep staged (already partly is): **σ₁ = max_{‖x‖=1}‖Ax‖ =
‖A‖₂** (the variational characterization). This is the Rayleigh quotient of the
eigen deck in disguise, it is the lemma the Eckart–Young proof reuses, and it is
what makes spectral normalization meaningful. Make sure it appears as its own
crisp statement (currently it is implicit inside the existence slide); a single
fragment "σ₁ is the most A can stretch any unit vector = ‖A‖₂" is enough.

### Lead diagram(s): all exist, reuse
- `@fig:mdl-la-svd-action` (and `../img/` plain ref): cover + rotate-scale-rotate.
- `../img/mdl-la-svd-subspaces.svg`: four fundamental subspaces.
- `../img/mdl-la-eckart-young.svg`: spectrum + rank-1/5/20/full image reconstructions.
- `../img/mdl-la-pca.svg`: correlated cloud + principal axes.
- `../img/mdl-la-condition.svg`: well- vs ill-conditioned bowl, straight vs zig-zag.
No new figure required.

### Code that earns its place (mix of untagged + 4-framework; no pytorch-only)
- `svd-defective-shear` (untagged) → `[1.618034 0.618034]`, product `1.0`. Keep;
  this is the "SVD repairs what eigendecomposition could not" payoff.
- `svd-verify` (4-fw) → reconstruction error `0.0`, `σ² = [18.32 5.68] =
  eig(AᵀA)`. Keep; verifies the two facts the section rests on.
- `svd-numerical-rank` (untagged) → `[4.35 0.91 0 0] -> rank 2`. Keep with the
  threshold `.d2l-note`.
- `svd-pca` (untagged) → explained variance `[8.0873 0.5705]`, ratio `[0.9341
  0.0659]`, and SVD axes == covariance eigenvectors. Keep on the PCA slide.
- `svd-least-squares` (4-fw) → `pinv == lstsq`, `cond(A)=7.469`,
  `cond(AᵀA)=55.782=cond²`. Keep; the `κ²` number is the on-slide proof of the
  warning.
- `svd-weight-spectrum` (untagged) → `rank 18 / LoRA 10.5%`. Keep on the
  effective-rank slide (this is the honest counter to the "0.39%" headline).

### Cross-links / forward-points (surface on-slide)
- Existence slide → back-link "this is the spectral theorem of §22.2 applied to
  AᵀA" (keep; it is the spine).
- σ₁ = ‖A‖₂ → **spectral normalization / Lipschitz** (lands later in the deck;
  add a one-line forward pointer where σ₁ is first defined).
- κ slide → "same number, two consequences" (error amplification + GD
  contraction `(κ−1)/(κ+1)`) and forward to the optimization chapter (keep).
- Eckart–Young → image compression, PCA, denoising (Gavish–Donoho), matrix
  completion / recommenders, and the **LoRA ceiling** (keep; the LoRA slide
  should explicitly say "Eckart–Young bounds how well *any* rank-r update can
  track the full one", it already does, keep).

### Slide arc (target ~24, down from 27; cover → recap). Bold = change.
1. Cover: the factorization that never fails (keep).
2. The picture, made universal (keep; svd-action figure + 1-factorization `.d2l-note`).
3. Divider 01: The factorization (keep).
4. Rotate–scale–rotate + `Avᵢ = σᵢuᵢ` (keep; **add σ₁ = ‖A‖₂ fragment**, supporting thm).
5. **Where the singular values come from: existence via AᵀA, re-staged in 3 fragments** (Gap 2).
6. The defective shear, finally decomposed: golden ratio (keep).
7. Two frames, one stretch: svd-verify (keep).
8. Rank & the four fundamental subspaces (keep).
9. Numerical rank in practice (keep).
10. Divider 02: Low-rank approximation (keep).
11. **Eckart–Young: optimal low rank: now with the dimension-counting proof-sketch** (Gap 1).
12. A visual proof on an image (keep; the eckart-young figure).
13. PCA = Eckart–Young on centered data (keep).
14. Divider 03: Solving & conditioning (keep).
15. Pseudoinverse & least squares (keep).
16. The condition number: κ figure + `κ(AᵀA)=κ²` warning (keep).
17. Divider 04: In modern deep learning (keep).
18. The SVD in modern DL: **LoRA + spectral-norm kept; Muon compressed to a
    fragment / `.d2l-note`** (structural note).
19. Effective rank, measured: weight-spectrum cell (keep).
20. Recap (keep; carry the 4-point spine; **strip em-dashes**).

(Net: removing ~3 slides of application sprawl and Muon detail makes room for the
two proof-sketches; the deck lands ~20–24 instead of 27.)

---

## Cross-deck observations (the top 3 + the must-do hygiene item)

**1. The decks are already north-star; the work is "stage the signature theorem
in each, surface the proof the chapter already wrote."** The pattern is
identical across all three: each section has exactly one crown-jewel theorem
that is currently a bare `.rule` callout , 
- geometry: the **geometric dot-product / angle** derivation (and the
  near-orthogonality `Var = 1/d`),
- eigen: the **spectral theorem** (parts i & ii),
- svd: **Eckart–Young** (dimension-counting lower bound) + existence via AᵀA.
All four proof-sketches are two-to-three fragments and fit a slide. This is the
highest-leverage, lowest-risk change and should be the spine of the
regeneration brief. It is precisely the rubric's "the ONE theorem that matters,
with an elegant proof-sketch staged with fragments, not a wall."

**2. There is a single mathematical spine threading all three decks, and the
decks under-advertise it.** It is: *circle → ellipse* (geometry, eigen) becomes
*two frames* (SVD); *orthogonal matrices are rigid, det ±1* (geometry) are the
building blocks of *both* `QΛQᵀ` and `UΣVᵀ`; the *Rayleigh quotient* (eigen)
*is* `σ₁ = ‖A‖₂` (SVD); the *condition number* `κ = λmax/λmin` (eigen Rayleigh)
*is* `σ₁/σ_r` (SVD) and is the same number in both "slow gradient descent"
pictures; `AᵀA` symmetric PSD (eigen) is what *manufactures* the SVD. Each deck
should keep one explicit forward/back pointer on the relevant slide (most are in
prose already; promote them onto the slide). The reader who sees all three
decks should feel one continuous argument, not three topics.

**3. Length and application-sprawl are the only real *quality* risks, and they
are worst in the SVD deck.** All three are at or above the §2 reference deck
length, and the SVD deck's modern-DL section (LoRA/Muon/spectral-norm/
effective-rank) is the biggest overflow-and-dilution risk. Trim the SVD deck
(compress Muon, drop ~3 slides) to fund the proof-sketches; lightly tighten the
geometry deck's similarity section; the eigen deck is the best-balanced and
needs the least structural change. Run the §8 720px overflow sweep on the
theorem slides specifically, since a 3-fragment proof beside a figure is exactly
where height creeps.

**Must-do hygiene (applies to all three, non-negotiable per Alex's rule):
strip every `---` em-dash.** The current blocks use em-dashes in many captions
and callouts (e.g. geometry "an arrow that may start anywhere", eigen
"insurmountable seeming problem", svd "powerful but *picky*", and almost every
recap). The regenerated blocks must use commas / parentheses / colons or reword.
This is a global find-and-fix across the three `<!-- slides -->` blocks, not a
per-slide judgment call.

### Practical notes for the regeneration (so the build does not surprise the author)
- **No new figures are required.** All 18 `img/mdl-la-*.svg` exist; the 8 used as
  `@fig:` resolve from `img/auto/`. The only *optional* new figures flagged
  (geometry `vector-add` enrichment; eigen symmetric 3-panel action) are
  recommended **against** in favor of fragment text reusing the existing
  ellipse/svd-action figures.
- **No `only=`/`except=` scoping is needed.** No pytorch-only cells exist; the
  only per-framework wrinkle is MXNet's real (not complex) eig output, handled by
  caption wording, not a slide split.
- **All shown outputs are already captured** in `outputs/<fw>/...`; the cited
  numbers in this plan are the real committed values. A regeneration that keeps
  the same cell ids needs no notebook re-run.
- Keep the `.d2l-note .rule` (theorem/blue-purple) and `.d2l-note .warn` (amber)
  conventions already in use; put each staged proof under the relevant `.rule`,
  not in a new class.
