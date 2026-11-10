# Writing pass log: Ordinary Differential Equations and Numerical Solvers

File: `chapter_mdl-dynamics/mdl-odes-solvers.md`. Reviews followed:
`43-dynamics/mdl-odes-solvers.md` (ODE-S1 to S5, ODE-1 to ODE-20) and
`43-dynamics.md`.

## Words and checks

Checker count (prose with exercises and tables): 7,992 before, 7,990 after
(-2). The cuts (about 330 words) paid for the new subsection on the driven
linear equation (about 250 words), the two-line derivation of ODE-8, and the
clause of ODE-12. Checker: "OK: only prose changed", one split marker.
`lint_source.py` exits 0 with no messages. The checker's "em dash" warning is
present before and after: it is triggered by the separator rows of the three
Markdown tables (`|---|---|`); there is no em dash in the prose (grep for
`—` and for ` --- ` finds none).

Numeric checks were done with `python3` and NumPy in scratch files outside
the repository (no `.venv-pytorch` on this machine); the rest by hand.

## Correctness changes

- **ODE-5.** Was: linear fields "describe every other field near a point
  where it vanishes". Now: "most other fields near a point where the field
  vanishes". Agreed on reading: the section's own $\dot x=\pm x^3$ example.
- **ODE-11.** Was: $\mathbf v_2=(-0.995,0.0995)$, "only $5.7^\circ$ apart",
  $\mathbf x_0=10.0\,\mathbf v_1+10.05\,\mathbf v_2$ called a difference. Now:
  $\mathbf v_2=(0.995,-0.0995)$, the eigenvectors "span lines only
  $5.7^\circ$ apart", $\mathbf x_0=10.0\,\mathbf v_1-10.05\,\mathbf v_2$.
  Recomputed: $10(1,0)-10.05(0.995,-0.0995)=(0.0003,1.0000)$;
  $\arctan 0.1=5.71^\circ$. The stored output prints absolute coefficients
  (10, 10.05), so it agrees with either sign.
- **ODE-15.** Was: region with strict inequality, A-stability with
  $|R(z)|\le1$ on $\operatorname{Re}z\le0$, "since its region is everything
  outside the unit disc". Now: A-stable "if its region contains the whole
  open left half-plane", and backward Euler's region is "everything outside
  the closed unit disc around $+1$". By hand: $|1/(1-z)|<1$ iff $|1-z|>1$,
  which holds for $\operatorname{Re}z<0$ (the proposition's proof).
- **ODE-4.** The parenthesis about "completeness of the continuous paths"
  is replaced by a plain statement of what is taken on trust (a uniform limit
  of continuous paths is continuous).
- **Corollary proof (my addition).** The composition law was argued for
  forward times only and then used with $-t$. One clause now says its
  argument did not use the sign of $s$ or $t$.
- **RK orders for nonlinear fields**: added "we do not prove this" after the
  citation (it was stated in the same register as the proved linear case).
- No finding disputed.

## Stored outputs

All numbers quoted from cells were compared with
`outputs/pytorch/chapter_mdl-dynamics/mdl-odes-solvers.json`: 0.90,
(-0.3062, 0.6690), 0.735759, peak 2.513 at 0.68, slopes 1.040 and 4.022,
3.6e-3, 1.3e-9, the stiffness table (0.3487, 0.3855, 1.0e6, -1.0), and the
gradient-descent table (1.01 at 0.201, norm 2.70). The quoted RK4 error at
512 steps (5.0e-12) and "each halving divides by 16" were dropped as not
needed. New: forward Euler's fast component at $h=0.1$ is stated as
$4^{10}\approx10^6$ (= 1,048,576; stored 1.0e+06).

Numbers in prose that no cell prints, kept after recomputing: the
compound-interest table and $e/(2n)=1.36h$ ($n(e-(1+1/n)^n)=1.358$ at
$n=1000$); bound $2.3h$; pendulum eigenvalues (reviewer checked, not redone);
$e^{-2.5}=0.08$ (0.0821).

## Six before/after pairs

1. Before: "Every network in this book is trained by gradient descent, and
   anyone who has trained one has seen a run blow up because the learning
   rate was too large."
   After: "Anyone who has trained a network has seen a run blow up because
   the learning rate was too large."
2. Before: "Do the series, the eigendecomposition and the compound-interest
   limit $(\mathbf I+t\mathbf A/n)^n$ give the same matrix, and does the
   length fall like $e^{-t/2}$? The cell computes all three for the spiral at
   $t=2$."
   After: "There is a third route to $e^{\mathbf At}$: the compound-interest
   limit $(\mathbf I+t\mathbf A/n)^n$, which needs no series and no
   eigenvectors. The cell computes all three for the spiral at $t=2$. Before
   running it, guess how fast the compounding converges in $n$."
3. Before: "Are the slopes $1$ and $4$ measurable? We integrate the spiral to
   $T = 2$, where the matrix-exponential cell gave the exact answer, and
   halve the step six times."
   After: "On log--log axes, $\textrm{error} \approx C h^p$ is a line of
   slope $p$, so the measured slopes should be $1$ and $4$. We test this on
   the spiral ... halving the step six times."
4. Before: "With the fast mode decayed after $t\approx0.1$, how large a step
   does each method allow for the slow mode? The cell runs forward Euler and
   backward Euler, the latter with one linear solve per step, at five step
   sizes."
   After: "We now run both methods on the two-mode system at five step
   sizes, backward Euler with one linear solve per step. The reader can
   predict the first row: at $h=0.1$ forward Euler multiplies the fast
   coordinate by $1-5=-4$ ten times."
5. Before: "We can now return to the question of the opening."
   After: "The opening treated one coordinate. With several, the same factor
   appears once per eigenvalue of the Hessian."
6. Before: "(This is the one place where the completeness of the continuous
   paths is used.)"
   After: "We take one fact from analysis on trust here: a uniform limit of
   continuous paths is continuous."

## Repetition

- **ODE-17 (owner: chapter 40).** The cell
  `#mdl-odes-solvers-gradient-descent-is-a-solver` stays. The prose no longer
  rederives the ceiling or narrates the sweep: it cites
  `sec_mdl-gradient-based-optimization` ("We do not repeat that analysis"),
  keeps the three-arrow display, and reads the table as points
  $z=-\eta\lambda$ in the stability figure. Kept, as owned here: "the
  condition number of the Hessian is the stiffness of the gradient flow" and
  "A diverging learning rate is a solver instability."
- **ODE-10 (owner: linear algebra).** Transient growth is reduced to the
  closed form (now obtained from the new variation-of-constants formula, so
  it doubles as its worked example), one sentence on the eigenvectors with a
  pointer to `subsec_mdl-transient-growth`, and one sentence on why it
  matters (a converging momentum run can grow for its first steps; I ran
  exercise 7d: $\beta=0.9$, $\eta\lambda=0.1$, norm grows for 4 steps to
  3.64, $|\lambda|=0.949$). The normal-matrix sentence and the second
  discrete example are gone.
- **ODE-6.** Matrix exponential: this file owns it; added "(stated without
  proof in `sec_mdl-eigendecompositions`)". The pointer the other way is for
  the linear algebra file.
- **ODE-18.** "We can now read from the eigenvalues ..." deleted; the two
  hand-off sentences kept as the end of the last subsection.
- **ODE-19.** Roadmaps: one in the opening (both parts, with the ranking) and
  one after the seam. The three part-level and subsection-level roadmaps are
  gone.
- **ODE-7.** "The one step that ..." appears once ("The step that is not
  routine"); the other two are reworded.
- Inside the file: the classification's third bullet and the following
  paragraph said the rotation case twice; merged. The $e^{-1}$ worked numbers
  for decay (0.74, 0.27) and the check of the integral form on decay were
  cut. The cost-of-implicitness paragraph was merged into the definition of
  "implicit".

## Ranking

- Demoted: "Proposition (matrix exponential by eigendecomposition)" is now a
  sentence with its one-line reason; the label
  `eq_mdl-ode-matrix-exp-eig` stays on the equation.
- Kept as blocks: Picard-Lindelöf (now called the result of the first part
  that everything later leans on), the bijection corollary, the matrix
  exponential solves linear ODEs, Euler's order, stability functions, the
  test-equation proposition.
- The opening names the three results that carry the chapter and says that
  the pendulum and transient growth can be skipped on a first reading.

## Labels added

- `subsec_mdl-ode-driven-linear` (new subsection "Linear Systems with an
  Input", between "Stability from Eigenvalues" and "Transient Growth")
- `eq_mdl-ode-variation-of-constants`
- `eq_mdl-ode-zoh`

References added, all resolve: `sec_gd`, `sec_resnet`,
`sec_diffusion-samplers` (twice), `sec_ssm`, `subsec_zoh`,
`sec_mdl-neural-odes-cnf` (in the body), `sec_mdl-euler-runge-kutta` (from
part one, as pointers).

## The driven linear equation (ODE-S3)

Statement: for $\dot{\mathbf x}=\mathbf A\mathbf x+\mathbf B\mathbf u(t)$,
$\mathbf x(t)=e^{\mathbf At}\mathbf x_0+\int_0^te^{\mathbf A(t-s)}\mathbf B\mathbf u(s)\,ds$;
with $\mathbf u$ constant on a step,
$\mathbf x_{n+1}=e^{\mathbf Ah}\mathbf x_n+\mathbf A^{-1}(e^{\mathbf Ah}-\mathbf I)\mathbf B\mathbf u_n$
($\mathbf A$ invertible for the last form). Derivation by the integrating
factor $e^{-\mathbf At}$. Verified with NumPy on the spiral matrix: the
zero-order-hold step with a constant input agrees with RK4 at 20,000
substeps to $6\times10^{-15}$; the general formula with a time-varying input
(trapezoid quadrature) agrees with RK4 to all printed digits. The scalar
case matches `eq_zoh` of `ssm.md` ($\bar a=e^{\Delta\lambda}$,
$\bar b=(e^{\Delta\lambda}-1)/\lambda$ with $b=1$). The stability remark
($|e^{h\lambda}|=e^{h\operatorname{Re}\lambda}<1$) is stated there and
recalled in one sentence beside the stability figure.

## Split

Marker `<!-- split: new section "Numerical Solvers" starts here -->` on the
line before `## Numerical Solvers` (line 787 of the revised file). Part one
is to be titled "Ordinary Differential Equations".

- Part two opens with its own question and roadmap, and the order cell now
  restates the spiral ($\mathbf A$ by figure label, its eigenvalues, the
  exact endpoint) before using it.
- Part one no longer says "later in this section" for Euler; it points to
  `sec_mdl-euler-runge-kutta` twice (integral form, compounding) as pointers
  only. It ends on the limit of linearization ($\dot x=\pm x^3$).
- The Summary has two paragraphs with italic lead-ins, one per part.
- Left for the development machine: the opening (the bowl, $1-50h$) belongs
  in spirit to part two but must stay in part one's position; part two's
  cells use `A`, `x0`, `expAt_eig` and `euler` from part one's cells; part
  two refers to "the opening's" step three times; the notation table.

## Reorder (ODE-3, ODE-16)

- "The Flow Map" now follows "Existence and Uniqueness" (no cell lies
  between them; headings and labels unchanged). The corollary, "trajectories
  never meet" and the neural-network field moved under "The Flow Map"; the
  blow-up example $\dot x=x^2$ stays with the theorem. No forward promises
  remain.
- Stiffness: the two-mode system and the definition of "stiff" now come
  first, then the proposition, implicit steps, regions and A-stability, then
  the cell (which was and is the last thing before its reading).

## Cells to cut or redesign on the development machine

- `#mdl-odes-solvers-gradient-descent-is-a-solver` (ODE-17): duplicate of
  the chapter 40 sweep. Replace by backward Euler (proximal point) or RK4 on
  the gradient flow, as the review proposes; I removed a sentence on backward
  Euler on the gradient flow to keep the length, it belongs with that cell.
- `#mdl-odes-solvers-transient-growth` (ODE-10): confirms a closed form; its
  plot adds nothing the printout lacks. Cut, or keep only if the linear
  algebra cell goes.
- `#odes-solvers-matrix-exponential`: the series-against-eigenformula line
  is a confirmation; the compounding rate is the part that teaches. Trim.

## Other deferred findings

- **ODE-9.** The portrait table stays, because the figure caption says "one
  per row of the table" and captions were not to be changed. Drop the table
  and edit the caption together.
- **ODE-S5.** The chapter notation table stays (its label is cited from
  `chapter_notation/index.md`; moving it is a cross-file change).
- **ODE-14.** Done in substance (cost sentence moved after the cell with the
  measured numbers).
- **Exercises 2a and 8a** are answered in the text; left, as the brief says
  to change only exercises that are wrong. Difficulty marks not reviewed.
- **Main text, `resnet.md`.** It lacks the Euler reading. Suggested sentence
  for `subsec_residual-blks`: "The update
  $\mathbf x\leftarrow\mathbf x+g(\mathbf x)$ is one step of Euler's method
  for the differential equation $\dot{\mathbf x}=g(\mathbf x)$, a reading
  that :numref:`sec_mdl-neural-odes-cnf` develops
  (:eqref:`eq_mdl-ode-resnet-euler`)." For Euler's method itself the label
  is `sec_mdl-euler-runge-kutta`.
- **Main text, `ssm.md`.** `subsec_zoh` should cite
  `subsec_mdl-ode-driven-linear` (or `eq_mdl-ode-zoh`) for the derivation,
  and its exercise on forward and backward Euler should cite
  `sec_mdl-stiffness-implicit`.
- **Chapter index** (`chapter_mdl-dynamics/index.md`) should name the SSM
  chapter as a reader (IDX-3), and the OU proof in `mdl-sdes.md` can cite
  `eq_mdl-ode-variation-of-constants` for its integrating factor.
- Stale reference in `mdl-matrix-calculus-autodiff.md` (adjoint method):
  not my file.

## Notation clashes

- $\mathbf B$ is now the input matrix (as in `eq_ssm_cont`), while the code
  and the slides call the transient-growth matrix `B`, and exercise 7 has
  $\mathbf B_\nu$. The prose no longer names that matrix and says "called
  `B` in the code below".
- $b$: imaginary part of an eigenvalue ($a\pm\mathrm ib$) and, once, the
  scalar input coefficient in $\bar b$.
- The SSM chapter writes $h$ for the state and $\Delta$ for the step; noted
  in a parenthesis.

## Claims added or changed

1. Variation of constants and the zero-order-hold step (above; verified).
2. $|e^{h\lambda}|<1$ for all $h>0$ iff $\operatorname{Re}\lambda<0$; on
   $\dot x=-50x$, $h=0.05$: $0.08$ against Euler's $-1.5$.
3. $\mathbf A\mathbf p=a\mathbf p-b\mathbf q$,
   $\mathbf A\mathbf q=b\mathbf p+a\mathbf q$, and $\mathbf A$ is
   $a\mathbf I+b\mathbf K$ in the basis $(\mathbf q,\mathbf p)$; $\mathbf p,
   \mathbf q$ independent when $b\ne0$. Checked by hand and numerically on
   the spiral.
4. $x_1(t)=10\int_0^te^{-(t-s)}e^{-2s}ds$ (quadrature at $t=0.9$: 2.41271
   both ways).
5. $(1+1/n)^n\approx e(1-\tfrac1{2n})$ from $n\log(1+1/n)=1-\tfrac1{2n}+\dots$.
6. Simpson's rule "integrates cubics exactly" (standard; by hand on $x^3$
   over $[0,1]$: $\tfrac16(0+4/8+1)=\tfrac14$).
7. Non-diagonalizable $\mathbf A$: polynomial factors "change the verdict
   only for eigenvalues on the imaginary axis" (shortened from the old
   paragraph, same content).
8. "a momentum run that converges can grow for its first few steps" (item
   under ODE-10).
9. "to that order gradient descent follows the flow of a modified loss that
   penalizes large gradients": restates exercise 5 (Barrett and Dherin), not
   rechecked beyond the exercise's own statement.
10. Forward Euler's fast component at $h=0.1$ is $(-4)^{10}\approx10^6$.
11. Judgments: Picard-Lindelöf as "the result of the first part that
    everything later leans on"; "The pendulum and transient growth can be
    skipped on a first reading."

## Slides that now disagree with the body

- Transient-growth slide: "$\mathbf x_0=10\,\mathbf v_1+10.05\,\mathbf v_2$"
  (body now has a minus sign with the flipped $\mathbf v_2$), and it names
  the matrix $\mathbf B$.
- A-stability slide: "$|R(z)|\le1$ whenever $\operatorname{Re}z\le0$" (body
  now uses the open half-plane).
- No slide covers the driven equation or the zero-order hold.
- The cover kicker and deck are for one section; after the split they need
  dividing.
