# Writing pass log: Stochastic Differential Equations

File: `chapter_mdl-dynamics/mdl-sdes.md`. Not split (no markers). No code
cell, cell ID, figure include, label or slide changed. Checker: "OK: only
prose changed", no em-dash warning; `tools/lint_source.py` clean. Numeric
checks were done with `python3` + NumPy in the scratchpad or by hand (no
`.venv-pytorch` on this machine); which is noted per item.

## Words

Checker count (body, captions, exercises): 9,176 before, 9,074 after (-102).
Display-equation delimiters 80 to 80 (one display removed with the
covariance proof, its replacement kept as a display). The reduction is small
because the cells the review wants cut (about 1,300 words with their prose)
cannot be cut in this pass; their prose is shortened, and the saving pays for
the ranking paragraph, the repaired definition of the integral, the
main-text pointers and the SDE-9/SDE-11 repairs.

## Correctness changes

| Finding | Before | Now | Verified |
|---|---|---|---|
| SDE-11 | "This is also why data are normalized to unit variance before training a diffusion model." | "The name describes unit-variance data only. For data of variance $s^2$ the marginal variance drifts from $s^2$ to $1$ (problem ...), so the scale of the data matters. Images scaled to $[-1, 1]$, as in `sec_diffusion-denoising`, have variance below one, while latent diffusion models rescale their latents to about unit variance (cite Rombach et al.)." | By hand: $\textrm{Var}(X_t)=\bar\alpha_t s^2+1-\bar\alpha_t$. `denoising.md` scales Fashion-MNIST to $[-1,1]$; `latent-diffusion.md` has the scale factor. The LDM statement is from memory of the paper (its latent rescaling by the estimated standard deviation). |
| SDE-14 | "In this moment, these data start as far from the Gaussian as unit-variance data can." | deleted; kept only "the smallest fourth moment a unit-variance law can have" | $\mathbb E[X^4]\ge(\mathbb E[X^2])^2=1$ (by hand). The two-line justification was cut with it; exercise 8 has the context. |
| SDE-9 | lemma applied to $\log x$ with no positivity assumption | "The logarithm is smooth only for $x>0$, so assume for the moment that $X$ stays positive" and, after the formula, "The formula we obtain is positive, and the lemma applied to it as a function of $W_t$ and $t$ confirms that it solves the equation, which removes the assumption." | By hand: $\phi(w,t)=X_0e^{(\mu-\nu^2/2)t+\nu w}$, $\phi_t+\tfrac12\phi_{ww}=\mu\phi$, $\phi_w=\nu\phi$. |
| SDE-8 | "Proposition (Itô's lemma)", "Proposition (orders of Euler--Maruyama)" | "**Theorem (Itô's lemma).**", introduced by "What we have given is a heuristic derivation, so we state the result as a theorem without proof", proof cited to Øksendal ch. 4; "**Theorem (orders of Euler--Maruyama; stated without proof).**" with the conditions indicated and Kloeden and Platen cited for the precise ones. "Proposition ... Proof ... $\blacksquare$" now marks only what is proved here (quadratic variation, isometry for step processes, OU kernel). | n/a |
| SDE-4 | integral for general $H$ defined via "the isometry below"; the isometry's proof then used the extension | order is now: step processes, zero mean, isometry stated and proved for step processes, then one paragraph: the isometry extends the definition, both properties survive the limit, "We do not prove these statements; see Øksendal, chapter 3". "Predictable" defined in plain words once. No cell moved. | Chapter number from memory (ch. 3 "Itô Integrals"). |
| SDE-1 | noise alone a failure in the bullet, "also works" five paragraphs later | bullet: "The start is forgotten only relative to this growing width, and there is no limiting law"; after the figure: "Only the first fails outright. The second is usable, and is in use, if it runs until its variance $2t$ is far larger than that of the data ... *variance-exploding*". The later paragraph is gone. | n/a |
| SDE-5 | "A simulation that moves forward in time cannot look ahead, so ..." | two reasons: zero mean, and the simplest simulation step is an Itô sum | n/a |
| SDE-15 | Euler--Maruyama "is the forward step of DDPM to first order", exactness as an afterthought | leads with exactness: kernel over one interval is DDPM's step with $1-\beta=e^{-2\theta\Delta t}$; Euler--Maruyama is its first-order approximation; numbers $0.99005$, $0.0198$ against $0.99$, $0.02$ | $e^{-0.01}=0.990050$, $1-e^{-0.02}=0.019801$ (by hand/NumPy). The intro sentence "agrees with the forward step of DDPM to first order" is gone. |
| SDE-17 | "The orders follow from counting ..." | "A count of single-step errors explains the exponents. It is not a proof: ... the proofs must also control how the later dynamics carry an early error forward." Hypotheses of the theorem reduced to indications. | n/a |
| SDE-3 | covariance "is what we test to validate a simulator" | "a Gaussian process is determined by its mean and covariance, so mean zero, covariance $\min(s,t)$ and continuous paths characterize Brownian motion" | standard fact |
| QV cell lead-in (mine) | n/a | first draft said the proposition is "about one path"; it is a mean-square statement, so the text says "At fine meshes the fluctuations are so small that a single path shows the limit." | |

No finding disputed. SDE-13 and SDE-19 want cells cut; see below.

## Numbers in prose that no cell prints

Kept after recomputation: "about half on this grid" of paths stay inside
$\pm2\sqrt t$ over the whole interval (new; NumPy, 20,000 paths, 500 steps:
0.54); standard errors $0.005$ and $0.013$ for mean and variance of
$\int W\,dW$ (by hand: fourth central moment 3.75, $\sqrt{3.5/20000}$);
gap "typically within about $0.05$ and rarely by more than $0.1$" (NumPy:
median of the path maximum 0.050, 90% 0.086, 5% exceed 0.1); $0.014$ for the
sample variance at $t=4$; $0.99005$ and $0.0198$; kernel values at
$t=0.5,1,4$; $3-2\bar\alpha_t^2$ values; $e^{-1}-0.99^{100}=0.00185$.
Dropped: the limit $1.005$ in the opening (now only under "The Stationary
Distribution"), the discrete variance $0.4995$, the hand step $1.881$, the
standard errors $0.006$ and $0.044$ of the Rademacher table.

## Six before/after pairs

1. Before: "Does the simulated walk have variance $t$, and what fraction of
   the paths sits inside $\pm 2\sqrt{t}$ at a fixed time? We generate
   $20{,}000$ paths on $[0, 1]$ and check." After: "We generate $20{,}000$
   paths on $[0, 1]$ this way, and with them the random generator `rng` that
   every later cell draws from." (and after the cell: "the printout tests the
   code, not the theory.")
2. Before: "Is the variance of the left-endpoint sum really $\tfrac12$? The
   next cell draws $20{,}000$ paths on the same $1000$-step grid and computes
   the sum on each." After: "A sample of $20{,}000$ left-endpoint sums on the
   same $1000$-step grid shows the size of the fluctuations."
3. Before: "Does the recurrence of the opening, run from $x_0 = 2$, follow
   this kernel? The next cell runs it as a plain loop ..." After: "The
   recurrence is only a discretization of the process, so its cloud should
   follow the kernel closely but need not follow it exactly. We run it as a
   plain loop ..."
4. Before: "The variance lies within a few standard errors ($0.014$ each) of
   both $1$ and $1.005$, so this experiment cannot tell the exact law from
   the recurrence's." After: "The half percent, however, is out of reach: the
   sample variance has a standard error of $0.014$ and cannot separate $1$
   from $1.005$."
5. Before: "The absolute errors of the two processes are not comparable,
   since the processes have different scales, but their ratio is: the gap
   between the two curves widens by about $\sqrt{2}$ each time the step is
   halved." After: "The two curves have different heights because the
   processes have different scales. What can be compared is how they fall:
   each halving of the step halves the OU error and divides the error of
   geometric Brownian motion by about $\sqrt{2}$."
6. Before: "We now check that this fails for Brownian motion, and by how
   much." After: "For Brownian motion this fails, and the result of this
   subsection says by how much. It is the one on which the rest of the
   section rests."

Ranking paragraph added to the opening: "Three results carry the section,
and the rest is preparation or consequence: the quadratic variation
$(dW)^2 = dt$, Itô's lemma, and the Gaussian kernel of the Ornstein--Uhlenbeck
process. Only the first is hard to believe, and its proof takes six lines. A
reader in a hurry can take the two properties of the Itô integral on trust
and leave the orders of convergence for a second reading." (A judgment;
soften if too strong.) At the kernel: "Of all the results of this section,
this is the one the rest of the book uses most."

Cell introductions now: plain statement (Brownian paths), clause
(covariance), claim about fluctuations (quadratic variation), end of the
derivation (left/right sums), one sentence (isometry), expectation "should
follow the line $t$" (Itô correction), prediction that can be slightly wrong
(OU cloud), one sentence (stationary), formula then cell (VP), test of the
helper (Euler check), plain statement (strong order), continuation of the
closed forms (weak order). "The next cell" no longer occurs.

## Repetition

Owned here and kept: the identification of the recurrence with the OU
process (now once in full, at the OU definition; one clause after the first
figure; "the recurrence we have been running all along" at
`eq_mdl-sde-em-ou`); the fixed point $1.005$ (derived once under "The
Stationary Distribution", cited as $v_\infty$ in the weak-order paragraph);
why the VP process is called variance-preserving; the OU cloud against the
kernel (the Fokker-Planck section should cite `eq_mdl-sde-ou-kernel`).
Removed: three of four identifications with their arithmetic, the fixed
point in the opening, the second and third $W^2$ set-up in "Itô's Lemma"
(SDE-7), the variance-exploding paragraph, the local-time paragraph
(SDE-10), the "We can now" recap (SDE-20; the hand-off sentences stay as the
ending), "the simulation part of this section" twice (now labels, SDE-6).

## Demoted from Proposition blocks

- Covariance of Brownian motion: now a three-line derivation in the text
  (still in the Summary and on the slide).
- Stationary law of the OU process: now "Indeed, if ..." with the same
  display, no proof mark.
- Relabelled as Theorem (not proved here): Itô's lemma, orders of
  Euler--Maruyama.

Labels added: none. New references: `sec_diffusion-gaussian-paths` (four
times: opening, kernel coefficients, DDPM step, VP-SDE), `sec_diffusion-denoising`,
`sec_mdl-sde-strong-weak`, `sec_mdl-ito-integral`; new citation
`Rombach.Blattmann.Lorenz.ea.2022`. All resolve.

## Main text (correction to the reviews' premise)

`chapter_diffusion-models/` cites two labels of this file, both from
`gaussian-paths.md`: `sec_mdl-ornstein-uhlenbeck` ("the VP path ... is the
Ornstein--Uhlenbeck process of ...") and `sec_mdl-euler-maruyama` ("the
ancestral step is, to first order, the Euler--Maruyama step"). Both passages
still bear the citation: the OU part keeps kernel, $\alpha_t$, $\sigma_t$,
$\bar\alpha_t$ and the VP-SDE; the Euler--Maruyama subsection keeps the
scheme and now states DDPM's forward step as the exact transition. The
section says where the book uses the forward SDE and the OU process (opening,
kernel, VP-SDE paragraph, Summary).

## Cells to cut or redesign on the development machine

- `#sdes-brownian-paths` (SDE-2): checks the generator. Either compute the
  fraction of paths inside the envelope over the whole interval (0.54 on this
  grid) or cut the variance lines. It defines `rng`, `dt_bm`, `W_bm`; keep
  those.
- `#sdes-brownian-covariance` (SDE-2/SDE-3): confirms a one-line identity;
  cut, or move to exercise 7.
- `#sdes-isometry-check` (structural 1): confirms $\tfrac12$, which Example
  1 derives exactly; cut.
- `#sdes-euler-check` (SDE-16): unit test; cut. Prose is now two sentences.
- `#sdes-ou-stationary` (SDE-19): cannot see the half percent; cut or keep
  the quantiles only.
- `#sdes-vp-normalization` (SDE-13): reviewer asks to cut cell and table;
  prose shortened from about 330 to about 190 words. Exercise 8a refers to
  the cell.

## Other deferred

- Caption of `fig_mdl-dyn-sde-paths` still begins "Two failures and the
  remedy" (SDE-1); captions were treated as part of the figure include.
- Structural finding 6 (four framework import cells for NumPy-only code):
  code.
- Exercise order (12 after 2) and the fate of exercise 11: left.
- Notation: $\theta$ is the OU rate here and the parameters in the neural
  ODE section; $h$ is the step here and a scalar field in the Fokker-Planck
  section. Not changed.
- Chapter-wide convention for "Theorem (name; stated without proof)":
  neighbouring files should match (FPK-9, FMT-14).

## Claims added or changed

Listed above under correctness and numbers; in addition: "Exact steps need a
linear equation; for a general drift there is only the approximate scheme"
(exercise 10 supports the first half; the second is loose, since exponential
integrators treat the linear part exactly); "the simplest simulation step is
an Itô sum"; "its endpoint variance must be made huge because it has no
stationary law".

## Slides now disagreeing with the body

- Slide "Independent Increments Give Cov(W_s, W_t) = min(s, t)" and the
  summary slide are consistent in content; the body no longer has a
  Proposition block for it.
- The Euler--Maruyama slide says the OU step is "DDPM's step with
  $\beta=2\theta\Delta t=0.02$"; the body now says DDPM's step is the exact
  transition with $1-\beta=e^{-2\theta\Delta t}$ and Euler--Maruyama its
  first-order approximation.
