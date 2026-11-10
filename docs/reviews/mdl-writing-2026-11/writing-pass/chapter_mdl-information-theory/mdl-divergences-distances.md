# Writing pass log: Divergences and Distances Between Distributions

File: `chapter_mdl-information-theory/mdl-divergences-distances.md`. To be
split in three; two markers placed. No code cell, cell ID, figure path or
slide changed. Checker: "OK: only prose changed", no em-dash warning;
`tools/lint_source.py` clean. All numeric checks were done with `python3` and
NumPy in scratch files (no SciPy or PyTorch on this machine) or by hand, as
noted per item.

## Words

Checker count (body, captions and exercises, to the slides marker): 11,753
before, 12,010 after (+257). The checker does not strip inline mathematics
that wraps over a line, so both numbers are inflated. Counted with mathematics
and reference roles removed, the body is 8,691 before and 9,009 after (+318,
+3.7%); the exercises are unchanged in length (2,077 and 2,086).

**The section did not get shorter, and I could not make it so honestly.** The
review's cut list (kernel Stein discrepancy, Sinkhorn, $\alpha$-divergences,
about a quarter of the section) was overridden by the author, and the
assignment asked for additions: 33 new label references (114 to 147), the
uses of each kept topic, the GAN objective in three steps (DD-12), $W_p$ with
FID and KID (DD-26), the concrete reason TV cannot be estimated (DD-18), a
motivated data-processing subsection (DD-15), and an opening and an ending
for each future section. Those come to about 800 words. The repetition and
ranking cuts removed about 480. Display equations fell from 46 to 44.

## Correctness changes

| Finding | Before | Now | Verified |
|---|---|---|---|
| DD-22 | "The section uses a dozen symbols, each with one meaning" | sentence deleted. $\delta$ of the local-quadratic proposition is now $r$ (also in :eqref:`eq_mdl-f-div-local`); the binary KL $d(a\|b)$ is $D_{\textrm{KL}}(\textrm{Ber}(a)\|\textrm{Ber}(b))$; the log-sum inequality uses $v_i, w_i$ instead of $a_i, b_i$; sample size is $N$ and dimension $n$ in the body ("In $d$ dimensions" is "In $\mathbb{R}^n$"; RBF feature index is $m$) | by reading; the opening now states the two conventions |
| DD-28 | from Stein's identity on, $P$ was the model and $Q$ the sample; $\mathrm{KSD}^2(Q, P)$, $k_P$, $\mathcal{A}_P$ | $P$ is the data and $Q$ the model everywhere. Stein's identity is stated for the model: $\mathbb{E}_Q[g' + g\,s_Q] = 0$; operator $\mathcal{A}_Q$; Stein kernel $k_Q$; $\mathrm{KSD}^2(P, Q) = \mathbb{E}_{x,x'\sim P}[k_Q]$; score-difference form with $s_Q - s_P$ under $P$; energy-based model written $\tilde q$. "KSD is a kernel-weighted Fisher divergence" is now "a kernel-smoothed relative of $D_{\textrm{F}}(P\|Q)$", which is true with the letters as they now stand | every equation retyped and checked term by term against the code of `ksd2_ustat` (the four terms of the product $\langle\psi(x),\psi(x')\rangle$); worked value $1/\sqrt3$ unchanged since the score difference is still $1$. Exercise 7(c) changed to the same letters |
| DD-20 | "For such *characteristic* kernels, MMD = 0 only if $P = Q$" | "A kernel for which MMD$(P,Q) = 0$ only if $P = Q$ is called *characteristic*. The RBF kernel is characteristic [cite], so its MMD is a metric; the linear and quadratic kernels above are not." "moments of every order" is "damped monomials of every order" | definition and cited fact; the linear and quadratic kernels were shown not to separate in the text itself |
| DD-5 | $f$ on $(0,\infty)$, $f(0)$ defined a page later | $f(0) = \lim_{u\downarrow0} f(u)$ is part of the definition | |
| DD-6 | measure-theoretic sentence on $P_\perp$; $f'(\infty)$ written as a derivative without comment | sentence deleted; "$f'(\infty)$ ... is the limit of $q\,f(p/q)$ as $q \downarrow 0$. For a convex $f$ the ratio $f(t)/t$ and the slope $f'(t)$ have the same limit, hence the notation." The paragraph now leads with its purpose | by hand: $q f(p/q) = p\,f(p/q)/(p/q)$; convex $f$ has nondecreasing slope with the same limit as $f(t)/t$ |
| DD-14 | optimal critic applied to disjoint supports without comment | "If we extend $T^\star = f'(p/q)$ formally beyond the proposition's hypothesis $P \ll Q$, the ratio is $\infty$ on the data and $0$ on the model's samples, so $T^\star$ is the constant $f'(\infty)$ on one support and $f'(0)$ on the other." | by hand |
| DD-12 | GAN paragraph gave $f$ and jumped to the value | conjugate $f^*(t) = -\log(1-e^t)$, substitution $T = \log D$, the familiar objective in one display, $D^\star = p/(p+q)$, then the constant $f(1) = -\log 4$ "which only shifts the value" | conjugate checked numerically (sup on a grid against the formula at $t=-0.7$: 0.68634 both); $\mathbb{E}_Q f(u) = -1.20405$ on the running pair against $2(0.0911)-\log4$ |
| DD-16 | Step 1 for $a, b \in (0,1)$, used at the endpoints; Pinsker countable, data processing finite | Step 1 for $a \in [0,1]$, $b \in (0,1)$ with the endpoints by continuity and "If $b$ is $0$ or $1$, the divergence is infinite unless $a = b$"; both propositions "finite or countable" | by hand |
| DD-17 | "asks whether the inequality is ever violated"; attribution of the constant to Csiszár | two questions (tightness for typical pairs; is $\tfrac12$ best); attribution sentence dropped | "typically at about three quarters of it": 10,000 seeded Dirichlet pairs, ratio quartiles 0.69, 0.78, 0.86, max 0.987 (NumPy) |
| DD-25 | Fubini justified in Step 1, not in Step 4; Folland cited for absolute continuity | Step 1: "The integrand is non-negative, so expectation and integral may be exchanged"; Step 4: Fubini "which the finite first moments justify"; "It suffices to consider $g(x) = g(0) + \int_0^x g'$ with $|g'| \le 1$: every such $g$ is 1-Lipschitz, and Step 3 already bounds all the others" | by hand |
| DD-24 | weak duality proved in prose before the proposition that cites it; "Thus $W_1$ is the IPM" drawn from weak duality alone | duality stated, then what it gives (WGAN), then "We prove the duality only on the line", then one proof with weak duality as Step 3 | |
| DD-30 | "which is why diffusion models perturb the data" | "which is one reason why"; pointer to `sec_diffusion-denoising` | checked that `denoising.md` reports the Fisher/KL contrast on trained networks |
| DD-21 | dimension drift | stated once at the head of part two; $\mathbb{R}^n$ in $W_1$ and the Stein kernel | |
| DD-9 | $\chi^2(q_\theta\|q_{\theta+h}) \approx h^\top I h$ asserted | one-line derivation with $u \approx 1 + \mathbf{h}^\top\nabla_\theta\log q_\theta$, written for $\chi^2(q_{\theta+h}\|q_\theta)$ so that the expectation is under $q_\theta$ | by hand |

Stored outputs: every number quoted beside a cell was compared with
`outputs/pytorch/.../mdl-divergences-distances.json`; all agree. No finding
disputed.

## Numbers in prose that no cell prints, kept after recomputation

Forward-fit prediction: variance $0.36 + 0.7\cdot0.3\cdot4^2 = 3.72$ (hand).
$-1.204$ for the GAN game (NumPy). "about three quarters" of the Pinsker
bound (NumPy, above). $\alpha = 2$ gives half of $\chi^2$, and the limits and
midpoint of $f_\alpha$ (NumPy at two values of $u$). "about $3\%$ of the
optimal cost" for $\varepsilon = 0.1$ against $3.5$. Ten routes of the
monotone plan and its cost 3.5 (NumPy, north-west-corner construction).
$3.35 - 1.65$. Six divergences "disagree by a factor of ten" (0.0911 to
1.0133).

## Six before/after pairs

1. Before: "The Jensen--Shannon row fixes the loss of the original generative
   adversarial network (GAN) at its best discriminator, which is
   $2\,\textrm{JS} - \log 4$ and therefore zero in every column." After: "A
   model trained on any of these three would receive the same loss at
   $d = 0.01$ as at $d = 100$, hence no gradient in $d$."
2. Before: "The following cell computes both fits for the mixture ..." After:
   "For the mixture ... we can therefore predict the forward fit: mean
   $0.7(-2) + 0.3(2) = -0.8$ and variance ... $= 3.72$. The reverse fit we
   leave to a derivative-free optimizer ..."
3. Before: "The following experiment asks whether the inequality is ever
   violated and how tight it is." After: "The proof leaves two questions
   open: how tight is the bound for typical pairs, and can the constant
   $\tfrac12$ be improved?"
4. Before: "Thus MMD is the distance between the mean features." After: "MMD
   is the distance between the mean features, and the supremum has gone."
5. Before: "The generators in :numref:`fig_mdl-f-div-generators` differ
   greatly away from $u = 1$. Do they still differ when $P$ is close to $Q$?"
   After: "... Near $u = 1$ they turn out to be one function up to a constant,
   which is perhaps the most surprising fact about the family."
6. Before: "Three properties make KSD useful. First, ..." After: "What
   recommends the KSD is what it does not need."

Also new, and possibly too strong: "It is rather surprising that a linear
program should collapse this far."; "No training objective in this book uses
the KSD."; Pinsker "is the inequality that later proofs use most".

## Repetition

- **Constant on disjoint supports**: told once, after the conventions ("One
  line has explained the first three rows ..."). The opening keeps only "no
  gradient in $d$"; the f-GAN caveat is the critic's view in two sentences;
  the optimal-transport opening is one sentence with a reference (DD-23).
- **Total variation**: the divergence-function paragraph gives the
  half-$\ell_1$ form and points to `sec_mdl-tv-pinsker`; the supremum
  subsection says it is the same quantity and cites the table. The
  proposition "TV is half the $\ell_1$ distance" is now a paragraph.
- **"Generator"**: $f$ is the *divergence function*, as in
  `sec_gan_objectives`; "generator" is the network. One sentence says the
  f-GAN literature (and the code's dictionary `generators`) use the other
  word. Heading "Common Generators" is "Common Divergence Functions"; table
  caption, column header and two figure captions follow. The exercises were
  changed to match (14 places). Labels `tab_mdl-f-div-generators`,
  `fig_mdl-f-div-generators`, `eq_mdl-alpha-generator` are unchanged.
- **Forward KL fits moments** (DD-11): proposition and proof replaced by two
  sentences citing `subsec_mdl-moment-matching` (chapter 41 owns it).
- **Gaussian KL** (DD-4): the display stays because its label is cited three
  times; the reading of its terms is gone and :eqref:`eq_mdl-gaussian_kl` is
  cited.
- **Convex conjugate**: the biconjugation sentence is gone (the optimization
  chapter owns it and the proof does not use it).
- **Fisher divergence and Stein's identity against chapter 43**: the
  definitions stay here. The body now cites `sec_mdl-score-matching` and
  :eqref:`eq_mdl-hyvarinen` for the integration by parts in $\mathbb{R}^n$;
  chapter 43 already cites `sec_mdl-fisher-divergence`. Stein's identity here
  is the one-dimensional case in three lines and is said to be that.
- **Closing** (DD-35): the "We can now ..." sentence is gone; the paragraph
  on estimation error stays as the ending.

Kept because this file owns them: the f-divergence definition and
conventions, the local quadratic and its Fisher-information form (further
reading should keep only Chentsov), the f-GAN bound, Pinsker, data
processing, MMD, $W_1$ and its dual, Sinkhorn, the Fisher divergence, the
kernel Stein discrepancy, the $\alpha$-family.

Author override applied: $\alpha$-divergences (moved before the cell, with
Rényi variational inference and differential privacy as uses), Sinkhorn
(Sinkhorn divergences, minibatch couplings in
`sec_mdl-flow-matching-transport`) and the kernel Stein discrepancy
(goodness-of-fit tests, sampler quality, SVGD; the central step of the
derivation is now written out) are kept at full length.

## Ranking

- The opening names the two results that matter most (f-GAN bound,
  Kantorovich--Rubinstein), the two that later proofs use (Pinsker, data
  processing), and what can be skipped on a first reading.
- Propositions: 10 before, 7 after. Removed: forward KL fits moments. Demoted
  to prose: TV is half the $\ell_1$ distance; Stein's identity (statement with
  hypotheses, labelled equation, three-line justification).

## Labels and references added

Label: **`subsec_mdl-f-div-data-processing`** on a new `### Data Processing`
subsection (the proposition, its proof, and the reading as a lower bound on
features). The paragraph was moved from before the Pinsker cell to after its
reading; no cell moved. `relativistic.md`, `mdl-mutual-information.md` (2)
and `mdl-mi-estimation.md` (2) should be repointed to it.

References added (all resolve): `sec_gan_objectives` (4),
`sec_gan_convergence`, `eq_gan_fid`, `eq_gan_kid`, `sec_lm_distillation`,
`sec_regularized`, `sec_diffusion-denoising`, `sec_mdl-score-matching`,
`eq_mdl-hyvarinen`, `subsec_mdl-moment-matching`, `eq_mdl-gaussian_kl`,
`subsec_mdl-dpi`, `sec_mdl-mutual-information`, and internal ones. Citations
added: `Li.Turner.2016`, `Mironov.2017`, `Liu.Wang.2016`,
`Genevay.Peyre.Cuturi.2018` (all in `d2l.bib`). Removed: `Folland.1999`.

## Split

- Line 740: `<!-- split: new section "Test Functions and Transport" starts here -->`
- Line 1402: `<!-- split: new section "Scores" starts here -->`
- "Divergences and Metrics" (label kept) is now the first `###` under
  `## f-Divergences`; the two headings changed places, no cell is between
  them.
- Part one ends on its own argument (the two defects of ratio methods) and
  points forward by label only. Parts two and three open with their question
  and the convention for $P$ and $Q$, and cite part one by label; part two
  restates the running pair where it uses it.
- The Summary has three paragraphs, one per future section.
- **"Training Objectives and Their Divergences" should go with the third
  section (Scores)**, as its close: only there are all eight rows defined. It
  refers to the rest by label and to `tab_mdl-point-mass-table`, so it can
  equally close the chapter's last divergence section, wherever that is.
- Code dependencies across the seams, for the development machine: part two
  uses `kl_div`, `p`, `q` and SciPy's `optimize` from part one; part three
  uses `log_components`, `log_mix`, `log_p`, `kl`, `x`, `w`, `mus`, `sigmas`
  and `integrate`.
- The opening roadmap and the taxonomy figure refer to all three parts.

## Cells to cut or redesign on the development machine

- `#divergences-distances-score` (DD-29): unit test of a helper. Define
  `mix_score` in the next cell and drop the check.
- `#divergences-distances-stein` (DD-31): confirms the identity where it
  holds. Break it instead (sample from $\mathcal{N}(0.5,1)$ or
  $\mathcal{N}(0,4)$). Its comment calls the model $P$.
- `#divergences-distances-ksd` (DD-28): the variable `x_q` and the comment
  "the sample: Q = N(0, 1)" use the old letters; rename to `x_p`. The prose
  around the cell avoids the letters.
- `#divergences-distances-w1`: LP against the CDF formula confirms a proved
  result. I kept it as one of the "once or twice" (it sets up `A_eq` for
  Sinkhorn); prose shortened.
- `#divergences-distances-f-gan-bound`: prints the closed form
  $\chi^2(2c - c^2)$; kept because the half-trained critic at 75% is what
  readers remember, but it decides nothing.
- `#divergences-distances-f-table`: the last line recomputes JS (one clause
  now).
- Imports (DD-36): four framework tabs for a NumPy section.

## Other deferred findings

- DD-33: the taxonomy figure does not draw the score family and has a
  "boxed = metric" footer; the caption now claims only three families.
- DD-27: the Sinkhorn cell still switches to the squared cost; $W_p$ is now
  defined before it.
- S4: total variation and Pinsker stay in part two (headings may not cross
  the seam). Data processing therefore lands in part two as well, although
  it is a statement about f-divergences; consider moving both when cutting.
- Exercises: sample sizes are still $n$, exercise 8 uses $d$ for a degree
  and $\delta_{jk}$; exercise labels are leftovers; the Rényi and Sinkhorn
  overlaps with further reading (FR-4, FR-5) are untouched.

## Claims added or changed

Besides the table above: the natural gradient "would be the same method under
any other smooth f-divergence" (same metric up to $f''(1)$); $u\log u$ has
$f(0)=0$, $f'(\infty)=\infty$ and $-\log u$ the reverse; the empirical TV of
two samples from one continuous distribution is 1; MMD and $W_1$ can grow
under a map (exercise 6a); "What an approximately Lipschitz critic reports is
no longer exactly $W_1$"; FID is a squared $W_2$ between fitted Gaussians and
KID an unbiased MMD$^2$ with a cubic kernel (as `dcgan.md` states); for
$p > 1$ on the line the monotone plan is the unique optimum; the minibatch
couplings of flow matching "solve small transport problems of this kind";
the printed witness gap is MMD$^2$ with self-pairs included; KSD is used for
goodness of fit, sampler quality and SVGD.

## Inconsistent with neighbours and slides

- Slides of this file: "Generators" in a title and in the f-GAN slide;
  $d(a\|b)$ in the Pinsker slide; the Stein/KSD slide has
  $\mathbb{E}_P[g' + g\,s_P] = 0$, $\mathrm{KSD}^2(Q,P)$, $k_P$ (old letters);
  the score slide writes $\tilde p$.
- `further-reading.md` ("Stein Variational Gradient Descent") writes the
  target as $P$ with score $s_P$ and particles $Q$, the old letters; its
  "Information Geometry" entry uses $\mathbf{F}$ for the Fisher information.
- The GAN chapter writes the density ratio $\rho$; this section keeps $u$.
