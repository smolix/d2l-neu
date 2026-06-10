# Wave-2 report: 24.1 Gradient-Based Optimization

## Structure written (1154 lines; sibling 24.3 is 1145)
5 top-level `##`: Descent Directions · Gradient Descent and Smoothness (descent
lemma, non-convex stationarity rate, Armijo backtracking) · The Quadratic Model
and the Condition Number (per-mode contraction, 2/L ceiling + edge-of-stability
remark, optimal step, valley figure, deferred convex-rate statements) · Momentum
and Acceleration (damped oscillator, √κ stated + Lessard caveat, Nesterov +
optimality) · Stochastic Gradients (unbiased + 1/b prop, noise ball,
Robbins–Monro, `### Coda: Why Not Newton?`). Then Summary, 9 Exercises (incl.
new implicit-bias Ex. 9), Discussions, 7 slides. 5 propositions proved
(steepest direction, descent lemma, stationarity rate, optimal-step contraction,
minibatch variance); 2 theorems stated and deferred to `sec_mdl-convexity`.
Labels kept: `sec_mdl-gradient-based-optimization`, `fig_mdl-opt-gd-bowl-vs-valley`.
9 cells: tabbed imports + 8 untagged numpy cells (ids
`gradient-based-optimization-{steepest-direction,backtracking,eta-sweep,contraction,momentum,sgd-variance,sgd-schedule,newton}`).

## Figure specs (includes are in the .md; SVGs NOT created — for the figure pass)
**img/mdl-opt-momentum-damping.svg** (`:label:fig_mdl-opt-momentum-damping`).
One panel, house style, equal aspect. Gray contour ellipses of
f(x,y)=½(x²+10y²), x∈[−2.2,2.2], y∈[−1.1,1.1]; star at origin. Three heavy-ball
trajectories from x₀=(−2, 1), all with η=(2/(1+√10))²≈0.231, ~30 steps, dots +
thin lines: β=0.05 over-damped (GD-like zig-zag crawl along the slow axis),
β=((√10−1)/(√10+1))²≈0.27 critically tuned (cuts almost straight to the
minimum, fastest), β=0.85 under-damped (overshoots past the minimum and loops
around it). Label each curve "over-damped β=0.05" / "critical β*≈0.27" /
"under-damped β=0.85". Implementer should simulate to place labels.

**img/mdl-opt-sgd-noise-ball.svg** (`:label:fig_mdl-opt-sgd-noise-ball`).
One panel: circular contours of f=½‖x‖² (λ=1). From x₀=(−2, 1.6): (a) GD path
(η=0.15, smooth curve into the origin, color C0); (b) SGD path, same η, gradient
+ N(0, σ²I) noise, σ=0.6, seeded, ~80 steps — line for the descent transient,
scatter dots for the last ~50 iterates rattling near the origin (C1). Dashed
circle of radius √(ησ²/(2λ)) ≈ 0.16·(visual scale ~3× for legibility is fine if
both circles scale together) annotated "noise ball ∝ √η"; second dashed circle
at radius/√2 annotated "η/2" to show the ball shrinking with step size.

## BibTeX to append to d2l.bib (cited in the .md; not added by me)
```bibtex
@InProceedings{	  Cohen.Kaur.Li.ea.2021,
  title		= {Gradient descent on neural networks typically occurs at
		  the edge of stability},
  author	= {Cohen, Jeremy M and Kaur, Simran and Li, Yuanzhi and
		  Kolter, J Zico and Talwalkar, Ameet},
  booktitle	= {International Conference on Learning Representations},
  year		= {2021}
}
@Article{	  Lessard.Recht.Packard.2016,
  title		= {Analysis and design of optimization algorithms via
		  integral quadratic constraints},
  author	= {Lessard, Laurent and Recht, Benjamin and Packard, Andrew},
  journal	= {SIAM Journal on Optimization},
  volume	= {26},
  number	= {1},
  pages		= {57--95},
  year		= {2016}
}
@Article{	  Robbins.Monro.1951,
  title		= {A stochastic approximation method},
  author	= {Robbins, Herbert and Monro, Sutton},
  journal	= {The Annals of Mathematical Statistics},
  volume	= {22},
  number	= {3},
  pages		= {400--407},
  year		= {1951}
}
```
Existing keys used: Boyd.Vandenberghe.2004, Nesterov.2018, Polyak.1964,
Goodfellow.Bengio.Courville.2016, Sutskever.Martens.Dahl.ea.2013, Bottou.2010,
Liu.Nocedal.1989, Kingma.Ba.2014.

## Verification log
All 9 cells extracted cumulatively per framework and run end-to-end:
pytorch 3 s, tensorflow 5 s, jax 5 s, mxnet 4 s — all exit 0 (MPLBACKEND=Agg).
Printed output byte-identical for pytorch/tensorflow/jax; mxnet (numpy 1.26 vs
2.4) differs in one digit of the *final* Newton gradient norm (3.2e-17 vs
3.0e-17) — prose quotes only the first five values, identical everywhere.
`lint_source.py` and `lint_source.py --corpus`: exit 0, no warnings.

## Deviations from plan
1. Plan's 6 `##` compressed to 5: "Why Not Newton" is a `###` coda inside
   Stochastic Gradients (per task instruction to stay at 3–5).
2. 2/L ceiling + edge-of-stability remark live in the Quadratic section (where
   per-mode factors make the ceiling exact), not in GD-and-Smoothness; Armijo
   stayed in GD-and-Smoothness; convex-rate statements close the Quadratic
   section. Same content, better flow.
3. 1154 lines vs the ~800–950 target — matches sibling 24.3's density.
4. GD-vs-momentum-vs-Nesterov is a printed table (no plot); the one data plot
   is the SGD fixed-η-vs-decay log-log gap curve.
5. For wave-2 reconciliation: eqlabels `eq_mdl-opt-rate-convex` /
   `eq_mdl-opt-rate-strongly-convex` are the *statements* 24.2 proves — if the
   convexity writer also labeled them, dedupe (corpus lint clean as of now).
6. Discussions link uses sibling slug style:
   https://d2l.discourse.group/t/gradient-based-optimization
