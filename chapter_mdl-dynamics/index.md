# Dynamics: Differential Equations and Generative Flows
:label:`chap_mdl-dynamics`

Many of the most powerful generative models are, at heart, differential
equations. This capstone chapter develops the dynamics needed to read them:
ordinary differential equations and their solvers (and the neural-ODE view of a
residual network), stochastic differential equations and the Itô calculus, the
Fokker--Planck equation and the probability-flow ODE that make diffusion
reversible, and finally score matching, diffusion, and flow matching---unified
as different choices of probability path, training objective, and sampler. It
draws on calculus (:numref:`chap_mdl-calculus`), probability
(:numref:`chap_mdl-probability-statistics`), optimization
(:numref:`chap_mdl-optimization`), and divergences
(:numref:`chap_mdl-information-theory`).

```toc
:maxdepth: 2

mdl-odes-solvers
mdl-sdes
mdl-fokker-planck-probability-flow
mdl-score-matching-diffusion-flow
```

## Resources and Further Reading

If this chapter sparked an interest in the dynamics behind modern generative
models, the following are the canonical places to go deeper.

**Books and lecture notes** (ODEs / SDEs)

- [Nonlinear Dynamics and Chaos — Steven Strogatz](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering) — the standard, geometric introduction to ODEs, flows, fixed points, and bifurcations; builds the intuition a vector field deserves.
- [Applied Stochastic Differential Equations — Särkkä & Solin](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf) — a free, application-minded treatment of Itô calculus, SDEs, and Fokker--Planck (full PDF released by the authors).
- [Stochastic Differential Equations: An Introduction with Applications — Bernt Øksendal](https://link.springer.com/book/10.1007/978-3-642-14394-6) — the classic rigorous reference for Brownian motion, the Itô integral, and the existence theory behind SDEs.
- [MIT 6.S184 Lecture Notes: Flow Matching and Diffusion Models — Holderrieth & Erives](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) — a self-contained set of notes deriving diffusion and flow matching from SDEs and the Fokker--Planck / continuity equation.

**Courses and tutorials**

- [MIT 6.S184: Generative AI with Stochastic Differential Equations (2026)](https://diffusion.csail.mit.edu/) — lectures, slides, and labs that build a latent diffusion model from first principles; openly licensed.
- [Flow Matching Guide and Code — Lipman et al. (Meta FAIR, 2024)](https://arxiv.org/abs/2412.06264) — the comprehensive, current reference on flow matching, paired with the [`facebookresearch/flow_matching`](https://github.com/facebookresearch/flow_matching) PyTorch library.
- [Understanding Diffusion Models: A Unified Perspective — Calvin Luo (2022)](https://arxiv.org/abs/2208.11970) — a careful tutorial deriving the variational (ELBO) and score-based views of diffusion side by side.

**Foundational papers** (diffusion and flow matching)

- [Neural Ordinary Differential Equations — Chen et al. (NeurIPS 2018)](https://arxiv.org/abs/1806.07366) — parameterizes a network's dynamics as an ODE and trains it with the adjoint method; the bridge from residual nets to continuous flows.
- [On Neural Differential Equations — Patrick Kidger (2022)](https://arxiv.org/abs/2202.02435) — the book-length treatment of neural ODEs, CDEs, and SDEs, with the solver and adjoint (discretize-then-optimize vs. optimize-then-discretize) machinery laid out cleanly.
- [Generative Modeling by Estimating Gradients of the Data Distribution — Song & Ermon (NeurIPS 2019)](https://arxiv.org/abs/1907.05600) — introduces score-based generation via denoising score matching and Langevin sampling.
- [Denoising Diffusion Probabilistic Models — Ho, Jain & Abbeel (2020)](https://arxiv.org/abs/2006.11239) — the DDPM paper that made discrete-time diffusion practical and high-quality.
- [Score-Based Generative Modeling through SDEs — Song et al. (ICLR 2021)](https://arxiv.org/abs/2011.13456) — unifies score matching and diffusion as forward/reverse SDEs and introduces the probability-flow ODE.
- [Denoising Diffusion Implicit Models — Song, Meng & Ermon (ICLR 2021)](https://arxiv.org/abs/2010.02502) — the deterministic sampler that reuses a trained DDPM with an order of magnitude fewer steps, plus the dial that interpolates back to ancestral sampling.
- [Elucidating the Design Space of Diffusion-Based Generative Models — Karras et al. (NeurIPS 2022)](https://arxiv.org/abs/2206.00364) — factors diffusion into orthogonal choices of schedule, preconditioning, parameterization, and sampler (Heun), and tunes each; the reference on making samplers fast.
- [Flow Matching for Generative Modeling — Lipman et al. (2022)](https://arxiv.org/abs/2210.02747) — simulation-free training of continuous flows by regressing conditional vector fields.
- [Flow Straight and Fast: Rectified Flow — Liu, Gong & Liu (2022)](https://arxiv.org/abs/2209.03003) — learns near-straight transport paths for fast, few-step sampling.
- [A Mathematical Perspective on Transformers — Geshkovski, Letrouit, Polyanskiy & Rigollet (2023)](https://arxiv.org/abs/2312.10794) — reads self-attention as an interacting particle system: tokens are particles on the sphere evolving under attention, and the theory characterizes their clustering in the long-time limit — this chapter's lens applied to transformers themselves.

**Blogs and visual explainers**

- [Generative Modeling by Estimating Gradients of the Data Distribution — Yang Song](https://yang-song.net/blog/2021/score/) — the author's own walkthrough connecting score matching, Langevin dynamics, and the SDE picture, with figures.
- [What are Diffusion Models? — Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — a thorough, regularly updated survey of diffusion from DDPM through score SDEs and guidance.
- [Perspectives on Diffusion — Sander Dieleman](https://sander.ai/2023/07/20/perspectives.html) — shows how diffusion models are simultaneously autoencoders, score predictors, reverse-SDE solvers, and flow models.
- [The Annotated Diffusion Model — Hugging Face](https://huggingface.co/blog/annotated-diffusion) — a line-by-line PyTorch reimplementation of DDPM, ideal for connecting the math to runnable code.
