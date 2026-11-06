# Dynamics: Differential Equations and Generative Flows
:label:`chap_mdl-dynamics`

Every network in this book was trained by taking small steps downhill, and
a residual network adds a small update to its input at every layer. Both are
step-by-step solutions of a differential equation, an equation that says how
fast a state changes from where it is now. A diffusion model that turns
Gaussian noise into an image works the same way. It calls a network a few
dozen times; each call returns a direction, and the sample moves a short step
along it. How few steps can such a model take, and where does the direction
come from when all we have are example images?

Differential equations were invented to predict motion, from Newton's
planets to Einstein's explanation of the jitter of small particles suspended
in water. The second example is the one that matters here. A cloud of noisy
particles is unpredictable one particle at a time, yet its density changes
deterministically, and running that change backward turns noise into data.
The tools of the Basics part stop short of this. Backpropagation
differentiates a fixed computation, not the solution of an equation.
Probability describes a random variable at one time, not a density that
moves. And noise whose size grows like the square root of time breaks the
ordinary chain rule.

:numref:`sec_mdl-odes-solvers` asks when a differential equation has
exactly one solution, and how accurately and stably a computer can follow
it. :numref:`sec_mdl-neural-odes-cnf` asks how to train a model defined by a
differential equation and how such a model changes a probability density.
:numref:`sec_mdl-sdes` asks how to compute with noise, and how noise carries
any data distribution to a Gaussian.
:numref:`sec_mdl-fokker-planck-probability-flow` asks how the density of a
noisy cloud evolves, and finds that running it backward needs one unknown
field, the score. :numref:`sec_mdl-score-matching-diffusion-flow` asks how
to learn the score from samples and how to sample with it.
:numref:`sec_mdl-flow-matching-transport` asks how to learn a velocity
directly, and why straight paths need fewer steps.

We assume calculus, Gaussians and conditional expectation at the level of
the Basics part, and we leave out the measure-theoretic foundations of
stochastic calculus. Discrete diffusion for text and the engineering of
image generators are treated in :numref:`chap_diffusion`.

```toc
:maxdepth: 2

mdl-odes-solvers
mdl-neural-odes
mdl-sdes
mdl-fokker-planck-probability-flow
mdl-score-matching-diffusion-flow
mdl-flow-matching-transport
further-reading
```

## Resources and Further Reading {.unnumbered}

The following references accompany the material this chapter teaches;
topics it does not teach are collected in
:numref:`sec_mdl-dynamics-further-reading`.

**Books and lecture notes**

- [Nonlinear Dynamics and Chaos --- Steven Strogatz](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering): a geometric introduction to flows, fixed points and their stability, the material of :numref:`sec_mdl-odes-solvers`.
- [On Neural Differential Equations --- Patrick Kidger (2022)](https://arxiv.org/abs/2202.02435): the book-length treatment of neural ODEs, controlled and stochastic differential equations, with the solver and adjoint machinery of :numref:`sec_mdl-neural-odes-cnf` laid out in full.
- [Applied Stochastic Differential Equations --- Särkkä & Solin](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf): an application-minded treatment of Itô calculus, SDEs and the Fokker--Planck equation for :numref:`sec_mdl-sdes` and :numref:`sec_mdl-fokker-planck-probability-flow`; the authors provide the full PDF.
- [Stochastic Differential Equations: An Introduction with Applications --- Bernt Øksendal](https://link.springer.com/book/10.1007/978-3-642-14394-6): a rigorous reference for Brownian motion, the Itô integral and existence theory, behind :numref:`sec_mdl-sdes`.
- [Numerical Solution of Stochastic Differential Equations --- Kloeden & Platen](https://link.springer.com/book/10.1007/978-3-662-12616-5): the reference on Euler--Maruyama, Milstein and higher-order schemes and their strong and weak orders, for the simulation part of :numref:`sec_mdl-sdes`.
- [MIT 6.S184 Lecture Notes: Flow Matching and Diffusion Models --- Holderrieth & Erives](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf): self-contained notes that derive diffusion and flow matching from SDEs and the continuity equation, companions to :numref:`sec_mdl-score-matching-diffusion-flow` and :numref:`sec_mdl-flow-matching-transport`.

**Courses and tutorials**

- [MIT 6.S184: Generative AI with Stochastic Differential Equations](https://diffusion.csail.mit.edu/): lectures, slides and labs that build a diffusion model from first principles, following the route of :numref:`sec_mdl-fokker-planck-probability-flow` to :numref:`sec_mdl-flow-matching-transport`; openly licensed.
- [Understanding Diffusion Models: A Unified Perspective --- Calvin Luo (2022)](https://arxiv.org/abs/2208.11970): derives the variational and score-based views of diffusion side by side, for :numref:`sec_mdl-score-matching-diffusion-flow`.
- [Flow Matching Guide and Code --- Lipman et al. (2024)](https://arxiv.org/abs/2412.06264): a detailed reference on flow matching for :numref:`sec_mdl-flow-matching-transport`, paired with the [`facebookresearch/flow_matching`](https://github.com/facebookresearch/flow_matching) PyTorch library. It runs time from noise at $0$ to data at $1$, as do the MIT notes and the flow-matching papers below; this chapter runs it the other way (:numref:`fig_mdl-dyn-time-conventions`).

**Foundational papers**

- [Neural Ordinary Differential Equations --- Chen et al. (NeurIPS 2018)](https://arxiv.org/abs/1806.07366): parameterizes a network's dynamics as an ODE and trains it with the adjoint method, as in :numref:`sec_mdl-neural-odes-cnf`.
- [Generative Modeling by Estimating Gradients of the Data Distribution --- Song & Ermon (NeurIPS 2019)](https://arxiv.org/abs/1907.05600): score-based generation with denoising score matching and annealed Langevin sampling, for :numref:`sec_mdl-score-matching-diffusion-flow`.
- [Denoising Diffusion Probabilistic Models --- Ho, Jain & Abbeel (2020)](https://arxiv.org/abs/2006.11239): the discrete-time DDPM formulation that :numref:`sec_mdl-score-matching-diffusion-flow` reads as a discretized SDE.
- [Score-Based Generative Modeling through SDEs --- Song et al. (ICLR 2021)](https://arxiv.org/abs/2011.13456): unifies score matching and diffusion as forward and reverse SDEs and introduces the probability-flow ODE of :numref:`sec_mdl-fokker-planck-probability-flow`.
- [Denoising Diffusion Implicit Models --- Song, Meng & Ermon (ICLR 2021)](https://arxiv.org/abs/2010.02502): the deterministic sampler of :numref:`sec_mdl-score-matching-diffusion-flow`, which reuses a trained DDPM with fewer steps.
- [Flow Matching for Generative Modeling --- Lipman et al. (2022)](https://arxiv.org/abs/2210.02747): simulation-free training of continuous flows by regressing conditional vector fields, the method of :numref:`sec_mdl-flow-matching-transport`.
- [Flow Straight and Fast: Rectified Flow --- Liu, Gong & Liu (2022)](https://arxiv.org/abs/2209.03003): straight interpolation paths and reflow for few-step sampling, for :numref:`sec_mdl-flow-matching-transport`.

**Blogs and visual explainers**

- [Generative Modeling by Estimating Gradients of the Data Distribution --- Yang Song](https://yang-song.net/blog/2021/score/): the author's walkthrough connecting score matching, Langevin dynamics and the SDE picture of :numref:`sec_mdl-score-matching-diffusion-flow`, with figures.
- [What are Diffusion Models? --- Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): a regularly updated survey of diffusion from DDPM through score SDEs and guidance, for :numref:`sec_mdl-score-matching-diffusion-flow`.
- [Perspectives on Diffusion --- Sander Dieleman](https://sander.ai/2023/07/20/perspectives.html): shows how a diffusion model is at once a denoiser, a score estimator, a reverse-SDE solver and a flow, the dictionary of :numref:`sec_mdl-flow-matching-transport`.
- [The Annotated Diffusion Model --- Hugging Face](https://huggingface.co/blog/annotated-diffusion): a line-by-line PyTorch implementation of DDPM that connects the mathematics of :numref:`sec_mdl-score-matching-diffusion-flow` to runnable code.
