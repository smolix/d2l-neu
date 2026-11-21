# Further Reading
:label:`sec_diffusion-further-reading`

This section collects monographs, tutorials, reference implementations, and primary literature extending the methods developed in this chapter.

## Monographs, Courses and Code

- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) :cite:`Lai.Song.Kim.ea.2025` derives diffusion models along the variational, score-based, and flow-based routes and shows where they meet.
- The [notes, lectures, and labs](https://diffusion.csail.mit.edu/) of MIT's course 6.S184 :cite:`Holderrieth.Erives.2025` cover flow matching, latent diffusion transformers, and discrete diffusion.
- [*Diffusion Meets Flow Matching*](https://diffusionflow.github.io/) :cite:`Gao.Hoogeboom.Heek.ea.2025` illustrates with interactive figures that DDIM and the Euler flow-matching sampler are equivalent under a schedule rescaling.
- The [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) :cite:`Lipman.Havasi.Holderrieth.ea.2024` and its [library](https://github.com/facebookresearch/flow_matching) provide full proofs and reference code for :numref:`sec_diffusion-gaussian-paths`.
- [*Step-by-Step Diffusion*](https://arxiv.org/abs/2406.08929) :cite:`Nakkiran.Bradley.Zhou.ea.2025` derives deterministic and stochastic samplers (:numref:`sec_diffusion-integration` and :numref:`sec_diffusion-samplers`) from elementary probability without stochastic calculus.
- Sander Dieleman's [blog](https://sander.ai/) examines frequency-space denoising :cite:`Dieleman.2024`, latent representations :cite:`Dieleman.2025`, flow maps :cite:`Dieleman.2026`, and continuous versus discrete diffusion for language :cite:`Dieleman.2026b`.
- Reference implementations include the [official DDPM repository](https://github.com/hojonathanho/diffusion) :cite:`ho2020denoising`, the [Score SDE repository](https://github.com/yang-song/score_sde_pytorch) :cite:`song2021score`, and the [MDLM repository](https://github.com/kuleshov-group/mdlm) :cite:`Sahoo.Arriola.Schiff.ea.2024`.
- The [*Deep Generative Models* course](https://kuleshov-group.github.io/dgm-website/) :cite:`Kuleshov.2023`, Yang Song's [score-based tutorial](https://yang-song.net/blog/2021/score/) :cite:`Song.2021`, Lilian Weng's [*What Are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) :cite:`Weng.2021`, and :citet:`Luo.2022` provide accessible reviews of score matching and variational diffusion.

Five primary papers are particularly foundational: :citet:`Karras.Aittala.Aila.ea.2022` (EDM) disentangle the sampler, preconditioning, and noise schedule; :citet:`Esser.Kulal.Blattmann.ea.2024` (Stable Diffusion 3) combine rectified flow with multimodal diffusion transformers; :citet:`Lu.Zhou.Bao.ea.2025` (DPM-Solver++) analyze high-order solvers under guidance; :citet:`Lu.Song.2025` (sCM) stabilize continuous-time consistency models at scale; and :citet:`Sahoo.Arriola.Schiff.ea.2024` (MDLM) give a clean formulation of masked discrete diffusion. Neighboring topics appear in :numref:`sec_gan-further-reading` (adversarial and distribution-matching distillation, FID evaluation) and :numref:`sec_mdl-dynamics-further-reading` (exponential integrators, score-error bounds, and Schrödinger bridges).

## Energy-Based Models and Score Matching

**Training energy-based models.** :citet:`Song.Kingma.2021` survey maximum-likelihood training for unnormalized densities. Contrastive divergence initializes short MCMC chains at the data :cite:`Hinton.2002`, which prevents mixing across separated modes (:numref:`sec_diffusion-denoising`), whereas persistent contrastive divergence maintains chain states across parameter updates :cite:`Tieleman.2008,Du.Mordatch.2019`.

**Noise-contrastive estimation.** Rather than eliminating the normalizer by differentiation, :citet:`Gutmann.Hyvarinen.2012` treat it as a learnable parameter and train the model via logistic regression against a known noise distribution.

**Limits of score matching.** The blindness of the Fisher divergence to mode weights across low-density valleys (:numref:`sec_diffusion-denoising`) is analyzed by :citet:`Wenliang.Kanagawa.2020`, and :citet:`Koehler.Heckett.Risteski.2022` bound the statistical efficiency of score matching via isoperimetric constants.

**Noise-ladder heuristics.** :citet:`Song.Ermon.2020` gave four practical rules for geometric noise ladders (:numref:`sec_diffusion-gaussian-paths`): setting the maximum noise scale to the maximum pairwise training distance, choosing adjacent levels close enough that high-probability shells overlap (Exercise :exref:`ex_gaussian-paths-ladder-spacing` of :numref:`sec_diffusion-gaussian-paths`), tuning the Langevin step size, and applying an exponential moving average to the weights.

**Empirical Bayes.** Tweedie's formula originated in empirical Bayes estimation :cite:`Robbins.1956`, and :citet:`Efron.2011` popularized its use for correcting selection bias in high-dimensional estimation.

## Image Networks and Guidance

**U-Net architectures at scale.** :citet:`Dhariwal.Nichol.2021` (ADM) augmented the DDPM U-Net with multi-resolution attention, residual resampling, and adaptive group normalization. :citet:`Karras.Aittala.Lehtinen.ea.2024` (EDM2) redesigned the U-Net to preserve activation and weight magnitudes and introduced post-hoc EMA reconstruction from periodic checkpoints.

**Guidance without saturation.** Large guidance scales can saturate pixel intensities (Exercise :exref:`ex_image-diffusion-thresholding` of :numref:`sec_diffusion-images`). :citet:`Sadat.Hilliges.Weber.2025` decompose the guidance update into components parallel and orthogonal to the conditional prediction, damping the parallel component to prevent saturation.

**Spatial conditioning.** ControlNet :cite:`Zhang.Rao.Agrawala.2023` conditions a pretrained text-to-image model on spatial maps (edges, depth, pose) using a trainable copy of the encoder connected via zero-initialized layers.

## Evaluation, Memorization and Generalization

**Prompt alignment.** Because Fréchet distance measures distributional similarity rather than prompt adherence, CLIPScore :cite:`Hessel.Holtzman.Forbes.ea.2021` measures image-caption similarity in a joint embedding, while GenEval :cite:`Ghosh.Hajishirzi.Schmidt.2023` verifies object counts, positions, and attributes with an object detector.

**Memorization.** :citet:`Carlini.Hayes.Nasr.ea.2023` extract near-duplicates of training images from large text-to-image models, and :citet:`Biroli.Bonnaire.deBortoli.ea.2024` identify distinct dynamical regimes in the reverse process corresponding to class commitment (speciation) and training-point memorization (collapse).

**Generalization of denoisers.** Although the exact empirical minimizer of the denoising objective memorizes a finite training set, :citet:`Bertrand.Gagneux.Massias.ea.2025` show that target noise alone does not explain generalization. Denoisers trained on disjoint subsets learn nearly identical score fields :cite:`Kadkhodaie.Guth.Simoncelli.ea.2024`, largely governed by convolutional locality and equivariance :cite:`Kamb.Ganguli.2025`; see :citet:`Massias.Bertrand.2026` for a survey.

## Samplers, Editing and Inverse Problems

**Solvers for guided sampling.** Large guidance scales destabilize high-order noise-prediction solvers (Exercise :exref:`ex_samplers-dpm-solver` of :numref:`sec_diffusion-samplers`). DPM-Solver++ :cite:`Lu.Zhou.Bao.ea.2025` instead parameterizes multistep updates in terms of the clean-image estimate, and UniPC :cite:`Zhao.Bai.Rao.ea.2023` adds a zero-overhead corrector step.

**Stochasticity in ODE solvers.** :citet:`Karras.Aittala.Aila.ea.2022` inject a small amount of noise before each Heun step, while Restart sampling :cite:`Xu.Deng.Cheng.ea.2023` alternates deterministic ODE integration with larger forward-noise jumps to contract accumulated discretization error.

**Exact inversion.** Because DDIM inversion (:numref:`sec_diffusion-samplers`) accumulates discretization error under classifier-free guidance, null-text inversion :cite:`Mokady.Hertz.Aberman.ea.2023` optimizes the unconditional embedding around an unguided trajectory, while EDICT :cite:`Wallace.Gokul.Naik.2023` uses coupled affine transformations for exact invertibility.

**Inverse problems without retraining.** For linear inverse problems such as deblurring or super-resolution, Diffusion Posterior Sampling (DPS) :cite:`Chung.Kim.Mccann.ea.2023` approximates the measurement likelihood gradient at the posterior mean $\hat{\mathbf{x}}_0(\mathbf{x}_t)$, and DDRM :cite:`Kawar.Elad.Ermon.ea.2022` operates in the spectral domain of the measurement operator.

## Schedules, Weighting and Likelihood

**Learned reverse variances.** Alongside the cosine schedule (:numref:`sec_diffusion-gaussian-paths`), :citet:`Nichol.Dhariwal.2021` parameterize the reverse variance as a learned interpolation between $\tilde\beta_k$ and $\beta_k$ trained on the variational bound.

**Min-SNR weighting.** :citet:`Hang.Gu.Li.ea.2023` weight the clean-image squared error by $\min(\mathrm{SNR}_t, \gamma)$ (equivalently weighting the noise error by $\min(1, \gamma e^{-\lambda})$), preventing low-noise steps from dominating gradient updates.

**Likelihood optimization.** Variational diffusion models (VDM) :cite:`Kingma.Salimans.Poole.ea.2021` optimize the noise schedule to minimize the variance of the ELBO estimator; :citet:`Song.Durkan.Murray.ea.2021` derive the likelihood-weighting schedule for continuous-time score matching; and :citet:`Zheng.Lu.Chen.ea.2023` improve exact likelihood training and evaluation for the probability-flow ODE.

## Latents, Text and Video

**Text conditioning.** Large text-to-image models condition on frozen CLIP :cite:`radford2021learning` or T5 :cite:`raffel2020exploring` text encoders via cross-attention (:numref:`sec_diffusion-latent`) or joint image-text self-attention :cite:`Esser.Kulal.Blattmann.ea.2024`; :citet:`saharia2022photorealistic` showed that scaling the text encoder improves prompt alignment more than scaling the U-Net.

**Representation alignment and autoencoders.** REPA :cite:`Yu.Kwak.Jang.ea.2025` accelerates latent DiT training by aligning intermediate hidden states with a pretrained vision encoder. Deep Compression Autoencoders (DC-AE) :cite:`Chen.Cai.Chen.ea.2025` increase spatial downsampling beyond $8\times$, while representation autoencoders :cite:`Zheng.Ma.Tong.ea.2026` pair a frozen semantic encoder (such as DINO) with a learned decoder.

**Video generation.** Video diffusion models apply 3D spatiotemporal autoencoders and factorized or joint attention over space and time :cite:`Ho.Salimans.Gritsenko.ea.2022,Blattmann.Dockhorn.Kulal.ea.2023,Polyak.Zohar.Brown.ea.2024`.

## Few-Step Generation

**Progressive and guidance distillation.** Progressive distillation :cite:`Salimans.Ho.2022` iteratively halves the step count of a teacher and introduced $\mathbf{v}$-prediction for numerical stability at low SNR. :citet:`Meng.Rombach.Gao.ea.2023` distill classifier-free guidance into a single pass conditioned on the guidance scale $\gamma$.

**Consistency and flow maps at scale.** Improved consistency training (iCT) :cite:`Song.Dhariwal.2024` removes the teacher in discrete time, while sCM :cite:`Lu.Song.2025` stabilizes continuous-time consistency training at ImageNet scale using trigonometric paths and tangent normalization. Improved MeanFlow :cite:`Geng.Lu.Wu.ea.2026` regresses directly onto instantaneous velocities, and :citet:`Boffi.Albergo.VandenEijnden.2025` analyze self-distillation across Eulerian, Lagrangian, and compositional flow-map identities.

## Discrete Diffusion

**Alternative kernels and remasking.** Discrete flow matching :cite:`Gat.Remez.Shaul.ea.2024` formulates discrete generation via probability paths on the simplex. Uniform corruption kernels allow tokens to be revised at any step, and :citet:`Sahoo.Deschenaux.Gokaslan.ea.2025` connect uniform discrete diffusion to thresholded Gaussian diffusion. ReMDM :cite:`Wang.Schiff.Sahoo.ea.2025` introduces remasking into absorbing diffusion, and :citet:`Kim.Shah.Kontonis.ea.2025` demonstrate the strong effect of confidence-ordered unmasking on reasoning tasks such as Sudoku.

**Diffusion language models.** LLaDA :cite:`Nie.Zhu.You.ea.2025` trains an 8B masked diffusion language model from scratch, while Dream :cite:`Ye.Xie.Zheng.ea.2025` initializes from an autoregressive checkpoint. Block diffusion :cite:`Arriola.Gokaslan.Chiu.ea.2025` combines autoregressive generation across blocks with parallel diffusion within blocks, and Fast-dLLM :cite:`Wu.Zhang.Xue.ea.2026` adds KV-cache reuse (:numref:`sec_lm_architectures_fr`). :citet:`Zheng.Chen.Mao.ea.2025` caution that 32-bit Gumbel-max sampling artificially lowers sampling temperature and inflates reported generative perplexity.

## Post-Training and Applications Beyond Vision and Text

**Reinforcement learning and preference optimization.** DDPO :cite:`Black.Janner.Du.ea.2024` and Flow-GRPO :cite:`Liu.Liu.Liang.ea.2025` apply policy-gradient updates (:numref:`sec_rl_sequences`) to the stochastic reverse transitions of diffusion and flow models. Diffusion-DPO :cite:`Wallace.Dang.Rafailov.ea.2024` adapts direct preference optimization by substituting the diffusion ELBO for the intractable model log-likelihood.

**Audio, robotics and proteins.** DiffWave :cite:`Kong.Ping.Huang.ea.2021` applies diffusion to raw audio waveforms. Diffusion policies :cite:`Chi.Feng.Du.ea.2023` model multimodal robot action trajectories conditioned on visual observations, avoiding the mode-averaging failure of direct MSE policy regression. Multimodal protein models :cite:`Campbell.Yim.Barzilay.ea.2024` jointly diffuse continuous 3D atomic coordinates and discrete amino-acid sequences.
