# Further Reading
:label:`sec_diffusion-further-reading`

The chapter taught one idea on problems small enough to check: a network that predicts the clean point from a noisy one at every noise level, trained by regression and turned into a sampler. Each entry states what a later work changes; nothing in the book depends on it. Distillation by distribution matching, adversarial post-training and the pitfalls of FID are on :numref:`sec_gan-further-reading`; the design space in continuous time, Schrödinger bridges, optimal couplings and sliced score matching on :numref:`sec_mdl-dynamics-further-reading`.

## Origins and the Three Routes

**Diffusion from thermodynamics.** :citet:`sohl2015deep` introduced the model: a forward chain of small Gaussian steps from data to noise, and a reverse chain trained on a variational bound that predicted the mean and covariance of each step. :citet:`ho2020denoising` simplified it to the noise regression of :numref:`sec_diffusion-gaussian-paths`.

**Scores at many noise levels.** :citet:`song2019generative` estimate the score at a ladder of noise levels and descend it by annealed Langevin dynamics; :citet:`Song.Ermon.2020` gave rules for the ladder, and :citet:`song2021score` took it to a stochastic differential equation whose time reversal needs only the score.

**Velocities of a path.** Flow matching :cite:`Lipman.Chen.BenHamu.ea.2022`, rectified flow :cite:`Liu.Gong.Liu.2022` and stochastic interpolants :cite:`Albergo.Boffi.VandenEijnden.2023` regress the velocity of a prescribed path; interpolants allow any two endpoint distributions. :citet:`Lai.Song.Kim.ea.2025` derive all three routes and show where they coincide.

## Energy-Based Models and Score Matching

**Training energy-based models.** :citet:`Song.Kingma.2021` survey the ways to fit a density known up to its normalizer. The likelihood gradient needs samples from the model, and the methods differ in where their chains start: contrastive divergence restarts them at the data :cite:`Hinton.2002` and so cannot correct mode weights, persistent chains keep their states :cite:`Tieleman.2008`, and :citet:`Du.Mordatch.2019` scaled Langevin chains with a replay buffer to images. Short chains restarted from noise give a good sampler but a poor density :cite:`Nijkamp.Hill.Zhu.ea.2019`. :citet:`Grathwohl.Wang.Jacobsen.ea.2020` read a classifier's logits as the energy of inputs and labels, so one network classifies and generates.

**Noise-contrastive estimation.** :citet:`Gutmann.Hyvarinen.2012` fit an unnormalized model by logistic regression of data against samples from a known noise distribution, the classifier of :numref:`sec_basic_gan` with fixed noise in place of a generator.

**What score matching cannot see.** :citet:`Wenliang.Kanagawa.2020` show that the score is blind to the weights of well-separated components, as :numref:`sec_diffusion-denoising` measures, and :citet:`Koehler.Heckett.Risteski.2022` tie the efficiency of score matching to isoperimetric constants, which are poor when modes are far apart.

**Correcting and reusing Langevin steps.** An accept-reject test removes the step-size bias of the Langevin update :cite:`Roberts.Tweedie.1996` but needs density ratios, which a score network does not give. Stochastic-gradient Langevin dynamics runs the update on the posterior of a network's weights with minibatch gradients :cite:`Welling.Teh.2011`.

**Empirical Bayes.** Tweedie's formula comes from empirical Bayes, where a prior is estimated from the noisy observations it is applied to :cite:`Robbins.1956`; :citet:`Efron.2011` showed how the formula corrects selection bias, as in the largest of many noisy effect estimates.

## Schedules, Weighting and Networks

**Cosine schedule and learned variances.** :citet:`Nichol.Dhariwal.2021` destroy information more evenly across steps with a cosine schedule and learn each reverse step's variance from the bound, which improves likelihoods at fewer steps.

**Choosing the training levels.** :citet:`Esser.Kulal.Blattmann.ea.2024` find that a logit-normal distribution of training times, concentrated at intermediate noise, works best for rectified flow at every model size; later text-to-image models adopt it.

**Weighting as multi-task learning.** :citet:`Hang.Gu.Li.ea.2023` clamp the weight of each noise level at the minimum of its signal-to-noise ratio and a constant, one point in the weighting space of :numref:`sec_diffusion-design-space`, reported to converge several times faster.

**Zero terminal SNR.** Against the leftover signal of :numref:`sec_diffusion-design-space`, :citet:`Lin.Liu.Li.ea.2024` rescale the schedule to end at zero signal, which needs a target other than the noise.

**Resolution and averaging.** At higher resolution the same noise level destroys less of an image, so :citet:`Hoogeboom.Heek.Salimans.2023` shift the schedule toward more noise. :citet:`Karras.Aittala.Lehtinen.ea.2024` keep the magnitudes of activations and weights constant and store snapshots of two moving averages, from which an average of any half-life, fixed before training in :numref:`sec_diffusion-images`, is reconstructed afterwards.

**The U-Net at ImageNet scale.** :citet:`Dhariwal.Nichol.2021` ablated the U-Net of :citet:`ho2020denoising` and kept multi-resolution attention, residual resampling and adaptive group normalization, in which the time and class embedding set a scale as well as a shift. The result, ADM, is the usual baseline.

## Samplers

**Exponential integrators.** Dividing by $\alpha$ removes the part of the probability-flow ODE that is linear in $\mathbf{x}$. DPM-Solver :cite:`Lu.Zhou.Bao.ea.2022` approximates the remaining integral of the predicted noise by a polynomial in $\log \bar\sigma$; its first-order version is DDIM (exercise :exref:`ex_samplers-dpm-solver`). DPM-Solver++ :cite:`Lu.Zhou.Bao.ea.2025` applies the idea to the clean-image prediction and stays stable at large guidance scales, and UniPC :cite:`Zhao.Bai.Rao.ea.2023` adds a corrector that gains an order of accuracy without an extra network call.

**Noise inside a deterministic solver.** :citet:`Karras.Aittala.Aila.ea.2022` raise the noise level slightly before each Heun step, adding a stochastic sampler's correction to a second-order method; it helps on some datasets and hurts on others. Restart sampling :cite:`Xu.Deng.Cheng.ea.2023` alternates deterministic steps with one large jump back up the interval by forward noise, which contracts accumulated error.

**The optimal reverse variance.** :citet:`Bao.Li.Zhu.ea.2022` derive the likelihood-maximizing variance of each ancestral step in closed form from the score, which tightens the chain's bound where a step is large in log-SNR.

**Parallel sampling.** :citet:`Shih.Belkhale.Ermon.ea.2023` refine all steps of a guessed trajectory at once by Picard iteration, trading compute for latency.

## Guidance, Editing and Inverse Problems

**Better guidance.** Autoguidance :cite:`Karras.Aittala.Kynkaanniemi.ea.2024` guides a model with a smaller, less trained version of itself, which keeps variety. :citet:`Kynkaanniemi.Aittala.Karras.ea.2024` guide only at intermediate noise levels. Against the saturated colors of large guidance scales, dynamic thresholding :cite:`saharia2022photorealistic` clips the predicted clean image, and :citet:`Sadat.Hilliges.Weber.2025` down-weight the part of the update parallel to the conditional prediction.

**Inversion and editing.** The round trip of `ddim_invert` is inexact, more so under guidance. EDICT :cite:`Wallace.Gokul.Naik.2023` keeps two coupled copies of the state that invert each other exactly, and null-text inversion :cite:`Mokady.Hertz.Aberman.ea.2023` optimizes the unconditional text embedding so that the guided sampler retraces the inversion, which allows text-driven editing of real photographs. SDEdit :cite:`Meng.He.Song.ea.2022` noises a rough guide image partway and denoises it.

**Restoration without retraining.** RePaint :cite:`Lugmayr.Danelljan.Romero.ea.2022` inpaints by re-imposing the noised known pixels at every step; DDRM :cite:`Kawar.Elad.Ermon.ea.2022` solves linear inverse problems through the SVD of the measurement operator; DPS :cite:`Chung.Kim.Mccann.ea.2023` follows the gradient of the measurement likelihood at the predicted clean image. ControlNet :cite:`Zhang.Rao.Agrawala.2023` instead trains a copy of the encoder on edges, depth or pose.

## Latent Spaces and Backbones

**Latent systems.** SDXL :cite:`Podell.English.Lacey.ea.2024` scales the latent U-Net of :citet:`Rombach.Blattmann.Lorenz.ea.2022` with two text encoders; SD3 :cite:`Esser.Kulal.Blattmann.ea.2024` and FLUX.1 :cite:`BlackForestLabs.2024` are joint-attention transformers trained with rectified flow.

**Better latents.** REPA :cite:`Yu.Kwak.Jang.ea.2025` aligns the transformer's hidden states with a pretrained image encoder and trains many times faster. DC-AE :cite:`Chen.Cai.Chen.ea.2025` compresses far more than eightfold with sharp reconstructions, and representation autoencoders :cite:`Zheng.Ma.Tong.ea.2026` use a frozen pretrained encoder such as DINO.

**Transformers on pixels.** Two lines of work drop the autoencoder that limits the latent models of :numref:`sec_diffusion-latent`. Hourglass diffusion transformers :cite:`Crowson.Baumann.Birch.ea.2024` give the transformer the U-Net's multi-resolution shape, so its cost grows linearly with the pixels. JiT :cite:`Li.He.2026` keeps pixel patches of $16 \times 16$ or more and predicts the clean image, while the loss stays on the velocity.

## Few-Step Generation

**Distillation and consistency.** Progressive distillation :cite:`Salimans.Ho.2022` repeatedly halves a teacher's steps; it introduced $\mathbf{v}$-prediction because a few-step student spends most of its evaluations at low signal-to-noise ratio, where a noise predictor's $\hat{\mathbf{x}}_0$ divides by a small $\alpha$. Improved consistency training :cite:`Song.Dhariwal.2024` needs no teacher. sCM :cite:`Lu.Song.2025` trains consistency models in continuous time at ImageNet scale, which needs more than the identity of :numref:`sec_diffusion-few-step`: a path that rotates between data and noise, a normalized tangent term and a warm-up.

**General flow maps.** Consistency trajectory models :cite:`Kim.Lai.Liao.ea.2024` jump between any two times; shortcut models :cite:`Frans.Hafner.Levine.ea.2025` require one step of twice the size to equal two steps; improved MeanFlow :cite:`Geng.Lu.Wu.ea.2026` regresses on the instantaneous velocity and takes the guidance scale as an input. :citet:`Boffi.Albergo.VandenEijnden.2025` turn any distillation scheme into training from scratch.

## Likelihood

**Bounds and weightings.** Variational diffusion models :cite:`Kingma.Salimans.Poole.ea.2021` learn the schedule to reduce the variance of the bound. :citet:`Song.Durkan.Murray.ea.2021` find the weighting under which score matching bounds the likelihood, and :citet:`Lu.Zheng.Bao.ea.2022` and :citet:`Zheng.Lu.Chen.ea.2023` train and evaluate the likelihood of the probability-flow ODE. :citet:`Kingma.Gao.2023` show that every monotone weighting is the bound of :numref:`sec_diffusion-design-space` on noise-augmented data, and that monotone versions of popular weightings lose no sample quality.

## Evaluation, Memorization and Generalization

**Text–image agreement.** CLIPScore :cite:`Hessel.Holtzman.Forbes.ea.2021` compares image and caption in a joint embedding; GenEval :cite:`Ghosh.Hajishirzi.Schmidt.2023` checks counts, colors and positions with an object detector.

**Memorization.** :citet:`Carlini.Hayes.Nasr.ea.2023` extract near-copies of training images from text-to-image models, and :citet:`Somepalli.Singla.Goldblum.ea.2023` find replicated content by retrieval. :citet:`Biroli.Bonnaire.deBortoli.ea.2024` find two times in the reverse process: speciation, when a sample commits to a class, and collapse, when it commits to a training point.

**Why networks generalize.** The exact optimum of the regression on a finite set reproduces training points, and :citet:`Bertrand.Gagneux.Massias.ea.2025` show that the noisy target is not what prevents this. Denoisers trained on disjoint halves of a large set learn nearly the same score :cite:`Kadkhodaie.Guth.Simoncelli.ea.2024`, and locality and equivariance alone predict their outputs :cite:`Kamb.Ganguli.2025`. :citet:`Massias.Bertrand.2026` survey the question.

## Video

**Video diffusion.** :citet:`Ho.Salimans.Gritsenko.ea.2022` factorize attention over space and time; Stable Video Diffusion :cite:`Blattmann.Dockhorn.Kulal.ea.2023` shows that curating video data matters as much as the architecture; Movie Gen :cite:`Polyak.Zohar.Brown.ea.2024` trains one transformer on joint image and video latents.

## Discrete Diffusion and Diffusion Language Models

**Continuous time and ratios.** In continuous time the masked chain of :numref:`sec_diffusion-discrete` becomes a continuous-time Markov chain :cite:`Campbell.Benton.DeBortoli.ea.2022`, and the score becomes ratios $p_t(\mathbf{y})/p_t(\mathbf{x})$ between sequences that differ in one position. Concrete score matching :cite:`Meng.Choi.Song.ea.2022` and SEDD :cite:`Lou.Meng.Ermon.2024` learn these ratios; for the absorbing kernel a ratio is a time factor times the clean conditional, what the masked denoiser learns. Discrete flow matching :cite:`Gat.Remez.Shaul.ea.2024,Campbell.Yim.Barzilay.ea.2024` builds the same models from probability paths, including joint discrete and continuous data such as protein sequences and structures.

**The time input.** :citet:`Ou.Nie.Xue.ea.2025` show that the continuous-time bound equals the loss of an order-agnostic autoregressive model. :citet:`Zheng.Chen.Mao.ea.2025` show that sampling categories in low precision lowers the effective temperature and flatters reported generative perplexity.

**Uniform noise and revision.** Under a uniform kernel every token can change at every step: :citet:`Schiff.Sahoo.Phung.ea.2025` add guidance to such models, and :citet:`Sahoo.Deschenaux.Gokaslan.ea.2025` relate them to a Gaussian process and distill them to few steps. ReMDM :cite:`Wang.Schiff.Sahoo.ea.2025` adds remasking to the masked sampler so that early mistakes can be revised. On puzzles such as Sudoku, a reveal order chosen by the model helps substantially :cite:`Kim.Shah.Kontonis.ea.2025`, the other side of the mode-seeking measured in :numref:`sec_diffusion-discrete`.

**Scale.** Masked diffusion follows autoregressive scaling laws with a constant-factor compute gap :cite:`Nie.Zhu.Du.ea.2025` and overtakes autoregression under repeated data :cite:`Prabhudesai.Wu.Zadeh.ea.2025`. LLaDA :cite:`Nie.Zhu.You.ea.2025` trains an 8-billion-parameter model from scratch and decodes in blocks; Dream :cite:`Ye.Xie.Zheng.ea.2025` and DiffuLLaMA :cite:`Gong.Agarwal.Zhang.ea.2025` adapt autoregressive models instead. Block diffusion :cite:`Arriola.Gokaslan.Chiu.ea.2025` makes the blocks part of the model, interpolating between a GPT and a masked model, and Fast-dLLM :cite:`Wu.Zhang.Xue.ea.2026` adds caching and parallel decoding without retraining.

**Images and the debate.** MaskGIT :cite:`Chang.Zhang.Jiang.ea.2022` unmasks the codes of an image tokenizer in eight to twelve confidence-ordered steps, and Muse :cite:`Chang.Zhang.Barber.ea.2023` conditions it on text. :citet:`Dieleman.2026b` traces why masked diffusion displaced continuous diffusion for language and why continuous models are returning.

## Post-Training by Reward and Preference

**Policy gradients.** After pretraining, a reward can replace the regression target, with the reverse steps as actions. DDPO :cite:`Black.Janner.Du.ea.2024` applies the clipped policy-gradient update of :numref:`sec_rl_sequences` to the Gaussian reverse steps, and Flow-GRPO :cite:`Liu.Liu.Liang.ea.2025` brings the group-relative update to flow models through a stochastic sampler with the same marginals.

**Preferences.** Diffusion-DPO :cite:`Wallace.Dang.Rafailov.ea.2024` adapts direct preference optimization, built on the Bradley--Terry model of :numref:`sec_regularized`, by replacing the intractable log-likelihood with the variational bound of :numref:`sec_diffusion-design-space`.
