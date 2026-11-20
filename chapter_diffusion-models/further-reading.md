# Further Reading
:label:`sec_diffusion-further-reading`

You have now trained every part of a current image generator except its text encoder, at a size where one training run takes minutes on one GPU. The papers behind the full-size systems are many and young, and some of them will not last. Each entry below states what a later work changes relative to the chapter; nothing in the book depends on it.

Five papers repay a careful reading more than the rest. Before changing anything in a real training run, read :citet:`Karras.Aittala.Aila.ea.2022`, who separate the choices of :numref:`sec_diffusion-samplers` and :numref:`sec_diffusion-design-space` into independent knobs and set each by experiment. For what a current text-to-image system is, read :citet:`Esser.Kulal.Blattmann.ea.2024`: rectified flow, a transformer that reads image and caption together, and a scaling study in one paper. For ten to twenty-five evaluations, read DPM-Solver++ :cite:`Lu.Zhou.Bao.ea.2025`, which explains why guidance breaks high-order solvers. For one or two evaluations, read :citet:`Lu.Song.2025`, who list what consistency training needed before it worked at ImageNet scale. For tokens, read :citet:`Sahoo.Arriola.Schiff.ea.2024`, the plainest statement of the masked objective on which the language models below build. The rest of the page is for a reader with a particular question.

Two other pages carry neighboring topics. The GAN chapter's further reading (:numref:`sec_gan-further-reading`) has distillation by distribution matching, adversarial post-training and the pitfalls of FID; that of the dynamics appendix (:numref:`sec_mdl-dynamics-further-reading`) has exponential integrators in general, bounds on the sampling error in terms of the score error, Schrödinger bridges and optimal couplings.

## Energy-Based Models and Score Matching

**Training energy-based models.** :citet:`Song.Kingma.2021` survey the ways to fit a density known only up to its normalizer. Maximum likelihood needs samples from the model, and its variants differ in where their chains start. Contrastive divergence restarts them at the data :cite:`Hinton.2002`. A short chain that never leaves its starting mode is a stuck chain of :numref:`sec_diffusion-denoising`, blind to how much weight each mode deserves. Persistent chains keep their states between updates and so have time to travel between modes :cite:`Tieleman.2008`; :citet:`Du.Mordatch.2019` scaled them to images with a replay buffer.

**Noise-contrastive estimation.** Score matching removes the normalizer by differentiating it away. :citet:`Gutmann.Hyvarinen.2012` keep it as one more parameter and fit the model by logistic regression of data against samples from a known noise distribution: the classifier of :numref:`sec_basic_gan`, with fixed noise in place of a generator.

**Limits of score matching.** The blindness to mode weights that :numref:`sec_diffusion-denoising` measured is proved for well-separated components by :citet:`Wenliang.Kanagawa.2020`. :citet:`Koehler.Heckett.Risteski.2022` turn it into a rate, tying the statistical efficiency of score matching to isoperimetric constants, which are poor exactly when the modes are far apart.

**Rules for a noise ladder.** The ladder of :numref:`sec_diffusion-gaussian-paths` ran geometrically between ends picked by hand, 3 and 0.05. :citet:`Song.Ermon.2020` gave four rules for images: a top level as large as the largest distance between two training points; levels close enough that a sample of one is typical of the next, 232 of them for CIFAR-10 (Exercise :exref:`ex_gaussian-paths-ladder-spacing` of :numref:`sec_diffusion-gaussian-paths`); a step size that leaves the last level's samples with the right variance; and a moving average of the weights. A path with its sampler absorbed the first three. The top level and the spacing became the sampler's grid, which starts at 80 in :numref:`sec_diffusion-samplers`, and the step size left with the Langevin steps. The fourth stayed.

**Empirical Bayes.** Tweedie's formula was born in empirical Bayes, where a prior is estimated from the very noisy observations it is applied to :cite:`Robbins.1956`, and :citet:`Efron.2011` used it to correct selection bias: the largest of many noisy estimates of an effect is too large, and the formula says by how much.

## Image Networks and Guidance

**The U-Net at ImageNet scale.** :citet:`Dhariwal.Nichol.2021` ablated the U-Net of :citet:`ho2020denoising` and kept attention at several resolutions, residual resampling blocks and adaptive group normalization. Their network, ADM, was the usual baseline until transformer backbones (:numref:`sec_diffusion-latent`) and the U-Net of :citet:`Karras.Aittala.Lehtinen.ea.2024` replaced it. The latter keeps the magnitudes of activations and weights constant and rebuilds a moving average of any length after training from periodic snapshots of two stored averages, so the half-life that :numref:`sec_diffusion-images` fixed in advance can be chosen afterwards.

**Guidance without saturation.** Large guidance scales saturate colors; Exercise :exref:`ex_image-diffusion-thresholding` of :numref:`sec_diffusion-images` clips the predicted clean image against it. :citet:`Sadat.Hilliges.Weber.2025` split the guidance update instead, into a part parallel to the conditional prediction and a part orthogonal to it. The first does most of the saturating, the second most of the improving, so damping the first allows larger scales at almost no cost.

**Spatial conditions.** ControlNet :cite:`Zhang.Rao.Agrawala.2023` conditions a trained text-to-image model on edges, depth or pose through a trained copy of its encoder, joined to the frozen model by layers initialized at zero, so that training starts from the unchanged model.

## Evaluation, Memorization and Generalization

**Agreement with a caption.** The FD of :numref:`sec_diffusion-images` asks whether generated images look like the data, not whether each one shows what its caption asks for. CLIPScore :cite:`Hessel.Holtzman.Forbes.ea.2021` compares image and caption in a joint embedding. GenEval :cite:`Ghosh.Hajishirzi.Schmidt.2023` finds the objects a caption names with an object detector, checks their counts and positions, and classifies their colors with CLIP.

**Memorization.** :citet:`Carlini.Hayes.Nasr.ea.2023` extract near-copies of training images from text-to-image models. :citet:`Biroli.Bonnaire.deBortoli.ea.2024` find two times in the reverse process: speciation, when a sample commits to a class, and collapse, when it commits to one training point.

**Generalization of denoisers.** This is the deepest question on the page. The exact optimum of the regression on a finite training set reproduces the training points, and :citet:`Bertrand.Gagneux.Massias.ea.2025` show that the noise in the target is not what prevents this. Denoisers trained on disjoint halves of a large set learn nearly the same score :cite:`Kadkhodaie.Guth.Simoncelli.ea.2024`, and locality and equivariance alone predict much of a convolutional denoiser's output :cite:`Kamb.Ganguli.2025`. So far the evidence says that a network generalizes because its inductive biases keep it away from the exact optimum. :citet:`Massias.Bertrand.2026` survey the question.

## Samplers, Editing and Inverse Problems

**Solvers for guided sampling.** Large guidance scales make the high-order solvers of the original DPM-Solver (Exercise :exref:`ex_samplers-dpm-solver` of :numref:`sec_diffusion-samplers`) unstable. DPM-Solver++ :cite:`Lu.Zhou.Bao.ea.2025` works with the clean-image estimate, which can be clipped to the range of the data, and uses multistep updates like those of `multistep_sample`, which stay stable at large scales. UniPC :cite:`Zhao.Bai.Rao.ea.2023` adds a corrector that raises the order by one without an extra network call.

**Noise inside a deterministic solver.** :citet:`Karras.Aittala.Aila.ea.2022` raise the noise level slightly before each Heun step; it helps on some datasets and hurts on others. Restart sampling :cite:`Xu.Deng.Cheng.ea.2023` alternates deterministic steps with one large jump back up the path by forward noise, which contracts the error accumulated so far.

**Exact inversion.** The round trip of `ddim_invert` in :numref:`sec_diffusion-samplers` is inexact, and under classifier-free guidance it breaks down, as :citet:`Mokady.Hertz.Aberman.ea.2023` found. Their null-text inversion therefore inverts without guidance and then tunes the empty caption's embedding until the guided sampler retraces the inversion, so that real photographs can be edited by text. EDICT :cite:`Wallace.Gokul.Naik.2023` avoids the error altogether by keeping two coupled copies of the state that invert each other exactly.

**Restoration without retraining.** The inpainting of :numref:`sec_diffusion-samplers` overwrote the known pixels at every step. A blurred or low-resolution copy has no pixels to overwrite. Bayes' rule instead adds the gradient of the measurement's log-likelihood to the score, as classifier guidance added a classifier's in :numref:`sec_diffusion-images`, and DPS :cite:`Chung.Kim.Mccann.ea.2023` evaluates that likelihood at the network's estimate $\hat{\mathbf{x}}_0(\mathbf{x}_t)$. DDRM :cite:`Kawar.Elad.Ermon.ea.2022` solves linear inverse problems through the singular value decomposition of the measurement operator.

## Schedules, Weighting and Likelihood

**Learned variances.** The cosine path of :numref:`sec_diffusion-gaussian-paths` came from :citet:`Nichol.Dhariwal.2021` together with a second change. Where :numref:`sec_diffusion-samplers` chose the variance of each reverse step from two fixed values, the forward step's and the posterior's, they learn an interpolation between the two from the bound. This improves likelihoods and lets sampling use far fewer steps.

**Choosing the training times.** :citet:`Esser.Kulal.Blattmann.ea.2024` compared time distributions for rectified flow on the latents of $256 \times 256$ images and found a logit-normal one, concentrated at intermediate noise, among the best. In the short pixel-space runs of :numref:`sec_diffusion-design-space` the logit-normal centered at $t = \tfrac12$ learned more slowly than uniform times, and one shifted half a unit toward the data caught up. The two experiments differ in resolution, latents, model size and training length, so they do not say which of these moves the best center; they do say that the center is a setting to tune, not a constant to copy.

**Weighting as multi-task learning.** :citet:`Hang.Gu.Li.ea.2023` treat the noise levels as tasks whose gradients conflict. They weight the clean-image error of each level by the smaller of its signal-to-noise ratio and a constant $\gamma$, which in the notation of :numref:`sec_diffusion-design-space` multiplies the noise error by $\min(1, \gamma e^{-\lambda})$. The clamp keeps the low-noise levels from dominating, and they report convergence several times faster.

**Likelihood.** Sample quality and the bound of :numref:`sec_diffusion-design-space` reward different weightings, and these papers take the bound's side. Variational diffusion models :cite:`Kingma.Salimans.Poole.ea.2021` learn the schedule to reduce the variance of the bound. :citet:`Song.Durkan.Murray.ea.2021` find the weighting under which score matching bounds the likelihood, and :citet:`Zheng.Lu.Chen.ea.2023` train and evaluate the likelihood of the probability-flow ODE.

## Latents, Text and Video

**Text encoders.** The frozen text encoder that feeds the cross-attention of :numref:`sec_diffusion-latent` is in practice the text half of CLIP :cite:`radford2021learning` or the encoder of T5 :cite:`raffel2020exploring`. The multimodal DiT of :citet:`Esser.Kulal.Blattmann.ea.2024` gives image and caption tokens separate weights within its one sequence. :citet:`saharia2022photorealistic` found that a larger text encoder improved their images more than a larger U-Net did.

**Faster training in latent space.** REPA :cite:`Yu.Kwak.Jang.ea.2025` aligns the hidden states of a latent transformer with those of a pretrained image encoder and reports the baseline's quality in a fraction of the training steps.

**Other autoencoders.** An autoencoder trained for reconstruction alone gives latents that decode well, but there are many of them and they need not be easy to model. DC-AE :cite:`Chen.Cai.Chen.ea.2025` compresses far more than the usual eightfold per side, which cuts the number of tokens. Representation autoencoders :cite:`Zheng.Ma.Tong.ea.2026` take a frozen pretrained encoder such as DINO, whose features already group images by content, and train only the decoder.

**Video.** A three-dimensional autoencoder compresses space and time, yet a few seconds of video still make tens of thousands of tokens. :citet:`Ho.Salimans.Gritsenko.ea.2022` factorize attention over space and time; :citet:`Blattmann.Dockhorn.Kulal.ea.2023` show how strongly the curation of the training videos changes the result; Movie Gen :cite:`Polyak.Zohar.Brown.ea.2024` trains one transformer on joint image and video latents.

## Few-Step Generation

**Progressive and guided distillation.** Progressive distillation :cite:`Salimans.Ho.2022` repeatedly halves a teacher's steps. It introduced $\mathbf{v}$-prediction, because a few-step student evaluates mostly at low signal-to-noise ratio, where a noise predictor's $\hat{\mathbf{x}}_0$ divides by a small $\alpha$. :citet:`Meng.Rombach.Gao.ea.2023` first distill classifier-free guidance into one network that takes the guidance scale as an input, halving the cost of each step, and then cut the steps.

**Consistency at scale.** Improved consistency training :cite:`Song.Dhariwal.2024` made training without a teacher competitive in discrete time. sCM :cite:`Lu.Song.2025` trains consistency models in continuous time at ImageNet scale, which needs more than the Eulerian identity of :numref:`sec_diffusion-few-step`: a path that rotates between data and noise, a normalized tangent term and a warm-up.

**Flow maps after MeanFlow.** Improved MeanFlow :cite:`Geng.Lu.Wu.ea.2026` regresses on the instantaneous velocity, which makes the target independent of the network, and takes the guidance scale as an input. :citet:`Boffi.Albergo.VandenEijnden.2025` train flow maps from scratch by self-distillation.

## Discrete Diffusion

**Other kernels and paths.** Discrete flow matching :cite:`Gat.Remez.Shaul.ea.2024` builds the models of :numref:`sec_diffusion-discrete` from probability paths. Under a uniform kernel every token can change at every step, so an early mistake can be revised. :citet:`Sahoo.Deschenaux.Gokaslan.ea.2025` obtain such a model as the argmax of a Gaussian diffusion of one-hot vectors, a link that carries consistency distillation over to tokens. ReMDM :cite:`Wang.Schiff.Sahoo.ea.2025` lets the masked sampler mask tokens again for the same reason. The order of revealing matters too. On Sudoku, a masked model that fills first the cells it is surest of solves about 90% of the puzzles instead of under 7% :cite:`Kim.Shah.Kontonis.ea.2025`.

**Language models.** LLaDA :cite:`Nie.Zhu.You.ea.2025` trains a masked model of 8 billion parameters from scratch, while Dream :cite:`Ye.Xie.Zheng.ea.2025` adapts an autoregressive one. Block diffusion :cite:`Arriola.Gokaslan.Chiu.ea.2025` generates blocks left to right and the tokens within a block by diffusion, and Fast-dLLM :cite:`Wu.Zhang.Xue.ea.2026` adds caching and parallel decoding without retraining; :numref:`sec_lm_architectures_fr` returns to these models.

**A caution on reported numbers.** :citet:`Zheng.Chen.Mao.ea.2025` show that drawing categories by Gumbel sampling in 32-bit floating point, the usual precision, lowers the sampler's effective temperature and flatters the generative perplexity reported in several papers.

## Post-Training by Reward and Preference

**Policy gradients.** A reward can replace the regression target, with the reverse steps as actions. DDPO :cite:`Black.Janner.Du.ea.2024` applies the clipped policy-gradient update of :numref:`sec_rl_sequences` to the Gaussian reverse steps, and Flow-GRPO :cite:`Liu.Liu.Liang.ea.2025` brings the group-relative update to flow models through a stochastic sampler with the ODE's marginals.

**Preferences.** Diffusion-DPO :cite:`Wallace.Dang.Rafailov.ea.2024` adapts direct preference optimization (:numref:`sec_rl_sequences`), built on the Bradley-Terry model of :numref:`sec_regularized`, by replacing the intractable log-likelihood of an image with the variational bound of :numref:`sec_diffusion-design-space`.

## Diffusion beyond Images and Text

Nothing in the chapter's regression assumed pixels. DiffWave :cite:`Kong.Ping.Huang.ea.2021` generates speech waveforms the same way, conditioned on a spectrogram or on nothing at all. A diffusion policy :cite:`Chi.Feng.Du.ea.2023` denoises a short sequence of robot actions conditioned on camera images. If the demonstrations pass an obstacle sometimes on the left and sometimes on the right, a policy regressed onto the action drives straight into it, just as one Euler step from noise lands on the data mean (:numref:`sec_diffusion-integration`); the diffusion policy commits to one side. :citet:`Campbell.Yim.Barzilay.ea.2024` generate a protein's sequence and its three-dimensional structure together, a discrete path and a continuous one under one clock.
