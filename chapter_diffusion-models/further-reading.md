# Further Reading
:label:`sec_diffusion-further-reading`

The chapter taught one idea on problems small enough to check: a network that predicts the clean point from a noisy one at every noise level, trained by regression and turned into a sampler. The field has built far more schedules, samplers, backbones, distillation methods and systems on this idea than the chapter needs. This page points to the ones a reader of the chapter is best placed to read next. Each entry states the problem it addresses, what it changes relative to the chapter, and why to read it; nothing in the book depends on it. Two neighboring pages hold entries that are not repeated here: :numref:`sec_gan-further-reading` for distillation by distribution matching, adversarial post-training and the pitfalls of FID, and :numref:`sec_mdl-dynamics-further-reading` for the diffusion design space in continuous time, Schrödinger bridges, stochastic interpolants, optimal couplings and sliced score matching.

## Origins and the Three Routes

**Diffusion from thermodynamics.** :citet:`sohl2015deep` introduced the model: a forward chain of many small Gaussian steps that turns data into noise, and a learned reverse chain trained on a variational bound. Its network predicted both the mean and the covariance of each reverse step. Read it to see the whole model before :citet:`ho2020denoising` simplified it to the noise regression of :numref:`sec_diffusion-gaussian-paths`.

**Scores at many noise levels.** :citet:`song2019generative` reached the same model from the score: one network estimates the score of the data at a ladder of noise levels, and annealed Langevin dynamics descends the ladder. :citet:`Song.Ermon.2020` gave rules for choosing the ladder, and :citet:`song2021score` took it to a continuum, a stochastic differential equation whose time reversal needs only the score.

**Velocities of a path.** Flow matching :cite:`Lipman.Chen.BenHamu.ea.2022`, rectified flow :cite:`Liu.Gong.Liu.2022` and stochastic interpolants :cite:`Albergo.Boffi.VandenEijnden.2023` skip the stochastic process and regress the velocity of a prescribed path between noise and data. The monograph of :citet:`Lai.Song.Kim.ea.2025` derives all three routes and shows where they coincide.

## Energy-Based Models and Score Matching

**Training energy-based models.** :citet:`Song.Kingma.2021` survey the ways to fit a density known only up to its normalizer: maximum likelihood with Markov chain Monte Carlo, score matching and its denoising form, and noise-contrastive estimation, the family of which :numref:`sec_diffusion-denoising` keeps one fact.

**Noise-contrastive estimation.** :citet:`Gutmann.Hyvarinen.2012` fit an unnormalized model by logistic regression between the data and samples from a known noise distribution, with the log-normalizer as one more parameter. The classifier is the one of :numref:`sec_basic_gan`, whose logit is a log density ratio, with a fixed noise distribution in place of a generator.

**What score matching cannot see.** :citet:`Wenliang.Kanagawa.2020` show that the score is blind to the weights of well-separated components, the phenomenon that :numref:`sec_diffusion-denoising` measures on the mixture. :citet:`Koehler.Heckett.Risteski.2022` make it quantitative: the statistical efficiency of score matching relative to maximum likelihood is governed by the isoperimetric constants of the distribution, which are poor when modes are far apart.

## Schedules, Weighting and Networks

**Cosine schedule and learned variances.** :citet:`Nichol.Dhariwal.2021` introduced the cosine schedule, which destroys information more evenly across steps than a linear one, and learned the variance of each reverse step from the variational bound, which improved likelihoods and allowed fewer sampling steps.

**Weighting as multi-task learning.** :citet:`Hang.Gu.Li.ea.2023` treat the noise levels as tasks whose gradients conflict and clamp the weight of each level at the minimum of its signal-to-noise ratio and a constant. It is one point in the weighting space of :numref:`sec_diffusion-design-space`, reported to converge several times faster.

**Zero terminal SNR.** :citet:`Lin.Liu.Li.ea.2024` found that common schedules leave some signal at the last step, so a model trained on them never sees the pure noise it starts from at inference; Stable Diffusion therefore could not produce very bright or very dark images. They rescale the schedule to end at zero signal.

**Resolution and capacity.** At higher resolution the same noise level destroys less of an image, so :citet:`Hoogeboom.Heek.Salimans.2023` shift the schedule toward more noise and train pixel models end to end. :citet:`Karras.Aittala.Lehtinen.ea.2024` rebuild the network so that activations and weights keep their magnitudes, and they store snapshots of two moving averages so that the averaging length, fixed before training in :numref:`sec_diffusion-images`, can be chosen afterwards. :citet:`Li.He.2026` argue that a network of finite capacity should predict the clean image, which lies near a low-dimensional manifold, rather than the noise, which does not.

**The U-Net at ImageNet scale.** :citet:`Dhariwal.Nichol.2021` ablated the U-Net of :citet:`ho2020denoising` and kept attention at several resolutions, residual up- and down-sampling and adaptive group normalization, in which the time and class embedding set a scale as well as a shift for each channel. The resulting network, ADM, is the baseline later image models compare against.

## Samplers

**Exponential integrators and their successors.** DPM-Solver and DEIS, which :numref:`sec_mdl-dynamics-further-reading` describes, integrate the linear part of the probability-flow ODE exactly. DPM-Solver++ :cite:`Lu.Zhou.Bao.ea.2025` applies the same idea to the clean-image prediction and reuses earlier evaluations, which keeps it stable at the large guidance scales of :numref:`sec_diffusion-images`. UniPC :cite:`Zhao.Bai.Rao.ea.2023` adds a corrector that reuses the evaluation at the new point, gaining an order of accuracy without an extra network call.

**The optimal reverse variance.** :citet:`Bao.Li.Zhu.ea.2022` derive the variance of each ancestral step that maximizes the likelihood, in closed form through the mean squared norm of the score, and estimate it from a trained model without retraining. It completes the choice of reverse variance that :numref:`sec_diffusion-design-space` makes by hand.

**Parallel sampling.** :citet:`Shih.Belkhale.Ermon.ea.2023` guess the whole trajectory and refine all of its steps at once by Picard iteration. This trades compute, not quality, for latency.

## Guidance and Inverse Problems

**Better guidance.** Autoguidance :cite:`Karras.Aittala.Kynkaanniemi.ea.2024` guides a model with a smaller, less trained version of itself instead of its unconditional half, which improves quality without the loss of variety of classifier-free guidance. :citet:`Kynkaanniemi.Aittala.Karras.ea.2024` apply guidance only at intermediate noise levels, where it helps most and harms least. Dynamic thresholding :cite:`saharia2022photorealistic` clips the predicted clean image at a per-sample percentile, keeping strongly guided samples in range. :citet:`Sadat.Hilliges.Weber.2025` answer the same failure differently: they split the guidance update into a part parallel to the conditional prediction, which mostly raises saturation, and an orthogonal part, which mostly improves agreement with the condition, and down-weight the first. Guidance also doubles the cost of a step; :citet:`Meng.Rombach.Gao.ea.2023` distill the guided prediction into one network that takes the guidance scale as an input.

**Editing and restoration without retraining.** SDEdit :cite:`Meng.He.Song.ea.2022` noises a rough guide image to an intermediate level and denoises it, the partial inversion of :numref:`sec_diffusion-samplers`. RePaint :cite:`Lugmayr.Danelljan.Romero.ea.2022` inpaints by re-imposing the noised known pixels at every step. DDRM :cite:`Kawar.Elad.Ermon.ea.2022` solves linear inverse problems through the singular value decomposition of the measurement operator, and DPS :cite:`Chung.Kim.Mccann.ea.2023` handles noisy and nonlinear measurements by following the gradient of the measurement likelihood at the predicted clean image. All four use a pretrained model as a prior, unchanged.

**Trained conditions.** ControlNet :cite:`Zhang.Rao.Agrawala.2023` takes the other route: it trains a copy of the encoder on edges, depth or pose, joined to the frozen model through zero-initialized convolutions.

## Latent Spaces and Backbones

**Latent diffusion at scale.** Latent diffusion :cite:`Rombach.Blattmann.Lorenz.ea.2022` trains its autoencoder with the perceptual and adversarial losses of :numref:`sec_gan_beyond` and adds cross-attention to a text encoder. SDXL :cite:`Podell.English.Lacey.ea.2024` scales the U-Net, uses two text encoders and conditions on image size and crop.

**Transformers for diffusion.** DiT :cite:`Peebles.Xie.2023` conditions transformer blocks on the noise level and class through adaptive normalization and shows that quality improves steadily with compute. SD3 :cite:`Esser.Kulal.Blattmann.ea.2024` gives text and image tokens separate weights in one joint attention and trains with rectified flow, and FLUX.1 :cite:`BlackForestLabs.2024` belongs to the same family.

**Better latents.** REPA :cite:`Yu.Kwak.Jang.ea.2025` aligns the transformer's hidden states with the features of a pretrained image encoder and reports training many times faster. DC-AE :cite:`Chen.Cai.Chen.ea.2025` compresses images far more strongly than the usual eightfold while keeping reconstructions sharp. Representation autoencoders :cite:`Zheng.Ma.Tong.ea.2026` replace the autoencoder's encoder with a frozen pretrained one, such as DINO, and adapt the transformer to its wide latents.

## Few-Step Generation

**Distillation and consistency.** Progressive distillation :cite:`Salimans.Ho.2022` trains a student to take two teacher steps in one and repeats, halving the steps each round. Consistency models :cite:`Song.Dhariwal.Chen.ea.2023` map every point of a trajectory to its endpoint; improved consistency training :cite:`Song.Dhariwal.2024` makes them competitive when trained from data without a teacher, and sCM :cite:`Lu.Song.2025` stabilizes them in continuous time.

**General flow maps.** Consistency trajectory models :cite:`Kim.Lai.Liao.ea.2024` jump between any two times, as the flow maps of :numref:`sec_diffusion-few-step` do. Shortcut models :cite:`Frans.Hafner.Levine.ea.2025` condition on the step size and require one step of twice the size to equal two steps. MeanFlow :cite:`Geng.Deng.Bai.ea.2025` regresses the average velocity, and its improved version :cite:`Geng.Lu.Wu.ea.2026` turns the loss into a regression on the instantaneous velocity and makes the guidance scale an input.

**One framework.** :citet:`Boffi.Albergo.VandenEijnden.2025` sort every method into Eulerian, Lagrangian and progressive families, according to the identity it enforces, and turn any distillation scheme into training from scratch. Distillation by distribution matching is on :numref:`sec_gan-further-reading`.

## Likelihood

**Variational diffusion models.** :citet:`Kingma.Salimans.Poole.ea.2021` learn the schedule to reduce the variance of the bound and reached the best image likelihoods of their time; the bound's invariance to the schedule is on :numref:`sec_mdl-dynamics-further-reading`.

**Likelihood weighting.** :citet:`Song.Durkan.Murray.ea.2021` show that one weighting of the score-matching loss bounds the negative log-likelihood and train with it. The probability-flow ODE has a different likelihood from the stochastic chain, and :citet:`Lu.Zheng.Bao.ea.2022` and :citet:`Zheng.Lu.Chen.ea.2023` add higher-order terms, a velocity parameterization and a better dequantization to train and evaluate it.

**Weightings as bounds.** :citet:`Kingma.Gao.2023` show that every monotone weighting is the bound of :numref:`sec_diffusion-design-space` applied to noise-augmented data, the paper behind that section's figure of weightings in log-SNR.

## Evaluation and Memorization

**Text–image agreement.** CLIPScore :cite:`Hessel.Holtzman.Forbes.ea.2021` measures the cosine similarity of image and caption in a joint embedding. GenEval :cite:`Ghosh.Hajishirzi.Schmidt.2023` runs an object detector on the image to check counts, colors and positions named in the prompt.

**Memorization.** :citet:`Carlini.Hayes.Nasr.ea.2023` extract near-copies of training images from large text-to-image models by generating many samples and filtering, and :citet:`Somepalli.Singla.Goldblum.ea.2023` find replicated objects and backgrounds by image retrieval.

**Why networks generalize.** The exact minimizer of the regression on a finite training set reproduces training points. :citet:`Bertrand.Gagneux.Massias.ea.2025` show that the noise in the regression target is not what prevents this: training on the closed-form target gives comparable models. The answer lies in the network. :citet:`Kadkhodaie.Guth.Simoncelli.ea.2024` find that two denoisers trained on disjoint halves of a large set learn nearly the same score, and :citet:`Kamb.Ganguli.2025` predict the outputs of convolutional denoisers from locality and equivariance alone. The tutorial of :citet:`Massias.Bertrand.2026` surveys this question.

## Video

**Video diffusion.** :citet:`Ho.Salimans.Gritsenko.ea.2022` extend the U-Net with factorized space-time attention and train jointly on images and videos. Stable Video Diffusion :cite:`Blattmann.Dockhorn.Kulal.ea.2023` adds temporal layers to an image latent model and shows that curating the video data matters as much as the architecture. Movie Gen :cite:`Polyak.Zohar.Brown.ea.2024` trains a large transformer with flow matching on joint image and video latents. Adversarial post-training for one-step video is on :numref:`sec_gan-further-reading`.

## Discrete Diffusion and Diffusion Language Models

**Objectives.** D3PM :cite:`Austin.Johnson.Ho.ea.2021` defined diffusion with general transition matrices, of which masking is one. SEDD :cite:`Lou.Meng.Ermon.2024` learns ratios of probabilities in place of a score. MDLM :cite:`Sahoo.Arriola.Schiff.ea.2024` and MD4 :cite:`Shi.Han.Wang.ea.2024` reduce masked diffusion to the weighted masked cross-entropy of :numref:`sec_diffusion-discrete`, and :citet:`Ou.Nie.Xue.ea.2025` show that the network then needs no time input. Discrete flow matching :cite:`Gat.Remez.Shaul.ea.2024` builds the same models from probability paths.

**Scale.** :citet:`Nie.Zhu.Du.ea.2025` measure scaling laws for masked diffusion, and LLaDA :cite:`Nie.Zhu.You.ea.2025` trains an 8-billion-parameter model from scratch. Dream :cite:`Ye.Xie.Zheng.ea.2025` starts from an autoregressive model instead. Block diffusion :cite:`Arriola.Gokaslan.Chiu.ea.2025` generates blocks left to right and denoises within each, which restores variable length and caching. Under repeated data, masked diffusion overtakes autoregressive training :cite:`Prabhudesai.Wu.Zadeh.ea.2025`.

**Images and the debate.** MaskGIT :cite:`Chang.Zhang.Jiang.ea.2022` generates images by the confidence-ordered unmasking that :numref:`sec_diffusion-discrete` uses, on the tokens of a learned image codebook. :citet:`Dieleman.2026b` traces why masked diffusion displaced continuous diffusion for language and why continuous models are returning.

## Post-Training by Reward and Preference

What changes relative to the chapter is the target: after pretraining, a reward or a preference replaces the regression target. The denoising chain becomes a Markov decision process whose actions are the reverse steps.

**Policy gradients.** DDPO :cite:`Black.Janner.Du.ea.2024` treats each Gaussian reverse step as a policy with a tractable log-likelihood and applies the clipped policy-gradient update of :numref:`sec_rl_sequences` to rewards such as compressibility, aesthetic scores and agreement judged by a vision-language model. Flow-GRPO :cite:`Liu.Liu.Liang.ea.2025` brings the group-relative update of that section to flow models by replacing the deterministic sampler with a stochastic one that has the same marginals.

**Preferences.** Diffusion-DPO :cite:`Wallace.Dang.Rafailov.ea.2024` adapts direct preference optimization, built on the Bradley--Terry model of :numref:`sec_regularized`, by replacing the intractable log-likelihood with the variational bound of :numref:`sec_diffusion-design-space`.
