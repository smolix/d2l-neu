# Further Reading
:label:`sec_gan-further-reading`

Each section of this chapter taught one idea and tested it on a small problem. This page collects the variants, systems and reported results that the sections name without teaching, in the order of the chapter.

## Objectives beyond the Table

The output activation of an $f$-GAN critic is read off the domain of the conjugate :cite:`Nowozin.Cseke.Tomioka.2016`. The forward KL conjugate is finite on all of $\mathbb{R}$ and needs no activation; the reverse KL conjugate is finite only for $t < 0$, enforced by $-\mathrm{softplus}$; the Jensen--Shannon generator of the original GAN is finite for $t < \log 2$, enforced by $\log 2 - \mathrm{softplus}$.

## Architectures 2015--2024

Projected GAN discriminates in the feature pyramid of a frozen pretrained classifier and reports reaching previous best FID values up to forty times faster :cite:`Sauer.Chitta.Muller.ea.2021`. StyleGAN-T retains this construction but uses self-supervised features to reduce overlap with the evaluation network :cite:`Sauer.Karras.Laine.ea.2023`.

## Evaluation

A single score also combines two distinct failures: low-fidelity samples and missing modes. The precision and recall metrics of :citet:`Kynkaanniemi.Karras.Laine.ea.2019` separate them through feature-space neighborhoods. Precision measures the fraction of generated samples that lie within the support of the real features; recall measures the fraction of real samples covered by generated features. A collapsed generator can retain nonzero precision if its single output lies near the data manifold, while its recall approaches zero. FID does not expose this asymmetry. R3GAN therefore reports recall as a separate measure of diversity alongside FID.

## Conditions, Translation and Text

SPADE is a spatial variant in which a semantic layout supplies a separate scale and offset at each location :cite:`Park.Liu.Wang.ea.2019`.

Text and other structured conditions are commonly incorporated through cross-attention from condition tokens to image features, as in the text-to-image GAN of :citet:`Kang.Zhu.Zhang.ea.2023`. By 2026, diffusion and autoregressive models dominate general text-to-image synthesis; adversarial variants remain useful when single-pass sampling speed is important.

## Fixed Discrepancies and Learned Ones

Generative moment matching networks minimize the fixed-kernel MMD from :numref:`sec_gan_objectives`, requiring neither a critic nor an inner optimization :cite:`Li.Swersky.Zemel.2015`. IMLE associates each data point with its nearest generated sample, directly penalizing any data mode left without a nearby sample :cite:`Li.Malik.2018`. Sliced Wasserstein generators compute one-dimensional transport costs along random projections :cite:`Deshpande.Zhang.Schwing.2018`, while Sinkhorn divergences replace the critic with a differentiable fixed-point solver that interpolates between optimal transport and MMD :cite:`Genevay.Peyre.Cuturi.2018`.

On natural images, several fixed discrepancies became more effective after adding a learned component. Fixed-kernel moment matching was not competitive with contemporary GANs and was extended to MMD-GAN by learning the kernel adversarially :cite:`Li.Chang.Cheng.ea.2017`. Sliced-Wasserstein generators added a log-loss discriminator to select informative projection directions because random directions provide little signal in high dimensions :cite:`Deshpande.Zhang.Schwing.2018`. Diffusion models avoid a game during initial training, but some few-step distillation methods introduce a discriminator. Sinkhorn generators illustrate the estimation trade-off within one method: on CIFAR-10, the two regularization settings nearest unregularized optimal transport score about one Inception point below the setting nearest MMD :cite:`Genevay.Peyre.Cuturi.2018`. This agrees with the estimation ordering derived in :numref:`sec_gan_objectives`: a discrepancy can be theoretically discriminating yet difficult to estimate reliably in high dimensions. Learning the critic or its features is one recurring response.

## Adversarial Terms in 2024--2026 Systems

Later systems use the adversarial term for different reasons. DMD2 adds an adversarial loss to train on real data rather than only sharpen teacher outputs, allowing its ImageNet-64 student to reach FID 1.28, below its teacher :cite:`Yin.Gharbi.Park.ea.2024`. LADD reuses the teacher diffusion model as the discriminator backbone and finds synthetic teacher samples effective enough to remove the distillation loss entirely, leaving only the adversarial objective :cite:`Sauer.Boesel.Dockhorn.ea.2024`. The approach is also used in deployed models. The FLUX.1-schnell model card states that its twelve-billion-parameter rectified-flow transformer was trained by latent adversarial diffusion distillation to generate in one to four steps :cite:`BlackForestLabs.2024`.

Few-step generation does not require an adversarial term. MeanFlow trains a one-evaluation generator from scratch, without a teacher, distillation, or discriminator, and reports FID 3.43 on ImageNet-256 :cite:`Geng.Deng.Bai.ea.2025`. Continuous-time consistency models report FID 1.88 on ImageNet-512 using two evaluations and no adversarial loss :cite:`Lu.Song.2025`. Both methods replace the critic with a regression target designed to remain informative at low step counts. Thus adversarial training is one solution to the capacity constraints of few-step generation, but it is not the only one.

RAF applies the relativistic pairing loss of :numref:`sec_gan_relativistic` to waveforms; a pairing-trained BigVGAN-base outperforms an LSGAN-trained BigVGAN while using twelve percent as many parameters :cite:`Lee.Choi.2026`. Diffusion and flow-matching vocoders provide non-adversarial alternatives. For example, WaveFM uses flow matching followed by consistency distillation to obtain single-step synthesis :cite:`Luo.Miao.Duan.2025`.

A recent real-time video generator uses the same mechanism. Adversarial post-training fine-tunes the generator against real data, with a zero-centered gradient penalty in the family of :numref:`sec_gan_convergence`, and produces two seconds of $1280 \times 720$ video at 24 frames per second in a single forward evaluation :cite:`Lin.Xia.Ren.ea.2025`.

As a fine-tuning stage for pretrained EDM diffusion models, direct discriminative optimization (:numref:`sec_gan_beyond`) lowers FID from 1.79 to 1.30 on CIFAR-10 and from 1.58 to 0.97 on ImageNet-64 :cite:`Zheng.Chen.Chen.ea.2025`.
