# GAN Literature Review 2018–2026 — for the D2L GAN Chapter Rewrite

**Compiled:** 2026-08-02
**Purpose:** inform a ground-up rewrite of the GAN chapter of *Dive into Deep Learning*, organized
around the closed-form analysis in `/home/smola/d2l-neu/gan-notes/gan-objectives.tex`
(log-loss GAN → proper losses / f-divergences → IPMs → relativistic pairing objective →
zero-centered gradient penalties as Sobolev / linearized-W₂ geometry).

**Evidence policy used throughout.** Every factual claim carries a URL. Where a claim could not be
verified against a primary source it is explicitly marked `UNVERIFIED`. Quantities quoted as
"exact quotes" were extracted from the arXiv LaTeX source or the typeset PDF, not from secondary
summaries. Sections noted as coming from a *delegated research pass* carry a weaker verification
level and are flagged inline; §§1–3 were verified directly against downloaded sources. A
section-by-section verification status table appears at the end.

---

## Table of contents

1. [Task 2 — VERIFICATION of R3GAN Eq. (12) and Appendix B](#1-verification-r3gan-eq-12-and-appendix-b)
2. [Task 1 — Primary papers](#2-primary-papers)
3. [Task 3 — NOVELTY CHECK: is `d_Rp(p,q) = JS(p⊗q, q⊗p)` new?](#3-novelty-check)
4. [Task 4 — Where GANs matter, 2024–2026](#4-where-gans-matter-20242026)
5. [Task 5 — Evaluation practice](#5-evaluation-practice)
6. [Task 6 — Conference tutorials 2023–2026](#6-conference-tutorials-20232026)
7. [Recommendations for the chapter](#7-recommendations-for-the-chapter)
8. [The other exits: how else researchers removed the heuristics](#8-the-other-exits-how-else-researchers-removed-the-heuristics)

---

## 1. VERIFICATION: R3GAN Eq. (12) and Appendix B

> **Task:** check whether the R3GAN paper's Eq. (12) eigenvalue formula reads
> `−γ/2 ± sqrt(γ²/4 − f′(0))` or `−γ/2 ± sqrt(γ²/4 − f′(0)²)`. Our note (footnote to
> §"Two point masses with a penalty") flags a suspected typo and asks that it be checked
> against the typeset paper before being cited.

### 1.1 Verdict

**The suspected typo is real, and it is in the paper — not an artifact of PDF text extraction.**
The paper prints `f′(0)` **without** the square. It is wrong: the Jacobian the paper itself
displays two lines later has determinant `f′(0)²`, so the correct eigenvalues carry the square.
The note's footnote should be rewritten from "either a typographical error or an artifact of text
extraction — should be checked" to a flat statement that the published paper contains a
typographical error, with the evidence below.

This was verified at **four independent levels**, all agreeing:

| Source | What it prints | Verified how |
|---|---|---|
| arXiv:2501.05441v1 **LaTeX source**, `tex/appendix.tex` line 78 | `\sqrt{\frac{\gamma^2}{4}-f'(0)}` | downloaded `arxiv.org/e-print/2501.05441`, read the file |
| arXiv:2501.05441v1 **typeset PDF**, Eq. (12), p. 18 | `−γ/2 ± √(γ²/4 − f′(0))` | `pdftotext` on `arxiv.org/pdf/2501.05441v1` |
| **NeurIPS 2024 proceedings camera-ready**, Eq. (12) | `−γ/2 ± √(γ²/4 − f′(0))` | `pdftotext` on the [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4e2acb1e1c8e297d394ae29ed9535172-Paper-Conference.pdf) |
| **Precursor course project** (Brown CSCI 1952Q), Eq. (12) | `−γ/2 ± √(γ²/4 − f′(0))` | `pdftotext` on [the PDF](https://cs.brown.edu/people/ycheng79/csci1952qs23/Top_Project_1_Nick%20Huang_Jayden%20Yi_Convergence%20of%20Relativistic%20GANs%20With%20Zero-Centered%20Gradient%20Penalties.pdf) |

The equation number is **(12)** in all four, confirming the note's citation of "their Eq. (12)".
An independent recount of numbered equations across the paper's `\input` order
(`abstract → introduction → loss → roadmap → experiments → discussion → appendix`) also lands on
(12) for the `eq:ev` label.

### 1.2 Exact quote — R3GAN, Lemma B.4 and Eq. (12)

From `arXiv:2501.05441v1`, Appendix B ("DiracRpGAN: A demonstration of non-convergence"),
typeset PDF and NeurIPS camera-ready, verbatim:

> **Lemma B.4.** *The eigenvalues of the Jacobian of the gradient vector field for the
> gradient-regularized DiracRpGAN at the equilibrium point are given by*
>
> $$\lambda_{1/2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^{2}}{4}-f'(0)} \qquad (12)$$
>
> *In particular, for $\gamma > 0$ all eigenvalues have a negative real part. Hence, gradient
> descent is locally convergent for small enough learning rates.*

The LaTeX source (`tex/appendix.tex`, lines 75–80) confirms this is authored, not rendered:

```latex
\paragraph{Lemma B.4.} \emph{The eigenvalues of the Jacobian of the gradient vector field for the
gradient-regularized DiracRpGAN at the equilibrium point are given by
\begin{equation}
\label{eq:ev}
    \lambda_{1/2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4}-f'(0)}
\end{equation}
```

### 1.3 Why it is wrong — the paper contradicts itself two lines later

Immediately after Eq. (12), the same appendix displays the regularized Jacobian at the equilibrium
(Eq. 15):

$$\mathbf{J}_{\tilde v}\Big|_{(0,0)} = \begin{pmatrix} 0 & -f'(0) \\ f'(0) & -\gamma \end{pmatrix}$$

with `trace = −γ` and `det = f′(0)²`. The characteristic polynomial is therefore
`λ² + γλ + f′(0)² = 0`, giving

$$\lambda_{1/2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^{2}}{4}-f'(0)^{2}},$$

which is what our note derives. The paper's own text says "given some calculations, we arrive at
Eq. 12" — the calculation was not carried through correctly into the display.

### 1.4 The source they adapted has the square

Mescheder, Geiger & Nowozin (ICML 2018), [arXiv:1801.04406](https://arxiv.org/abs/1801.04406),
prints the correct formula **twice** in its LaTeX source
(`parts/regularization.tex` line 214 and `supplementary/proofs.tex` line 271):

> **Lemma.** *The eigenvalues of the Jacobian of the gradient vector field for the
> gradient-regularized Dirac-GAN at the equilibrium point are given by*
> $$\lambda_{1/2}=-\frac{\gamma}{2}\pm\sqrt{\frac{\gamma^{2}}{4}-f^{\prime}(0)^{2}}.$$
> *In particular, for $\gamma>0$ all eigenvalues have negative real part. Hence, simultaneous and
> alternating gradient descent are both locally convergent for small enough learning rates.*

Mescheder additionally notes a consequence the R3GAN version loses by dropping the square: there is
a **critical regularization strength** `γ_critical = 2|f′(0)|` "that results in a locally rotation
free vector field" — i.e. the value at which the eigenvalues stop being complex and the
oscillation disappears. This is a genuinely useful teaching point (it is the critically-damped
oscillator) and the chapter should state it with the square.

### 1.5 Provenance of the error

The precursor Brown course project by Nick (Yiwen) Huang and Jayden Yi, *"Convergence of
Relativistic GANs With Zero-Centered Gradient Penalties"*, already contains the identical typo at
the identical equation number (12), with the identical Jacobian display. So the error was
introduced when Mescheder's lemma was transcribed for the DiracRpGAN setting and then propagated
unchanged into the workshop versions, the NeurIPS camera-ready, and the arXiv posting.

### 1.6 Appendix B — the two-point-mass (DiracRpGAN) analysis, verbatim

The note's §"Two point masses with a penalty" claims that the relativistic and classical objectives
"differ by a constant and therefore have identical gradient vector fields". **This is exactly what
the paper says.** From `tex/appendix.tex`:

> **Summary.** To obtain DiracRpGAN, we apply Eq. 2 to the DiracGAN problem setting. After
> simplification, DiracRpGAN and DiracGAN are different only by a constant. They have the same
> gradient vector field, therefore all proofs are identical to Mescheder et al.

> **Definition B.1.** *The DiracRpGAN consists of a (univariate) generator distribution
> $p_{\theta} = \delta_{\theta}$ and a linear discriminator $D_{\psi}(x) = \psi \cdot x$. The true
> data distribution $p_{\mathcal{D}}$ is given by a Dirac distribution concentrated at 0.*

The objective (Eq. 6) is `L(θ, ψ) = f(ψθ)`. (Sanity check: the classical DiracGAN objective is
`f(ψθ) + f(−ψ·0) = f(ψθ) + f(0)`, so the two do differ by exactly the constant `f(0)` — the
note's claim is correct.)

> **Lemma B.2.** *The unique equilibrium point of the training objective in Eq. 6 is given by
> $\theta = \psi = 0$. Moreover, the Jacobian of the gradient vector field at the equilibrium point
> has the two eigenvalues $\pm f'(0)i$ which are both on the imaginary axis.*

> **Lemma B.3.** *The integral curves of the gradient vector field $v(\theta, \psi)$ do not
> converge to the equilibrium point. More specifically, every integral curve
> $(\theta(t), \psi(t))$ of the gradient vector field $v(\theta, \psi)$ satisfies
> $\theta(t)^2 + \psi(t)^2 = const$ for all $t \in [0, \infty)$.*

The unregularized Jacobian at equilibrium (Eq. 9) is `[[0, −f′(0)], [f′(0), 0]]`, and the
regularizer reduces to `R(ψ) = (γ/2)ψ²` (Eq. 11) — all matching the note. **Every statement the
note makes about Appendix B is correct except that the note is too cautious about the typo.**

### 1.7 Action item for the note

Replace the footnote in §"Two point masses with a penalty" with something like:

> The corresponding expression in Huang et al., their Eq. (12), reads
> $-\tfrac{\gamma}{2}\pm\sqrt{\tfrac{\gamma^{2}}{4}-f'(0)}$, without the square. This is a
> typographical error, present identically in the arXiv v1 source, the arXiv typeset PDF, and the
> NeurIPS 2024 camera-ready: the Jacobian they display in the following equation has determinant
> $f'(0)^{2}$, and the corresponding lemma in Mescheder et al. carries the square.

---

## 2. Primary papers

### 2.1 R3GAN — Huang, Gokaslan, Kuleshov, Tompkin (NeurIPS 2024)

**"The GAN is dead; long live the GAN! A Modern GAN Baseline"** — Yiwen Huang (Brown),
Aaron Gokaslan (Cornell), Volodymyr Kuleshov (Cornell), James Tompkin (Brown). NeurIPS 2024.
- arXiv: <https://arxiv.org/abs/2501.05441> (v1, 9 Jan 2025)
- Proceedings: <https://proceedings.neurips.cc/paper_files/paper/2024/file/4e2acb1e1c8e297d394ae29ed9535172-Paper-Conference.pdf>
- Code: <https://github.com/brownvc/R3GAN>

**Title caution.** The arXiv metadata title is "**A Modern GAN Baseline**"; the LaTeX `\title{}`
in the source (and hence the typeset PDF) reads "**A Modern Baseline GAN**". Cite the arXiv/
proceedings form ("A Modern GAN Baseline") to match indexes. The task brief used "A modern
baseline GAN" — that matches the PDF but not the citation databases.

#### The loss

Their Eq. (1), the general GAN (note: `D` is a *fakeness* logit here, opposite to our note's
convention — translate by `D ↦ −D`):

$$\mathcal{L}(\theta,\psi)=\mathbb{E}_{z\sim p_z}\left[f\left(D_\psi(G_\theta(z))\right)\right]+\mathbb{E}_{x\sim p_\mathcal{D}}\left[f\left(-D_\psi(x)\right)\right]$$

with `f(t) = −log(1+e^{−t})` recovering Goodfellow. Their Eq. (2), RpGAN:

$$\mathcal{L}(\theta,\psi)=\mathbb{E}_{\substack{z\sim p_z\\x\sim p_\mathcal{D}}}\left[f\left(D_\psi(G_\theta(z))-D_\psi(x)\right)\right]$$

Their Eq. (3), the two zero-centered penalties:

$$R_1(\psi)=\frac{\gamma}{2}\mathbb{E}_{x\sim p_\mathcal{D}}\left[\left\|\nabla_x D_\psi\right\|^2\right],\qquad R_2(\theta,\psi)=\frac{\gamma}{2}\mathbb{E}_{x\sim p_\theta}\left[\left\|\nabla_x D_\psi\right\|^2\right]$$

**Full loss = RpGAN + R₁ + R₂.** Two informal propositions frame it: Prop. I "Unregularized RpGAN
does not always converge using gradient descent" (proved in Appendix B, §1.6 above); Prop. II
"RpGAN with R₁ or R₂ regularization is locally convergent subject to similar assumptions as in
Mescheder et al." (Appendix C). The Appendix C proof is explicitly a port: *"our proof is exactly
the same as Mescheder et al., Theorem 4.1."*

**One equilibrium-condition subtlety worth teaching** (Assumption I): because RpGAN sees only
critic *differences*, *"we no longer require $D_{\psi^*}$ to produce 0 on $\mathrm{supp}\ p_\mathcal{D}$,
instead any constant $C$ would suffice."* This is exactly the additive-constant symmetry our note
identifies in §"The objective and its symmetries".

#### The StackedMNIST loss ablation (their Table 1) — the key pedagogical experiment

Small ResNet G and D, no normalization layers, 1000 uniformly-distributed modes:

| Loss | # modes ↑ | D_KL ↓ |
|---|---|---|
| RpGAN + R₁ + R₂ | **1000** | **0.0781** |
| GAN + R₁ + R₂ | 693 | 0.9270 |
| RpGAN + R₁ | Fail | Fail |
| GAN + R₁ | Fail | Fail |

"Fail" = training diverged early. Their caption to the loss-curve figure: *"Regardless of which
objective is used, training diverges with only R₁ and succeeded with both R₁ and R₂."* Footnote:
*"Varying γ from 0.1 to 100 does not stabilize training."* This **cleanly separates the two
mechanisms**: the penalties buy convergence, the relativistic form buys coverage. Our note's
§"Why R₁ alone is insufficient" reproduces these numbers correctly.

Their point of contrast: StyleGAN's minibatch-stddev trick improves StackedMNIST mode coverage only
"from 857 to 881" (numbers from ProGAN, Karras et al., Table 4).

#### The roadmap (their Table 2) — what "hacks" get removed

Evaluated on FFHQ-256, ~25M params each for G and D, each config trained until D sees 5M real
images:

| Config | What changes | FID ↓ | G params | D params |
|---|---|---|---|---|
| **A** | StyleGAN2 (= StyleGAN2-ADA baseline) | 7.516 | 24.767M | 24.001M |
| **B** | *Stripped StyleGAN2* — remove: z normalization, minibatch stddev, equalized learning rate, mapping network, style injection, weight mod/demod, noise injection, mixing regularization, path length regularization, lazy regularization | 12.46 | 18.890M | 23.996M |
| **C** | + RpGAN loss → 11.77; + R₂ gradient penalty → **11.65** | 11.65 | 18.890M | 23.996M |
| **D** | + ResNet-ify G & D (1-3-1 bottleneck) → 10.17; − output skips → **9.950** | 9.950 | 23.378M | 23.282M |
| **E** | + ResNeXt-ify (grouped conv) → 7.507; + inverted bottleneck → **7.045** | **7.045** | 23.058M | 23.010M |

**The removed "hacks", grouped as the paper groups them:**
- *Style-based generation*: mapping network, style injection, weight modulation/demodulation, noise injection.
- *Image-manipulation enhancements*: mixing regularization, path length regularization.
- *Tricks*: z normalization, minibatch stddev, equalized learning rate, lazy regularization.

Config B additionally reduces `dim(z)` to 64 and, because equalized learning rate is gone, drops
the learning rate from `2.5e-3` to `5e-5`.

**The six principles they keep** (Config B vs DCGAN — these are the actual takeaways):
a) convergent training objective with R₁ regularization; b) smaller learning rate, no momentum
(Adam β₁ = 0); c) no normalization layer in G or D; d) proper resampling via bilinear
interpolation instead of strided/transposed convolution; e) leaky ReLU in both G and D, no tanh in
G's output; f) 4×4 constant input for G, output skips for G, ResNet D.
*"Violating (a), (b), or (c) often leads to training failures."* Violations of (d) or (e) hurt
sample quality but not stability. (f) is the backbone and gets replaced in D/E.

**On lazy regularization** (a StyleGAN2 trick they explicitly reject, Appendix E): applying R₁/R₂
once every 8 minibatches *"led to slightly worse FID performance on real world datasets like FFHQ
and CIFAR-10. However, it resulted in complete convergence failure on Stacked MNIST and several
two dimensional toy datasets (line, circle, 25 Gaussians, etc.), indicating potential concerns
regarding the mathematical legitimacy of this trick."*

#### Architecture of Config E

Fully symmetric G and D. Per resolution stage: one transition layer (bilinear resample + optional
1×1 conv) and **two** residual blocks. Residual block = `Conv1×1 → LeakyReLU → Conv3×3 →
LeakyReLU → Conv1×1` with the final 1×1 bias-free. At 4×4 the transition layer is replaced by a
*basis layer* for G (4×4 learnable feature maps modulated by z through a linear layer) and a
*classifier head* for D (global 4×4 depthwise conv → linear). The 3×3 conv is a **grouped**
convolution with group size 16 (not depthwise: *"depthwise convolution is highly inefficient on
GPUs"*), and the bottleneck is **inverted**. Fix-up initialization: zero-init the last conv in
each residual block, scale the other two by `L^{-0.25}` where L is the number of residual blocks.
Roughly 3× as deep and 1.5–3× as wide as StyleGAN2 at the same parameter count.

#### Headline results

| Benchmark | R3GAN (Config E) | NFE | Best comparison |
|---|---|---|---|
| FFHQ-256 | **2.75** FID | 1 | StyleGAN2 3.78, LDM 4.98 (200 NFE), StyleGAN3-R 3.92 |
| FFHQ-64 | **1.95** FID | 1 | EDM 2.39 (79 NFE), StyleGAN2 3.32 |
| CIFAR-10 (cond.) | **1.96** FID | 1 | StyleGAN2+ADA 2.42, VP 2.48 (35 NFE), DDPM 3.21 (1000 NFE) |
| ImageNet-32 (cond.) | **1.27** FID | 1 | DDPM-IP 2.87 (1000 NFE), ADM 3.60 |
| ImageNet-64 (cond.) | **2.09** FID | 1 | EDM 2.23 (79 NFE), DMD 2.62 (1 NFE), ADM 2.91 |
| StackedMNIST | **1000** modes, **0.029** D_KL | — | DDGAN 1000/0.071, MEG 1000/0.031, StyleGAN2 940/0.42 |

**Important caveat the paper itself makes and the chapter must repeat**: models marked `*` in
their tables (StyleGAN-XL 1.85/1.52/1.10, StyleSAN-XL 1.68, PolyINR 2.72) use a pretrained
ImageNet classifier in the discriminator, which leaks ImageNet features into FID
([Kynkäänniemi et al., ICLR 2023](https://arxiv.org/abs/2203.06026)). R3GAN gets its numbers
*without* ImageNet pretraining. This is a good teaching moment about benchmark hygiene.

**Recall** (diversity), from their §"Recall": CIFAR-10 peaked at 0.57 (StyleGAN-XL 0.47);
FFHQ-64 0.53, FFHQ-256 0.49 (StyleGAN2 0.43); ImageNet-32 0.63 (comparable to ADM);
ImageNet-64 0.59 (vs ≈0.63 for many diffusion models, but better than BigGAN-deep's 0.48).

**Resolved:** R3GAN does **not** report FD-DINOv2. Its metric suite is FID-50k, StackedMNIST
modes + reverse-KL, and Recall (Kynkäänniemi et al. 2019). Verified by grepping the full
proceedings text.

#### Hyperparameter table (their Table 6, Appendix D) — quote-ready

| Hyperparameter | StackedMNIST 32² | CIFAR-10 32² | FFHQ 256² | FFHQ 64² | ImageNet 32² | ImageNet 64² |
|---|---|---|---|---|---|---|
| Class conditional | – | ✓ | – | – | ✓ | ✓ |
| GPUs | 8 | 8 | 8 | 8 | 32 | 64 |
| Duration (Mimg) | 10 | 250 | 200 | 100 | 1000 | 1000 |
| Burn-in (Mimg) | 2 | 20 | 20 | 20 | 200 | 200 |
| Minibatch | 512 | 512 | 256 | 256 | 4096 | 4096 |
| Learning rate | 2e-4 | 2e-4 → 5e-5 | 2e-4 → 5e-5 | 2e-4 → 5e-5 | 2e-4 → 5e-5 | 2e-4 → 5e-5 |
| γ (R₁ and R₂) | 1 → 0.1 | 0.05 → 0.005 | 150 → 15 | 2 → 0.2 | 0.5 → 0.05 | 1 → 0.1 |
| Adam β₂ | 0.9 → 0.99 | 0.9 → 0.99 | 0.9 → 0.99 | 0.9 → 0.99 | 0.9 → 0.99 | 0.9 → 0.99 |
| EMA half-life (Mimg) | 0 → 0.5 | 0 → 5 | 0 → 0.5 | 0 → 0.5 | 0 → 50 | 0 → 50 |
| G params | 20.73M | 20.78M | 23.06M | 22.43M | 82.91M | 103.57M |
| D params | 20.68M | 21.28M | 23.01M | 22.38M | 86.55M | 107.21M |
| Augment prob. | – | 0 → 0.55 | 0 → 0.3 | 0 → 0.3 | 0 → 0.5 | 0 → 0.4 |

Adam β₁ = 0 throughout (no momentum). `→` denotes a **cosine schedule** over the burn-in phase.
EMA decay: `β = 0.5^(minibatch / EMA half-life)`.

**Note the γ range: 0.05 to 150.** γ is *not* a universal constant; it scales with resolution and
dataset. A chapter that hard-codes one γ will mislead.

**Compute:** StackedMNIST 7 h on 8×L40; CIFAR-10 4 days on 8×L40; FFHQ-256 ≈3 weeks on 8×A6000;
ImageNet 1 day on 32×H100 (≈5000 H100-hours). Worth quoting — it is honest about the cost.

**Mixed precision:** FP16 "cripples the training"; BFloat16 works. Practical and citable.

**Negative results** (Appendix E, following BigGAN's convention): GELU/Swish/SMU all deteriorate
FID vs leaky ReLU; group normalization did not help; removing the activation after the 3×3
grouped conv (as ConvNeXt does) worsened FID; Pixel-Shuffle without low-pass filtering produced
checkerboard-like artifacts; Adam β₂ = 0.999 caused instability on ImageNet; scaling capacity at
*low* resolution did not help but at *high* resolution always did; no attention/transformer
variants were tried.

---

#### The reference implementation — and a paper/code discrepancy the note should know about

The entire R3GAN loss is ~15 lines, which makes it ideal for a code textbook. From
<https://github.com/brownvc/R3GAN/blob/main/R3GAN/Trainer.py> (fetched 2026-08-02), verbatim:

```python
@staticmethod
def ZeroCenteredGradientPenalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
    return Gradient.square().sum([1, 2, 3])

def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1, Preprocessor=lambda x: x):
    FakeSamples = self.Generator(Noise, Conditions)
    RealSamples = RealSamples.detach()
    FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
    RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
    RelativisticLogits = FakeLogits - RealLogits          # <-- note the order
    AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
    (Scale * AdversarialLoss.mean()).backward()

def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Conditions, Gamma, Scale=1, Preprocessor=lambda x: x):
    RealSamples = RealSamples.detach().requires_grad_(True)
    FakeSamples = self.Generator(Noise, Conditions).detach().requires_grad_(True)
    RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
    FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
    R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
    R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
    RelativisticLogits = RealLogits - FakeLogits          # <-- opposite order
    AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
    DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
    (Scale * DiscriminatorLoss.mean()).backward()
```

Three things to note.

**(a) The code uses the note's *realness* convention, not the paper's.** `softplus(-u) = -log σ(u)`,
so the discriminator minimizes `softplus(D_fake − D_real)`, i.e. **maximizes
`E[log σ(D(x) − D(y))]` — exactly the note's `Φ(D)`.** The sign flip the note describes in its
notation section is already applied in the official code. Good news for the chapter: the code and
the note agree, and it is the paper's Eq. (2) that is in the opposite convention.

**(b) `R₁` and `R₂` are a three-line function**, and the sum over `[1,2,3]` (not the mean) with the
`γ/2` prefactor matches the paper's Eq. (3) exactly.

**(c) ⚠️ The generator uses the NON-SATURATING variant — contradicting a claim in our note.**
The generator minimizes `softplus(D_real − D_fake)`, i.e. **maximizes `E[log σ(D(y) − D(x))]`**,
whereas the discriminator maximizes `E[log σ(D(x) − D(y))]`. **These are not a zero-sum pair.**
The paper's Eq. (2) presents a single minimax `L(θ,ψ)` (zero-sum); the implementation does not.

Working out the generator's gradient weights confirms the two differ exactly as saturating vs
non-saturating do in the classical case:

| generator variant | per-sample weight on `∇_y D(y)` | behaviour on a badly-ranked fake |
|---|---|---|
| zero-sum / saturating (paper's Eq. 2) | `σ(D(y) − D(x))` | weight → 0, **gradient vanishes** |
| non-saturating (the actual code) | `σ(D(x) − D(y))` | weight → 1, **gradient is largest** |

Our note's remark on saturation currently states that both weights vanish for a badly ranked fake
sample, that the objective "admits the analogous non-saturating variant", and that **"R3GAN uses no
such variant, relying instead on the gradient penalties."** The last clause is **contradicted by
the official implementation.** The note's `w_Rp(y) = E_x[σ(D(y) − D(x))]` is the *saturating*
weight, i.e. it analyses the paper's stated objective rather than the code that produced the
results.

**Recommended fix for the note:** keep the analysis (it is correct for the objective as written),
but change the last sentence to something like: *"The paper states a zero-sum objective, but the
reference implementation uses the non-saturating form, in which the generator maximizes
`E[log σ(D(y) − D(x))]`; the gradient penalties of §7 address a different failure."* This matters
for the chapter because a reader who implements the paper's Eq. (2) literally will get the
saturating variant and worse results.

`UNVERIFIED`: whether the reported FID numbers were produced with this exact code revision. The
repo is the official one linked from the paper, but I did not check the commit history against the
submission date.

### 2.2 Jolicoeur-Martineau — "The relativistic discriminator" (ICLR 2019)

**"The relativistic discriminator: a key element missing from standard GAN"** — Alexia
Jolicoeur-Martineau. ICLR 2019. arXiv: <https://arxiv.org/abs/1807.00734>

The construction our note builds on. The Relativistic Standard GAN (RSGAN) losses, verbatim from
the source:

$$L_D^{RSGAN} = -\mathbb{E}_{(x_r,x_f)\sim(\mathbb{P},\mathbb{Q})}\left[\log\left(\mathrm{sigmoid}(C(x_r)-C(x_f))\right)\right]$$
$$L_G^{RSGAN} = -\mathbb{E}_{(x_r,x_f)\sim(\mathbb{P},\mathbb{Q})}\left[\log\left(\mathrm{sigmoid}(C(x_f)-C(x_r))\right)\right]$$

— i.e. the generator uses the **non-saturating swap**, exactly the variant our note describes in
its "Saturation of the rank weight" remark and notes R3GAN does *not* use.

She also introduces the **relativistic average** variant (RaGAN), where the critic is centered by
the *mean* of the opposing batch rather than paired sample-by-sample; RaSGAN's D loss uses
`D̄(x_r) = sigmoid(C(x_r) − E_{x_f}C(x_f))`. The distinction matters for the chapter: R3GAN and our
note treat **Rp** (paired), not **Ra** (averaged).

**Experimental results (CIFAR-10, standard CNN, FID):** RSGAN 36.61, RaSGAN 31.98,
**RSGAN-GP 25.60** — the last "on par with the lowest FID obtained for this architecture using
spectral normalization, as reported by Miyato et al. (25.5)", and achieved with only one
discriminator update per generator update. RaSGAN-GP failed badly (331.86). Her own summary of
the unstable-setup experiments is notably measured: *"this provide[s] good support for the
improved stability of using the relative discriminator with LSGAN, but not with HingeGAN and
SGAN... differences are minimal and probably reflect natural variations."* **The chapter should
not oversell the 2019 empirical case for relativism** — it was weak; the strong case came later
from Sun et al.'s landscape theorem and R3GAN's ablation.

### 2.3 Jolicoeur-Martineau — "On Relativistic f-divergences" (ICML 2020)

arXiv: <https://arxiv.org/abs/1901.02474>. Verified against the LaTeX source.

**Theorem 3.1 (the divergence property).** Let `f: ℝ → ℝ` be concave with `f(0)=0`, differentiable
at 0, `f′(0) ≠ 0`, `sup_x f(x) = M > 0`, and `argsup_x f(x) > 0`. Then

$$\mathrm{D}^{Rp}_f(\mathbb{P},\mathbb{Q}) = \sup_{C:\mathcal{X}\to\mathbb{R}} 2\,\mathbb{E}_{x\sim\mathbb{P},\,y\sim\mathbb{Q}}\left[f(C(x)-C(y))\right]$$

is a divergence (along with the Ra, Ralf, and Rc variants). **Note the leading factor of 2** — this
is the normalization our note's "Normalization" remark accounts for, and the remark is correct:
`D^Rp_{f_S} = 2 d_Rp`, with `f_S(z) = log(sigmoid(z)) + log 2`, so her upper bound is `2 log 2`
where the note's is `log 2`.

**Theorem 3.2 (weakness ordering).** `D^W` (Wasserstein) is weakest; `D^W` weaker than `D^Sy`;
`D^Sy` weaker than `D^Rp`; `D^Rp` weaker than `D^Ra`. Her definition: "D₁ is weaker than D₂ if
D₂(Pₙ,P) → 0 ⟹ D₁(Pₙ,P) → 0." So `D^Rp → 0 ⟹ D^Sy → 0` — the note's reading is correct. Her own
comment on this is a good chapter quote: the result runs *opposite* to what the WGAN weakness
argument would predict given the observed performance ordering, which she reads as evidence that
*"the argument made by [Arjovsky et al.] is insufficient."*

**Corollary 4.1 (MVUE).** The all-pairs `k²` estimator, not the diagonal `k` estimator, is the
minimum-variance unbiased estimator, by the two-sample U-statistic theorem.

**The estimator experiments** (verified verbatim): *"Using the MVUE for RpGAN resulted in the
generator having a worse performance on CIFAR-10 with f_LS (β=.37, p=.72), CelebA with f_Hinge
(β=2.08, p=.07), and CAT with f_S (β=4.02, p=.003)."* The note's characterization is exactly
right, including which payoff went with which dataset. Her conclusion: *"These results are
surprising as they suggest that using noisy or slightly biased estimators may be beneficial."*
Setup: spectral-GAN architecture at 32×32, lr = 2e-4, batch 32, Adam(0.5, 0.999), 100k iterations,
one critic update per generator update, FID for evaluation, same seed for all models.

**Crucially for Task 3: the paper contains no `⊗`, no "product measure", no "product
distribution", and no "permutation" anywhere in its source.** She establishes that the game's
value *is* a divergence; she never computes *which* divergence.

### 2.4 Mescheder, Geiger, Nowozin — "Which Training Methods for GANs do actually Converge?" (ICML 2018)

arXiv: <https://arxiv.org/abs/1801.04406>. Verified against the LaTeX source.

The Dirac-GAN: `p_θ = δ_θ`, linear discriminator `D_ψ(x) = ψ·x`, true data `δ_0`. Unregularized,
the Jacobian at equilibrium is `[[0, −f′(0)],[f′(0), 0]]` with purely imaginary eigenvalues
`±f′(0)i`, and the training trajectories are exact circles. With the zero-centered penalty
`R(ψ) = (γ/2)ψ²`, the eigenvalues become `−γ/2 ± sqrt(γ²/4 − f′(0)²)`, with negative real part
for all `γ > 0` (§1.4 above for the verbatim lemma).

**The instance-noise connection they draw** (motivating text, verbatim): *"Motivated by the success
of instance noise to make the f-divergence between two distributions well-defined, Roth et al.
derived a local approximation to instance noise that results in a zero-centered gradient penalty
for the discriminator."* With the footnote: *"In contrast to the gradient regularizers used in
WGAN-GP and DRAGAN which are not zero-centered."*

**`γ_critical = 2|f′(0)|`** — the value at which the vector field becomes locally rotation-free.
This is a nice, teachable fact (critical damping) that only exists if you keep the square.

**Why this matters for the chapter's spine:** this is the paper that establishes the one-centered
vs zero-centered distinction our note formalizes as `W₁` geometry vs linearized-`W₂` geometry.
They also show WGAN-GP is *not* locally convergent for exactly the reason the note gives: the
one-centered penalty keeps the critic's slope nonzero at the equilibrium.

### 2.5 Sun, Fang, Schwing — "Towards a Better Global Loss Landscape of GANs" (NeurIPS 2020)

arXiv: <https://arxiv.org/abs/2011.04926>. Code: <https://github.com/AilsaF/RS-GAN>.
Verified against the LaTeX source.

**Abstract (verbatim):** *"We prove that a class of separable-GAN, including the original JS-GAN,
has exponentially many bad basins which are perceived as mode-collapse. We also study the
relativistic pairing GAN (RpGAN) loss which couples the generated samples and the true samples.
We prove that RpGAN has no bad basins."*

**Theorem (separable-GAN).** For distinct `x₁,…,xₙ ∈ ℝ^d` and separable-GAN loss `g_SP(Y)`:
(i) the global minimum value is `−½ sup_t (h₁(t) + h₂(−t))`, achieved iff `{y₁,…,yₙ} = {x₁,…,xₙ}`;
(ii) if `yᵢ ∈ {x₁,…,xₙ}` and `yᵢ = y_j` for some `i ≠ j`, then `Y` is a sub-optimal strict local
minimum. **Therefore `g_SP(Y)` has `(nⁿ − n!)` sub-optimal strict local minima.**

**Theorem (RpGAN).** For RpGAN loss `g_R`: (i) the global minimum value is `h(0)`, achieved iff
`{y₁,…,yₙ} = {x₁,…,xₙ}`; (ii) **any `Y` is global-min-reachable**.

The parameter-space versions (their Propositions) carry this over: separable-GAN has at least
`(nⁿ − n!)` distinct `w` that are not global-min-reachable; for RpGAN *any* `w` is
global-min-reachable.

This is the precise content behind the informal claim "exponentially many bad local minima
corresponding to mode-dropping configurations". `nⁿ − n!` is the count of ways to assign n
generated points to n data points with at least one collision — i.e. **every** mode-dropping
configuration is a strict local minimum. That is a very quotable fact and the chapter should give
the combinatorial reading. Note these are results about the **empirical** (finite-sample) loss
landscape, not about the population divergence — which is exactly why our note's
Proposition (d) (the population relativistic divergence still saturates under disjoint support) is
not in conflict with them.

### 2.6 Classical objective papers

> Sourced by a delegated pass with primary-PDF verification. Items it could not verify are marked.
> **Two findings here contradict premises in our note or the task brief — see §2.7.**

#### f-GAN — Nowozin, Cseke, Tomioka (NeurIPS 2016)

<https://arxiv.org/abs/1606.00709>. The variational bound (their Eq. 4):

$$D_f(P\|Q) = \int q(x)\sup_{t\in\operatorname{dom}_{f^*}}\Big\{t\tfrac{p(x)}{q(x)} - f^*(t)\Big\}dx \;\geq\; \sup_{T\in\mathcal{T}}\Big(\mathbb{E}_{P}[T] - \mathbb{E}_{Q}[f^*(T)]\Big)$$

tight at `T*(x) = f′(p(x)/q(x))` — **exactly the note's Eq. for `T*`**. Their Eq. (7) introduces the
**output activation** `T_ω(x) = g_f(V_ω(x))`, which is the mechanism the note describes in
"Implementing the variational form".

*Their Table 1* (f and T*) and *Table 2* (g_f, dom f*, f*, and f′(1)) are the direct ancestors of
the note's Table of six f-divergences. Selected rows:

| Name | f(u) | f*(t) (dom) | g_f(v) | T*(x) |
|---|---|---|---|---|
| KL | u log u | exp(t−1) (ℝ) | v | 1 + log(p/q) |
| Reverse KL | −log u | −1−log(−t) (ℝ₋) | −exp(−v) | −q/p |
| Pearson χ² | (u−1)² | ¼t²+t (ℝ) | v | 2(p/q − 1) |
| Squared Hellinger | (√u−1)² | t/(1−t) (t<1) | 1−exp(−v) | (√(p/q)−1)·√(q/p) |
| Jensen–Shannon | −(u+1)log((1+u)/2) + u log u | −log(2−e^t) (t<log2) | log2 − log(1+e^{−v}) | log(2p/(p+q)) |
| GAN | u log u − (u+1)log(u+1) | −log(1−e^t) (ℝ₋) | −log(1+e^{−v}) | log(p/(p+q)) |

Table 1's caption: **"GAN is related to the Jensen-Shannon divergence through
`D_GAN = 2 D_JS − log(4)`."**

**Cross-check with our note (passes).** The note's GAN row uses
`f(u) = u log u − (u+1)log((u+1)/2)` and evaluates to `2 JS`; f-GAN uses
`f(u) = u log u − (u+1)log(u+1)` and evaluates to `2 JS − log 4`. The two differ by
`(u+1)log 2 = (u−1)log 2 + 2log 2`, i.e. an affine term (invisible to `D_f`, per the note's
Remark on affine ambiguity) plus the constant `2 log 2 = log 4`. **The note's normalization
remark is exactly what reconciles them** — worth citing this as the worked example.

Also relevant: their **Algorithm 1 "Single-Step Gradient Method"** replaces Goodfellow's inner-loop
discriminator optimization with a single simultaneous gradient step, with Theorem 1 proving
geometric convergence near a saddle point under local strong convexity/concavity. This is the
origin of the now-universal one-step alternating training loop.

#### WGAN — Arjovsky, Chintala, Bottou (ICML 2017)

<https://arxiv.org/abs/1701.07875>. **Example 1 ("learning parallel lines")** is the canonical
version of the note's opening two-point-mass problem, in a form that also exhibits the
low-dimensional-manifold structure: `Z ~ U[0,1]`, `P₀` = law of `(0,Z)`, `g_θ(z) = (θ,z)`. Then
`W(P₀,P_θ) = |θ|`; `JS = log 2` for `θ≠0`; `KL` both directions `= +∞`; `TV = 1`.
*"When `θ_t → 0`, the sequence `(P_{θ_t})` converges to `P₀` under the EM distance, but does not
converge under JS, KL, reverse KL, or TV divergences."* **The note's Example (two point masses)
is the 0-dimensional version of this; the chapter could use either, but WGAN's has the advantage
of exhibiting the manifold intuition.**

Recipe: RMSProp `α = 5e-5`, weight clipping to `[−0.01, 0.01]`, batch 64, `n_critic = 5`. Headline
practical claim: *"the plots clearly show that these curves correlate well with the visual quality
of the generated samples"*, versus JS-GAN where *"the JS estimate usually stays constant or goes up
instead of going down."*

**Companion:** Arjovsky & Bottou, *Towards Principled Methods for Training Generative Adversarial
Networks*, <https://arxiv.org/abs/1701.04862>. This is the theoretical half and matters more for
the note than WGAN itself:
- **Thm 2.1:** if `P_r, P_g` have disjoint compact supports, there is a smooth optimal
  discriminator with accuracy 1 and `∇_x D*(x) = 0` on both supports.
- **Thm 2.4 (vanishing gradients):** `‖∇_θ E[log(1−D(g_θ(z)))]‖₂ < Mε/(1−ε)` as `D` approaches
  optimality — the generator's gradient vanishes precisely as the discriminator gets good.
- **Thm 2.5:** the `−log D` trick's gradient direction equals minimizing
  `KL(P_g‖P_r) − 2 JSD(P_g‖P_r)` — the second term is *maximized*, explaining simultaneous mode
  dropping and instability. **This is a sharper statement than the note's current remark that
  the non-saturating substitution "changes the gradient field without moving the fixed point",
  and is worth citing there.**
- Their proposed remedy is **adding Gaussian noise to real and generated samples** — the direct
  ancestor of Roth et al.'s analytic version.
- `UNVERIFIED`: formal venue (widely cited as ICLR 2017 workshop; arXiv listing only).

#### WGAN-GP — Gulrajani et al. (NeurIPS 2017)

<https://arxiv.org/abs/1704.00028>. Eq. (3):

$$L = \mathbb E_{\tilde x\sim\mathbb P_g}[D(\tilde x)] - \mathbb E_{x\sim\mathbb P_r}[D(x)] + \lambda\,\mathbb E_{\hat x\sim\mathbb P_{\hat x}}\left[(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2\right]$$

`x̂ = εx + (1−ε)x̃`, `ε ~ U[0,1]` — uniformly along straight lines between real and fake pairs.
Justification: their **Proposition 1** proves the optimal critic has gradient norm 1 almost
everywhere along lines connecting optimally-coupled points `(x,y) ~ π*`; **Corollary 1**: *"f* has
gradient norm 1 almost everywhere under `P_r` and `P_g`."* This is the exact fact the note cites in
motivating the one-centered penalty.

Defaults: `λ = 10`, `n_critic = 5`, Adam(`α = 1e-4`, `β₁ = 0`, `β₂ = 0.9`). **No batch norm in the
critic** — because the penalty is per-input while batch norm couples the batch; layer norm
recommended as the drop-in replacement. Two-sided penalty (toward 1), not one-sided (below 1).
Notably, they train 200 random architectures on 32×32 ImageNet and show WGAN-GP trains far more of
them than either weight-clipped WGAN or DCGAN loss — including a 101-layer ResNet G/D.

#### Spectral Normalization — Miyato, Kataoka, Koyama, Yoshida (ICLR 2018)

<https://arxiv.org/abs/1802.05957>. `W̄_SN(W) := W/σ(W)` with `σ` the largest singular value; their
Eq. (7): `‖f‖_Lip ≤ ∏_l σ(W^l)`. Power iteration with `ũ, ṽ` **persisted across training steps**,
and *"one round of power iteration was sufficient in the actual experiment"* per SGD step — hence
essentially free. CIFAR-10 / STL-10 (their Table 2):

| Method | CIFAR IS | CIFAR FID | STL IS |
|---|---|---|---|
| Weight clipping | 6.41 ± .11 | 42.6 | — |
| WGAN-GP | 6.68 ± .06 | 40.2 | 8.42 ± .13 |
| **SN-GAN** | **7.42 ± .08** | **29.3** | **8.28 ± .09** |
| SN-GAN, hinge loss | 7.58 ± .12 | **25.5** | — |

The 25.5 with hinge loss is the number Jolicoeur-Martineau's RSGAN-GP (25.60) was compared against.

#### Geometric GAN (hinge) — Lim & Ye

<https://arxiv.org/abs/1705.02894>. arXiv-only; `UNVERIFIED` peer-reviewed venue. Frames the
discriminator as a **soft-margin SVM** on feature space; in the large-sample limit the
discriminator cost becomes exactly the standard hinge-loss GAN:

$$R(D,g) = \mathbb E_{x\sim P_x}\big[\max(0,1-D(x))\big] + \mathbb E_{z\sim P_z}\big[\max(0,1+D(g_\theta(z)))\big]$$

Three geometric operations: separating-hyperplane search, discriminator update *away from* the
hyperplane, generator update *toward* it along the normal. They unify GAN/f-GAN/EB-GAN/W-GAN as
differing only in how the normal vector `w` is chosen (earlier GANs implicitly use the
mean-difference classifier; Geometric GAN uses the max-margin SVM hyperplane). Their Theorem 3.1
gives `R(D*,g*) = 2` at equilibrium with `p_{g*} = p_x` a.e.

**Caution:** the delegated pass reports that **no explicit total-variation connection is stated in
the paper**. Our note attributes "the hinge loss gives total variation exactly, which is the
content of the Geometric GAN" to `lim2017geometric`. The TV identity is correct as a Bayes-risk-gap
computation (hinge Bayes risk `L(η)=2min(η,1−η)`, `L(½)=1`, gap `= TV`), but it is **our
derivation, not theirs**. The note should say "the hinge loss gives total variation exactly; the
Geometric GAN of Lim & Ye motivates the same loss from a maximum-margin perspective" rather than
implying the TV computation is theirs.

#### LSGAN — Mao et al. (ICCV 2017)

<https://arxiv.org/abs/1611.04076>. Discriminator targets `a` (fake), `b` (real); generator target
`c`:

$$\min_D V(D) = \tfrac12\mathbb E_{p_{data}}[(D(x)-b)^2] + \tfrac12\mathbb E_{p_z}[(D(G(z))-a)^2],\qquad \min_G V(G) = \tfrac12\mathbb E_{p_z}[(D(G(z))-c)^2]$$

**Their claim: minimizing this is equivalent to minimizing the Pearson χ² divergence provided
`b−c = 1` and `b−a = 2`.** Recommended coding `a = −1, b = 1, c = 0`. Adam `β₁ = 0.5`, lr 0.001
(scenes) / 0.0002 (handwriting).

**This contradicts our note's Table — see §2.7.**

#### Roth, Lucchi, Nowozin, Hofmann (NeurIPS 2017)

<https://arxiv.org/abs/1705.09367>. **The paper the note leans on for the smoothing reading of
`R₁`/`R₂`, and the delegated pass verified the mechanism precisely.** They define the
noise-convolved objective `F_γ(P,Q;ψ) := F(P*Λ, Q*Λ; ψ)` with `Λ = N(0, γI)`, then Taylor-expand
(their Eq. 12):

$$\mathbb E_\Lambda[\psi(\mathbf x+\boldsymbol\xi)] = \psi(\mathbf x) + \frac{\gamma}{2}\triangle\psi(\mathbf x) + \mathcal O(\gamma^2)$$

giving (Eq. 13) `F_γ = F + (γ/2){E_P[Δψ] − E_Q[Δ(f^c∘ψ)]} + O(γ²)`. Their gloss: *"the Laplacian
measures how much the scalar fields ψ and f^c∘ψ differ at each point from their local average. It
is thereby an infinitesimal proxy for the (exact) convolution."* Using the first-order optimality
condition of the maximizer they reach (Eq. 19):

$$F_\gamma(\mathbb P,\mathbb Q;\psi)= \mathbb E_{\mathbb P}[\psi]-\mathbb E_{\mathbb Q}[f^c\circ\psi]-\frac\gamma2\Omega_f,\qquad \Omega_f(\mathbb Q;\psi):=\mathbb E_{\mathbb Q}\left[(f^{c\prime\prime}\circ\psi)\|\nabla\psi\|^2\right]$$

**Key nuance the note should add:** their penalty is *not* the plain `E‖∇D‖²` of `R₁`/`R₂` — it
carries a **divergence-specific weighting function** `f^{c″}∘ψ`. In their own words: *"As opposed
to a naïve norm penalization, each f-divergence has its own characteristic weighting function over
the input space, which depends on the discriminator output."* The note already hedges correctly
("up to a weighting factor and a Laplacian error term"); this quote is the citable form. Note also
that the Laplacian *is* the error term the note mentions, and that it appears as the first-order
Taylor coefficient — which is a nicer statement than "approximates".

#### MMD GANs

**Bińkowski, Sutherland, Arbel, Gretton, *Demystifying MMD GANs*, ICLR 2018**,
<https://arxiv.org/abs/1801.01401>. **KID (Kernel Inception Distance)** = squared MMD between
Inception representations with the polynomial kernel `k(x,y) = ((1/d)xᵀy + 1)³`, estimated with the
unbiased U-statistic — the same estimator the note derives. Their argument for KID over FID: it
*"possesses a simple unbiased estimator"* with asymptotic normality and *"does not assume a
parametric form for the distribution of activations."*

**Correction to the task brief:** the delegated pass verified that **they do not critique WGAN-GP's
gradient penalty — they adopt it**: *"It thus seems preferable to adopt Gulrajani et al.'s proposal
of regularising the critic witness by constraining its gradient norm to be nearly 1 along randomly
chosen convex combinations of generator and reference points."* Their critique target is
**weight clipping**, not the penalty.

**Li, Chang, Cheng, Yang, Póczos, *MMD GAN*, NeurIPS 2017**,
<https://arxiv.org/abs/1705.08584>. `min_θ max_φ M_{f_φ}(P_X, P_θ)` — an **adversarially learned
kernel** `k ∘ f_φ`. Their **Theorem 4**: `max_φ M_{f_φ}(P_X, P_n) → 0 ⟺ P_n → P_X` in
distribution — the same "loss tracks distributional closeness" guarantee that motivated WGAN.
Kernel: mixture of 5 RBFs with fixed bandwidths {1,2,4,8,16}. RMSProp lr 5e-5, batch 64.
`UNVERIFIED`: Yu Cheng's exact affiliation.

#### The StyleGAN lineage — the precise list of what R3GAN removes

This is the enumeration the chapter needs to make the R3GAN "roadmap" story land.

**Progressive GAN** (Karras, Aila, Laine, Lehtinen, ICLR 2018,
<https://arxiv.org/abs/1710.10196>) contributes:
- **Equalized learning rate** — weights init `N(0,1)`, rescaled at runtime by the per-layer He
  constant so *"the dynamic range, and thus the learning speed, is the same for all weights"*,
  counteracting Adam's per-parameter gradient normalization. (`UNVERIFIED` scaling direction —
  spot-check `w_i/c` vs `w_i·c` before quoting an equation.)
- **Minibatch standard deviation layer** — std over the minibatch per feature per location,
  averaged to a scalar, replicated as one extra constant feature map into the discriminator.
- **Progressive growing** — fade in each resolution with a ramped blend weight.

**StyleGAN1** (Karras, Laine, Aila, CVPR 2019, <https://arxiv.org/abs/1812.04948>) adds:
8-layer **mapping network** `f: Z → W` (both 512-d, and trained at a 100× lower learning rate);
**AdaIN style injection** `AdaIN(x_i, y) = y_{s,i}(x_i − μ)/σ + y_{b,i}`; **noise inputs**
(per-layer single-channel Gaussian images with learned per-feature scale); **style mixing /
mixing regularization** (switch latent codes at a random crossover point in the layer stack).
FFHQ FID ablation A→F: 8.04 → 5.25 → 4.85 → 4.88 → 4.42 → **4.40**.

**StyleGAN2** (CVPR 2020, <https://arxiv.org/abs/1912.04958>) adds/changes:
- **Weight demodulation** replacing AdaIN: `w′ = s_i·w`, `w″ = w′/√(Σw′² + ε)` — kills the
  droplet artifacts.
- **Lazy regularization** — `R₁` evaluated **once every 16 minibatches**. *(R3GAN tried this at
  every 8 and reports it caused complete convergence failure on StackedMNIST and 2-D toy data —
  a direct, quotable disagreement.)*
- **Path length regularization** — `E_{w,y}(‖J_wᵀ y‖₂ − a)²` with `a` an EMA of the observed norm.
- **No progressive growing**, replaced by output skips in G and residual connections with `1/√2`
  variance correction in D.
- **`R₁`** as the gradient penalty, cited to Mescheder et al.
- Minibatch stddev, equalized LR, noise inputs, mapping network all **retained**.

FFHQ-1024 / LSUN Car FID (their Table 1): a 4.40 → b 4.39 → c 4.38 → d 4.34 → e 3.31 → f
**2.84**. **No CIFAR-10 numbers appear in StyleGAN2** (confirmed by full-text search) — CIFAR-10
belongs to StyleGAN2-ADA.

**StyleGAN2-ADA** (NeurIPS 2020, <https://arxiv.org/abs/2006.06676>) adds **adaptive discriminator
augmentation**: 18 transformations in six categories, applied to every image D sees (real *and*
generated, differentiably, so the generator is not biased), with probability `p` tuned to hold
`r_t = E[sign(D_train)] ≈ 0.6`, adjusted every four minibatches. CIFAR-10: unconditional FID
8.32 → 5.33, conditional 6.96 → 3.49; headline *"a new record FID of 2.42"*. FFHQ at 1k images:
100.16 → 21.29. **R3GAN keeps non-leaky augmentation but replaces the adaptive controller with a
fixed cosine schedule, reporting no degradation.**

**StyleGAN3** (NeurIPS 2021, <https://arxiv.org/abs/2106.12423>) is the alias-free rebuild:
diagnoses **"texture sticking"** as aliasing from non-ideal upsampling and from pointwise
nonlinearities; fixes it by treating feature maps as samples of a continuous signal and wrapping
every nonlinearity in upsample → nonlinearity → downsample with **Kaiser-windowed sinc filters at
>100 dB stopband attenuation** (vs ~42 dB standard). StyleGAN3-R uses radially symmetric jinc
filters. FFHQ-1024 FID: StyleGAN2 2.70 → SG3-T 2.79 → SG3-R 3.07 — **alias-freeness costs FID**.
SG3-R costs **+103% GPU training time**.

**The complete "hacks" checklist R3GAN reacts against**: 8-layer mapping network, AdaIN /
weight modulation-demodulation styles, noise injection, equalized learning rate, minibatch-stddev
discriminator statistic, progressive growing (→ skip/residual architecture), style mixing
regularization, lazy regularization, path length regularization, adaptive discriminator
augmentation, and StyleGAN3's alias-free signal-processing rebuild. R3GAN's Config B removes ten
of these in one step at a cost of 7.52 → 12.46 FID, then recovers to 7.05 with the loss and a
modern backbone.

---

### 2.7 Reconciliations and one attribution fix

#### The note's proper-loss table is correct — all five rows verified numerically

I checked every row of the note's table of proper losses on random discrete distributions
(exact agreement to machine precision, several trials):

| loss | Bayes-risk gap `Δ_ℓ` | claimed value | agrees? |
|---|---|---|---|
| logistic | `log2 − E_m[H_b(η)]` | `JS(p,q)` | ✓ exact |
| square (Brier) | `¼ − E_m[η(1−η)]` | `⅛∫(p−q)²/(p+q)` (triangular) | ✓ exact |
| exponential | `1 − E_m[2√(η(1−η))]` | `1 − ∫√(pq)` (sq. Hellinger) | ✓ exact |
| hinge | `1 − E_m[2min(η,1−η)]` | `TV(p,q)` | ✓ exact |
| 0–1 | `½ − E_m[min(η,1−η)]` | `½TV(p,q)` | ✓ exact |

**No changes needed to that table.**

#### LSGAN's "Pearson χ²" and the note's "triangular discrimination" are the *same divergence*

This looked like a conflict and is not one — but the chapter should say so explicitly, because a
reader who compares the note against the LSGAN paper will hit it.

LSGAN states its objective minimizes "the Pearson χ² divergence" under `b−c=1`, `b−a=2`. Their
derivation gives `2C(G) = ∫((b−c)(p+q) − (b−a)q)²/(p+q)`, which under those conditions is
`∫(p−q)²/(p+q)`. Numerically verified:

```
LSGAN 2C(G)                        : 0.9901327570208975
triangular  ∫(p−q)²/(p+q)          : 0.9901327570208975
χ²_Pearson(2q ‖ p+q)               : 0.9901327570208975
note's table entry ⅛∫(p−q)²/(p+q)  : 0.1237665946276122   (ratio exactly 8)
```

**The resolution:** LSGAN's χ² is taken between the *mixture* `p+q` and `2q`, not between `p` and
`q`. Triangular discrimination *is* `χ²_Pearson(2q ‖ p+q)`. So the two names denote the same
quantity, and the note's sentence *"The square loss gives triangular discrimination, the divergence
underlying LSGAN"* is **correct as written**. The only difference is an overall factor of 8 from
normalization conventions.

**Suggested (optional) addition to the note**, since this is a genuine reader trap: a parenthetical
noting that LSGAN reports this divergence as `χ²_Pearson` between the mixture and `2q`, which is the
same object as triangular discrimination up to scale. Note also that the naive reading
`χ²_Pearson(p+q ‖ 2q) = ∫(p−q)²/(2q)` is a *different* (and much larger) quantity — the argument
order matters, and getting it wrong is easy.

#### The TV ↔ hinge identity is ours, not Lim & Ye's — reword the attribution

The identity `Δ_hinge = TV` is correct (verified above). But the delegated pass found **no
total-variation statement anywhere in the Geometric GAN paper**; their result is the SVM/max-margin
geometry and `R(D*,g*) = 2` at equilibrium. The note currently reads *"The hinge loss gives total
variation exactly, which is the content of the Geometric GAN"*, which over-attributes.

**Suggested fix:** *"The hinge loss gives total variation exactly; Lim & Ye arrive at the same loss
from a maximum-margin perspective, treating the discriminator as a soft-margin SVM in feature
space."*

#### Minor, for the record

The task brief's premise that Bińkowski et al. critique WGAN-GP's gradient penalty is wrong — they
**adopt** it and critique **weight clipping** instead. Do not repeat that framing in the chapter.

---

## 3. NOVELTY CHECK

> **Claim under test.** Our note's Theorem derives
> $$d_{\mathrm{Rp}}(p,q) := \sup_D \mathbb{E}_{x\sim p,\,y\sim q}[\log\sigma(D(x)-D(y))] + \log 2 = \mathrm{JS}(p\otimes q,\ q\otimes p) = \mathrm{H}\!\left[\tfrac{p\otimes q + q\otimes p}{2}\right] - \mathrm{H}[p] - \mathrm{H}[q]$$
> as the closed-form value of the relativistic pairing game, and believes this identity is new.

### 3.1 Verdict and confidence

**Confidence that the identity as stated is not in the literature: MEDIUM-HIGH (≈80%).**

I could not find the identity, or any equivalent statement, anywhere. The two papers that *should*
contain it if it were known — Jolicoeur-Martineau (2020), which proves the relativistic objective
is a divergence, and Huang et al. (2024), which builds on it — demonstrably do not: neither
contains the symbol `⊗`, the phrase "product measure"/"product distribution", nor any evaluation
of the game's value. The searches below turned up nothing closer than the near-miss in §3.3.

**Why not higher than 80%.** Three reasons for residual doubt, and the note is right to hedge:
1. The derivation is genuinely short (a two-line lifting argument once you see it), which is
   exactly the profile of a result that exists as an unremarked exercise or a remark inside a
   paper on a different topic.
2. Full-text search over the literature is not something I could do exhaustively — I have
   metadata/abstract search plus targeted source downloads, not a full-text index of arXiv. A
   result stated only inside a proof or an appendix would be invisible to this method.
3. The quantity `JS(p⊗q, q⊗p)` is natural enough in information theory (it is the information
   content of a paired two-sample permutation test) that it may have a name in a statistics
   literature that does not use GAN vocabulary. My searches on that side (paired permutation
   tests, exchangeability tests, matched-pair hypothesis testing) returned nothing, but that
   literature is large and old.

**Practical recommendation.** The note's existing footnote is well-calibrated and should be kept
essentially as written. If anything, strengthen it slightly by *citing the near-miss* in §3.3,
which is the honest "closest prior art" and makes the claim more credible, not less.

### 3.1a The identity itself is correct (independent check)

Separate from the novelty question: I re-ran the note's `gan-notes/verify_rpgan.py` independently.
Across four random trials on a finite sample space, the numerically-maximized
`sup_D Φ(D) + log 2` agrees with `JS(p⊗q, q⊗p)`, with the entropy form
`H[(p⊗q + q⊗p)/2] − H[p] − H[q]`, and with the cross-ratio form, **to ten decimal places**;
the maximizer matches `log(p/q)` up to an additive constant to ~1e-7; the local expansion ratio
`d_Rp / JS` converges to exactly 2.000000 as `ε → 1e-3`; and disjoint supports give
`d_Rp = JS = log 2` exactly. So the mathematics is not in question — only the priority claim is.

### 3.2 What was searched

- Jolicoeur-Martineau, *On Relativistic f-divergences* (arXiv:1901.02474) — **LaTeX source
  downloaded and grepped**. No `otimes`, no "product measure", no "product distribution", no
  "permutation". Theorem 3.1 establishes the divergence property only.
- Huang et al., *R3GAN* (arXiv:2501.05441) — **LaTeX source downloaded and read in full**. No
  value computation anywhere; the theory is entirely local-convergence (Jacobian spectra), exactly
  as the note says.
- Jolicoeur-Martineau, *The relativistic discriminator* (arXiv:1807.00734) — source downloaded.
  No divergence-value derivation.
- Sun et al., *Towards a Better Global Loss Landscape of GANs* (arXiv:2011.04926) — source
  downloaded. Landscape results on the *empirical* loss; no population-divergence value.
- **The full citation graph of Jolicoeur-Martineau 2020** (23 citing papers via the Semantic
  Scholar API,
  <https://api.semanticscholar.org/graph/v1/paper/arXiv:1901.02474/citations>), inspected title-
  by-title. The only theory-flavoured entries are: *Towards a Better Global Loss Landscape of GANs*
  (§2.5), *Gradient Penalty from a Maximum Margin Perspective*, *Connections between Support Vector
  Machines, Wasserstein Distance and Gradient-Penalty GANs*, *Generative Adversarial Ranking Nets*
  (JMLR 2024), and *The Benefits of Pairwise Discriminators for Adversarial Training* (§3.3).
- *Generative Adversarial Ranking Nets*, Yao, Pan, Li, Tsang, Yao, JMLR 25, 2024
  (<https://jmlr.org/papers/v25/23-0461.html>) — checked. It is about generating samples matching
  a user-specified preference/score vector, with a theorem that "the learned distribution of
  GARNet rigorously coincides with the distribution specified by the given score vector". **Not a
  divergence-value result**; no JS of product measures.
- Keyword searches (WebSearch, ~10 distinct phrasings) over: "Jensen-Shannon divergence of product
  measures"; relativistic GAN + closed form + value of the game; Bradley-Terry pairwise logistic
  loss + JS divergence between `p(x)q(y)` and `q(x)p(y)`; InfoNCE optimal value + generalized JS +
  `log(K+1)`; paired permutation test + mutual information + "which of the two is real"; two-sample
  test distinguishing `P×Q` from `Q×P`. **No hits.**
- Note: the session's WebSearch budget (200 calls, shared with the delegated passes) was exhausted
  before I could run a further round; the remaining checks were done via direct source download and
  API queries. **A further full-text search pass would be the cheapest way to raise confidence.**

### 3.3 The closest prior art found — PairGAN (Tong, Garipov, Jaakkola)

**"The Benefits of Pairwise Discriminators for Adversarial Training"** — Shangyuan Tong,
Timur Garipov, Tommi S. Jaakkola (MIT CSAIL). arXiv: <https://arxiv.org/abs/2002.08621> (2020).
Verified against the LaTeX source.

**This is the nearest miss and should be cited in the note as related work.** They define, for a
pairwise discriminator `D(·,·)` interpreted as "the estimated probability of a pair being sampled
from the same distributions", three measures on `X × X`:

$$M^+_{p,q}(x,y) = \tfrac12\big(p(x)p(y) + q(x)q(y)\big)$$
$$M^-_{p,q}(x,y) = \tfrac12\big(p(x)q(y) + q(x)p(y)\big)$$
$$M_{p,q}(x,y) = \tfrac12\big(M^+_{p,q}(x,y) + M^-_{p,q}(x,y)\big)$$

**Observe: `M⁻` is exactly the mixture `(p⊗q + q⊗p)/2` whose entropy appears in our note's
Theorem.** So the *object* is in the literature. But the *divergence they compute is different*.
Their Proposition:

$$\widehat L^1_{\mathcal G}(q) = 4\big(\mathrm{KL}(M^+_{p,q}\|M_{p,q}) + \mathrm{KL}(M_{p,q}\|M^+_{p,q})\big),\qquad \widehat L^2_{\mathcal G}(q) = -\log(\varepsilon)\cdot\delta_{\mathrm{TV}}(M^+_{p,q}\|M^-_{p,q})$$

**Why this does not pre-empt the note's identity — three structural differences:**

1. **Different discriminator task.** Their `D(x,y)` answers *"were these two drawn from the same
   distribution?"* — an independence/homogeneity test. The relativistic critic answers *"which of
   these two is the real one?"* — an ordering test. The first compares `M⁺` against `M⁻`; the
   second compares `p⊗q` against `q⊗p`. These are different partitions of the pair space.
2. **Different divergences.** Symmetrized KL between `M⁺` and `M`, and TV between `M⁺` and `M⁻`.
   Neither is `JS(p⊗q, q⊗p)`. Their optimal discriminator is a symmetric function of the pair;
   the relativistic one is antisymmetric and additively separable, which is the exact property our
   note's Lifting Lemma (iii) exploits.
3. **No separability analysis.** The whole point of the note's Proposition on separability — that
   restricting the pair critic to `D(a₁) − D(a₂)` costs nothing *for the logistic loss and only
   for the logistic loss* — has no counterpart there.

**Recommended framing for the note:** add a sentence to the Theorem's footnote along the lines of
"The mixture `(p⊗q + q⊗p)/2` appears in Tong et al. (2020) under the name `M⁻`, in the analysis of
a pairwise discriminator trained to detect whether two samples come from the same distribution;
the divergences they compute (`KL(M⁺‖M) + KL(M‖M⁺)` and `TV(M⁺‖M⁻)`) are those of a homogeneity
test rather than an ordering test, and differ from `JS(p⊗q, q⊗p)`." This is honest, strengthens
the claim, and gives the reader the right neighbouring result.

### 3.4 Adjacent results that are genuinely known (and should be cited as such)

To make sure the note does not over-claim, these *are* in the literature and the note already
treats them correctly:

- **JS as mutual information with a Bernoulli label** — standard, and the basis of the
  classification view of GANs. The note uses it in §"Jensen–Shannon divergence as a mutual
  information".
- **`JS(joint ‖ product of marginals)` as an MI-estimation objective** — this is the
  Deep InfoMax / MINE-adjacent line (Hjelm et al., *Learning deep representations by mutual
  information estimation and maximization*, <https://arxiv.org/abs/1808.06670>) and the "IJS"
  bound. **Do not confuse it with the note's identity**: that one is `JS(p_{XY}, p_X⊗p_Y)`
  (a joint against a product); the note's is `JS(p⊗q, q⊗p)` (two products against each other,
  differing by a swap). Superficially similar notation, structurally different.
- **InfoNCE's `log(K+1)` ceiling** — standard (Oord et al., <https://arxiv.org/abs/1807.03748>).
  The note's Eq. for `d^(K)` reproduces this ceiling, which is a good consistency check *and*
  means the `K`-negative generalization is the part most likely to have a prior appearance.
- **Optimal critic = log density ratio for the relativistic objective** — the note proves this
  (its Proposition), and it is arguably folklore. `UNVERIFIED` whether it appears in print; I found
  no explicit statement, but it is a one-line stationarity computation.

**The specific thing that appears new** is the *lifting argument* — recognizing that the
relativistic game is the ordinary log-loss game played on `(p⊗q, q⊗p)` over `X × X`, that the
unrestricted optimum on the product space happens to be additively separable *because the log
ratio of two product measures is a sum*, and hence that the restriction to separable pair critics
is free. That step, and the closed form it yields, is what I could not find.

---

## 4. Where GANs matter, 2024–2026

> Sourced by a delegated pass. Its verification caveats are preserved. **Two of its open items I
> resolved myself:** R3GAN's NeurIPS 2024 proceedings entry **is** confirmed (I downloaded the
> camera-ready PDF, §1.1), and the title discrepancy is explained in §2.1 (arXiv metadata vs the
> LaTeX `\title{}`).

### 4.0 The one-sentence thesis

**The adversarial *architecture* lost; the adversarial *loss term* won.** Training a GAN from
scratch as your generative model is no longer competitive at the frontier. But the adversarial loss
is now a near-universal **finishing step** — bolted onto diffusion/flow models to collapse them to
1–4 steps, and bolted onto autoencoders to make lossy reconstruction look sharp. Both footholds are,
as of mid-2026, under credible attack from GAN-free alternatives, while a small 2025–2026 line has
made pure GANs competitive again at ImageNet scale.

### 4.1 Adversarial diffusion distillation

**ADD / SDXL-Turbo** — Sauer, Lorenz, Blattmann, Rombach (Stability AI),
<https://arxiv.org/abs/2311.17042>. Two losses: adversarial against real images + score
distillation against the frozen SDXL teacher (`λ = 2.5`, R1 `γ = 1e-5`). Discriminator = *frozen*
**DINOv2 ViT-S** with trainable lightweight heads at multiple layers.

**Their Table 1d is the single most pedagogically valuable result in this entire review:**

| Loss configuration | FID ↓ | CLIP ↑ |
|---|---|---|
| Adversarial only | 20.8 | 0.315 |
| **Distillation only** | **315.6** | **0.076** |
| Both | 20.6 | 0.319 |

Score distillation *alone* is catastrophic at one step; **the adversarial term carries essentially
all of the fidelity.** This is the cleanest empirical statement of what a GAN loss is *for* in the
modern era: it keeps a one-step sample **on the image manifold**. If the chapter quotes one number
from 2024–2026, it should be this one.

ADD-XL at 4 steps beat **its own teacher** SDXL 1.0 in an ELO human study on PartiPrompts;
≈0.09 s per 512² image on an A100.

**LADD / SD3-Turbo** — Sauer, Boesel, Dockhorn, Blattmann, Esser, Rombach, SIGGRAPH Asia 2024,
<https://arxiv.org/abs/2403.12015>. Names ADD's weakness explicitly: reliance on "a fixed
pretrained DINOv2 discriminator" that lives in pixel space, forcing a decode every step. LADD
instead **reuses the teacher diffusion model itself as the discriminator backbone**. Striking
finding: training on **synthetic** teacher samples "substantially outperforms training on real
data, rendering the distillation loss obsolete" — LADD trains with an **adversarial loss alone**.
SD3-Turbo = 8B SD3 distilled to 4 unguided steps. `UNVERIFIED`: specific FID/CLIP figures (the
paper de-emphasizes FID in favour of human preference).

**SDXL-Lightning** — Lin, Wang, Yang (ByteDance), <https://arxiv.org/abs/2402.13929>. Discriminator
= encoder + mid-block of the pretrained SDXL U-Net, **entirely in latent space**. Their argument is
the one to paraphrase in the chapter: a capacity-limited student under MSE produces **blurry
averages**; under an adversarial objective the discriminator learns the probability flow, so the
student *follows* it instead of averaging over it. Whole-image FIDs across SDXL-Lightning / Turbo /
LCM are within noise (21.5–23.7); the separation is in **patch FID** (33.5 vs 42.7) — i.e. the
adversarial term buys **high-frequency local detail, not global layout**. Honest framing.

**Production data point:** **FLUX.1 [schnell]**,
<https://huggingface.co/black-forest-labs/FLUX.1-schnell>, model card verbatim: *"Trained using
latent adversarial diffusion distillation, FLUX.1 [schnell] can generate high-quality images in
only 1 to 4 steps"* — a **12B** rectified-flow transformer. The strongest single "GANs are in
production in 2026" citation available.

### 4.2 One/few-step generators — who adds an adversarial term, and why

**The field splits cleanly, and this is the honest core of the section.**

*Added deliberately:* **DMD** (Yin et al., CVPR 2024, <https://arxiv.org/abs/2311.18828>) has
**no GAN loss** — FID 2.62 ImageNet-64. **DMD2** (NeurIPS 2024 **Oral**,
<https://arxiv.org/abs/2405.14867>) removes the regression loss and **adds a GAN loss**, for a
precise and quotable reason: it "lets them train the student model on **real** data, mitigating the
imperfect real-score estimation from the teacher." Different motive from ADD's — not "stay on the
manifold" but "**escape the teacher's ceiling**." Result: **FID 1.28** ImageNet-64, **8.35**
zero-shot COCO-2014, *surpassing the teacher*.

*The GAN-free camp, and it is strong:*
- **sCM** — Lu & Song (OpenAI), ICLR 2025 **Oral**, <https://arxiv.org/abs/2410.11081>. 1.5B params;
  at **two** steps: FID 2.06 CIFAR-10, 1.48 ImageNet-64, **1.88 ImageNet-512**. No adversarial term.
- **MeanFlow** — Geng, Deng, Bai, Kolter, He, <https://arxiv.org/abs/2505.13447>. **FID 3.43 at
  1-NFE on ImageNet-256, trained from scratch**, no pre-training, distillation, or discriminator.

These two are the strongest evidence that the adversarial term is **not logically necessary** for
few-step generation. The chapter must include them.

*Video — adversarial post-training is the dominant real-time recipe:*
**Seaweed-APT** (Lin et al., ICML 2025, <https://arxiv.org/abs/2501.08316>) — adversarial
post-training against real data with an approximated R1 penalty; **2-second 1280×720 24 fps video
in a single forward evaluation**. **Seaweed APT2** (NeurIPS 2025,
<https://arxiv.org/abs/2506.09350>) — 8B, 1 NFE per latent frame, real-time 736×416 @ 24 fps on
**one H100**. **Self Forcing** (NeurIPS 2025 spotlight, <https://arxiv.org/abs/2506.08009>) —
cites R3GAN.

*The generalization beyond distillation* — **Adversarial Flow Models**, Lin, Yang, Lin, Chen, Fan,
ICML 2026, <https://arxiv.org/abs/2511.22475>: *"Unlike traditional GANs, in which the generator
learns an arbitrary transport map between the noise and data distributions, our generator is
encouraged to learn a **deterministic noise-to-data mapping**. This significantly stabilizes
adversarial training."* 1 NFE ImageNet-256: XL/2 **FID 2.38**; deeper end-to-end models 2.08 and
1.94. Follow-up **Continuous Adversarial Flow Models** (<https://arxiv.org/abs/2604.11521>) lifts
**guidance-free** ImageNet-256 FID from **8.26 → 3.63** (latent SiT) and **7.17 → 3.57** (pixel
JiT) purely by replacing flow matching's fixed MSE criterion with a learned discriminator. **This
is the best "the adversarial objective is a general post-training tool, not a GAN-specific
artifact" citation in the review.**

### 4.3 Large-scale pure GANs (the pre-R3GAN baseline)

- **StyleGAN-XL** — Sauer, Schwarz, Geiger, SIGGRAPH 2022, <https://arxiv.org/abs/2202.00273>.
  ImageNet FID 1.81 @128², 2.30 @256², 2.41 @512². Latency 0.05–0.10 s vs ADM's 27–92 s.
  **Caveat to teach:** uses pretrained ImageNet classifiers as projected discriminators — the
  feature-leakage problem R3GAN flags.
- **StyleGAN-T** — Sauer, Karras, Laine, Geiger, Aila, ICML 2023,
  <https://arxiv.org/abs/2301.09515>. Zero-shot COCO FID ≈**7.30 @64²** (beats SD's 8.40) but
  ≈**13.90 @256²** (loses to Imagen's 7.27). **GANs held at low resolution and lost at high
  resolution — this is the moment text-to-image GANs lost.** Discriminator = frozen DINO ViT-S/16
  with 5 heads, the direct ancestor of ADD's.
- **GigaGAN** — Kang, Zhu, Zhang, Park, Shechtman, Paris, Park, CVPR 2023,
  <https://arxiv.org/abs/2303.05511>. G 652.5M / D 381.4M. Zero-shot COCO FID-30k **9.09** at
  512px; **0.13 s** for 512px, 3.66 s for 4K. Scaling ablation: StyleGAN2 29.91 → +adaptive conv
  19.97 → +CLIP loss 14.88 → full 9.18.

### 4.4 R3GAN's citation graph — and an honest reading of it

Semantic Scholar (fetched 2026-08-02): **99 citations, 8 influential**. But the citing set is
**dominated by two things that are not the minimalist-GAN line**: (a) AI-generated-image
*detection*/forensics papers (≥10 of 99), citing R3GAN merely as a modern generator *to detect*;
and (b) long-tail domain applications (SAR, EEG, PET synthesis, channel estimation, tabular data).
**This is itself a finding the chapter should report honestly:** R3GAN's citation graph is broad but
shallow, and mostly not people building on the method.

**The genuine continuation line:**

| Paper | arXiv | Venue | Why it matters |
|---|---|---|---|
| **GAT — Scalable GANs with Transformers** (Hyun, Lee, Heo) | [2509.24935](https://arxiv.org/abs/2509.24935) | ICML 2026 (comment field) | Pure transformer G *and* D in VAE latent space; explicitly "we deploy relativistic pairing loss with the approximated version of two-sided gradient penalty, **following R3GAN**." **FID 2.18 ImageNet-256 single-step, 60 epochs** |
| **CAT — Cross-scale Aligned Supervision** (same authors) | [2605.26449](https://arxiv.org/abs/2605.26449) | preprint | **FID-50K 1.56, class-conditional ImageNet-256, ONE step** — beats DiT-XL/2 (2.27 @ 250 NFE) and SiT-XL/2 (2.06 @ 250 NFE) at 1/250th the NFE |
| **UCD — Unconditional Discriminator** (Xia, Xue, Zhu, Shen) | [2510.00624](https://arxiv.org/abs/2510.00624) | — | **FID 1.47 ImageNet-64**, "surpassing StyleGAN-XL and several state-of-the-art one-step diffusion models" |
| **DDO — Direct Discriminative Optimization** (Zheng et al.) | [2503.01103](https://arxiv.org/abs/2503.01103) | **ICML 2025 Spotlight** | *"Your Likelihood-Based Visual Generative Model is Secretly a GAN Discriminator."* Implicit discriminator = likelihood ratio vs a frozen reference; **no joint G/D training.** EDM 1.79/1.58/1.96 → **1.30/0.97/1.26** on CIFAR-10/ImageNet-64/ImageNet-512 |
| **Revisiting GAN with Bayes-Optimal Discrimination** (Naeini et al.) | [2510.25609](https://arxiv.org/abs/2510.25609) | — | Reframes GAN objectives as parameterized bounds on discrimination Bayes error; **recovers TV and 1-Lipschitz Wasserstein as special cases** — directly adjacent to our note's proper-loss/Bayes-risk-gap framing |
| **Align Your Flow** (Sabour, Fidler, Kreis, NVIDIA) | [2506.14603](https://arxiv.org/abs/2506.14603) | NeurIPS 2025 | "an additional boost can be achieved by **adversarial finetuning, with minimal loss in sample diversity**" |
| **GAS — Generalized Adversarial Solver** | [2510.17699](https://arxiv.org/abs/2510.17699) | ICLR 2026 | Adversarial training added to ODE-solver distillation |
| **RAF — Relativistic Adversarial Feedback** (Lee & Choi) | [2603.11678](https://arxiv.org/abs/2603.11678) | **Interspeech 2026** | Ports R3GAN's relativistic pairing **to waveforms**; RAF-trained BigVGAN-base beats LSGAN-trained BigVGAN with **12% of the parameters** |

**Note the DDO result carefully** — it is arguably the field's verdict on our note's subject:
**the objective was the good idea; the two-player optimization was the tax.**

**Not found:** no paper scaling R3GAN itself to text-to-image or billion-parameter scale. GAT/CAT
are the scale-up, and they rebuild the architecture rather than scaling R3GAN's ResNet.

### 4.5 GAN losses inside tokenizers — and the 2026 revolt

**The established recipe.** VQGAN (Esser, Rombach, Ommer, CVPR 2021,
<https://arxiv.org/abs/2012.09841>) adds a **patch-based discriminator** + perceptual loss with an
adaptive λ; reconstruction FID on ImageNet val **7.94** (1024 codes) / **4.98** (16384) vs DALL·E's
dVAE **32.01**. The LDM/Stable Diffusion autoencoder (<https://arxiv.org/abs/2112.10752>, §3) is
explicit: a "combination of a perceptual loss and a **patch-based adversarial objective**", which
"ensures that the reconstructions are confined to the image manifold by enforcing **local
realism** and **avoids blurriness** introduced by relying solely on pixel-space losses such as L2
or L1."

**The best single ablation number.** **ViTok** (Hansen-Estruch et al., Meta,
<https://arxiv.org/abs/2501.09755>) stages training *because* of adversarial instability: Stage 1
= MSE + LPIPS + KL only; Stage 2 = freeze the encoder, fine-tune only the decoder adversarially.
For S-B/16 at E=4096: Stage 1 **rFID 1.6 / gFID 5.5** → Stage 2 **rFID 0.50 / gFID 4.9**. A **>3×
reconstruction-FID improvement** from the adversarial term, with a much smaller downstream gain.
**That asymmetry — big rFID win, small gFID win — is the honest framing.**

**l-DeTok** (<https://arxiv.org/abs/2507.15856>) confirms the recipe is unchanged in mid-2025:
`λ_GAN = 0.1`, GAN turned on at epoch 100 of 200, **and disabled entirely for ablations** — a
telling operational detail: the GAN loss is treated as finishing polish, not a core objective.

**The counter-evidence, and it is not fringe:**
- **DiTo** (Chen, Girdhar, Wang, Rambhatla, Misra, Meta, <https://arxiv.org/abs/2501.18593>) — a
  **single diffusion L2 loss**, no discriminator. Results genuinely mixed: DiTo-XL rFID 7.95 vs a
  GAN+LPIPS baseline's 4.14 (*loses*), but downstream gFID 7.57 vs 7.49 (tie), → 6.29 with noise
  synchronization (wins). Human raters preferred DiTo reconstructions 52.44% of the time.
- **ViTok-v2** — *Scaling Native Resolution Auto-Encoders to 5 Billion Parameters*,
  <https://arxiv.org/abs/2605.05331>. **The most important 2026 citation in this section**, because
  it is the *same team* that produced the rFID 1.6 → 0.50 GAN result a year earlier, now reversing
  course. Verbatim: *"reliance on adversarial losses prevents stable scaling"*; they introduce
  "a novel DINOv3 perceptual loss that **replaces both LPIPS and GAN objectives** for stable
  training at any scale." Largest image autoencoder to date; outperforms all baselines at ≥512p.
- **Mage-Flow** (<https://arxiv.org/abs/2607.19064>) makes the same DINOv3 move in its tokenizer —
  **but still uses "few-step distillation with adversarial perceptual guidance"** for its 4-step
  Turbo variants. **This is 2026 in one paper: GAN loss out of the tokenizer, GAN loss still in the
  distillation.**

`UNVERIFIED`: whether MAGVIT-v2, MacTok (CVPR 2026), AlignTok (ICLR 2026) retain an internal
adversarial term; whether SD3's VAE specifically is adversarially trained (assert only via the LDM
paper).

### 4.6 Audio — the last unquestioned stronghold

From HiFi-GAN (NeurIPS 2020, <https://arxiv.org/abs/2010.05646>, multi-period discriminator,
167.9× real-time on a V100) through EnCodec (<https://arxiv.org/abs/2210.13438>), DAC (NeurIPS
2023 spotlight, <https://arxiv.org/abs/2306.06546>, ~90× compression of 44.1 kHz at 8 kbps via
"improved **adversarial** and reconstruction losses"), BigVGAN (ICLR 2023,
<https://arxiv.org/abs/2206.04658>, 112M params, Snake activations, zero-shot to unseen speakers/
languages/singing), Vocos (<https://arxiv.org/abs/2306.00814>, iFFT head, **13× faster than
HiFi-GAN, 70× faster than BigVGAN** — but keeps MPD + MRD and switches to a *hinge* GAN objective),
to RAF at **Interspeech 2026**:

**No mainstream neural vocoder or audio codec found, 2020→2026, drops the adversarial objective.**
Every architecture change keeps the discriminator stack. **This is the strongest surviving
stronghold and the chapter should say so plainly.** Suggested explanation for students: 44.1 kHz
waveforms are perceptually dominated by phase and fine spectral texture, where an L1/L2 objective
provably blurs — and there is no "DINOv3 for audio" to substitute in.

`UNVERIFIED`: BigVGAN-v2 has no standalone arXiv paper; cite the
[HF card](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x).

### 4.7 Super-resolution and restoration

**Real-ESRGAN** (Wang, Xie, Dong, Shan, ICCVW 2021, <https://arxiv.org/abs/2107.10833>) contributed
two ideas that outlived it: high-order degradation modelling with pure synthetic pairs, and a
**U-Net discriminator with spectral normalization**.

By 2025–2026, diffusion displaced the GAN *prior* — but every competitive **deployable** restorer
collapses the diffusion model to one step, and almost all use an adversarial term to do it:
**AdcSR** (CVPR 2025, <https://arxiv.org/abs/2411.13383>) uses "adversarial distillation to
compensate for the performance lost to pruning" (−74% params, up to 9.3× speedup); **SeedVR2**
(ICLR 2026, <https://arxiv.org/abs/2506.05301>) is literally the APT recipe applied to restoration;
**ODTSR** and **FlashClear** both use adversarial distillation. Counterexample worth citing:
**OSEDiff** (NeurIPS 2024, <https://arxiv.org/abs/2406.08177>) is one-step and **not** adversarial.

**Framing:** Real-ESRGAN's discriminator did not disappear — **it moved one layer up, from "the
model" to "the compressor of the model."**

### 4.8 The honest 2026 assessment

**Where adversarial objectives clearly survive**
1. **Audio vocoders and neural codecs** — universal, unchallenged, 2020→2026.
2. **Real-time / one-step video** — every real-time 720p video generator found uses adversarial
   post-training.
3. **Production few-step image models** — FLUX.1 [schnell] (12B), SDXL-Turbo, SD3-Turbo,
   SDXL-Lightning.
4. **Post-training/finishing of diffusion and flow models generally** — DMD2, DDO, Align Your Flow,
   GAS, Continuous Adversarial Flow Models (8.26 → 3.63 guidance-free FID).
5. **Deployable real-world restoration.**

**Where they have lost, or are losing**
1. **As the primary generative model for text-to-image.** StyleGAN-T's 13.90 vs Imagen's 7.27 at
   256² was the moment. No pure GAN has been SOTA text-to-image since.
2. **In tokenizers — actively reversing in 2026.** ViTok-v2 at 5B params states outright that
   adversarial losses "prevent stable scaling."
3. **As a *necessary* ingredient for few-step sampling.** sCM: 1.88 FID at 2 steps on ImageNet-512.
   MeanFlow: 3.43 at 1-NFE from scratch. Neither has a discriminator.
4. **In mindshare.** 99 citations in 20 months, a tenth of them from AI-image-*detection* papers
   citing R3GAN as a thing to detect. Not the citation profile of a live research program.

**The strongest argument FOR teaching GANs in 2026** — three concrete claims, not nostalgia:
1. **The adversarial term is the only known general-purpose fix for "learned distribution meets
   limited student capacity."** ADD's Table 1d is the proof (315.6 → 20.8). A student who never
   sees this has no model for why every one-step generator on their laptop has a discriminator in
   its training log.
2. **The reasons GANs were hard are now understood, and the fixes are short.** R3GAN deletes
   StyleGAN2's folklore, keeps RpGAN + R₁ + R₂, and gets 1000/1000 modes on StackedMNIST. **Mode
   collapse is a fixable regularization failure, not an intrinsic property of the minimax game.**
   That is a genuinely satisfying teaching arc.
3. **Pure GANs are competitive again at ImageNet scale in 2026.** GAT 2.18 and CAT 1.56 at one
   step on ImageNet-256; UCD 1.47 on ImageNet-64. A chapter claiming GANs are dead would be
   contradicted by three 2026 preprints.

**The strongest argument that GANs are a historical footnote** — give this real estate:
1. **Nobody's frontier model is a GAN.** The GAN appears only as a fine-tuning stage on top of
   something else. Teaching GANs as a generative-modelling *paradigm* inverts the actual
   dependency: **you now need diffusion to have a GAN worth training.**
2. **GAN-free alternatives keep matching it**, and DDO gets GAN-*style* benefits with no joint
   G/D training at all.
3. **Where scale binds, the GAN loss is removed first** — ViTok-v2's stated reason is instability
   under scaling, the exact 2017 criticism, unresolved in 2026.
4. **The adversarial recipes are being absorbed.** LADD made the discriminator *the teacher*; DDO
   made it *a likelihood ratio*; Adversarial Flow Models is a flow model wearing a discriminator.
   The clean 2014 GAN abstraction is dissolving into diffusion/flow training.

---

## 5. Evaluation practice

### 5.1 FID — Heusel et al., NeurIPS 2017

**"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"** — Heusel,
Ramsauer, Unterthiner, Nessler, Hochreiter. arXiv: <https://arxiv.org/abs/1706.08500>;
[proceedings](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html).

The paper's headline contribution is actually the **Two Time-Scale Update Rule** (separate learning
rates for D and G, with a stochastic-approximation convergence proof); FID is introduced as the
metric to track it.

$$d^2\big((\boldsymbol{m},\boldsymbol{C}),(\boldsymbol{m}_w,\boldsymbol{C}_w)\big) = \lVert \boldsymbol{m}-\boldsymbol{m}_w\rVert_2^2 + \operatorname{Tr}\!\big(\boldsymbol{C}+\boldsymbol{C}_w-2(\boldsymbol{C}\boldsymbol{C}_w)^{1/2}\big)$$

**This is the Bures–Wasserstein formula** — the squared `W₂` between two Gaussians — which is
exactly the closed form our note records in §"Closed forms for Wasserstein distance". **The chapter
should make this connection explicitly**: FID is `W₂²` between Gaussians fitted to Inception
features, i.e. one of the only two cases where the `W₂` supremum is analytic. That is a strong
structural link between the objectives section and the evaluation section, and it is the kind of
thing a textbook is uniquely positioned to point out.

- Feature layer: *"we use the last pooling layer as coding layer"* = Inception-v3 pool3, 2048-d.
- Sample count: *"we generate 50,000 images, propagate them through the Inception-v3 model, and
  then compute the mean m and the covariance matrix C"* — origin of "FID-50k".
- The authors' reference implementation (<https://github.com/bioinf-jku/TTUR>) recommends
  "a minimum sample size of 10,000 ... otherwise the true FID of the generator is underestimated".

### 5.2 Inception Score and its critique

Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016,
<https://arxiv.org/abs/1606.03498>: `exp(E_x KL(p(y|x) ‖ p(y)))`.

Barratt & Sharma, *A Note on the Inception Score*, <https://arxiv.org/abs/1801.01973>: IS "fails to
provide useful guidance when comparing models". Concrete findings: mean IS is **3.5% higher for
ImageNet validation images and 11.5% higher for CIFAR validation images** depending on whether a
Keras or Torch Inception implementation is used; the `n_splits` convention "unnecessarily
introduce[s] an extra parameter that can change the final score"; applying IS off-domain (CIFAR-10)
yields nonsensical top classes ("Moving Van", "Sorrel", "Milk Can"); directly optimizing the score
reaches ~900 with visually meaningless images; and IS cannot detect training-set memorization.

### 5.3 FID's criticisms

**Finite-sample bias.** Chong & Forsyth, *Effectively Unbiased FID and Inception Score and Where to
Find Them*, CVPR 2020, <https://arxiv.org/abs/1911.07023>. FID and IS at finite `n` are biased
estimators, and the bias is **model-dependent** — so "model A may get a better score than model B
simply because model A's bias term is smaller". Fix: linear extrapolation in `1/n` to give
**FID∞ / IS∞**.

**ImageNet-class dependence.** Kynkäänniemi, Karras, Aittala, Aila, Lehtinen, *The Role of ImageNet
Classes in Fréchet Inception Distance*, ICLR 2023, <https://arxiv.org/abs/2203.06026>. FID "is
known to sometimes disagree with human judgement"; the cause is that the feature space "is so
close to the ImageNet classifications that aligning the histograms of Top-N classifications
between sets of generated and real images can reduce FID substantially — without actually
improving the quality of results." **This is the paper R3GAN cites for the `*` "ImageNet feature
leakage" annotation in its tables** — the two connect directly.

**Disagreement with humans; unfairness to diffusion.** Stein et al., *Exposing flaws of generative
model evaluation metrics and their unfair treatment of diffusion models*, NeurIPS 2023,
<https://arxiv.org/abs/2306.04675>; library <https://github.com/layer6ai-labs/dgm-eval>. Largest
such study to date (17 metrics × 9 encoders against human perceptual-realism judgments): "no
existing metric strongly correlates with human evaluations"; diffusion models' human-judged realism
"is not reflected in commonly reported metrics such as FID"; over-reliance on Inception-v3 is a
cause; **DINOv2-ViT-L/14 features "allow for much richer evaluation"**. This is the origin of
**FD-DINOv2**.

**CMMD.** Jayasumana, Ramalingam, Veit, Glasner, Chakrabarti, Kumar, *Rethinking FID: Towards a
Better Evaluation Metric for Image Generation*, CVPR 2024, <https://arxiv.org/abs/2401.09603>.
Four concrete charges: (i) contradicts humans — in a case where "Model-A was preferred in 92.5% of
the comparisons, while Model-B was preferred only 6.9%", "COCO 30K FID and its unbiased variant
FID∞ unfortunately say otherwise"; (ii) for an iteratively-improving model, "FID and FID∞
incorrectly suggest that image quality degrades, when the quality improvements are obvious";
(iii) needs "over 20,000 images for reliable estimates"; (iv) the Gaussianity assumption on
Inception features does not hold.

Their replacement **CMMD** is the unbiased MMD U-statistic with a Gaussian RBF kernel
(`σ = 10`) over **CLIP ViT-L/14@336px** embeddings:

$$\widehat{d}^2_{\mathrm{MMD}}(X,Y)=\tfrac{1}{m(m-1)}\sum_{i}\sum_{j\ne i}k(x_i,x_j)+\tfrac{1}{n(n-1)}\sum_i\sum_{j\ne i}k(y_i,y_j)-\tfrac{2}{mn}\sum_i\sum_j k(x_i,y_j)$$

**This is a second strong structural link to the objectives chapter**: CMMD is literally the
unbiased MMD U-statistic estimator our note derives in §"Maximum mean discrepancy", applied to CLIP
features. The chapter can therefore say: *the two metrics that dominate generative evaluation are
the two IPM/`f`-divergence closed forms from §4 — FID is `W₂²` between Gaussians, CMMD is `MMD²`
with an RBF kernel* — and note that CMMD's advantages (no Gaussianity assumption, unbiased,
stable at ~1000 samples vs FID's 20,000) are exactly the estimation properties the note's
"Behavior under disjoint support" section predicts for MMD.

### 5.4 FD-DINOv2 and feature-backbone alternatives

- Origin: Stein et al. 2023 (above); DINOv2 itself is Oquab et al.,
  <https://arxiv.org/abs/2304.07193>.
- **EDM2** (Karras, Aittala, Lehtinen, Hellsten, Aila, Laine, *Analyzing and Improving the Training
  Dynamics of Diffusion Models*, CVPR 2024 oral, <https://arxiv.org/abs/2312.02696>) reports and
  **selects models by** FD_DINOv2 — its repo (<https://github.com/NVlabs/edm2>) documents config
  presets `edm2-img512-{xs..xxl}-dino  # Table 5, minimize fd_dinov2`.
- **CLIP-FID** appears in the Kynkäänniemi et al. ICLR 2023 codebase as a complementary,
  less-gameable metric.
- **R3GAN does NOT report FD-DINOv2** (verified by grepping the full proceedings text). It reports
  FID-50k, StackedMNIST modes + reverse-KL, and Recall.

### 5.5 Precision/Recall-family metrics

- **PRD** — Sajjadi, Bachem, Lucic, Bousquet, Gelly, *Assessing Generative Models via Precision and
  Recall*, NeurIPS 2018, <https://arxiv.org/abs/1806.00035>. Splits a single divergence number into
  precision (are samples realistic?) and recall (is the data distribution covered?).
- **Improved P/R (k-NN manifolds)** — Kynkäänniemi, Karras, Laine, Lehtinen, Aila, NeurIPS 2019,
  <https://arxiv.org/abs/1904.06991>. Around each feature vector draw a hypersphere of radius equal
  to the distance to its own k-th nearest neighbour within its own set; membership is "inside any
  such sphere":
  $$f(\phi,\Phi)=\mathbb 1\big[\exists\,\phi'\in\Phi:\ \lVert\phi-\phi'\rVert_2\le\lVert\phi'-\mathrm{NN}_k(\phi',\Phi)\rVert_2\big]$$
  precision = fraction of generated samples inside the real manifold; recall = fraction of real
  samples inside the generated manifold. **This is the "recall" R3GAN reports.**
- **Density & Coverage** — Naeem, Oh, Uh, Choi, Yoo, *Reliable Fidelity and Diversity Metrics for
  Generative Models*, ICML 2020, <https://arxiv.org/abs/2002.09797>. Argues the k-NN P/R metrics
  "fail to detect the match between two identical distributions", "are not robust against
  outliers", and have arbitrarily-chosen hyperparameters; proposes *density* (count of containing
  spheres rather than binary membership) and *coverage* (real-sample-centered).

### 5.6 StackedMNIST mode-count / reverse-KL protocol

**Origin: Metz, Poole, Pfau, Sohl-Dickstein, *Unrolled Generative Adversarial Networks*, ICLR 2017,
<https://arxiv.org/abs/1611.02163>.** Construction: "stacking three randomly chosen MNIST digits,
so as to construct an RGB image with a different MNIST digit in each color channel" → `10³ = 1000`
modes. A pretrained MNIST classifier is applied per channel; alongside the raw mode count they
report a KL "estimated tractably between the generated samples and the data distribution over
classes, where the data distribution is a uniform distribution over all 1,000 classes."

Also used by **VEEGAN** (Srivastava, Valkov, Russell, Gutmann, Sutton, NeurIPS 2017,
<https://arxiv.org/abs/1705.07761>) — note VEEGAN's reported numbers for prior methods differ from
those papers' self-reported ones because of different discriminator sizes; the "150 modes / 2.95"
row in R3GAN's Table 3 is VEEGAN's.

**"Reverse KL" here means `KL(p_θ ‖ p_data)`** over the 1000-way class histogram — the mode-seeking
direction, which is what makes it sensitive to mode dropping. R3GAN's `D_KL` column is this.

### 5.7 What modern papers actually report (2024–2026)

The landscape has **fragmented by task** — this is worth saying plainly in the chapter:

- **Unconditional / class-conditional image synthesis** (ImageNet, FFHQ, CIFAR-10, StackedMNIST)
  still runs on **FID-50k** as lingua franca, increasingly paired with **FD-DINOv2**. EDM2
  headlines "the previous record FID of 2.41 in ImageNet-512 synthesis to 1.81" but selects models
  by FD_DINOv2. R3GAN headlines FID-50k + modes/KL + Recall.
- **Text-to-image systems** have moved the headline claim to **compositional-alignment +
  human-preference** metrics, with FID demoted to a legacy row. Stable Diffusion 3 headlines
  **GenEval** and **human-preference ELO**
  (<https://stability.ai/news/stable-diffusion-3-research-paper>). GenEval itself — Ghosh,
  Hajishirzi, Schmidt, NeurIPS 2023 D&B, <https://arxiv.org/abs/2310.11513> — is object-detector
  based over 553 compositional prompts, built precisely because "metrics like FID or CLIPScore only
  offer a holistic measure ... unsuited for fine-grained or instance-level analysis."
- **Human-preference Elo arenas** are now a standard headline: **GenAI Arena**, Jiang et al.,
  NeurIPS 2024 D&B, <https://arxiv.org/abs/2406.04485> — blind side-by-side voting aggregated via
  **Bradley–Terry** into Elo. *(Pleasing for this chapter: the aggregation model for human
  evaluation is the same Bradley–Terry pairwise-logistic model as the relativistic critic. Worth a
  sentence.)*
- `UNVERIFIED`: specific 2026 vote counts for LM Arena / Artificial Analysis image arenas; the
  delegated pass had only secondary sources. Do not quote numbers.
- `LIGHTLY VERIFIED`: the exact metric set attributed to FLUX.1 (GenEval, T2I-CompBench++, TIFA,
  Pick-a-Pic, ELO) came via secondary sources only.

---

## 6. Conference tutorials 2023–2026

**Headline finding: there is nothing to borrow. Across NeurIPS, ICML, and CVPR from 2023 through
2026, no tutorial gives GANs a modern unified treatment.** The tutorial track has reoriented
almost entirely onto diffusion and flow matching. This is itself a citable fact for the chapter's
framing, and it argues that a unified divergence/IPM treatment of GANs is a *gap the textbook can
fill* rather than something to summarize from elsewhere.

**NeurIPS**
- 2023 — "Latent Diffusion Models: Is the Generative AI Revolution Happening in Latent Space?"
  (Kreis, Gao, Vahdat), <https://neurips.cc/virtual/2023/events/tutorial>. Diffusion only.
- 2024 — "Flow Matching for Generative Modeling" (Chen, Lipman, Ben-Hamu),
  <https://neurips.cc/virtual/2024/tutorial/99531>. No GAN content.
- 2025 — no diffusion/flow/GAN tutorial found; nearest are "Autoregressive Models Beyond Language"
  (Li, Chang, He) and "Science of Trustworthy Generative Foundation Models".
- 2026 — tutorial decisions notified 2026-08-07, i.e. **after** today; no list exists yet
  (<https://neurips.cc/Conferences/2026/CallForTutorials>).

**ICML**
- 2025 — "Flowing Through Continuous-Time Generative Models: A Clear and Systematic Tour" (Qiang
  Liu), <https://icml.cc/virtual/2025/40011>. Unifies flow vs diffusion; **no GAN mention**. Also
  "Harnessing Low Dimensionality in Diffusion Models" and "Generative AI Meets RL" — neither
  treats GANs.
- 2026 — "Diffusion and Flow-Matching: From Memorization to Generalization & Beyond" (Massias,
  Bertrand), <https://icml.cc/virtual/2026/75374>; "Unifying Attention and Diffusion with Kan
  Extension Transformers" (Mahadevan). No GAN-inclusive unification.

**CVPR**
- 2023 — "Denoising Diffusion Models: A Generative Learning Big Bang" (Song, Meng, Vahdat),
  <https://cvpr.thecvf.com/virtual/2023/tutorial/18546>. No GAN discussion.
- 2024 — 24 tutorials (<https://cvpr.thecvf.com/virtual/2024/events/tutorial>); the two generative
  ones are diffusion/score-distillation-centric.
- 2026 — "The Principles of Diffusion Models: Real-Time Continuous & Discrete Diffusion" (Lai,
  Sahoo, Kim, Song, Mitsufuji, Ermon), <https://cvpr.thecvf.com/virtual/2026/tutorial/36147>.
  **The closest thing to a taxonomy tutorial** — billed as distilling "foundations into a small set
  of core ideas that unify variational, score-based, and flow-based approaches" — but the
  unification is *intra-diffusion-family* and explicitly **excludes** the adversarial family.

**ICLR** has no main-conference tutorial track comparable to the other three; as of ICLR 2026 an
"interactive tutorials track" appears only inside individual workshops. Report as a genuine
structural gap, not a search failure.

**Research-literature unifications worth borrowing from instead** (not tutorials):
- **MonoFlow** — Yi, Zhu, Liu, *MonoFlow: Rethinking Divergence GANs via the Perspective of
  Wasserstein Gradient Flows*, ICML 2023, <https://arxiv.org/abs/2302.01075>. Reframes adversarial
  training as a Wasserstein-gradient-flow particle evolution, arguing this "reveal[s] the
  fundamental difference between variational divergence minimization and adversarial training",
  covering both divergence-minimizing and IPM-based objectives. **The most GAN-taxonomy-explicit
  source found, and a natural companion to our note's `W₂`-geometry reading of the zero-centered
  penalty.**
- **DiffFlow** — Zhang, Shi, Yu, Xie, Li, *DiffFlow: A Unified SDE Framework for Score-Based
  Diffusion Models and Generative Adversarial Networks*, <https://arxiv.org/abs/2307.02159>.
  A single SDE whose drift interpolates between score-based diffusion and GAN dynamics.
  `UNVERIFIED`: peer-reviewed venue — treat as arXiv/workshop only.
- **Stochastic interpolants** — Albergo, Boffi, Vanden-Eijnden, JMLR 26(209):1–80, 2025; code
  <https://github.com/malbergo/stochastic-interpolants>. No dedicated conference tutorial found
  (`UNVERIFIED`/not found rather than confirmed absent).

---

## 7. Recommendations for the chapter

### 7.1 The framing that the evidence supports

**"The game was never the point; the critic was."** Every strand of this review converges on it:

- The note's own analysis shows the critic is doing **density-ratio estimation** in every objective
  (§Three axes of variation), and that the loss and the pairing structure only change how that
  estimate is parameterized and weighted.
- ADD's ablation (315.6 → 20.8) shows the adversarial loss is what puts a one-step sample on the
  manifold.
- DDO (ICML 2025 spotlight) gets the benefit with **no two-player optimization at all** — an
  implicit discriminator defined as a likelihood ratio against a frozen reference.
- ViTok-v2 removes the discriminator at 5B params and replaces it with a **DINOv3 perceptual
  loss** — i.e. a *fixed* critic instead of a learned adversarial one.

So the chapter should teach the adversarial loss as **a learned perceptual metric for "is this on
the data manifold?"** — the thing you reach for whenever a pixel/waveform-space `L2` would blur and
the target is a distribution your student cannot exactly represent. That framing survives the 2026
evidence in a way that "GANs are a model family" does not.

### 7.2 Structural suggestion

The note's spine (log loss → proper losses/f-divergences → IPMs → relativistic pairing → gradient
penalties) is sound and should be kept. Three additions the review supports:

1. **Close the loop between §objectives and §evaluation.** FID *is* the Bures–Wasserstein closed
   form from the note's `W₂` section; CMMD/KID *are* the unbiased MMD U-statistic from the note's
   MMD section. The two metrics that dominate generative evaluation are the two closed forms the
   chapter already derives. No other textbook makes this connection and it costs one paragraph.
2. **Use the R3GAN roadmap table as the chapter's empirical spine.** 7.52 → 12.46 (strip ten
   tricks) → 11.65 (better loss) → 9.95 → 7.05 (modern backbone) is a complete, honest,
   single-dataset narrative that lets you teach *why each hack existed* and *what replaced it*.
3. **Include the "where GANs matter in 2026" section with both sides.** The strongest teaching
   asset here is the tension, not a verdict. Give ADD Table 1d *and* MeanFlow/sCM.

### 7.3 Code the chapter should ship

R3GAN's entire loss is ~15 lines of PyTorch (§2.1). It is a near-perfect D2L notebook cell: it
computes something real, it is short and elegant, and it directly instantiates the note's `Φ(D)`,
`R₁`, and `R₂`. **But implement the non-saturating generator form as the code does, not the
paper's Eq. (2)** — and say why, since that discrepancy is itself a teaching moment about
saturating vs non-saturating objectives.

The note's `verify_rpgan.py` is also a good model for a "verify the closed form numerically" cell,
which fits the chapter's proof-plus-computation style.

### 7.4 Concrete fixes to the note, in priority order

| # | Item | Where | Severity |
|---|---|---|---|
| 1 | **Eq. (12) footnote** — upgrade from "should be checked" to a flat statement that it is a typo, with the four-source evidence | §Two point masses with a penalty | **High** — the note currently under-claims a verified finding |
| 2 | **R3GAN's generator is non-saturating in the official code**, contradicting "R3GAN uses no such variant" | §Saturation of the rank weight | **High** — factually wrong as written |
| 3 | **TV↔hinge attribution** — the identity is ours; Lim & Ye give the max-margin geometry, not the TV computation | §Changing the classification loss | Medium |
| 4 | **Cite PairGAN (Tong, Garipov, Jaakkola)** as nearest prior art for the mixture `(p⊗q + q⊗p)/2`; it strengthens rather than weakens the novelty claim | Theorem footnote | Medium |
| 5 | **Add a parenthetical reconciling LSGAN's "Pearson χ²" with triangular discrimination** (same object; argument order matters) | Table of proper losses | Low but a real reader trap |
| 6 | **Cite Arjovsky & Bottou Thm 2.5** for the `−log D` trick — sharper than the current remark | §Behavior under disjoint support | Low |
| 7 | **Add Mescheder's `γ_critical = 2\|f′(0)\|`** (only meaningful with the square) | §Two point masses with a penalty | Low, but a nice payoff |
| 8 | **Note Roth et al.'s penalty carries a divergence-specific weighting `f^{c″}∘ψ`** — the note hedges correctly but the exact quote is better | §Smoothing and disjoint support | Low |

**Not a fix:** the note's table of proper losses is correct — all five rows verified numerically to
machine precision. Do not change it.

### 7.5 What is genuinely missing from the literature (and the chapter can supply)

No conference tutorial 2023–2026 gives GANs a modern unified treatment (§6). The only
GAN-inclusive divergence/IPM unifications are individual arXiv papers (MonoFlow, DiffFlow, and now
Naeini et al.'s Bayes-error framing). **A textbook chapter organized around closed-form values of
adversarial objectives would be filling a real gap, not restating a known synthesis.** That is the
strongest argument for writing this chapter the way the note frames it.

---

## 8. The other exits: how else researchers removed the heuristics

R3GAN's framing is that instability forced a stack of heuristics, which in turn froze GAN
architecture development. That diagnosis was widely shared, and R3GAN's route — *fix the objective,
then modernize the backbone* — was only one of three responses the field actually pursued. This
section verifies and corrects a three-way taxonomy: **fix the game**, **tame the critic**,
**remove the game**.

**The organizing observation**, which the chapter can state up front: the three paths differ in
*what they treat as the source of the instability*. Path 1 blames the **dynamics** (a vector field
with imaginary eigenvalues), path 2 blames the **critic** (an unconstrained, co-adapting
adversary), path 3 blames the **minimax structure** itself. All three worked; only the third won
the frontier, and it did so by making the adversarial term optional rather than absent.

### 8.1 Path 1 — Fix the game

The intellectual spine runs: *diagnose the failure → identify it as a property of the gradient
vector field → add a term that moves the eigenvalues left → prove local convergence.*

**Diagnosis. Arjovsky & Bottou (2017)**, *Towards Principled Methods for Training Generative
Adversarial Networks*, <https://arxiv.org/abs/1701.04862>. Establishes that the failure is
structural, not a tuning problem: with disjoint supports a perfect discriminator exists with
`∇_x D* = 0` on both supports (Thm 2.1); the generator's gradient norm is bounded by `Mε/(1−ε)` as
`D` approaches optimality (Thm 2.4); and the `−log D` trick's gradient equals that of
`KL(P_g‖P_r) − 2·JSD(P_g‖P_r)`, whose second term is *maximized* (Thm 2.5). Proposed remedy:
additive Gaussian noise on real and generated samples. **Cost:** the analysis prescribes noise,
which blurs the target distribution and requires an annealing schedule nobody knew how to set.

**Local stability. Nagarajan & Kolter (2017)**, *Gradient descent GAN optimization is locally
stable*, NeurIPS 2017, <https://arxiv.org/abs/1706.04156>. Shows that despite being
non-convex-non-concave, equilibria of the **traditional** GAN are locally asymptotically stable
under conditions — while **WGAN "can have non-convergent limit cycles near equilibrium"**. They
add a regularizer that guarantees local stability. Per Mescheder et al.'s supplementary, their
term differs from consensus optimization in two specific ways: it *"proposed to only regularize the
component `∇_ψ L(θ,ψ)` of the gradient vector field corresponding to the discriminator
parameters"*, and *"the regularization term is only added to the generator objective to give the
generator more foresight."* **Cost:** an extra second-order-ish term on the generator, and
guarantees that are strictly local.

**Numerics / consensus optimization. Mescheder, Nowozin, Geiger (2017)**, *The Numerics of GANs*,
NeurIPS 2017, <https://arxiv.org/abs/1705.10461>. The cleanest statement of the mechanism:
convergence suffers from *"i) presence of eigenvalues of the Jacobian of the gradient vector field
with zero real-part, and ii) eigenvalues with big imaginary part."* Their fix penalizes the squared
norm of the whole vector field,

$$R(\theta,\psi)=\frac{\gamma}{2}\|v(\theta,\psi)\|^{2}=\frac{\gamma}{2}\big(\|\nabla_\theta L\|^{2}+\|\nabla_\psi L\|^{2}\big),$$

which on the Dirac-GAN gives eigenvalues `λ₁/₂ = −γf′(0)² ± i f′(0)` — negative real part for all
`γ > 0`. **Cost, in Mescheder et al. (2018)'s own words:** *"consensus optimization has the drawback
that it can introduce new spurious points of attraction to the GAN training dynamics. While this is
usually not a problem for simple examples, it can be a problem for more complex ones like deep
neural networks."* It also requires second derivatives of the full objective.

**Smoothing. Roth et al. (2017)** — §2.6 above. **Cost:** the principled version carries a
divergence-specific weighting `f^{c″}∘ψ`, which practitioners dropped; and the approximation is
first-order in the noise level.

**Zero-centered penalties. Mescheder, Geiger, Nowozin (2018)** — §2.4 above. This is the paper that
made the taxonomy *decidable*, because Dirac-GAN gives a two-line test that every method either
passes or fails. Its verdicts, all verified from source:

| method | Dirac-GAN verdict |
|---|---|
| GAN (saturating), simultaneous GD | eigenvalues `±f′(0)i` — **does not converge**, exact circles |
| GAN (**non-saturating**) | eigenvalues still `±f′(0)i`; *not* locally convergent for any `h>0`, though the **continuous** dynamics converge at a **logarithmic (sublinear)** rate |
| **WGAN-GP / DRAGAN** | **does not converge locally**; the regularized vector field *"has a discontinuity at the equilibrium point"* |
| Consensus optimization | converges, `λ = −γf′(0)² ± i f′(0)` |
| Instance noise | converges |
| **`R₁` / `R₂` (0-GP)** | converges, `λ = −γ/2 ± √(γ²/4 − f′(0)²)`, with `γ_critical = 2\|f′(0)\|` giving a locally rotation-free field |

**The non-saturating row is a genuinely useful teaching result** and is usually misreported: the
non-saturating trick does *not* fix local convergence — it only downgrades divergence to very slow
convergence.

**Game-dynamics algorithms** — the "change the optimizer, not the objective" branch. **Daskalakis,
Ilyas, Syrgkanis, Zeng (ICLR 2018)**, *Training GANs with Optimism*,
<https://arxiv.org/abs/1711.00141>: Optimistic Mirror Descent, proving that *"in the case of
bi-linear zero-sum games the last iterate of OMD dynamics converges to an equilibrium, in contrast
to GD dynamics which are bound to cycle."* **Gidel, Berard, Vignoud, Vincent, Lacoste-Julien
(ICLR 2019)**, *A Variational Inequality Perspective on Generative Adversarial Networks*,
<https://arxiv.org/abs/1802.10551>: recasts GAN training as a variational inequality and imports
**averaging, extrapolation, and "extrapolation from the past"** into SGD/Adam. **Balduzzi,
Racanière, Martens, Foerster, Tuyls, Graepel (ICML 2018)**, *The Mechanics of n-Player
Differentiable Games*, PMLR 80, <https://arxiv.org/abs/1802.05642>: decomposes the game Jacobian
into a **potential** part (gradient descent on an implicit function) and a **Hamiltonian** part
obeying a conservation law, motivating **Symplectic Gradient Adjustment**. *(The conservation law
is exactly why Dirac-GAN's trajectories are circles — the two papers are describing the same
phenomenon, which is worth pointing out.)* **Gidel et al. (AISTATS 2019)**, *Negative Momentum for
Improved Game Dynamics*, <https://arxiv.org/abs/1807.04740>: proves **alternating updates are more
stable than simultaneous**, and that alternating updates with a **negative** momentum term converge
on saturating GANs. **This line is why R3GAN uses Adam `β₁ = 0`** — the paper cites it explicitly:
*"Since optimal negative momentum is another challenging hyperparameter, we do not use any momentum
to avoid worsening GAN training dynamics."*
**Cost of the whole branch:** extra gradient evaluations per step (extragradient roughly doubles
them), extra state, and one more hyperparameter — for guarantees that are mostly bilinear/local.
It is the least-adopted of the three paths in practice.

**Landscape. Sun, Fang, Schwing (2020)** — §2.5. **Cost:** the guarantee is about the empirical
landscape, and pairing costs `O(k²)` per batch if you use the all-pairs estimator.

**Culmination: R3GAN (2024)** — §2.1. Its contribution to *this* path is the observation, verified
by ablation, that the two failures are **independent**: `R₁+R₂` buys convergence, RpGAN buys mode
coverage, and neither substitutes for the other.

#### The two papers the coordinator flagged — verified, with one correction

**⚠️ "GANs Settle Scores!" has been retitled, and the taxonomy placement needs adjusting.**
arXiv:2306.01654 was submitted 2 June 2023 as **"GANs Settle Scores!"** (Siddarth Asokan, Nishanth
Shetty, Aadithya Srikanth, Chandra Sekhar Seelamantula) and **replaced on 31 July 2025** by v2,
retitled **"Insights into Closed-form IPM-GAN Discriminator Guidance for Diffusion Modeling"**
(author order changed, Srikanth first). <https://arxiv.org/abs/2306.01654>;
v1 at <https://arxiv.org/abs/2306.01654v1>. Anyone citing the MIT 6.S978 reading list
(<https://mit-6s978.github.io/schedule.html>, which lists the v1 title) will hit a title mismatch.

**It does not belong in path 1 as a stabilization method** — it is a *unification* result, and it
sits between path 1 and path 3. From the v1 abstract, verbatim: *"In `f`-divergence-minimizing GANs,
we show that the optimal generator is the one that matches the score of its output distribution
with that of the data distribution, while in IPM GANs, we show that this optimal generator matches
score-like functions, involving the flow-field of the kernel associated with a chosen IPM
constraint space. Further, the IPM-GAN optimization can be seen as one of **smoothed
score-matching**, where the scores of the data and the generator distributions are convolved with
the kernel associated with the constraint."*

**This is directly relevant to our note and should probably be cited in it.** The note computes what
the *discriminator* computes at optimum; this paper computes what the *generator* optimum is, in
score terms — and its "smoothed score matching" reading of IPM-GANs is the same mechanism the note
identifies in §"Smoothing and disjoint support" (Roth et al.) and §"Zero-centered gradient
penalties". It is the closest thing in the literature to the bridge the note is building between
the adversarial and score-based families. The v2 turns the insight into a practical method:
closed-form kernel-based discriminator guidance improving CLIP-FID and KID atop DDIM and LDM
baselines.

**Diffusion-GAN belongs in path 1, as the modern descendant of instance noise.** Wang, Zheng, He,
Chen, Zhou, ICLR 2023, <https://arxiv.org/abs/2206.02262>. Its abstract states the lineage
explicitly: *"a promising remedy of injecting instance noise into the discriminator input has not
been very effective in practice."* Diffusion-GAN makes it work by using a **forward diffusion chain
to generate Gaussian-mixture instance noise**, with an **adaptive** chain length balancing noise and
data levels, and a **timestep-dependent discriminator** that sees diffused real vs diffused fake at
every noise level. So the arc is: Sønderby et al. instance noise → Arjovsky & Bottou's noise
prescription → Roth et al.'s analytic approximation → `R₁`/`R₂` → Diffusion-GAN's adaptive,
multi-level version. **Cost:** backpropagation through the diffusion chain, an adaptive schedule to
tune, and a discriminator that must handle all noise levels.

**Not to be confused with DDGAN** — Xiao, Kreis, Vahdat, *Tackling the Generative Learning Trilemma
with Denoising Diffusion GANs*, ICLR 2022, <https://arxiv.org/abs/2112.07804>. This is **not** a
stabilization method: it models each *denoising step's* distribution with a conditional GAN so that
large denoising steps become possible, cutting the step count. It belongs in §4 (few-step
generation), and indeed R3GAN's StackedMNIST table cites it (1000 modes, 0.071 KL).

### 8.2 Path 2 — Tame the critic

Here the diagnosis is different: instability comes from an **unconstrained adversary that co-adapts
with the generator**. Constrain it, or replace it with something that cannot co-adapt, and the
heuristics become unnecessary.

**Structural constraint. Spectral normalization** (§2.6). The key property for this taxonomy is
that SN is a **reparameterization, not a penalty**: `W̄_SN(W) = W/σ(W)` makes the Lipschitz bound
`‖f‖_Lip ≤ ∏_l σ(W^l)` hold **by construction at every step**, with no coefficient to tune and no
extra loss term — unlike WGAN-GP, which enforces the constraint only in expectation, only where it
samples, and only if `λ` is right. It costs one power iteration per step with a persistent `u`
vector. The paper's own claim that it displaces the stabilizer stack, verbatim: *"in the absence of
complimentary regularization techniques (e.g., batch normalization, weight decay and feature
matching on the discriminator), spectral normalization can improve the sheer quality of the
generated images better than weight normalization and gradient penalty"*, and its Lipschitz constant
*"is the only hyper-parameter to be tuned."* **Cost:** a hard global Lipschitz bound caps
discriminator capacity, which can under-fit complex data; and the bound is a product of per-layer
norms, which is loose.

**Frozen pretrained features as the critic.** The idea: if the discriminator's *features* are fixed,
it cannot co-adapt with the generator, so the arms race that destabilizes training is defused.

- **Projected GANs Converge Faster** — Sauer, Chitta, Müller, Geiger, NeurIPS 2021,
  <https://arxiv.org/abs/2111.01007>. Projects both real and generated samples into a **frozen
  pretrained EfficientNet-Lite1** feature pyramid (verified from `sec_ablations.tex`: *"In the
  following, we thus use EfficientNet-Lite1 as our feature network"*; they ablate
  EfficientNet-Lite0–4, ResNet-18/50, ResNet-50-CLIP, DeiT and ViT, and find *"compact EfficientNets
  outperform both ResNets and Transformers"*). Features are taken at four resolutions
  `{64², 32², 16², 8²}` and each gets its **own independent discriminator**. Two fixed
  random-projection modules, both frozen after Kaiming init, stop the discriminators latching onto a
  feature subset: **cross-channel mixing (CCM)**, random 1×1 convolutions, and **cross-scale mixing
  (CSM)**, random 3×3 convs plus bilinear upsampling forming a small random U-Net. Headline claim,
  verbatim: Projected GANs *"match the previously lowest FIDs up to 40 times faster, cutting the
  wall-clock time from 5 days to less than 3 hours given the same computational resources."* On
  LSUN-Church, FID 3.18 after **1.1M** images versus StyleGAN2's previous best 3.39 after **88M** —
  ~80× fewer images.
  **Cost — and the paper says it out loud in its own Discussion:** the generator inherits the frozen
  classifier's blind spots, producing systematic **"floating heads"** on AFHQ (sharp subject, bland
  or incoherent background) and visible proportion errors on FFHQ *even at state-of-the-art FID*,
  which they attribute to the classification network being largely background-invariant. Plus the
  metric problem in the verdict below.
- **Vision-Aided GAN** — exact title **"Ensembling Off-the-shelf Models for GAN Training"** —
  Kumari, Zhang, Shechtman, Zhu, CVPR 2022 (Oral), <https://arxiv.org/abs/2112.09130>. Rather than
  one backbone, **eight** frozen off-the-shelf models (CLIP, VGG-16, DINO, MoBY, a face-parsing net,
  a face-normals net, a Swin ADE-20K segmenter, a Swin MS-COCO detector) each get a small trainable
  head and act as *additional* discriminators alongside the standard StyleGAN2 one. Selection is
  automatic via **linear-probe separability** — train a logistic head on frozen features, pick the
  backbone with the lowest real-vs-fake validation error — and models are added **progressively**,
  one at a time. Inception is deliberately excluded to avoid confounding FID.
  **The crucial caveat, in their own words:** *"we also observe that only using off-the-shelf models
  as the discriminator leads to divergence."* **It is inherently a hybrid, not a replacement.**
  Results: with only 10k training samples, LSUN-Cat FID matches StyleGAN2 trained on 1.6M images;
  full-data FID improves ~1.5–2× (LSUN Cat 6.86→3.98, Church 4.28→1.72, Horse 4.09→2.11). Because
  these are ImageNet-adjacent backbones, the **human study matters more than the FID**: preference
  over StyleGAN2-ADA was 53.8% ± 1.3 (FFHQ), 60.5% ± 1.7 (LSUN Church), 63.5% ± 1.6 (LSUN Cat) —
  real gains, but on faces barely above chance. **Cost:** up to three large backbones running
  forward every discriminator step, plus the probe search to choose them.
- **StyleGAN-T** — Sauer, Karras, Laine, Geiger, Aila, ICML 2023 (PMLR v202, 30105–30118),
  <https://arxiv.org/abs/2301.09515>. Squarely in this line, and it makes a **deliberate refinement
  the chapter should highlight**: it retains "a frozen, pretrained feature network and ... multiple
  discriminator heads" but switches to a **self-supervised DINO ViT-S** with five heads spaced
  evenly across the transformer's layers, stating explicitly that *"an additional benefit of using a
  self-supervised feature network is that it circumvents the concern of potentially compromising
  FID"* — citing Kynkäänniemi et al. directly. The redesign alone improved FID/CLIP-score by ~40% in
  their ablation and made the discriminator ~2.5× faster than StyleGAN-XL's. **This is the field
  correcting the leakage problem from inside path 2.**
- **GigaGAN — correction: it is a hybrid, not a frozen-feature discriminator.** Kang, Zhu, Zhang,
  Park, Shechtman, Paris, Park, CVPR 2023, <https://arxiv.org/abs/2303.05511>. Its *primary* image
  discriminator is a **trainable** multi-scale convolutional pyramid. Frozen pretrained features
  enter only as two **auxiliary** terms: a CLIP contrastive loss on generator outputs, and an
  explicit **"vision-aided adversarial loss"** that is the Vision-Aided GAN mechanism above (cited
  directly), using a frozen CLIP encoder plus a Projected-GAN-style fixed random projection. Best
  described as sitting between path 1 and path 2.

**Feedback-controlled augmentation.** The other half of path 2: stop the discriminator memorizing
by changing what it sees, adaptively.

- **DiffAugment** — Zhao, Liu, Lin, Zhu, Han, NeurIPS 2020, <https://arxiv.org/abs/2006.10738>. The
  crucial design point is in the name: augmentations are applied to **both real and fake samples**
  and are **differentiable**, so gradients propagate through them to the generator. Augmenting only
  reals (the naive approach) "yields little benefit" because it shifts the real distribution only.
  Results: ImageNet-128 FID **6.80** / IS **100.8** on a BigGAN backbone; CIFAR-10/100 matched using
  **20% of the data**; 100-shot with no pretraining, StyleGAN2 FID 80.20 → **46.87** on Obama,
  34.27 → **12.06** on panda. **Cost:** the policy is a **fixed, hand-picked hyperparameter per
  dataset** with no runtime correction — too weak or too strong and nothing adapts.
- **StyleGAN2-ADA** — §2.6. Same non-leaking principle, but **closed-loop**: the overfitting
  statistic `r_t = E[sign(D_train)]` (chosen over alternatives as *"far less sensitive to the chosen
  target value and other hyperparameters"*) drives the augmentation probability `p`, adjusted **once
  every four minibatches** toward a target of **0.6**, sized so `p` can travel 0→1 in ~500k images.
  Their own ablation shows a grid-searched *fixed* `p` is "too strong in the beginning and too weak
  towards the end" — which is the argument for control. **Cost:** an extra control loop and its
  target constant; R3GAN found the controller unnecessary and replaced it with a fixed cosine
  schedule "without any performance degradation."
  > **Attribution note:** the fixed-policy-vs-closed-loop-control framing is *our* analytical
  > synthesis. ADA and DiffAugment are contemporaneous NeurIPS 2020 papers and **neither cites the
  > other**; do not attribute the contrast to either paper.

#### Assessment: frozen features vs the loss — how much does each explain?

The honest answer is **both matter, and the published FID numbers systematically overstate the
frozen-feature contribution.**

*For frozen features.* Projected GAN's up-to-40× wall-clock speedup (and 80× sample-efficiency gain
on LSUN-Church) is real and large, and the mechanism is sound: a frozen critic cannot enter an arms
race, and multi-scale feature discriminators give dense, well-conditioned gradients from step one.
StyleGAN-T attributes ~40% of its FID/CLIP improvement to the discriminator redesign alone. And
Vision-Aided GAN's gains show up in a **human preference study** (53.8–63.5%), not only in FID —
which is the evidence that survives the leakage objection. Every adversarial diffusion-distillation
method uses the same trick (ADD's frozen DINOv2, LADD's frozen teacher).

*Against — and this is now a measurement, not an allegation.* **Kynkäänniemi et al. (ICLR 2023)**,
<https://arxiv.org/abs/2203.06026>, ran the direct test: train Projected FastGAN (Projected GAN's
architecture, ImageNet-pretrained EfficientNet features) on FFHQ and compare against StyleGAN2 **at
matched standard FID**, then re-measure in a CLIP feature space that does not share the leakage
channel:

| model | FID | FID_CLIP | Recall |
|---|---|---|---|
| Projected FastGAN | **5.28** | 4.67 | 0.45 |
| StyleGAN2 | **5.30** | **2.76** | 0.46 |

Tied on standard FID; StyleGAN2 is **~1.7× better** in CLIP space, and the gap tracks their human
finding that Projected FastGAN "tends to generate lower quality and less diverse FFHQ samples than
StyleGAN2." Their diagnosis: *"the accidental leak of information from the pre-trained network,
causing the model to replicate the ImageNet-like aspects in the training data more keenly ... such
pre-training can make FID unreliable in practice."* **So for the Projected-GAN family, standard FID
overstates quality by a measurable margin.** R3GAN's `*` annotation on StyleGAN-XL / StyleSAN-XL /
PolyINR is well-founded, and R3GAN treats those rows as *not apples-to-apples* rather than merely
inflated.

*The decisive counter-evidence.* **R3GAN reaches FID 1.96 on CIFAR-10, 2.75 on FFHQ-256, and 1.27 on
ImageNet-32 with no pretrained features at all**, using ~25M-parameter networks — while
StyleGAN-XL, which does use them, needs a 125M-parameter discriminator. And **UCD** (arXiv:2510.00624)
reports FID 1.47 on ImageNet-64 "surpassing StyleGAN-XL". So frozen features are **not necessary**
for competitive FID.

**Verdict for the chapter.** Frozen pretrained features are best understood as buying
**optimization speed and data efficiency**, not correctness — they make the critic's gradients
immediately informative instead of requiring it to learn features from scratch, which is why they
dominate in low-data and few-step-distillation settings where you cannot afford a long
co-adaptation. The *loss* is what determines whether the fixed point is right and whether the
generator covers modes; R3GAN's StackedMNIST ablation (693 → 1000 modes from the loss alone, with
no pretrained anything) isolates this cleanly. A fair summary: **path 2 fixed the wall-clock
problem, path 1 fixed the correctness problem**, and the headline FID gaps between them are
partially an artifact of the metric.

The field's own behaviour confirms both halves at once: **StyleGAN-T kept the frozen-feature
mechanism but switched to a self-supervised DINO backbone precisely to escape the leakage
critique.** The mechanism survived; the ImageNet-classifier backbone did not.

#### 2025–2026: has path 2 continued past R3GAN?

**Largely no — the line has plateaued for general-purpose image generation.** Searching the
Projected-GAN citation graph for 2025–2026 work turns up **domain transplants and incremental
combinations, not a large-scale successor**:

- **Vocoder-Projected Feature Discriminator** — Kaneko, Kameoka, Tanaka, Kondo, **Interspeech 2025**,
  <https://arxiv.org/abs/2508.17874>. A direct transplant into speech: the discriminator operates on
  features from *"a pretrained and frozen vocoder feature extractor"* rather than raw waveforms,
  cutting training time/memory by **9.6× / 11.4×** at matched quality. *(Consistent with §4.6 — audio
  is where adversarial machinery keeps advancing.)*
- **HP-GAN: Harnessing pretrained networks for GAN improvement with FakeTwins and discriminator
  consistency** — Son, Lee, Hwang, *Neural Networks*, **2026**,
  <https://arxiv.org/abs/2602.03039>. Pretrained CNN and ViT features used both for a
  self-supervised "FakeTwins" loss and as parallel discriminators kept mutually consistent;
  evaluated on 17 datasets including limited-data regimes.
- **SCAD: Efficiency without Compromise — CLIP-aided Text-to-Image GANs with Increased Diversity** —
  Kobayashi, Takida, Shibuya, Mitsufuji, **IJCNN 2025**, <https://arxiv.org/abs/2506.01493>. Pairs
  CLIP-aided components with two Slicing-Adversarial-Network discriminators to fix a diversity
  collapse the authors identify in the pretrained-generator approach; competitive zero-shot FID at
  ~2 orders of magnitude lower training cost.
- **Not in this line, despite surface similarity:** GAT (arXiv:2509.24935) trains transformer G and
  D from scratch in VAE latent space with no frozen discriminator features — it is a scaling paper
  that follows R3GAN's *loss*; and *Improving GANs with Self-Distillation*
  (<https://arxiv.org/abs/2605.08577>, 2026) uses an EMA-generator guidance signal, which is a
  path-1 dynamics fix.

**Reading:** new SOTA claims in 2025–2026 come from the **loss/architecture** side (GAT, CAT, UCD,
DDO — §4.4), not from better frozen features. That is a mild vindication of R3GAN's thesis, and the
chapter can say so.

### 8.3 Path 3 — Remove the game

The most radical exit: if the inner maximization is the problem, delete it. Replace the learned
critic with a **fixed closed-form discrepancy**, or replace the minimax with a **regression
objective that has a unique optimum**.

**⚠️ The finding that organizes this whole path:** *every* attempt to replace the learned critic
with a fixed closed-form one was tried against real images, found wanting, and then **partially
re-admitted a learned or adversarial component** — at a different point in the pipeline. It happens
in Group A, again in Group B, and again in Group D. The chapter should foreground this: **the game
is remarkably hard to delete; it keeps reappearing one stage downstream.**

#### Group A — fixed closed-form critic (moment matching)

**Generative Moment Matching Networks** — Li, Swersky, Zemel, ICML 2015 (pp. 1718–1727),
<https://arxiv.org/abs/1502.02761> — and the concurrent, independent **Dziugaite, Roy,
Ghahramani**, *Training generative neural networks via Maximum Mean Discrepancy optimization*,
UAI 2015, <https://arxiv.org/abs/1505.03906>. (Concurrence is acknowledged in the latter, verbatim:
*"In independent work reported in a recent preprint, Li, Swersky, and Zemel also propose to use MMD
as a training objective for generative neural networks."*) Both train a generator by directly
minimizing **MMD with a fixed kernel** — no discriminator, no inner loop, a single well-defined
objective with an analytic gradient. This is exactly the note's MMD section used as a *training
objective* rather than an analytical device.

**The instructive part is how it failed.** GMMN's own motivation for its better variant, **GMMN+AE**
(MMD in the code space of a *separately pretrained, frozen* autoencoder), is statistical: *"the
amount of data required to produce a reliable [MMD] estimator grows with the dimensionality of the
data."* The numbers show the gap — Parzen-window log-likelihood on MNIST: GMMN **147 ± 2**,
GMMN+AE **282 ± 2**, GAN **225 ± 2**; on TFD 2085 / 2204 / 2057. So the raw-pixel version *loses* to
GANs and only the version with a learned feature map wins.

Then the field re-adversarialized it outright. **MMD GAN** (<https://arxiv.org/abs/1705.08584>)
states it plainly — *"the empirical performance of GMMN is still not as competitive as that of GAN
on challenging and large benchmark datasets"* — and replaces the fixed kernel with an
**adversarially learned** one. Their experiment is damning: on CIFAR-10 *"both GMMN variants fail to
generate meaningful images"*, and raising the batch size from 64 to 1024 still leaves GMMN *"not
competitive to MMD GAN."* **Demystifying MMD GANs** (<https://arxiv.org/abs/1801.01401>) gives the
reason: fixed-kernel methods *"struggle with complex natural images, where pixel distances are of
little value, and fixed representations can easily be tricked, as in the adversarial examples ...
Adversarial training of the MMD loss is thus an obvious choice."*

**Cost:** a fixed kernel in pixel space is a weak discrepancy in high dimension — the witness
function cannot adapt to the directions that matter — plus `O(n²)` per batch and acute bandwidth
sensitivity. **This is the single best cautionary tale in the taxonomy**, and it instantiates the
note's estimation-vs-discrimination trade-off precisely: MMD has the better *estimator*
(`O(n^{-1/2})`, no inner problem) and the worse *discriminative power*, and on images the second
dominates.

#### Group B — optimal transport with a tractable objective

**Genevay, Peyré, Cuturi**, *Learning Generative Models with Sinkhorn Divergences*, AISTATS 2018,
<https://arxiv.org/abs/1706.00292>. Entropic-regularized OT solved by `L` Sinkhorn fixed-point
iterations, **unrolled and backpropagated through** — so the "critic" is a differentiable *algorithm*
rather than a network. The Sinkhorn divergence provably interpolates between OT (`ε → 0`) and MMD
(`ε → ∞`), inheriting MMD's `O(1/√n)` sample complexity at large `ε` versus unregularized OT's
`O(n^{-1/d})` curse.

**Their CIFAR-10 Inception scores are a gift to the chapter** (their Table 2), because they show the
trade-off *inside a single method*:

| objective | Inception Score ↑ |
|---|---|
| MMD baseline | 4.04 ± 0.07 |
| Sinkhorn, `ε = 1000` (≈ MMD end) | **4.14 ± 0.06** |
| Sinkhorn, `ε = 100` | 3.09 ± 0.04 |
| Sinkhorn, `ε = 10` (≈ true OT end) | 3.11 ± 0.03 |

**The setting closest to true optimal transport is the *worst*** within a practical compute budget —
a concrete demonstration that the theoretically stronger metric is not the practically better
objective when you must estimate it from minibatches. **Cost:** `O(Lmn)` per batch (effectively
quadratic in batch size), a sensitive `ε` (*"too much regularization ... leads to a loose fit"*,
*"not regularizing enough ... yields poor performance"*), and minibatch energy bias — the minibatch
objective is not the population OT distance.

**Sliced Wasserstein.** 1-D Wasserstein has a closed form via **sorting**, so average it over random
projections. Canonical references: Deshpande, Zhang, Schwing, *Generative Modeling using the Sliced
Wasserstein Distance*, CVPR 2018, <https://arxiv.org/abs/1803.11188> (*"a single objective rather
than a saddle-point formulation"*); Kolouri, Pope, Martin, Rohde, *Sliced-Wasserstein Autoencoder*,
<https://arxiv.org/abs/1804.01947>; and Wu et al., *Sliced Wasserstein Generative Models*, CVPR 2019,
<https://arxiv.org/abs/1706.02631>, which uses a small number of *learned* orthogonal projections.

> **Venue note.** SWAE is **ICLR 2019**, not ICLR 2018 (DBLP `conf/iclr/KolouriPMR19`, OpenReview
> forum `H1xaJn05FQ`). `PARTIALLY VERIFIED` — arXiv carries no venue metadata for it, and at write
> time DBLP returned 503 and OpenReview's API was behind a challenge, so this rests on the
> delegated pass's two independent identifiers rather than my own direct check.

**And here the game creeps back twice.** First, the CVPR 2018 paper's own remedy for uninformative
random directions is to **bolt on a discriminator network trained with a standard `−log D` loss,
purely to select good projection directions** — the flagship "no game" sliced-Wasserstein generator
re-admits an adversary. Second, **Max-Sliced Wasserstein** (Deshpande et al., CVPR 2019,
<https://arxiv.org/abs/1904.05877>) replaces averaging with the single *worst-case* direction found
by gradient ascent — an explicit, if cheap, inner maximization. Its headline result is worth
quoting: the max-sliced GAN *"using just one projection direction is able to produce results which
are either comparable or better than the sliced Wasserstein GAN even when using 10,000
[random] projections."* **Distributional Sliced-Wasserstein** (Nguyen, Ho, Pham, Bui, ICLR 2021,
<https://arxiv.org/abs/2002.07367>) learns a *distribution* over directions and provably generalizes
Max-SW. **Cost:** random projections are statistically inefficient in high dimension; every fix for
that reintroduces an inner optimization.

`UNVERIFIED` (do not cite without checking): arXiv IDs for Nguyen & Ho, *Amortized Projection
Optimization for Sliced Wasserstein Generative Models* (NeurIPS 2022) and Lezama, Chen, Qiu,
*Run-Sort-ReRun* (ICML 2021).

#### Group C — nearest-neighbour / implicit likelihood

**IMLE** — Ke Li & Jitendra Malik, *Implicit Maximum Likelihood Estimation*,
<https://arxiv.org/abs/1809.09087>. **Venue: arXiv preprint only** (DBLP lists `CoRR 2018`, no
peer-reviewed venue) — do not attribute a conference. The objective:

$$\hat\theta = \arg\min_\theta\ \mathbb E\Big[\textstyle\sum_i \min_j \big\lVert \tilde x^\theta_j - x_i\big\rVert^2\Big]$$

For each **real data point**, draw a fresh batch of `m ≥ n` generator samples, find the nearest
*generated* sample, and pull it toward the data point. **The direction is the entire point and the
chapter should dwell on it:** because every *data* point must have some generated sample near it or
keep incurring loss, no mode can be left uncovered — mode collapse is structurally impossible.
This is the same failure the relativistic loss attacks, approached from the opposite side, and the
contrast makes a good exercise. No discriminator anywhere.

**Cost:** `m ≥ n` fresh samples plus a nearest-neighbour search every outer iteration, so
tractability depends on approximate-NN machinery (the authors' own DCI / Prioritized-DCI), and cost
scales with batch/dataset size in a way GAN training does not.

Lineage, all verified: conditional IMLE for super-resolution (<https://arxiv.org/abs/1810.01406>,
also arXiv-only) → Multimodal CIMLE (IJCV 2020, <https://arxiv.org/abs/2004.03590>) → CAM-Net
(<https://arxiv.org/abs/2106.09015>) → CHIMLE (<https://arxiv.org/abs/2211.14286>) → **RS-IMLE**,
*Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis* (Vashist, Peng, Li,
Sept 2024, <https://arxiv.org/abs/2409.17439>), which diagnoses a train/test prior mismatch
(*"inadequate correspondence between the latent codes selected for training and those drawn during
inference"*) and reports gains over GAN and IMLE baselines on nine few-shot datasets.
`UNVERIFIED`: no 2025/2026 IMLE paper beyond RS-IMLE could be confirmed.

> **Name-collision warning.** "Adaptive IMLE (AIMLE)" (Minervini, Franceschi, Niepert, AAAI 2023,
> <https://arxiv.org/abs/2209.04862>) is **unrelated** — a gradient estimator for discrete
> black-box layers, from Niepert et al.'s separate I-MLE line. Do not conflate.

#### Group D — the field-level answer: regression objectives with no equilibrium

Denoising score matching and flow matching removed the game not by finding a better fixed critic but
by changing *what is regressed*. **NCSN** (Song & Ermon, NeurIPS 2019 oral,
<https://arxiv.org/abs/1907.05600>), **DDPM** (Ho, Jain, Abbeel, NeurIPS 2020,
<https://arxiv.org/abs/2006.11239>), the **score SDE** framework (Song et al., ICLR 2021 oral,
<https://arxiv.org/abs/2011.13456>), and **Flow Matching** (Lipman, Chen, Ben-Hamu, Nickel, Le,
ICLR 2023, <https://arxiv.org/abs/2210.02747>) all train a network with a **per-sample squared-error
regression against a closed-form target** — a noise/score vector, or a conditional velocity field.
There is no inner maximization, no equilibrium, and a **unique global minimizer**, so the loss curve
is *readable*. That single property — being able to tell whether training is going well — arguably
ended the GAN era more decisively than any FID number.

Representative numbers: DDPM CIFAR-10 IS 9.46 / FID 3.17; NCSN IS 8.87 ± 0.12 / FID 25.32; score-SDE
IS 9.89 / FID 2.20; Flow Matching with an OT path on CIFAR-10, 2.99 bpd / FID 6.35 (versus their own
DDPM reimplementation at 3.12 bpd / FID 7.48) and ImageNet-128 FID 20.9.

**Cost, with a number:** Flow Matching's own Table 1 reports **NFE ≈ 122–193** with an adaptive
dopri5 solver even on the *efficient* OT path (≈183–441 for diffusion/score paths) — against **one**
forward pass for a GAN generator. **And that cost is exactly what brought the adversarial term
back.** ADD says so explicitly, combining score distillation *"with an adversarial loss to ensure
high image fidelity even in the low-step regime"* (<https://arxiv.org/abs/2311.17042>). The entire
§4 story — ADD, LADD, DMD2, APT, adversarial flow models — is the field paying the sampling debt by
reintroducing a discriminator as a *finishing* objective. **The game was removed from training and
returned at distillation** — the third instance of the pattern in this section.

### 8.4 What the three paths cost, side by side

| Path | Treats the cause as | Representative fix | What you give up |
|---|---|---|---|
| **1. Fix the game** | vector-field dynamics | `R₁`+`R₂`, consensus opt., extragradient/optimism/negative momentum, RpGAN | extra gradient/Hessian work; guarantees mostly local; `γ` to tune (0.05–150 in R3GAN alone) |
| **2. Tame the critic** | unconstrained co-adapting adversary | spectral norm; frozen pretrained features; ADA/DiffAugment | capacity caps, pretrained-feature dependence and FID leakage, extra frozen backbones, control-loop constants |
| **3. Remove the game** | the minimax itself | MMD/Sinkhorn/sliced-W/IMLE; diffusion & flow matching | weak fixed discrepancies (A–C) or expensive multi-step sampling (D) — the latter repaid by re-adding an adversarial term |

**The chapter's punchline, supported by all three columns:** nobody removed the heuristics by
being more careful. Each path removed them by making one structural commitment and paying for it
elsewhere — and the winning combination in 2026 is *path 3 for training, path 1 for the objective,
path 2 for the distillation critic.*

**A second punchline, which only becomes visible once path 3 is laid out in full:** the adversarial
game is extraordinarily hard to delete. It was removed from the objective and came back as a learned
kernel (MMD → MMD GAN); removed from the metric and came back as discriminator-selected or
worst-case projections (sliced-W → Max-SW); removed from training entirely and came back at
distillation (diffusion → ADD/LADD/DMD2). Three independent attempts, three returns. That is the
strongest single argument for teaching adversarial objectives in 2026 — not that GANs are the best
generative model, but that **whenever a fixed discrepancy proves too weak in high dimension, the fix
has repeatedly been to learn the critic**, and a student needs to understand what that critic
computes.

---

## Appendix: verification status summary

| Claim class | Status |
|---|---|
| R3GAN Eq. (12) typo | **Verified 4 ways** (LaTeX source, arXiv PDF, NeurIPS camera-ready, precursor project) |
| R3GAN Appendix B statements | **Verified verbatim** from LaTeX source + typeset PDF |
| R3GAN tables, hyperparameters, ablations | **Verified verbatim** from LaTeX source |
| R3GAN NeurIPS 2024 proceedings entry | **Verified** (camera-ready PDF downloaded) |
| Mescheder eigenvalue formula (with square) | **Verified** from LaTeX source, two locations |
| Jolicoeur-Martineau 2020 theorems + experiments | **Verified** from LaTeX source |
| Sun et al. 2020 theorem statements | **Verified** from LaTeX source |
| Note's proper-loss table (5 rows) | **Verified numerically**, exact |
| LSGAN ≡ triangular discrimination | **Verified numerically**, exact (ratio 8) |
| R3GAN non-saturating generator | **Verified** from official repo + numerical gradient check |
| Note's Theorem (`d_Rp = JS(p⊗q, q⊗p)`) | **Verified numerically** (10 dp, 4 trials) |
| Novelty of the identity | **~80% confidence** — see §3.1 for the three residual doubts |
| §2.6 classical papers | Delegated pass, primary-PDF verified; 4 items flagged `UNVERIFIED` |
| §4 GANs 2024–2026 | Delegated pass; several items flagged `UNVERIFIED` inline |
| §5 evaluation | Delegated pass; arena vote counts and FLUX metric set flagged |
| §6 tutorials | Delegated pass; absence claims are "not found", not "confirmed absent" |
| §8.1 path 1 (fix the game) | **Verified** — all 6 core papers' abstracts fetched from arXiv; Mescheder 2018's Dirac-GAN verdict table read from LaTeX source |
| §8.1 "GANs Settle Scores!" retitling | **Verified** — v1 and v2 abs pages both fetched; submission history confirms 2023-06-02 → 2025-07-31 |
| §8.2 path 2 (tame the critic) | **Verified** from LaTeX sources of Projected GAN, Vision-Aided GAN, DiffAugment, StyleGAN2-ADA, SN, StyleGAN-T, GigaGAN, Kynkäänniemi et al. I independently re-checked the Projected GAN backbone (**EfficientNet-Lite1**, `sec_ablations.tex` — an earlier draft of this report said Lite0 and was wrong). GigaGAN recharacterized as a hybrid after source check. ADA-vs-DiffAugment contrast marked as our synthesis (neither paper cites the other). |
| §8.3 path 3 (remove the game) | Delegated pass with primary-source quotes. Two venue corrections applied: **SWAE is ICLR 2019** (`PARTIALLY VERIFIED` — DBLP was 503 and OpenReview gated at write time) and **IMLE has no peer-reviewed venue** (arXiv-only; I re-checked both arXiv pages myself — neither carries venue metadata). Sliced-W amortized-projection and Run-Sort-ReRun arXiv IDs left `UNVERIFIED`; no post-2024 IMLE work confirmed. |

**Known method limitation:** the session's shared WebSearch budget (200 calls) was exhausted before
the novelty check could be exhaustively pursued. The cheapest way to raise confidence on §3 is a
full-text search pass (Google Scholar full text, or an arXiv full-text index) for the specific
formula and for "product measure" + "relativistic" combinations.
