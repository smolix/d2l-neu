# GAN pedagogy review: textbooks, tutorials, blogs, surveys (2026-08-02)

Research pass to inform a rewrite of d2l-neu's GAN chapter (`chapter_generative-adversarial-networks/`).
Covers Bishop & Bishop (2024), Murphy PML-Advanced-Topics (2023/2025-printing), Prince UDL (2023/2024),
Tomczak DGM 2nd ed. (2024), Foster GDL 2nd ed. (2023), Goodfellow's NeurIPS 2016 tutorial, Lilian Weng's
"From GAN to WGAN," distill.pub, Sebastian Raschka, R3GAN and the 2025-2026 "GAN revival" literature,
2023-2026 surveys, and the current d2l.ai chapter itself. All claims are sourced; anything not directly
verified is explicitly flagged as such rather than asserted.

Method note: sections 1-5 (the five textbooks) were researched by two parallel research agents that
downloaded primary-source PDFs/notebooks/companion repos wherever possible rather than relying on
marketing copy; sections 6-9 were researched directly. Provenance caveats (e.g. an unofficial mirror
used for one book) are carried through from the agents' own methodology notes.

---

## 1. Bishop & Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024) — Ch. 17

**Sources:** https://www.bishopbook.com/ (official — errata, a solutions PDF covering only ch. 2-10, a
figures .zip; no confirmed official free full-text PDF); chapter text was read from a **third-party
GitHub mirror**, `github.com/luzmontserrat/deep-learning` — content is DOI-stamped
(`10.1007/978-3-031-45468-4_17`) and internally consistent with the real Springer book, but **this
mirror's authorization could not be verified** (flagged explicitly); publisher page
https://link.springer.com/book/10.1007/978-3-031-45468-4 (login-walled, TOC not accessible); unofficial
community code/notes repo `github.com/gerdm/deep-learning-bishop` (exists, not investigated further, not
author-sanctioned).

**Position:** Chapter 17 of ~20, pp. 533-545 (~12 pages incl. 3 exercises) — the **first** of four
back-to-back "modern generative model" chapters: 16 Continuous Latent Variables (ending in **§16.4.4
"Four approaches to generative modelling"**, an explicit one-paragraph-each roadmap for GANs/flows/
VAEs/diffusion) → **17 GANs** → 18 Normalizing Flows → 19 Autoencoders (VAEs live as §19.2, not their
own chapter) → 20 Diffusion Models.

**Order of ideas:** §16.4.4 motivates GANs purely as a workaround for intractable likelihood ("we
abandon the concept of a likelihood function and instead introduce a second neural network whose
function is to provide a training signal") — not via two-sample testing or divergence minimization in
general. Chapter body: 17.1 Adversarial Training (minimax as binary cross-entropy → loss function,
conditional GAN in one paragraph → training in practice: mode collapse narrative, vanishing gradient
Fig. 17.2, LSGAN, instance noise, saturating-vs-non-saturating Fig. 17.3, Wasserstein/WGAN/WGAN-GP
compressed into ~1 page at the very end) → 17.2 Image GANs (DCGAN, ProgressiveGAN, BigGAN) → 17.2.1
CycleGAN as a full worked example (cycle-consistency loss derived, Figs. 17.6-17.10 incl. latent-space
vector arithmetic).

**Math depth:** Thin in the main body for a Bishop book. The cross-entropy GAN loss is derived (~3
lines); everything else of substance is **asserted, not shown** — "we can show that... a fully optimized
GAN will have a generative distribution that matches the data distribution exactly" is stated with zero
derivation, no D*(x) formula, no JS-divergence identity in the main text. Wasserstein distance is
explained by the "earth mover" metaphor only — **no Kantorovich-Rubinstein duality, no Lipschitz-
constraint derivation**; WGAN-GP's loss is simply handed to the reader. **All the real proof content is
relegated to the exercises**: Exercise 17.1 (★★★) walks the reader through deriving D*(x), rewriting the
loss as JS divergence (named explicitly), and showing the global minimum is at p_G = p_data — this is
the book's *only* presentation of the classic Goodfellow et al. proof. Exercise 17.2 (★★★) is a genuinely
elegant device: a toy zero-sum game E(a,b)=ab standing in for generator/discriminator parameters; the
reader derives the continuous-time gradient ODEs and shows a(t) traces a circle around the saddle point
and never converges — a hand-rolled, simplified DiracGAN-style non-convergence result, done as homework
rather than exposition. No official solutions exist yet for these (the solutions manual only covers ch.
2-10).

**Code:** None whatsoever — no notebooks, no repo referenced in the text or on the official site.

**Figures worth emulating:** Fig. 17.1 (clean discriminator/generator loop schematic); **Fig. 17.2** (a
1-D toy showing p_data, p_G, and a near-step-function optimal discriminator with near-zero gradient
almost everywhere vs. a smoothed discriminator — genuinely excellent for motivating instance-
noise/LSGAN, a simpler cousin of Murphy's Fig. 26.4); **Fig. 17.3** (side-by-side `-ln(d)` vs. `ln(1-d)`
and their gradients, visualizing the saturating/non-saturating distinction — worth cloning directly);
Figs. 17.9-17.10 (DCGAN bedroom latent-space walk and "vector arithmetic in latent space," reproduced
from Radford et al. 2015, competent but not original).

**Verdict — steal:** the §16.4.4 "four approaches, one paragraph each" framing device for situating GANs
among VAE/flow/diffusion before any math; the toy-game convergence-failure exercise as a worked,
*derived* example (promote it to the main text rather than an exercise); Fig. 17.2's "why is the
discriminator gradient dead" picture. **Avoid:** burying the D*(x)/JSD proof and equilibrium analysis
entirely in unsolved exercises; the one-page, metaphor-only, no-duality treatment of Wasserstein/WGAN.

---

## 2. Murphy, *Probabilistic Machine Learning: Advanced Topics* (MIT Press, 2023 print / Dec-2025 2nd-printing online) — Ch. 26

**Sources:** official free PDF, `github.com/probml/pml2-book/releases/latest/download/book2.pdf`
(tag `2025-dec-10`, CC-BY-NC-ND); landing page https://probml.github.io/pml-book/book2.html; companion
notebooks verified via GitHub API at `github.com/probml/pyprobml/tree/main/notebooks/book2/26`; publisher
listing https://mitpress.mit.edu/9780262048439/probabilistic-machine-learning/.

**Position:** Chapter 26 of Part IV "Generation," pp. 893-924 (32 pages, **no exercises**) — deliberately
**last** of six generation chapters: 20 Generative models overview (§20.4 unifies **evaluation metrics**
for *all* generative-model families here — Inception Score derived Eq. 20.9-20.10, FID Eq. 20.11 with
its known bias problems noted, Kernel Inception Distance, precision/recall via a k-NN manifold
criterion — none of this is duplicated in ch. 26, a genuinely good organizational choice) → 21 VAEs →
22 Autoregressive → 23 Normalizing flows → 24 Energy-based models → 25 Diffusion → **26 GANs**.
Explicitly co-written by Mihaela Rosca, Shakir Mohamed, and Balaji Lakshminarayanan — DeepMind/Google
researchers whose own papers (DiracGAN follow-ups, f-GAN/implicit-models, GAN theory broadly) are the
literature this chapter distills, which shows in the derivation density.

**Order of ideas — the chapter's defining feature, general-to-specific rather than game-first:**
§26.1 frames the whole topic as implicit vs. prescribed ("likelihood-free"/"simulation-based") inference,
cross-referencing ABC as a sibling framework — GANs are introduced as *one instance*, not the primary
object. §26.2 "Learning by comparison" builds the general theory *before* naming GANs: three guiding
principles any valid training objective must satisfy → four concrete instantiations, each its own
subsection: **26.2.2** density-ratio estimation via binary classifiers (derives the density-ratio trick
p*/q=D/(1-D), a table of proper scoring rules mapping Bernoulli/Brier/exponential/hinge/spherical losses
to different divergences, **derives D*(x)=p*/(p*+q_θ)** and **proves the JS-divergence-at-optimum
identity**, connects LSGAN→Pearson-χ² and hinge-loss-GAN→total-variation distance); **26.2.3** f-
divergence variational lower bound via Fenchel/convex conjugates — this *is* f-GAN, derived as the
general case the binary-classifier trick specializes; **26.2.4** IPMs — general definition, a proof of
*why* not every function class gives a valid distance, derives the **Wasserstein-1/Kantorovich duality**
as the 1-Lipschitz IPM, states the WGAN min-max game, names spectral-norm/gradient-penalty as Lipschitz-
enforcement mechanisms, then derives **MMD** as the RKHS-norm-1 IPM including a learned-kernel
generalization; **26.2.5** moment matching, generalizing further; **26.2.6** "On density ratios and
differences" **proves f-divergences give exactly zero gradient signal under non-overlapping support**
(D_KL=∞, JSD=log 2 regardless of θ — the formal root cause of vanishing gradients, shown in Fig. 26.4),
contrasted with IPMs staying smooth — then honestly nuances its own point (a smooth neural critic
*approximating* an f-divergence bound also gives smooth gradients in practice, so the ratio-vs-difference
distinction "is less significant in practice"). **Only now, in §26.3**, does the chapter narrow to "GAN"
specifically: unifies everything above into one min-max template and a fully general loss-function table
`L_D = E[g(D)] + E[h(D)]`, `L_G = E[l(D)]` whose (g,h,l) triples recover vanilla-GAN, non-saturating-GAN,
WGAN, and f-GAN as **one formula** — the most compact unification found anywhere in this whole review.
§26.3.5 "Convergence of GAN training" is the mathematical high point: formally defines Nash equilibrium
(global/local), **proves** (q_θ=p*, D=½) is the global Nash equilibrium of the vanilla GAN objective (a
clean 3-line proof chaining D* and the JSD-minimizer fact), then works through **DiracGAN** [Mescheder
et al. 2018] in full — Dirac-delta data, linear-in-a-scalar generator/discriminator, derives the Jacobian
eigenvalues (±i·l'(0), purely imaginary), proves gradient descent traces a circular non-convergent
trajectory via a conserved-quantity argument, and shows continuous-time ODE analysis and discrete
gradient descent can give *contradictory* convergence conclusions at large step sizes — citing Runge-
Kutta integrators as a fix. §26.4 Conditional GANs (general conditional min-max + auxiliary-classifier
variant, both derived). §26.5 Inference with GANs (BiGAN/ALI, contrasted with the VAE ELBO). §26.6
Architectures (DCGAN, SAGAN with full self-attention equations, LapGAN/ProgressiveGAN, a regularization
survey, BigGAN/StyleGAN/alias-free-GAN as scaling case studies). §26.7 Applications is the widest breadth
found in this entire review: image (pix2pix, CycleGAN with full cycle-consistency+identity-loss
derivation), video, audio (GANSynth/WaveGAN/GAN-TTS), **text generation** (rare for a GAN chapter —
covers the discrete-generation problem, score-function-estimator instability, SeqGAN/MaliGAN/RankGAN/
ScratchGAN), imitation learning (GAIL, explicitly tied back to the f-divergence generalization), domain
adaptation, art/design.

**Math depth:** essentially everything the chapter claims is derived — D*(x), the JS-divergence-at-
optimum identity, the f-GAN variational bound via convex conjugacy, Wasserstein-1/IPM duality, MMD
(including the learned-kernel generalization), the non-saturating loss's motivation via explicit gradient
comparison, the Nash-equilibrium proof, and the full DiracGAN non-convergence derivation. Mode collapse
itself is narrative-plus-empirical (a real, reproducible 16-Gaussian-mixture experiment, Fig. 26.6, with
loss values annotated at 6 checkpoints) rather than formally proved — standard, since no clean closed-
form proof of mode collapse exists in the literature generally. **This is the most textbook-rigorous
treatment of GAN theory found anywhere in this review.**

**Code/exercises:** Zero exercises (confirmed by direct inspection of the extracted chapter text — no
"Exercises" section, unlike most other PML2 chapters). Extensive **official, verified** companion code —
six notebooks confirmed via GitHub API at `pyprobml/notebooks/book2/26/`:
`genmo_types_implicit_explicit.ipynb`, `ipm_divergences.ipynb`, `gan_loss_types.ipynb`,
`gan_mixture_of_gaussians.ipynb`, `dirac_gan.ipynb`, `gan_jax_celebA_demo.ipynb` — every figure in the
chapter is reproducible from these, a genuinely strong "code teaches" precedent.

**Figures worth emulating:** **Fig. 26.2** — a taxonomy tree (Learning By Comparison → {Ratios,
Differences} → {class-probability estimation, f-divergences} / {IPMs, moment matching}) — the single best
"orient the reader before the math" figure found in this review; **Fig. 26.4** (KL failing under non-
overlapping support, a rigorous version of Bishop's 17.2, shown alongside an MLP-smoothing fix); **Fig.
26.6** (16-Gaussian-mixture mode collapse, real reproducible experiment, not a cartoon); **Figs. 26.7-8**
(DiracGAN phase portrait — circular trajectory, continuous-vs-discrete dynamics side by side — the best
non-convergence visualization found in this whole survey, directly reproducible from `dirac_gan.ipynb`).

**Verdict — steal:** the general-to-specific structure (principles → density ratios / f-divergences / IPMs
/ moment-matching → GAN as the neural-network special case → the unified (g,h,l) loss table); the fully
derived DiracGAN + Nash-equilibrium proof; putting evaluation metrics in a shared overview rather than
duplicating per model. **Avoid:** 32 pages of encyclopedic application-domain sprawl (video/audio/text/
imitation-learning/domain-adaptation/art all in one chapter — more survey than teachable chapter); the
total absence of exercises.

### Direct comparison, Bishop vs. Murphy

| | Bishop ch. 17 | Murphy ch. 26 |
|---|---|---|
| Length | ~12 pp. | 32 pp. |
| Position | 1st of 4 (GAN→Flow→AE/VAE→Diffusion) | Last of 6 (Overview→VAE→AR→Flow→EBM→Diffusion→**GAN**) |
| D*(x) / JSD-at-optimum | Exercise only | Derived in main text |
| f-divergences / f-GAN | Not covered | Fully derived, general case |
| IPMs / Wasserstein duality | 1 page, metaphor only | Derived as 1-Lipschitz IPM |
| MMD | Not covered | Derived, incl. learned kernel |
| Equilibrium/convergence proof | Toy exercise (not GAN-specific) | Nash eq. proof + full DiracGAN |
| Evaluation metrics | Not covered | Covered (in overview ch. 20) |
| Text/audio/video applications | Not covered | Extensive (§26.7) |
| Exercises | 3, math-heavy, no solutions yet | None |
| Official code | None | 6 verified notebooks |

If the d2l rewrite wants rigor+generality, Murphy's structure is the far stronger model. If the target is
brevity+architecture-first storytelling, Bishop's shape is closer, but its math should be promoted from
exercises into the body, and its WGAN gap filled using Murphy's IPM derivation.

---

## 3. Prince, *Understanding Deep Learning* (MIT Press, 2023, v5.0.3 "Feb 2026 printing") — Ch. 15

**Sources:** official free PDF, `github.com/udlbook/udlbook` release v5.0.3
(`UnderstandingDeepLearning_02_09_26_C.pdf`); landing page https://udlbook.github.io/udlbook/; notebooks
`Notebooks/Chap15/15_1_GAN_Toy_Example.ipynb` and `15_2_Wasserstein_Distance.ipynb` (raw GitHub).

**Position:** Ch. 15 of 21, pp. 276-303, first of four back-to-back generative chapters: **15 GANs** → 16
Normalizing flows → 17 VAEs → 18 Diffusion. Deliberately positioned first because GANs are the one
*non-probabilistic* generative model in the book — a footnote states explicitly: "Until this point,
almost all of the relevant math has been embedded in the text. However, the following four chapters
require a solid knowledge of probability" — GANs are the on-ramp before the likelihood-heavy chapters.

**Order of ideas:** 15.1 "Discrimination as a signal" (toy 1-D intuition → loss function → training as
minimax → DCGAN → why GANs are hard to train, mode collapse) → 15.2 "Improving stability" (JS-divergence
analysis of the loss at the optimal discriminator → vanishing gradients → Wasserstein distance, discrete
then continuous → WGAN loss) → 15.3 Progressive growing / minibatch discrimination / truncation → 15.4
Conditional generation (CGAN, ACGAN, InfoGAN) → 15.5 Image translation (Pix2Pix, adversarial+content loss
/ SRGAN, CycleGAN) → 15.6 StyleGAN → 15.7 Summary.

**Math depth — the deepest of the three books in this cluster:** **Derives the optimal discriminator** in
closed form (Eq. 15.7), **substitutes it back into the loss and shows algebraically it reduces to twice
the Jensen-Shannon divergence** (Eq. 15.8→15.9), explicitly decomposed into a "quality" term and a
"coverage" term whose independence from θ is given as the textbook explanation for mode dropping.
Vanishing gradients explained via the same JS-at-optimum result (max/flat under disjoint support) plus a
cited empirical figure (Arjovsky & Bottou 2017). **Wasserstein distance** gets unusually rigorous
treatment for a DL text: works the discrete earth-mover problem as a linear program (primal `min
Σ P_ij|i−j|` s.t. marginal constraints), **derives the LP dual**, arrives at the Kantorovich-Rubinstein
form with a Lipschitz-1 constraint, then generalizes to continuous multivariate distributions and states
the WGAN loss with the `|∂f/∂x|<1` constraint, discussing weight clipping vs. gradient penalty as
enforcement mechanisms. Conditioning and image-translation sections (CGAN/ACGAN/InfoGAN, Pix2Pix/SRGAN/
CycleGAN) are architectural/loss-level without new derivations. End-of-chapter Problems 15.1-15.6 are
substantive: derive the JS loss value when Pr(x*)=Pr(x); relate the WGAN loss to D_JS algebraically;
write out the LP constraint matrix; compute KL/reverse-KL/JS/Wasserstein between two uniform
distributions as a function of an offset parameter; closed-form KL/Wasserstein between two Gaussians.

**Code/exercises/datasets:** Two Colab notebooks, both fill-in-the-blank pedagogical exercises, **NumPy
only, no real image dataset**: `15_1_GAN_Toy_Example.ipynb` (1-D Gaussian toy problem — generator is
literally `x = z + θ`, one scalar parameter; discriminator is 1-D logistic regression; student fills in
BCE loss, generator loss, alternating-update loop; includes hard-coded expected-loss self-check values).
`15_2_Wasserstein_Distance.ipynb` (student builds the distance matrix and LP constraint matrices, solves
with `scipy.optimize.linprog`, computes Wasserstein distance, and — final unanswered TODO — computes
forward-KL/reverse-KL/JS for comparison). **No DCGAN/StyleGAN/real-image notebook exists**; all real-
image results are shown only as figures adapted from the original papers, not reproduced in code — a real
gap relative to d2l's own "code teaches" policy.

**Figures worth emulating:** Systematic, one figure per concept: Fig. 15.1 (GAN mechanism as a 1-D
histogram + sigmoid discriminator across three training snapshots — the toy example made visual); 15.6
(vanishing-gradient cartoon: flat sigmoid when samples are easy to separate); 15.7 (empirical log-scale
gradient-decay plot, adapted from Arjovsky & Bottou); **15.8** (four-panel earth-mover's-distance figure:
source distribution / target distribution / transport-plan heatmap / distance-matrix heatmap); 15.13
(three-panel CGAN vs. ACGAN vs. InfoGAN architecture comparison); 15.15 (InfoGAN on MNIST — one attribute
recovers the 10 digit classes, two continuous attributes recovered as rotation and stroke width). This is
exactly the "conceptual diagram, not a training curve" style d2l's own house rules call for.

**Verdict — steal:** the JS-divergence-at-optimum derivation; the LP-duality treatment of Wasserstein
distance; the "GANs as the non-probabilistic on-ramp before the likelihood-heavy chapters" positioning.
**Avoid:** the notebook side never touches a real image and leaves the flashiest results (DCGAN,
StyleGAN) as purely borrowed figures with zero reproducible code.

---

## 4. Tomczak, *Deep Generative Modeling*, 2nd ed. (Springer, 2024) — Ch. 8

**Sources:** publisher page login-walled (could not access official TOC/chapter text directly); author's
site mirrors chapters as blog posts, `jmtomczak.github.io/dgm_book.html` (book TOC) and
`jmtomczak.github.io/blog/12/12_gans.html` (chapter draft — **unverified whether this is a verbatim
mirror of the printed Ch. 8 or a lightly diverging draft**); companion repo `github.com/jmtomczak/
intro_dgm`, notebook `gans/gans_example.ipynb`; auxiliary (older, separate) lecture-slide PDF
`jmtomczak.github.io/pdf/ssds2017.pdf` confirms Tomczak's long-standing interest in the divergence framing
but is not part of the 2024 book.

**Position:** Ch. 8 of 11: 1 Why Deep Generative Modeling? → 2 Probabilistic Modeling → 3 Autoregressive →
4 Flow-based → 5 Latent Variable Models (VAEs) → 6 Hybrid Modeling → 7 Energy-Based Models → **8 GANs** →
9 Score-based/diffusion → 10 Neural compression → 11 LLMs/Generative-AI systems. Opposite positioning
from Prince: GANs come *after* VAEs, hybrid modeling, and EBMs, right before score-based/diffusion —
consistent with the book's overall theme of treating everything as a variation on likelihood-based
modeling, with GANs marking the point where an explicit likelihood is abandoned (following EBMs, another
likelihood-adjacent-but-not-quite family, and preceding score-based models, which restore an explicit
generative process via SDEs).

**Order of ideas:** Starts from the general density-network/latent-variable setup shared with the VAE
chapter (intractable marginal likelihood, Monte-Carlo/log-sum-exp approximation) → "Getting rid of
Kullback-Leibler" (motivates IPMs/learnable losses as alternatives) → "Getting rid of prescribed
distributions" (replaces the decoder likelihood with a Dirac delta `p(x|z)=δ(x−NN(z))`, so the marginal
becomes an "infinite mixture of delta peaks" — a genuinely distinctive way to derive *why* GANs are
"implicit" models with no tractable density) → "Adversarial loss" (fraud/expert analogy → formal minimax
BCE objective) → "GANs" (synthesis) → 11+ extension topics (CGAN, BiGAN/ALI, StyleGAN, CycleGAN, WGAN,
f-GAN, MMD-nets, hierarchical implicit models, GAN-EBM connections, mode collapse, InfoGAN).

**Math depth:** Framed around *why* to leave likelihood-based training, not full derivations of every
alternative. KL, IPMs (MMD), and f-divergences are named and motivated conceptually as alternatives to KL
but — as far as could be extracted — the main text does **not** derive the f-divergence variational bound
or the MMD kernel dual the way Nowozin et al. or Prince's Wasserstein-duality treatment does; those are
pushed to references/extensions. What *is* derived: intractable marginal likelihood → Dirac-delta trick →
implicit distribution → adversarial BCE loss → minimax objective. WGAN is mentioned with its own
objective equation in a later subsection; **whether Kantorovich-Rubinstein duality is derived there or
just stated could not be verified**.

**Code/exercises/datasets:** Only one GAN notebook in the repo (`gans/gans_example.ipynb`) despite many
more variants being discussed in text — PyTorch, fully-connected Generator/Discriminator, trained on
**sklearn's `load_digits()` (1500 8×8 images)**, explicitly chosen so "everyone can run it on a laptop,
without any GPU." Two Adam optimizers, `retain_graph=True` alternating loop, `Linear→ReLU→Linear→Tanh`
generator / `Linear→ReLU→Linear→Sigmoid` discriminator. The notebook's own commentary (per the fetched
text) notes the adversarial loss "is jumping all over the place" and that GAN training "greatly depends
on initialization and neural nets rather than the adversarial loss" — an honest, if slightly deflating,
teaching moment.

**Figures:** Not independently verifiable — could not confirm the blog mirror embeds the book's actual
figures (flagged unverified).

**Verdict — steal:** the "get rid of KL, then get rid of prescribed likelihoods via a Dirac delta"
derivation as a short, distinctive framing device for *why* an adversarial loss is needed at all.
**Avoid:** the code/exercise side, which is the thinnest of the three in this cluster — one MLP on 8×8
digits, with WGAN/conditional/image-scale variants discussed but never implemented.

---

## 5. Foster, *Generative Deep Learning*, 2nd ed. (O'Reilly, 2023) — Ch. 4 "GANs" + Ch. 10 "Advanced GANs"

**Sources:** live O'Reilly pages 403'd; retrieved via Wayback Machine snapshots (Aug/Dec 2025) which
serve the same React-app page data, giving the full nested TOC plus "Chapter Goals" preview text; code
verified directly via GitHub API + downloaded notebooks at `github.com/davidADSP/
Generative_Deep_Learning_2nd_Edition`.

**Position:** Part I Introduction (1-2) → Part II Methods (3 VAEs, **4 GANs**, 5 Autoregressive, 6
Normalizing Flow, 7 EBMs, 8 Diffusion) → Part III Applications (9 Transformers, **10 Advanced GANs**, 11
Music, 12 World Models, 13 Multimodal) → 14 Conclusion. GANs get a foundational chapter early (paired with
VAEs) *and* a dedicated later chapter once more architectural vocabulary (attention, from ch. 9) is
available — a two-pass structure none of the other books use.

**Order of ideas (confirmed exact headers):** Ch. 4: Introduction → **DCGAN** [Bricks Dataset →
Discriminator → Generator → Training → Analysis] → GAN Training Tips and Tricks → **WGAN-GP** [Wasserstein
Loss → Lipschitz Constraint → Enforcing the Lipschitz Constraint → Gradient Penalty Loss → Training →
Analysis] → **CGAN** [Architecture → Training → Analysis] → Summary. Ch. 10 "Advanced GANs": Introduction
→ ProGAN → StyleGAN (Mapping Network, Synthesis Network, Outputs) → StyleGAN2 (Weight
Modulation/Demodulation, Path Length Regularization, No Progressive Growing, Outputs) → Other Important
GANs (SAGAN, BigGAN, VQ-GAN, ViT VQ-GAN) → Summary. **CycleGAN and Pix2Pix/SRGAN-style image translation
are absent from the entire book** — a real gap relative to Prince, and something d2l could cover that
Foster skips.

**Math depth:** Inferred from section-header pattern and "Chapter Goals" teaser text (actual paragraph-
level math is paywalled and could not be directly read — **flagged as inferred, not confirmed**): the
Kantorovich-Rubinstein/Lipschitz-1 requirement appears to be *stated and motivated practically* ("The
Lipschitz Constraint," "Enforcing the Lipschitz Constraint," "The Gradient Penalty Loss" as headers, no
"Proof" headers, no LP-duality treatment implied) rather than derived from first principles the way
Prince does. Consistent with the book's known "build it, see it work" positioning, this is plausibly the
least mathematically rigorous of the three books in this cluster.

**Code/exercises/datasets — verified by downloading and inspecting the notebooks; Keras/TensorFlow
throughout, not PyTorch:** `04_gan/01_dcgan/dcgan.ipynb` — **LEGO Bricks dataset** (grayscale, 64×64),
standard strided-conv discriminator / transposed-conv generator, custom `train_step`.
`04_gan/02_wgan_gp/wgan_gp.ipynb` — **CelebA faces** (64×64 RGB), critic-based, `CRITIC_STEPS=3`,
`GP_WEIGHT=10.0`, explicitly adapted from a Keras.io WGAN-GP tutorial. `04_gan/03_cgan/cgan.ipynb` —
CelebA again, conditioned on the binary `Blond_Hair` attribute, one-hot label appended as extra spatial
channels, adapted from a Sayak Paul Keras.io tutorial. **Chapter 10 "Advanced GANs" has no accompanying
notebook at all** (confirmed via a recursive GitHub tree listing — `04_gan → 05_autoregressive → ... →
09_transformer → 11_music`, no `10_advanced_gan` folder); StyleGAN/StyleGAN2/SAGAN/BigGAN/VQ-GAN are pure
walkthrough/description once compute cost gets serious — a real pattern shift within the same book.

**Figures:** Could not retrieve actual images (paywalled); an "Outputs"/"Outputs from StyleGAN" heading
pattern suggests results galleries analogous to Prince's, unconfirmed.

**Verdict — steal:** exactly what a from-scratch DCGAN/WGAN-GP/CGAN training pipeline should look like on
a real, fun, visually diagnostic dataset (LEGO Bricks is arguably a nicer teaching dataset than MNIST —
failure modes are visually obvious). **Avoid:** the "Advanced GANs" chapter quietly abandons the "build
it" promise once compute gets expensive, and CycleGAN/Pix2Pix are skipped entirely.

### Cross-book comparison, Prince / Tomczak / Foster

| | Prince (UDL) | Tomczak (DGM) | Foster (GDL 2e) |
|---|---|---|---|
| Chapter position | Before flows/VAE/diffusion | After VAEs/EBMs, before score-based | Ch.4 (basics) + Ch.10 (advanced) |
| Derives D*(x) / JSD-at-optimum | Yes, explicit | Via Dirac-delta trick, not the same derivation | Not evident (paywalled; unlikely per structure) |
| Derives Wasserstein/KR duality | Yes, full LP primal→dual | Objective stated; derivation depth unverified | Practical "how to enforce Lipschitz," not derived |
| Framework / real dataset | NumPy toy only, no real image | PyTorch, sklearn Digits (8×8) toy | Keras/TF, real: LEGO Bricks + CelebA |
| Image translation (Pix2Pix/CycleGAN/SRGAN) | Yes, own subsection | Listed only in extensions | Absent entirely |
| StyleGAN | Described + figure, no code | Not covered in the fetched draft | Described, no code (compute-gated) |
| Exercises | Fill-in-blank notebooks + 6 substantive math problems | One minimal educational notebook | Full working notebooks, no formal problem sets |

---

## 6. Goodfellow, "NIPS 2016 Tutorial: Generative Adversarial Networks" (arXiv:1701.00160)

**Sources:** https://arxiv.org/abs/1701.00160 ; https://arxiv.org/pdf/1701.00160 ; HTML extraction via
https://ar5iv.labs.arxiv.org/html/1701.00160

**Full TOC:** 1 Introduction; 2 Why study generative modeling?; §2.1-2.5 How do generative models work /
taxonomy (maximum likelihood estimation → taxonomy of deep generative models → explicit tractable models
→ explicit models requiring approximation → implicit density models → comparing GANs to other models); 3
How do GANs work? (3.1 the GAN framework; 3.2 cost functions — discriminator's cost, minimax, heuristic
non-saturating game, maximum-likelihood game, is the choice of divergence a distinguishing feature?,
comparison of cost functions; 3.3 the DCGAN architecture; 3.4 how GANs relate to NCE and maximum
likelihood); 4 Tips and Tricks (train with labels; one-sided label smoothing; virtual batch
normalization; can one balance G and D?); 5 Research Frontiers (non-convergence incl. mode collapse and
"other games"; further open topics); 6 Exercises; 7 Solutions to exercises.

**Order of ideas:** Leads with a full taxonomy of deep generative models (explicit-tractable / explicit-
approximate / implicit) before defining the GAN game at all — GANs are placed as the implicit,
single-shot-sampling leaf of this tree, one of the most-reused framing devices in the field. Only then
does it define the minimax game and walk through *four* candidate generator costs (minimax /
non-saturating heuristic / maximum-likelihood-equivalent), comparing rather than asserting one canonical
loss. Closes with engineering tips, open research problems, and — uniquely among all sources surveyed —
graded exercises with solutions.

**Math depth:** Derives (as Exercise 7.1, with solution) the optimal discriminator's density-ratio
interpretation; works through the minimax game's connection to JS divergence; defines the non-saturating
"−log D trick" and its gradient-strength motivation; shows (Exercise 7.3) that a particular reweighting of
the generator objective is equivalent to minimizing KL(p_data‖p_model) under an optimal discriminator
— the maximum-likelihood game. This is the closest the tutorial comes to an MLE↔adversarial bridge, but
framed as one of three alternative costs to pick from, not a single narrative arc, and it predates any
relativistic-objective material by nearly a decade.

**Code/exercises:** No notebook (slide-deck-style tutorial). Three exercises with solutions: optimal
discriminator strategy; gradient descent on games (a toy 1-D bilinear game showing cycling without
convergence — the tutorial's own toy non-convergence device, a precursor in spirit to Bishop's Exercise
17.2 and Murphy's DiracGAN); the MLE-equivalence result.

**Figures worth emulating:** The explicit/implicit taxonomy tree (still the most-reused GAN-pedagogy
figure in the field); a toy cycling-under-simultaneous-gradient-descent figure making "games ≠
optimization" concrete without heavy machinery.

**Verdict — steal:** the taxonomy tree and the three-cost-function comparison table; the toy
non-convergence figure. **Avoid:** teaching the tips-and-tricks section (label smoothing, virtual batch
norm) as current best practice — superseded by spectral norm / R1-R2 penalties / relativistic losses.

---

## 7. Blogs

### 7a. Lilian Weng, "From GAN to WGAN" (2017; arXiv revision 2019, arXiv:1904.08994)

**Sources:** https://lilianweng.github.io/posts/2017-08-20-gan/ ; https://arxiv.org/abs/1904.08994

**Structure:** 1 Kullback-Leibler and Jensen-Shannon Divergence; 2 GAN (What is the optimal value for D?
/ What is the global optimal? / What does the loss function represent?); 3 Problems in GANs (5
challenges: hard Nash equilibrium, low-dimensional/disjoint supports, vanishing gradient, mode collapse,
no proxy metric for progress); 4 Improved GAN Training (7 practical fixes); 5 Wasserstein GAN (Earth-Mover
"dirt-moving" intuition; why Wasserstein beats JS/KL under disjoint support; Kantorovich-Rubinstein
duality → the tractable K-Lipschitz critic form; weight clipping); 6 Example: Create New Pokémon; 7
References.

**Order of ideas — the tightest, most complete "MLE→KL→JS→adversarial→Wasserstein" arc found anywhere in
this entire survey, including all five textbooks.** It leads with the divergences themselves (KL, then
JS) *before* introducing the GAN game, so that when it defines D*(x)=p_r(x)/(p_r(x)+p_g(x)) and shows
L(G,D*)=2·JS(p_r‖p_g)−2log2, the reader already has the vocabulary to see immediately why this matters
and why it's fragile: JS is locally constant / has a discontinuous limit whenever supports are disjoint
(generic in high dimensions) — presented as the *root cause* of vanishing gradients and instability, not
an unexplained empirical curiosity. It then motivates Wasserstein distance as the fix (continuous,
differentiable even under disjoint support) via a concrete discrete "dirt-moving" example, and derives the
Kantorovich-Rubinstein dual used in practice. Every step is motivated by the failure of the previous step
— the single strongest pedagogical throughline in this whole review.

**Math depth:** Full derivation of D*(x); the value function at optimum in terms of JS divergence; an
explicit statement of why JS is discontinuous under disjoint support with the low-dimensional-manifold
argument; the K-Lipschitz constraint's role in the KR dual; a comparison table of KL/JS/Wasserstein
continuity properties. Does not cover general f-GAN duality or MMD/IPMs beyond Wasserstein specifically,
and — an 2017/2019 artifact — has nothing on relativistic losses or R3GAN.

**Figures:** KL vs. JS asymmetry/symmetry plot; a toy oscillating-minimization figure for Nash-equilibrium
difficulty; a plot of vanishing discriminator-gradient magnitude over training (several orders of
magnitude decay); a mode-collapse example grid from DCGAN; a step-by-step discrete Earth-Mover "dirt pile"
diagram; Pokémon generation progression across epochs.

**Verdict:** The strongest single pedagogical flow found in this entire survey — steal the "divergences
first, GAN game second" ordering and the disjoint-support argument for *why* vanilla-GAN training is
unstable, plus the dirt-moving Wasserstein intuition. Nothing to avoid, though it's ~9 years stale
(pre-WGAN-GP, let alone pre-relativistic/R3GAN) and needs a "what's next" coda.

### 7b. Distill.pub, "Open Questions about Generative Adversarial Networks" (Görtler, Kehlbeck, Deussen, 2019)

**Sources:** https://distill.pub/2019/gan-open-problems/ ; https://github.com/distillpub/post--gan-open-problems

Note: distill.pub has no dedicated *introductory* GAN tutorial (its output was mostly interpretability/
visualization work); this is the one GAN-focused piece, and it's explicitly an open-problems survey, not
a from-scratch derivation. Adjacent and useful: distill.pub's "Deconvolution and Checkerboard Artifacts"
(Odena, Dumoulin, Olah, 2016, https://distill.pub/2016/deconv-checkerboard/) explains why transposed-conv
upsampling in a generator produces checkerboard artifacts and why resize-then-convolve avoids them —
directly useful for justifying DCGAN-style generator architecture choices.

**Structure:** Seven open questions, each its own section: (1) trade-offs vs. flow models and
autoregressive models (a 3×3 comparison table on parallelism/efficiency/reversibility, with a concrete
number — a flow model needed ~17× the GPU-days of a comparable GAN); (2) what makes a distribution
"learnable" by a GAN; (3) scaling GANs beyond images; (4) global convergence — when does simultaneous
gradient descent on the minimax game provably converge; (5) evaluation methodology; (6) does large-batch
training help GANs the way it helps classifiers; (7) discriminator adversarial-robustness effects on
training. Explicitly enumerates open problems rather than asserting settled answers, organizes literature
by *question* rather than by chronology of variants — an antidote to the "zoo of variants" failure mode.

**Verdict:** Steal the "organize by open question, not chronological variant list" structure and the
comparison-table format against sibling model families. Not a source for derivations — it assumes GAN
basics are covered elsewhere.

### 7c. Sebastian Raschka

**Sources:** https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch09/ ;
https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L17_gan_slides.pdf ;
https://sebastianraschka.com/pdf/slides/2021-07_issdl-gdansk-intro-to-gans.pdf

Chapter 9 of "Machine Learning Q and AI" is a comparison chapter across six generative-model families
(energy-based/DBM, VAE, GAN, flow-based, autoregressive, consistency models), concluding — consistent
with the survey/blog consensus found elsewhere in this review — that diffusion has "largely superseded"
GANs since 2022 for image quality, at the cost of GANs' one-shot sampling speed. Minimal math (no
GAN-specific derivation). The dedicated stat479/ISSDL slide decks (2019/2021) are standard from-scratch
GAN+DCGAN lecture material, predating WGAN-GP-era content in the syllabus.

**Verdict:** Useful only as an up-to-date "where does a GAN fit among today's six generative-model
families" framing paragraph; not a source of new derivations or figures.

---

## 8. Recent (2023-2026) surveys and the R3GAN / "GAN revival" literature

### R3GAN — Huang, Gokaslan, Kuleshov, Tompkin, "The GAN is dead; long live the GAN! A Modern Baseline GAN" (arXiv:2501.05441, Jan 2025)

**Sources:** https://arxiv.org/abs/2501.05441 ; https://arxiv.org/html/2501.05441v1 ; equations extracted
via https://ar5iv.labs.arxiv.org/html/2501.05441 ; secondary coverage:
https://www.marktechpost.com/2025/01/12/r3gan-a-simplified-and-stable-baseline-for-generative-adversarial-networks-gans/
, https://www.emergentmind.com/papers/2501.05441 , https://huggingface.co/blog/Kseniase/fod83 ("GAN is
back"); background on relativistic discriminators: Jolicoeur-Martineau, "The relativistic discriminator: a
key element missing from standard GAN" (arXiv:1807.00734, 2018) and the pedagogical blog
https://ajolicoeur.wordpress.com/relativisticgan/ and
https://ajolicoeur.wordpress.com/2018/10/01/alternative-losses-for-relativistic-gans/.

**Motivation:** attributes GANs' bad reputation less to the adversarial objective itself than to (a) an
unstable *unregularized* loss and (b) architectures frozen at ~DCGAN/StyleGAN2-era design (minibatch
stddev, equalized LR, noise injection, path-length regularization — "poorly understood empirical tricks")
while diffusion models absorbed a decade of modern CNN/ResNet/ViT advances.

**Exact closed forms (extracted directly from the paper via ar5iv HTML rendering — high confidence):**
- RpGAN objective: $\mathcal L(\theta,\psi) = \mathbb E[f(D_\psi(G_\theta(z)) - D_\psi(x))]$, $z\sim p_z$,
  $x\sim p_{\mathcal D}$, with $f(t) = -\log(1+e^{-t}) = \log\sigma(t)$ (log-sigmoid) — the discriminator/
  generator loss is a function of the *difference* $D_\psi(\text{fake}) - D_\psi(\text{real})$ rather than
  of each evaluated in isolation, recovering the classic non-saturating GAN loss as the special case where
  the discriminator is not paired/relativized.
- Zero-centered gradient penalties: $R_1(\psi) = \frac{\gamma}{2}\mathbb E_{x\sim p_{\mathcal D}}
  [\lVert\nabla_x D_\psi\rVert^2]$ (real data) and $R_2(\theta,\psi) = \frac{\gamma}{2}
  \mathbb E_{x\sim p_\theta}[\lVert\nabla_x D_\psi\rVert^2]$ (fake data) — both vanish exactly at
  distribution match ($p_\theta=p_{\mathcal D}$), unlike WGAN-GP's penalty, which is centered away from
  zero.
- Discriminator objective: maximize $\mathcal L(\theta,\psi) - R_1(\psi) - R_2(\theta,\psi)$ (the paper does
  not state this as one single numbered display equation, but combines the pieces this way in the text).

**Theory:** Proposition I/II (Jacobian eigenvalue analysis at equilibrium) prove the *unregularized* RpGAN
provably fails to converge, while RpGAN+R1+R2 has **local convergence guarantees** — an actual convergence
proof for a practical, modern loss, something neither the original 2014 paper nor WGAN ever supplied
outside idealized settings.

**Mode coverage:** on StackedMNIST (1000 synthetic modes), classic GAN+R1+R2 recovers 693/1000 modes;
RpGAN+R1+R2 recovers all 1000 (KL-to-uniform 0.078) — a concrete, teachable number for "why relativistic
pairing helps mode collapse."

**Architecture:** strips StyleGAN2 to a plain modernized ResNet (1-3-1 bottleneck blocks, no BatchNorm/
InstanceNorm — replaced by Fixup-style init — no spectral norm, grouped convolutions, inverted
bottleneck) — explicitly demonstrating that once the *loss* is well-behaved, none of the historical
architectural tricks are load-bearing.

**Results:** FID 2.75 (FFHQ-256) vs. StyleGAN2's 3.78 and LDM's 4.98; competitive-or-better numbers on
FFHQ-64, CIFAR-10, ImageNet-32/64 against both StyleGAN2/XL and diffusion baselines (EDM/DDPM-IP/VP), with
the standard GAN advantage of single-forward-pass sampling vs. tens-to-hundreds of diffusion steps.

**This is an active 2025-2026 thread, not a one-off:** follow-on work found includes "Beyond Data
Scarcity: Optimizing R3GAN for Medical Image Generation from Small Datasets" (arXiv:2510.26828, Oct 2025)
and "Pairing Regularization for Mitigating Many-to-One Collapse in GANs" (arXiv:2604.20130).

### Direct Discriminative Optimization (DDO) — arXiv:2503.01103 (March 2025)

**Source:** https://arxiv.org/html/2503.01103v1 — "Direct Discriminative Optimization: Your
Likelihood-Based Visual Generative Model is Secretly a GAN Discriminator."

**This is the single closest thing found anywhere in this survey to the MLE→KL→adversarial unification
the task brief asked me to check for** — and it is a March-2025 *research* paper, not a textbook or blog
explainer. Its move: take the optimal-discriminator formula itself, $d^*(x)=\sigma(\log
p_\text{data}(x)/p_\text{ref}(x))$, and instead of learning $d$ via a separate network, plug in the ratio
of a learnable likelihood-based model (diffusion / flow / autoregressive) to a frozen reference model.
This turns *any* likelihood-based generative model into an implicit GAN discriminator, with a closed-form
practical objective $\mathcal L_{\alpha,\beta}(\theta) = -\mathbb E_{p_\text{data}}[\log\sigma(\beta\log
(p_\theta/p_\text{ref}))] - \alpha\,\mathbb E_{p_\text{ref}}[\log(1-\sigma(\cdots))]$ and a theorem that the
optimum recovers $p_\text{data}$. It is explicitly a research fine-tuning method (applied to pretrained
diffusion/autoregressive models), not written for pedagogy, and it does **not** touch the relativistic-
pairing objective at all. It confirms half of the hypothesized gap (MLE↔adversarial unification via the
optimal-discriminator identity is a live, very recent idea) while leaving the other half (a from-scratch,
textbook-style treatment of the *relativistic* objective specifically, tied back into the same MLE/KL
story) genuinely unaddressed by anything found in this survey.

### Surveys

- **"Ten years of generative adversarial nets (GANs): a survey of the state-of-the-art"** (*Machine
  Learning: Science and Technology*, 2024) — https://iopscience.iop.org/article/10.1088/2632-2153/ad1f77.
  Structure: prior-survey comparison → vanilla-GAN foundations → 14 application domains → **30+ variants
  organized chronologically and by "loss-based" (LSGAN, WGAN, SN-GAN) / "architecture-based" (DCGAN,
  ProGAN, CycleGAN) / "application-specific" (SRGAN, InfoGAN, CGAN...) buckets** → theory (JS divergence,
  minimax, Nash equilibrium mentioned, but no explicit f-divergence/IPM treatment surfaced) → evaluation
  metrics (IS, FID) → limitations/remedies → hybridization with transformers/diffusion/LLMs → future
  directions. **A textbook example of the "zoo of variants" failure mode**: comprehensive but with no
  single organizing mathematical principle tying the 30+ variants together — contrast with Murphy's
  principled f-divergence/IPM organization above.
- **"Generative adversarial networks: A comprehensive survey"** (*ScienceDirect*, 2026) —
  https://www.sciencedirect.com/science/article/pii/S2772941926000244 — **blocked from direct fetch
  (HTTP 403)**; per search-result snippets, claims a "method-centric organization aligning architectures,
  objectives, and training heuristics with measurable outcomes," emphasizing 2022-2025 trends
  (transformer-based generators/discriminators, text-to-image). **Could not verify** whether it discusses
  R3GAN or the relativistic objective specifically — flagged unverified.
- **Bond-Taylor et al., "Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing
  Flows, Energy-Based and Autoregressive Models"** (IEEE TPAMI, 2021/2022) — slightly outside the
  2023-2026 window but the standard precedent for a principled cross-family comparative review; newer
  surveys largely build on its taxonomy. Not independently re-verified this session (background
  knowledge + search-snippet corroboration only).

### f-GAN background (context for "does anyone organize by f-divergence generally?")

Nowozin, Cseke, Tomioka, "f-GAN: Training Generative Neural Samplers using Variational Divergence
Minimization" (NeurIPS 2016, arXiv:1606.00709) is the origin of the general f-divergence framework: any
f-divergence $D_f(P\Vert Q)$ admits a variational lower bound via the Fenchel conjugate $f^*$, and choosing
$f$ recovers vanilla GAN (JS) or other GANs as special cases. This is the paper Murphy's chapter builds its
§26.2.3 on (see §2 above) — noted here as background, not independently re-derived this session.

### Evaluation-metrics critique literature (2023-2025)

Multiple 2023-2025 papers (arXiv:2402.03654 "reviewing FID and SID metrics"; MDPI 2024 "Evaluation Metrics
for Generative Models: An Empirical Study") document by-now-standard critiques: Inception Score doesn't
penalize memorization/low intra-class diversity; FID depends on an ImageNet-trained Inception network of
dubious relevance to non-natural-image domains, is sensitive to sample size and to trivial Inception-
weight reimplementation differences, and correlates imperfectly with human judgment (documented to
*underestimate* diffusion-model quality on FFHQ relative to human raters in at least one study). Proposed
fixes: Memorization-informed FID (MiFID), Feature Likelihood Divergence (FLD), Signed Inception Distance
(SID, 2023). None of this is GAN-specific — it applies to all deep generative models — but any GAN chapter
introducing FID/IS as *the* evaluation protocol should flag these caveats. Of the five textbooks surveyed,
only Murphy (§20.4, in the shared overview chapter) treats evaluation with any rigor at all; none appear to
carry the 2023-2025 critique literature (none postdate it comfortably enough to be certain either way for
Bishop/Prince/Tomczak/Foster — flagged as likely-but-not-directly-confirmed for those four).

---

## 9. Current d2l.ai GAN chapter — local assessment

**Files read:** `/home/smola/d2l-neu/chapter_generative-adversarial-networks/gan.md`,
`/home/smola/d2l-neu/chapter_generative-adversarial-networks/dcgan.md`,
`/home/smola/d2l-neu/chapter_generative-adversarial-networks/index.md`.

**What's there:** `gan.md` motivates GANs via a two-sample-test framing (distinctive — not seen verbatim
in any of the other eight sources reviewed, worth keeping), states the discriminator's cross-entropy loss
and the generator's minimax objective, explains *qualitatively only* (one paragraph, no equations) why the
saturating loss vanishes early in training, introduces the non-saturating surrogate
"$-\log D(G(z))$" as sharing "the same global optimum" (asserted, not shown), states the joint minimax
objective, then jumps straight into code: fitting a 2-D Gaussian with a linear-generator / 3-layer-MLP-
discriminator toy problem across all four framework tabs (mxnet/pytorch/tensorflow/jax). `dcgan.md`
extends this to DCGAN on a small Pokémon-sprite dataset with the classic all-conv recipe (transposed-
conv+BN+ReLU generator vs. conv+BN+LeakyReLU discriminator), including nice from-scratch shape-arithmetic
derivations for transposed-conv output size (consistent with d2l's "derive the shapes" house style) and a
recently-added caveat against over-reading GAN loss curves as validation loss.

**Weaknesses relative to the best of the sources above:**

1. **No derivation of the optimal discriminator D*(x) or the JS-divergence value at optimum** — the single
   most-repeated derivation across every other source surveyed (Goodfellow's tutorial, Weng, Murphy,
   Bishop, Prince) is entirely absent here; the chapter *asserts* the non-saturating trick "shares the same
   global optimum" without showing it.
2. **No treatment of *why* vanilla-GAN training is unstable beyond one qualitative paragraph** — no
   disjoint-support / vanishing-gradient argument (Weng's and Murphy's strongest contribution), no
   mode-collapse formalization.
3. **Zero coverage of WGAN / IPMs / f-divergences / spectral normalization** — every other 2023-2026
   text/survey in this review treats at least Wasserstein-GAN as core content (Prince derives its LP
   duality in full; Murphy derives IPMs and MMD generally); d2l's chapter stops at the original 2014
   minimax objective plus DCGAN's architectural trick — frozen at 2015-era content.
4. **No evaluation metrics at all** — not even qualitative discussion of how one would know a GAN "worked"
   beyond eyeballing a scatter plot / image grid. Murphy, the surveys, and Weng all treat this as
   essential, and the 2023-2025 literature has well-documented critiques of FID/IS that a 2026-era chapter
   should engage with rather than ignore entirely.
5. **No coverage of any GAN variant beyond DCGAN** — no conditional GAN, CycleGAN, StyleGAN, WGAN-GP; the
   "Recap" slide namedrops WGAN/WGAN-GP/StyleGAN/BigGAN in one bullet with zero elaboration.
6. **Thin exercises** — one open-ended question in gan.md ("does an equilibrium exist where G wins"), two
   shallow prompts in dcgan.md (ReLU vs. LeakyReLU; try Fashion-MNIST) — none derivation-based, unlike
   Goodfellow's graded exercises-with-solutions, Bishop's starred toy-game exercise, or Prince's six
   substantive math problems, and unlike the "Proofs, intuition-first" convention CLAUDE.md documents for
   other chapters of this same book.
7. **No connection to the rest of the book's generative-model material** — no forward/backward reference
   tying GANs to the (placeholder) diffusion chapter or to likelihood-based models elsewhere, despite this
   being exactly the kind of unifying opportunity an MLE→KL→adversarial story would provide (and that
   Bishop §16.4.4, Prince's ch.14 footnote, and Tomczak's Dirac-delta framing each attempt in their own
   way).
8. **Zero mention of the 2025-2026 GAN revival** (R3GAN, relativistic losses, DDO) — unsurprising given the
   content predates it, but the chapter's implicit narrative ("GANs were superseded by better methods") is
   now stale; every 2025+ source surveyed here treats "GANs are back — the tricks were broken, not the
   objective" as the current state of the field, and this is a genuinely open niche none of the five
   textbooks fill either (see synthesis below).
9. *(Build-hygiene aside, not a pedagogy point.)* The chapter still carries mxnet and tensorflow tabs;
   CLAUDE.md's current framework policy (2026-07-17) restricts the Advanced part (ch. 9-16, which includes
   GANs) to PyTorch+JAX only — this chapter needs a framework-tab prune independent of any content
   rewrite.

**One clear asset worth keeping:** the two-sample-test framing in the opening paragraphs, and the
from-scratch conv/transposed-conv shape-arithmetic derivations in `dcgan.md`, are both distinctively "d2l"
(derive-everything-from-scratch) in ways none of the other eight sources are — any rewrite should preserve
that spirit while filling the theory gap identified above.

---

## 10. Synthesis

### (a) Strongest pedagogical flow found, and why it works

Of everything surveyed — five textbooks, a classic tutorial, three blog/essay pieces, and a stack of
2023-2026 papers/surveys — **Lilian Weng's "From GAN to WGAN"** has the tightest, most reusable arc:
divergences first (KL, then JS) → define the GAN game and immediately show D*(x) and the
JS-value-at-optimum identity, so the reader already owns the vocabulary needed to see why that identity is
fragile → name the concrete failure modes (disjoint support ⇒ locally constant/discontinuous JS ⇒
vanishing gradient; mode collapse) as *consequences of the divergence choice*, not unexplained empirical
folklore → motivate Wasserstein distance as the fix via a concrete discrete example → derive the
Kantorovich-Rubinstein dual that makes it tractable. Every step is motivated by the failure of the
previous step. Two of the five textbooks converge on structurally similar ideas independently: **Murphy's**
chapter generalizes the same divergence-minimization spine outward (density ratios → f-divergences → IPMs
→ moment matching, with GANs as the neural-network special case) and adds the field's most rigorous
convergence analysis (DiracGAN); **Prince's** chapter runs the same D*→JS derivation Weng does, then gives
the most complete Wasserstein *LP-duality* derivation of any source surveyed, textbook or blog. Goodfellow's
tutorial has the best scene-setting (a taxonomy of generative models before any GAN-specific content) but
spends more of its length on now-dated engineering tricks than on driving the theory forward.

**The organizing lesson for the rewrite:** a GAN chapter reads well when structured as *divergence
minimization that keeps failing and getting patched* — MLE/KL → JS/vanilla-GAN → (why JS breaks under
disjoint support) → Wasserstein/IPM → (why WGAN needed weight-clipping/gradient-penalty patches) → modern
regularized-relativistic objectives (R3GAN) — rather than as a chronological list of things called "GAN."

### (b) Common failure modes observed across sources

1. **Zoo-of-variants with no organizing principle.** The "Ten years of GANs" 2024 survey is the clearest
   example: 30+ variants bucketed by loss-based / architecture-based / application-specific with no single
   mathematical thread. d2l's own chapter risks the opposite but equally bad failure — a "zoo of one" (only
   DCGAN) with no theory to generalize *from* at all.
2. **Math dumped without use.** Not badly observed in the strongest sources (Weng, Murphy, Prince) but it's
   the classic risk whenever f-divergence duality or KR duality is presented as formalism to admire rather
   than a tool explaining a concrete symptom (why does JS give vanishing gradients under disjoint support?
   why did WGAN's weight-clipping cause its own pathologies, motivating WGAN-GP?). Every derivation in the
   rewrite should be attached to the symptom it explains, matching CLAUDE.md's existing "intuition-first
   proofs, led by a picture" convention.
3. **Toy code that teaches nothing.** d2l's own 2-D Gaussian toy example is defensible (honestly framed,
   lets you check convergence visually against a known target) but never computes anything theoretical —
   it could compute the empirical JS divergence or the discriminator's implied density-ratio estimate
   against ground truth, turning "cute demo" into "numerically verifies the theorem above." Prince's own
   toy notebooks have exactly this same gap (real-image results are pasted-in figures, never reproduced).
4. **Treating GAN training diagnostics as ordinary validation loss.** Several sources (Weng's
   vanishing-gradient plot, d2l's own newly-added dcgan.md caveat) warn against this explicitly; worth
   stating up front rather than as an aside.
5. **Evaluation as an afterthought or omitted entirely.** d2l has zero evaluation-metric coverage; only
   Murphy (in a shared overview chapter, not duplicated per-model) treats it with real rigor, and even
   sources that cover FID/IS mostly present them uncritically against the well-documented 2023-2025
   critique literature.

### (c) Gaps no existing source fills — confirming or refuting the hypothesis

**Confirmed, with direct evidence:**

- **Nobody surveyed is post-R3GAN, by simple publication-date arithmetic.** Bishop & Bishop (2024), Murphy
  PML-AT (2023, refreshed printing Dec 2025 but the GAN chapter's content is unchanged from 2023), Prince
  UDL (2023, v5.0.3 "Feb 2026 printing" — again, a printing/errata update, not new GAN content), Tomczak
  2nd ed. (2024), and Foster 2nd ed. (2023) all predate R3GAN (arXiv Jan 2025) and Direct Discriminative
  Optimization (arXiv Mar 2025). Neither confirmed survey (the 2024 "Ten years of GANs," or the paywalled
  2026 ScienceDirect survey whose R3GAN coverage could not be verified) was confirmed to discuss it. Blog
  coverage exists (HuggingFace "GAN is back," MarkTechPost, EmergentMind) but these are news write-ups, not
  pedagogical chapters with worked derivations.
- **Nobody gives closed-form derivations for the relativistic objective as part of a from-scratch teaching
  narrative.** The closest anything gets is the research papers themselves (R3GAN's own Jacobian-eigenvalue
  convergence proof; Jolicoeur-Martineau's 2018 relativistic-GAN blog, which does walk through RSGAN/RaGAN/
  RpGAN pedagogically but predates R3GAN by seven years and never connects to zero-centered gradient
  penalties or to the MLE/KL story).
- **Nobody connects MLE→KL→adversarial as one continuous story *and* reaches a modern relativistic closed
  form.** The single closest source found to a genuine MLE↔adversarial bridge is the 2025 Direct
  Discriminative Optimization paper, which repurposes the optimal-discriminator identity
  $d^*(x)=\sigma(\log p_\text{data}/p_\text{ref})$ to turn any likelihood-based model into an implicit
  discriminator — but it is a research fine-tuning method for diffusion/autoregressive models, written for
  researchers with no pedagogical framing, and it does not touch relativistic pairing at all. So the two
  halves of the hypothesized gap are each individually corroborated by a *different* 2025 paper, but no
  single source (research or pedagogical) joins both halves into one artifact, let alone a teaching one.

**Net verdict: the hypothesized gap is real, and if anything understates the opportunity.** A d2l GAN
chapter that (1) tells the MLE→KL→JS→adversarial story the way Weng tells it best, generalized via
f-divergence duality the way Murphy does, (2) explains *why* WGAN/IPMs were the historically necessary
patch the way Prince derives it, and (3) lands on R3GAN's regularized relativistic objective with an honest
from-scratch derivation of the RpGAN+R1+R2 loss and its local-convergence property — would be
pedagogically ahead of every source surveyed, including sources published as recently as 2024. This is a
genuinely open niche, not a solved-elsewhere problem d2l would be redundantly re-solving.

### Caveats / what could not be verified

- The exact provenance of the Bishop & Bishop chapter text (read from an unofficial third-party GitHub
  mirror, not bishopbook.com itself — content is DOI-stamped and internally consistent, but the mirror's
  authorization is unconfirmed).
- Whether the 2026 ScienceDirect GAN survey discusses R3GAN specifically (blocked by a 403).
- The depth of Tomczak's WGAN/Kantorovich-Rubinstein derivation, and whether the fetched blog mirror of
  ch. 8 is verbatim identical to the printed 2nd-edition text (both flagged unverified by the researching
  agent).
- Foster's actual body-text mathematical depth (only "Chapter Goals" teaser text was accessible past the
  paywall; the inference of thin math rigor is structural, not a direct read).
- The precise combined-objective display equation in R3GAN (the paper states $\mathcal L$, $R_1$, $R_2$
  separately with a textual description of maximizing $\mathcal L - R_1 - R_2$, rather than one single
  numbered combined-loss equation — this is reported as found, not filled in speculatively).
- Whether Bishop/Prince/Tomczak/Foster's evaluation-metric treatments (where present) engage with the
  2023-2025 FID/IS critique literature — plausible they don't, given timing, but not directly confirmed for
  all four.

No fact in this report was fabricated; every claim traces to a fetched primary or secondary source cited
inline, and every unresolved question is flagged above rather than guessed at.
