# Generative Adversarial Networks (chapter introduction)

Source: `chapter_generative-adversarial-networks/index.md`, read to the slides marker. About 320 words of introduction in four paragraphs and a resources list of seven annotated entries (about 150 words). No code, no figure. The reference to :numref:`sec_regularized` was checked (the Bradley-Terry model is taught there under "Preferences and the Bradley-Terry Model").

## Verdict

Three good paragraphs and one list. The first three paragraphs do what an introduction should: they pose a real question (a sampler has no likelihood to raise, so what replaces it), give the idea in two sentences, and place the chapter honestly in 2026 ("have since replaced stand-alone GANs for most image synthesis, yet the learned comparison survives"). That third paragraph is the candour the voice guide asks for, and it is the reason the chapter is still worth reading. The fourth paragraph is then seven questions in a row with a label after each, the route template the Basics review found in every chapter. It does not say which sections carry the chapter and which can be skipped, it does not say that the chapter is built on one 2024 recipe, and it does not say what the experiments can and cannot establish. The most important fix is to replace the list by a route with a judgment in it (IDX-2, IDX-3).

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | The question and the idea are clear in 150 words |
| Clarity of thought and focus | 3 | The route is a list; the chapter's bet on one recipe is not stated |
| Pedagogy | 3 | No prerequisites, no guidance on what to skip, no compute note |
| Language | 4 | Plain and exact; one sentence that carries two ideas too many |
| Ease and enjoyment | 3 | The first three paragraphs invite; the fourth reads as a table of contents |

## What works

- **The question.** "Nothing in it computes the probability of the image it returns, so there is no likelihood to raise. Can a learned comparison take its place?" The reader knows in two sentences why a new training principle is needed. Keep it.
- **The idea in one sentence**: "The critic's output tells the generator where its samples are too dense and where too sparse, although neither density is ever written down." This is the density-ratio reading of the whole chapter, stated before any formula.
- **The placement in 2026.** The third paragraph names three places where an adversarial term is still trained (one-step generators, autoencoders, vocoders). It tells the reader why to read a chapter on a displaced method, and :numref:`sec_gan_beyond` pays it off.
- **The resources list** ties each entry to a section and says what the entry is for. The entry on :citet:`Lucic.Kurach.Michalski.ea.2018` ("given equal tuning budgets, many objectives ... attain similar scores") is the right warning to put in front of :numref:`sec_gan_objectives`.

## Structural findings

1. **The route is a list of seven questions.** See IDX-2. The questions themselves are well chosen (they are better than seven "asks how" sentences, because each is the section's real question), but seven interrogatives of equal weight give the reader no map: nothing says that the first, fourth and seventh sections carry the argument, that the second and third are theory a builder can skim, or that the fifth and sixth need a GPU.
2. **The chapter's main editorial decision is invisible here.** Sections 16.3 to 16.6 follow one paper (the pairing loss with two gradient penalties, on a plain backbone). The introduction mentions it only as a repository link in the resources list. See IDX-3.
3. **No prerequisites and no note on the appendix.** The chapter needs logistic regression, the covariate-shift classifier, KL divergence and entropy from Basics, and the Bradley-Terry model from :numref:`sec_regularized`; it cites the appendix for proofs (divergences, optimal transport) and reads without it. One sentence would say so.
4. **Nothing on how the chapter ends.** The third paragraph says what survives; it does not say that the last section hands over to :numref:`chap_diffusion` with a list of limits. One clause would tie the first and last pages together.

## Passage-level findings

**IDX-1** (minor, thought). First paragraph.
Anchor: "The reward model of :numref:`sec_regularized` learned differently, from which of two answers a judge preferred, through the Bradley--Terry model"
The antecedent offered is the Bradley-Terry model, which is the antecedent of the third section's pairing critic. The first section's critic is a classifier, and `gan.md` rightly ties it to the covariate-shift classifier of Basics instead (see GAN-1 in the review of `gan.md`). So the introduction prepares the reader for 16.3 and the first section starts from a different place. The sentence is also the longest of the page and holds three ideas. Fix: use the classifier as the antecedent here ("we have already trained a classifier to tell two samples apart and read a density ratio from its output, in :numref:`subsec_covariate-shift-correction`"), and leave Bradley-Terry to :numref:`sec_gan_relativistic`, which introduces it well.

**IDX-2** (moderate, pedagogy). Fourth paragraph.
Anchor: "Seven questions organize the chapter."
Seven questions in one paragraph, each followed by a label. Rewrite as a route with a voice, in three movements, and say what matters. Draft: "The chapter has three parts. The first asks what the game measures: the critic of :numref:`sec_basic_gan` estimates a log density ratio, and the generator's loss is a divergence that can lose its gradient; :numref:`sec_gan_objectives` and :numref:`sec_gan_relativistic` ask whether another loss repairs this (mostly it does not). The second asks whether training converges: :numref:`sec_gan_convergence` shows on a game with two parameters that it circles, and that a gradient penalty damps the rotation. The third builds: an image generator and its scores (:numref:`sec_dcgan`), a generator that takes a request (:numref:`sec_gan_conditional`), and the reason a critic survives in systems that are not GANs (:numref:`sec_gan_beyond`). A reader who wants to build can read 16.1, the first half of 16.4, and then 16.5 on; 16.2 and 16.3 are for the reader who wants to know what each loss measures."

**IDX-3** (moderate, thought). Whole introduction.
Anchor: "trains the loss and backbone of :numref:`sec_gan_convergence` and :numref:`sec_dcgan` at full scale"
This is the only place the introduction names R3GAN, and it does so as if the chapter's loss and backbone were common property. The chapter adopts one 2024 paper's recipe as its teaching baseline. Say so once, here, with the reason and the limit: "From :numref:`sec_gan_convergence` on we train with one recipe, the pairing loss with two zero-centered gradient penalties :cite:`Huang.Gokaslan.Kuleshov.ea.2024`, because it has few parts and each has a stated reason. The evidence that it matches older, more elaborate GANs is at the scale of face and ImageNet datasets; our notebooks can show that each part does what the analysis says on small problems, not that the recipe wins."

**IDX-4** (minor, pedagogy). End of fourth paragraph.
Anchor: "The experiments, in PyTorch and JAX, run on Fashion-MNIST garments, planar Gaussians whose densities we know, game sprites and CIFAR-10."
Good to list the data. Add the cost, since two notebooks of this chapter do not run on a laptop CPU: "The planar experiments run in seconds to minutes on a CPU; the sprite and CIFAR-10 runs of :numref:`sec_dcgan` and :numref:`sec_gan_conditional` take up to half an hour each on one GPU."

**IDX-5** (minor, correctness, unverified). Resources.
Anchor: "a 2016 training checklist, shows which stabilizers the penalties of :numref:`sec_gan_convergence` later replaced"
"Replaced" is the R3GAN paper's claim, not an established fact about practice; several items of that checklist (label smoothing, normalized inputs, Adam settings) are untouched by the penalties. "shows the stabilizers that the penalties of :numref:`sec_gan_convergence` are meant to make unnecessary".

**IDX-6** (minor, pedagogy). Resources.
Anchor: "These resources accompany particular sections; :numref:`sec_gan-further-reading` turns to the methods the chapter does not teach."
The division between this list and the further-reading section is clear and should stay. Two entries have no section attached (GAN Lab, StyleGAN3); say which section each goes with (GAN Lab with :numref:`sec_basic_gan` and :numref:`sec_gan_convergence`; StyleGAN3's metric tools with :numref:`sec_dcgan`).

## Code and experiments

None.

## Exercises

None.

## Still open from October

Not covered by the October review.

## For the chapter review

- The introduction and `gan.md` open on the same image (a hundred random numbers, one forward pass) with two different antecedents (IDX-1, GAN-1). One owner each.
- The introduction should carry the two statements several section reviewers asked for at chapter level: that the chapter adopts one recipe (IDX-3), and what small experiments can establish about it.
- The third paragraph ("the learned comparison survives") is the chapter's best statement of why it exists. The chapter's close (`adversarial-losses.md`, "Remaining Limits") should answer it and at present does not.
- The notation table that interrupts `gan.md` (GAN-2) could live at the end of this introduction, after the route, where a reader looks for it.
