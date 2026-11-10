# Further Reading

Source: `chapter_generative-adversarial-networks/further-reading.md` (label `sec_gan-further-reading`), read to the slides marker. About 3,000 words, 22 entries under six headings, about 70 citations. No code. Claims about papers were checked against what I know with certainty; those I could not confirm are marked.

## Verdict

This is a better further-reading section than most textbooks have: every entry says what problem a paper addresses and what it changes relative to the chapter, several entries carry a real judgment ("collapse is delayed, not removed"; "a trained critic certifies only a lower bound"; "A reader who trains a critic on pretrained features should evaluate with a different feature network"), and the claims I could check are accurate. Three things keep it from being as useful as it could be. It does not rank: 22 entries at equal weight, no "if you read three things" and no order within a heading. It is too long for its job, because about a third of it is content the teaching sections need or already have (the three routes to a Lipschitz critic, the caveat on one penalty, limited data, the systems of the last section). And its last heading quotes about twenty benchmark numbers from 2024 to 2026 papers, two of them from 2026, which will date faster than anything else in the chapter. The most important fix is to add a ranked short list at the top and to move the two passages that correct the main text into the main text (FR-1, FR-3, FR-4).

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | Each entry is self-contained; a few assume terms the chapter never defined |
| Clarity of thought and focus | 3 | No ranking; a third of the page duplicates or repairs teaching sections |
| Pedagogy | 3 | The reader is not told where to start or what can be ignored |
| Language | 4 | Plain and exact; the "Read them for" closing is a small formula |
| Ease and enjoyment | 3 | Dense; the last heading is a list of scores |

## What works

- **The opening contract**: "Each entry states the problem it addresses, what it changes relative to the chapter, and why to read it; the topics follow the order of the chapter, and nothing in the book depends on them." The entries keep it.
- **"Least squares as a divergence."** It says exactly which coding the published derivation covers, notes that "The zero--one coding of the chapter's code ... falls outside the derivation", and draws the lesson ("it shows how much the coding constants matter"). I checked the algebra: with codes two apart and the generator's target one above the fake code the generator's loss is $\tfrac12\int (p-q)^2/(p+q)$, bounded by one. Correct.
- **"Output activations for $f$-GAN critics."** The four domains and activations match the paper's table, and the last sentence (a minibatch estimate can flatter the bound) is a judgment a practitioner needs.
- **"The feature network behind the score."** The best entry. Five effects that change FID without changing the images, each with its source, ending in a rule the reader can follow.
- **"Fixed Discrepancies and Learned Ones."** It asks a question ("Could a fixed discrepancy, with no second network, serve as well?") and answers it from the chapter's own estimation argument. This is how a further-reading entry should connect to the text.
- **"Few steps without a critic."** It records that the adversarial term is not the only route, which the teaching section does not say as plainly.

## Structural findings

1. **No ranking.** See FR-1.
2. **Two entries correct the main text and should move into it.** "Three ways to a Lipschitz critic" holds the distinctions that the review of `objectives.md` found missing from the Wasserstein passage (OBJ-9(b)): what clipping, spectral normalization and the one-centered penalty each enforce. "The cost of the penalty" holds the caveat that qualifies the chapter's most repeated citation: "StyleGAN2 also trains well on faces with $R_1$ alone, so the StackedMNIST ablation ... does not settle whether one penalty suffices" (CONV-9). A caveat on a claim made five times in the teaching sections cannot live only on the last page. See FR-3.
3. **Overlap with the teaching sections.** "Training on limited data" restates `dcgan.md` "Limited Data" and adds the papers; "Conditional scores" and "Auxiliary classifiers" repeat sentences of `conditional.md`; "Tokenizers", "Vocoders and video" and "Likelihood models as critics at scale" cover the systems of `adversarial-losses.md` again. The section reviewers recommend moving systems out of the teaching sections into this page (DCG-12, ADV-11, structural findings 2 and 3 of `adversarial-losses.md`). If that is done, this page becomes the single owner and the duplication goes; the page must then be cut elsewhere to stay near its present length.
4. **The last heading is a scoreboard.** See FR-4.
5. **What a reader will look for and not find.** (a) The literature on the dynamics of games: the bilinear game, extragradient and optimistic gradient methods, and the first local stability result for GANs. The chapter's best section, :numref:`sec_gan_convergence`, has no entry here; nothing in the chapter mentions these methods. (b) One survey or book chapter for the reader who wants the history in one place.

## Passage-level findings

**FR-1** (moderate, pedagogy). Opening paragraph.
Anchor: "This page points to the ones a reader of the chapter is best placed to read next."
Twenty-two entries follow at equal weight. Add a ranked short list directly after this paragraph, with a reason per item. A proposal: (1) :citet:`Mescheder.Geiger.Nowozin.2018` for the convergence theory the chapter states without proof; (2) :citet:`Huang.Gokaslan.Kuleshov.ea.2024` for the recipe the chapter adopts, read with the caveats of :numref:`sec_gan_convergence`; (3) :citet:`Kynkaanniemi.Karras.Laine.ea.2019` on precision and recall, since the chapter never measures coverage on images; (4) the entry on the feature network behind the score, before reporting any FID; (5) one of the distillation papers for where the critic is used today. Then say which headings are reference material to consult when needed (architectural devices, modulation variants, translation systems).

**FR-2** (minor, correctness of a pointer). "Translation systems".
Anchor: ":numref:`sec_gan_conditional` explains why pix2pix adds a critic to its $L_1$ loss."
`conditional.md` gives pix2pix four sentences; the explanation (a regression loss returns the conditional mean, which blurs) is the subject of :numref:`sec_gan_beyond`. Point there. If the pix2pix stub is removed from `conditional.md` as its reviewer recommends, this pointer breaks entirely.

**FR-3** (moderate, thought). "Three ways to a Lipschitz critic" and "The cost of the penalty".
Anchors: "These guarantees decide how close a trained critic can come to the Kantorovich potential", "does not settle whether one penalty suffices"
See structural finding 2. Move the content of the first entry into the Wasserstein passage of `objectives.md` (about 100 words there, a pointer here), and the StyleGAN2 sentence into "One Penalty or Two" of `convergence.md`. In the first entry, "tightly there" overstates a soft penalty: the one-centered penalty pulls the gradient norm toward one at sampled points and guarantees nothing; "softly, and only at sampled points".

**FR-4** (moderate, focus and currency). "Adversarial Terms in Recent Systems".
Anchor: "the numbers are those their papers reported between 2024 and 2026"
Five entries, about twenty quoted scores (FID 0.50, 5.5 to 4.9, 1.28, 3.43, 1.88, 1.79 to 1.30, 1.58 to 0.97, "twelve percent of the parameters", and so on). The sentence quoted concedes that these are a snapshot. The scores teach nothing by themselves: a reader cannot compare FID 3.43 at $256 \times 256$ with 1.88 at $512 \times 512$, and each will be superseded before the book is printed. Keep the argument of each entry and at most one number where the number is the point (DMD2's student scoring below its teacher). The same applies to "Training at scale", which lists eleven hyperparameter values of one paper's configuration; keep the two facts a reader can use (the penalty weight is scheduled with the learning rate; the rebuilt network reaches the starting point's score without its devices) and drop the rest. Verification: the DMD2, MeanFlow, consistency-model, DDO and R3GAN roadmap numbers match my recollection of those papers; the ViTok figures, ViTok-v2 (arXiv, 2026), RAF (2026) and WaveFM claims I cannot confirm, and the R3GAN schedule values (learning rate, $\gamma$, $\beta_2$, burn-in) only approximately.

**FR-5** (minor, language). "The relativistic divergence for general payoffs".
Anchor: "Her normalization is twice the chapter's, so her divergence is $2\, d_{\mathrm{Rp}}$."
Refer to the paper, not to its author by pronoun: "The paper's normalization is twice the chapter's". The same sentence occurs in `relativistic.md` (REL-14). The conditions quoted for the general result match my recollection of the paper; unverified in detail.

**FR-6** (minor, language). Pattern.
The closing formula "Read it for ..." / "Read them for ..." / "are the place to start for ..." ends seven entries: "Read it for the general conditions", "Read them for evaluation protocols that a single number can report", "Read them for the alternative to projection and its repair", "They are the place to start for image-to-image applications", "the two papers are the place to start for a dataset that small", "Their table is the place to look before", "Read together, the two papers show". Each is useful once; as a template it reads as filler. Vary, or fold the reason into the entry's first sentence.

**FR-7** (minor, explanation). "Pretrained features in the critic".
Anchor: "the second entry under Evaluation explains why that overlap matters"
A positional reference inside a page that the review proposes to reorder. Name the entry ("the entry on the feature network below").

**FR-8** (minor, thought). "Architectural devices, 2015 to 2021".
Anchor: "reading them against R3GAN shows which ones a well-behaved loss makes unnecessary"
This states the R3GAN paper's thesis as a fact the reader will see for themselves. The paper shows that one network without these devices reaches a similar score on a few datasets; it does not show device by device which are unnecessary, and StyleGAN-family models with the devices remain in use. "the R3GAN paper argues that a well-behaved loss makes most of them unnecessary".

**FR-9** (minor, correctness, unverified). "The cost of the penalty".
Anchor: "because its fused attention kernels and sharded training did not support it"
I recall the reason given in that paper as missing support for double backpropagation in the attention implementation and the distributed training setup; "sharded training" may not be the paper's term. Check the wording against the source. The description of the replacement (a finite-difference approximation of $R_1$ by Gaussian perturbation of real samples) and the video figures are correct as far as I know.

## Code and experiments

None. No number on the page comes from the chapter's notebooks; one sentence refers to an exercise that measures the flattered bound, which is the right kind of link.

## Exercises

None.

## Still open from October

The October note that the chapter carries "benchmark tables" survives here as the last heading (FR-4).

## For the chapter review

- This page is where the teaching sections should send their systems, erratum-style remarks and paper numbers. It can only take them if it sheds its own duplicates and its scoreboard.
- Two of its sentences are the chapter's best caveats on its own argument (one penalty or two; a critic certifies only a lower bound). The chapter states its claims in the sections and its qualifications on the last page and in the exercises. That pattern should be reversed.
- It has no entry for the convergence literature beyond the two papers the chapter uses, and none for coverage measures beyond one paragraph, the two places where the chapter's own evidence is thinnest.
- R3GAN is named three times here, each time as the reference point against which other work is read (FR-8). Together with the teaching sections this confirms the weight the chapter puts on one paper.
