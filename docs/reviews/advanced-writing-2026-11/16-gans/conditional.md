# Conditional Generation

Source: `chapter_generative-adversarial-networks/conditional.md` (label `sec_gan_conditional`). About 3,400 words of prose, 1,300 words of exercises, 14 code cells per framework. Stored outputs checked: `outputs/pytorch/.../conditional.json` and `outputs/jax/.../conditional.json`.

## Verdict

A good section with a clear question ("how do we ask for a bird, and check that we got one"), a memorable counterexample (the cyclic label shift), and the best derivation of the projection discriminator I know of in a textbook: Bayes' rule splits the optimal critic into a label term and a realness term, and a log-linear model of the label term gives the head. The first half would not be out of place in a very good text. The second half, the measurement, is weaker than it looks. The section promises to show that "a generator can obey every request and still return a single image per class", but the trained generator does not show it; a constructed baseline does, and that baseline prints two numbers (alignment 1.00, variance 0) that are true by construction. The one measurement that is supposed to carry "mode coverage" is a pooled feature variance, which is not a coverage measure, and the prose reads it as "lost variety". The most important thing to fix: make the diversity measurement honest (say what a variance can and cannot show, or replace it by a recall-type measure), and replace the two cannot-fail baselines by one that can surprise. Second: the mechanism that dominates practice, modulation (class-conditional normalization), gets one sentence and no code, while the generator uses the weakest mechanism (concatenation at the $4 \times 4$ input) with a justification that does not hold up.

## Scores

| dimension | score |
|:--|--:|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening.** "to get a pair of trousers from the garment generator we can only draw again until one appears" makes the reader feel the missing request with the chapter's own models. The second paragraph states three questions and announces the twist (checking is the harder half). Keep both paragraphs.
- **The cyclic shift** ("a request for a bird returns a cat") under "Stronger than Marginal Matching". One example shows that the unconditional game is blind to labels and that the conditional game sits near its worst value. It is reused later as a measurement baseline, which is the right kind of running example.
- **"The Projection Head from Bayes' Rule".** Identity first (:eqref:`eq_gan_cond_bayes`), then the reading ("realistic images with scrambled labels differs from the data through the first term; ... poorly formed images of the requested class differs through the second"), then the modelling assumption, flagged as such: "Bayes' rule is an identity, but the shared feature map is a modeling assumption". Exercise 2b then shows the form is not a restriction once $\phi$ has $K$ dimensions. This is candid and correct; preserve it.
- **Per-class constants.** "The critic never calibrates its scores across classes, so our diagnostics must not depend on these constants" is a real practical consequence drawn from theory, and the held-out pair accuracy is then chosen because the constants cancel from it. Theory steering a measurement choice is what the chapter does best.
- **Honest reading of noise.** "Individual gaps move by several points between runs, so the table does not rank them" and "The real accuracy is a reference, not an upper bound" follow the house rule on precision. The stored outputs agree with every number quoted (alignment 0.71 in both frameworks, referee 0.776 and 0.781, FD ratio 3.2 and 3.6, variance deficit 20% and 21%).
- **Code.** `CondGenerator` and `ProjectionCritic` are ten lines each and map line by line onto :eqref:`eq_gan_cond_projection`. The zero-initialized embedding table with the comment "start as an unconditional critic" is a detail that matters and is explained.

## Structural findings

1. **The section is two sections of unequal quality: the game and its critic (about 1,300 words, strong), and measuring a conditional generator (about 1,600 words, weaker).** It need not be split, at 3,400 words, but the second half should be rebuilt around one question: "which failures does each number see?" At present that question is answered by a prose paragraph and two constructed baselines after the fact; it should organize the subsection from the start (a small table: rows = wrong label, within-class collapse, poor images; columns = alignment, FD, within-class measure).

2. **"Same-Class Pairs" comes too early and is too dense.** Before the reader has seen a conditional GAN, the section spends 250 words on a second value identity (:eqref:`eq_gan_cond_rp_value`) whose content is "the same thing holds for the pairing loss, class by class". The only consequence the section uses is the per-class constant $\zeta_c$. Move the subsection to "Training with Same-Class Pairs", where the pairing is implemented; state the value identity in one sentence without a display; keep the $\zeta_c$ paragraph.

3. **Modulation is named and dropped.** The caption of :numref:`fig_gan_conditioning` and one sentence are all the reader gets on class-conditional normalization, the mechanism of BigGAN and of StyleGAN and the one the reader meets again in every diffusion U-Net (time and class embeddings enter by scale and shift). A reader cannot implement it from "scales and shifts a feature map, $a(c) \odot \phi + b(c)$, with one entry per channel": where it sits in a block, that $a$ starts at one and $b$ at zero, and why per-block injection beats a single concatenation at the input are all missing. Either teach it (a six-line `Modulate` layer and one sentence on why the label is injected at every resolution), or say plainly that the section does not and point to `sec_gan-further-reading`. Since this part of the book is where the reader learns to build modern systems, I recommend teaching it and making concatenation versus modulation the exercise, in place of exercise 8b or beside it.

4. **Image conditioning is a stub that the next section leans on.** "An Image as the Condition" is four sentences. The closing bridge and the first paragraph of `adversarial-losses.md` both rely on pix2pix as something "the previous section" taught. Either cut the subsection here and let `sec_gan_beyond` introduce pix2pix where its question lives (my preference: it removes a repetition, since the blur argument is owned by the next section), or keep it and name the patch critic, which the next section uses without a name.

5. **Text conditioning is one clause.** For a 2026 reader "a caption" is the condition that matters. The clause on StyleGAN-T is fine as a pointer, but the opening promises "a class, a caption, a sketch to complete". Add one sentence saying that the same projection form with a text embedding in place of $e_c$ is how it is done, and that this section stays with classes; the clause on StyleGAN-T almost says this already.

6. **Scale.** "well under half an hour on one GPU" means the experiment does not run on a reader's laptop: 15,000 steps with two gradient penalties (double backpropagation) on a 256-channel generator. The brief asks for experiments that run on the reader's machine. Say what a CPU reader should do (for instance 2,000 steps at reduced width, and which conclusions survive), or check whether the conclusions of the section (alignment near 0.7, the baselines) already appear at a tenth of the budget.

## Passage-level findings

**COND-1** (minor, language). Anchor: "proposed the obvious repair: give the request to both players", introduction.
"Request" is a good plain word, but the section then uses request, condition, label and class for the same object, and "slice" for $p(\cdot \mid c)$ (also "requested labels", "label posteriors", "condition alignment"). Fix: after the first paragraph of "The Optimal Conditional Critic" say once that for classes the condition is the label and that "request" means the condition handed to the generator; then use "condition" in theory and "class" in the CIFAR-10 experiment.

**COND-2** (moderate, explanation). Anchor: "We draw the generator's conditions from the data marginal", under "The Optimal Conditional Critic".
This design choice carries the whole derivation (it is why $p(c)$ cancels in :eqref:`eq_gan_cond_dstar` and in the Bayes step), but it is stated as a fact about the setup with no consequence attached, and the reader learns only in "Training with Same-Class Pairs" that the code realizes it by reusing the real batch's labels. Fix: add one sentence here, "In code this is free: the generated batch reuses the labels of the real batch", and one sentence on what happens otherwise (a mismatch in $p(c)$ becomes a term $\log(p(c)/q(c))$ in the optimal critic, which is where the per-class constants could hide).

**COND-3** (minor, explanation). Anchor: "would equal $0$, the largest value the game can take", under "Stronger than Marginal Matching".
Correct ($2\log 2 - 2\log 2$), but "the largest value" and "close to the worst conditional generator" ask the reader to hold a sign convention (the generator minimizes $\max_D V$) that was set three sections ago. Fix: "would equal $0$, the value of a game the critic wins outright".

**COND-4** (moderate, thought). Anchor: "The weighting by $p(c)$ has a cost for imbalanced data", under "Stronger than Marginal Matching".
A two-sentence paragraph that raises a real issue, gives no number, and is answered only in a two-dot exercise. It dangles. Fix: either give the bound in the text (a slice of weight $\epsilon$ can cost at most $2\epsilon\log 2$, so a class with one image in a thousand is worth about 0.0014 nats) or move the remark into the exercise.

**COND-5** (moderate, pedagogy). Anchor: "The critic of each class can be optimized separately, so the conditional value averages these divergences over classes", under "Same-Class Pairs".
See structural finding 2. In addition, "can be optimized separately" is true only for a critic with unrestricted dependence on $c$; with a shared feature map the classes are coupled, which is exactly the modelling assumption introduced in the next subsection. Fix: say "for a critic free to choose a separate function for each class".

**COND-6** (minor, thought). Anchor: "likewise hold class by class, since each concerns one pair of distributions", under "Same-Class Pairs".
A blanket claim about "the chapter's other results" with one example. The penalty analysis of `sec_gan_convergence` is local and about dynamics, and nothing in this section uses the claim. Cut the sentence.

**COND-7** (moderate, pedagogy). Anchor: "BigGAN computes $a(c)$ and $b(c)$ from a class embedding in every block of its generator", under "Three Ways In".
See structural finding 3. Also, "Papers write $\gamma(c)$ and $\beta(c)$, but $\gamma$ is the penalty weight in this chapter" is material for the notation table (where it already is), not for the running text. Cut it here.

**COND-8** (minor, explanation). Anchor: "The third mechanism conditions only the critic, through a head that scores how well sample and condition fit together", under "Three Ways In".
The figure calls it "compatibility head", the next subsection "projection head", the citation "projection discriminator", the class `ProjectionCritic`. Also $\phi$ is the generator's feature map in the middle panel of the figure and the critic's pooled features in the right panel and in :eqref:`eq_gan_cond_projection`. Fix: one name (projection head) in text and figure; use $h$ for the modulated feature map.

**COND-9** (minor, correctness, unverified). Anchor: "R3GAN's conditional critic keeps only the inner product", under "The Projection Head from Bayes' Rule".
I believe this is right (the StyleGAN2-ADA-derived critic projects a feature vector onto a mapped class embedding with no unconditional head), and I believe the generator claim in the next paragraph ("the division R3GAN uses in its class-conditional experiments") is right as well, but I cannot confirm either from certain knowledge. The same holds for "trained for four days on eight GPUs" under "Class-Conditional CIFAR-10". A reviser should check the three against the R3GAN paper and repository. The StyleGAN-T, BigGAN, AC-GAN and Miyato and Koyama claims are correct as stated.

**COND-10** (moderate, thought). Anchor: "no such identity says how the generator should use $c$, so concatenation leaves that free", under "The Projection Head from Bayes' Rule".
This justifies the weakest generator mechanism by the absence of a theorem. The honest reasons are that concatenation is the least code on the saved backbone and that it follows R3GAN. Exercise 2b also shows the projection head restricts nothing for ten classes, so "the critic gets the structure that the derivation motivates" overstates what the structure buys. Fix: "We use concatenation in the generator because it needs no change to the backbone, and the projection head in the critic because it shares one feature map across classes; with ten classes neither restricts what the networks can represent (Exercise ...), so the choice is about what is easy to learn at a fixed budget."

**COND-11** (minor, explanation). Anchor: "passes it as the base class's `extra` input, which is concatenated with the learned constant and the projected latent code before the mixing convolution", under "A Conditional Backbone".
The embedding is `nn.Embedding(num_classes, embed_dim)  # unit variance` in PyTorch and an explicit `normal(1.0)` initializer in JAX. The scale of the embedding relative to the projected latent decides how strongly the class enters at initialization, and the comment "unit variance" is the only trace of that decision. One sentence of prose: why unit variance, and that the JAX default would be much smaller.

**COND-12** (minor, pedagogy). Anchor: "The runs use the penalty weight $\gamma = 0.5$, against $\gamma = 10$ for the sprites", under "Training with Same-Class Pairs".
A factor of 20 with the explanation "a new setup needs its own sweep" and a pointer to an exercise. The batch size (128 against 64) and the evaluation size (512 against 256) also changed and only the second is mentioned. The reader should be told how the authors found 0.5 (what went wrong at 10), in one sentence; that is the practical knowledge the brief asks for.

**COND-13** (moderate, thought). Anchor: "At this budget the generator has not caught up with the data", under "Training with Same-Class Pairs".
The paragraph reports that the critic orders 80% of held-out pairs correctly and stops. The reader cannot tell whether this is a budget effect or the normal state of a penalized game (the sprite runs of `sec_dcgan` presumably also end well above one half). Say which, with the sprite number for comparison, and say what a reader should expect to see if training were ten times longer. Also "Both losses settle within the first few thousand steps, after a high start of the generator loss" narrates the plot at more length than it deserves; the losses add nothing the pair accuracy does not show.

**COND-14** (minor, language). Anchor: "Vehicles are recognizable by shape: cars and trucks as wheeled bodies, ships as hulls on water, airplanes against the sky", under "Samples on Request".
Three sentences describe what the figure shows. The observation worth keeping is the last one (animal classes are told apart by palette and background, which is what a shortcut looks like) because it motivates the referee. Cut the vehicle list.

**COND-15** (major, correctness of interpretation). Anchor: "The generator has lost some of the variety within each class, and alignment cannot see that loss", under "Fidelity and Within-Class Diversity".
The evidence is that the pooled within-class feature variance is 0.17 against 0.21. A lower variance of the referee's penultimate features does not establish lost variety: (i) the features are post-ReLU activations of a classifier, and softer, lower-contrast generated images (the chapter itself notes "soft outlines" on this backbone) lower activation magnitudes and hence variances without any change in variety; (ii) a variance averaged over 128 features and ten classes cannot tell a dropped mode from a uniform shrinkage, and a generator that drops half the modes of a class while spreading the rest can match it; (iii) no run-to-run range is given for the generated value, only the real-versus-real 3 to 5%. The summary repeats the reading ("about a fifth less within-class variety than real data"). The caller asked whether mode coverage is measured: it is not. Fix: (a) rename the reading to what is measured ("the generated features are about a fifth less spread out within a class; lower contrast and lost variety both produce this"); (b) add a measurement that is about coverage, for example the fraction of real test images of class $c$ whose nearest neighbour among 500 generated images of class $c$ is closer than its nearest real neighbour at the same sample size (a recall in the sense of Kynkäänniemi et al., ten lines with the saved features), with the real-versus-real value as floor; (c) state the range over two or three training seeds.

**COND-16** (moderate, pedagogy; author decision 4). Anchor: "The confident-image generator reaches alignment 1.00 with zero within-class variance", under "Two Baselines".
Both numbers are true by construction: the baseline is the image the referee is most confident about, repeated 100 times, so the referee labels it correctly and identical images have zero variance. The first baseline is similar: after shifting the requested labels, "alignment" is the referee's rate of confusing class $c$ with $c+1$, and FD is not recomputed because it cannot change. The cell `conditional-measuring-condition-alignment-5` therefore prints three numbers none of which could have come out otherwise. The point the baselines make is right and worth making; the cell does not make it. Fix: replace the degenerate generator by a partial collapse that the numbers must actually detect, and print all measurements for it: for instance the 10 most confidently classified real test images per class, each repeated ten times. These are real photographs, so per-image fidelity is perfect; alignment rises above the real reference, the within-class measure falls, and FD rises well above the real-versus-real floor. That table shows "alignment rewards prototypes" with numbers that could have failed, and it previews the truncation trick. The label shift can stay as one sentence of prose with no cell.

**COND-17** (minor, language). Anchor: "This *within-class collapse* is mode collapse inside each conditional distribution", under "Two Baselines".
Coined terms in this section: referee, condition alignment, gap, held-out pair accuracy, within-class feature variance, within-class collapse, slice. Seven is too many for 3,400 words. "Gap" and "within-class collapse" can go (the second is defined by the phrase that follows it); "alignment" and "referee" earn their place.

**COND-18** (moderate, pedagogy; author decision 6). Anchor: "We can now ask a GAN for a class and check what we got", last paragraph before the Summary.
A recap paragraph before the Summary, in the form the author asked to remove; the previous paragraph ("Conditional evaluation thus has three axes") is a second recap, so the section ends three times. Fix: keep the three-axes sentence and the pointer to published scores; delete the "We can now" sentence; keep the bridge to `sec_gan_beyond` only if pix2pix stays in this section (structural finding 4).

**COND-19** (minor, language). Pattern: sentence-initial or medial "thus" as the hinge of a conclusion, 3 instances in 3,400 words: "Thus the conditional game is an average of per-condition games", "Thus the optimal conditional critic separates condition compatibility", "Conditional evaluation thus has three axes". Vary or drop; in each case the sentence stands without it.

## Code and experiments

- `conditional-measuring-condition-alignment-5`: three printed numbers that cannot fail (COND-16).
- `conditional-measuring-condition-alignment-4`: the within-class variance line is read for more than it shows (COND-15). The FD and MMD rows are read correctly and against a floor.
- `conditional-training-with-the-chapter-s-loss-3`: prose matches (0.52, 0.063/0.066, 1.07/1.09, 0.79/0.81). "down from about 90% early in training" is read off the plot; no cell prints it. Acceptable, but printing the mean of the first 2,000 steps beside the last 5,000 would cost nothing.
- `conditional-measuring-condition-alignment-1`: "its accuracy on a single class moves by tens of points between two training seeds" without augmentation is an authors' report that no cell shows. It is a useful warning; mark it as such ("in our runs").
- `conditional-class-conditional-cifar-10-1`: a cell that only defines the class names; merge it into the next cell.
- `conditional-a-conditional-backbone-at-32-by-32-2`: the shape check prints `[2, 3, 32, 32]` and `[2]`; fine, once.
- Framework tabs: the two tabs differ only in mechanics, and the prose never has to fork. No TensorFlow or MXNet material.
- The referee is 78% accurate. That is low enough that alignment near 0.7 says as much about the referee as about the generator; the section handles this with the real-accuracy column, correctly. A sentence on why a stronger referee was not used (budget) would pre-empt the obvious question.

## Exercises

Eight exercises, graded, most of them good and several excellent (2 on separate feature maps and the non-restriction result; 3 on the critic as a classifier, which makes the constants $\zeta_c$ tangible; 7 on the auxiliary-classifier bias). Three remarks:

- Exercise 1 is cited from the text as computing "both values for two Gaussian slices"; good. It is the only place the reader computes :eqref:`eq_gan_cond_value`; consider moving its part (c) numbers (two lines of quadrature) into the text beside the cyclic shift, where at present the claim "just below that maximum" has no number.
- Exercise 5c quotes "only 16% of its models trained without orthogonal regularization tolerated truncation ... against 60% with it". This matches my memory of the BigGAN paper, but "`ema_G` had none" is ambiguous (no regularization, or no artifacts?). Reword.
- Exercises 6 and 8 each need three or more training runs of 4,000 steps on a GPU. Mark the cost. Exercise 8 mixes two unrelated sweeps (the penalty weight and the critic head); split it or drop 8a, which repeats the sprite exercise on $\gamma$.
- Add: an exercise that implements modulation in the generator if the text does not (structural finding 3).

## Still open from October

October's chapter note ("the two teachable ideas ... are buried under divergence algebra") applies here only to "Same-Class Pairs". Its complaint about per-framework narration does not apply to this section.

## For the chapter review

- Order is right: `sec_dcgan` hands over a generator that "answers no request", and this section supplies the request. It depends on `sec_gan_relativistic` (pairing value, shift invariance), `sec_gan_convergence` (the penalty weight, saved steps) and `sec_dcgan` (backbone, `train_r3gan`, FD, MMD, the feature CNN). It does not need the appendix.
- Overlap: the pix2pix paragraph duplicates the opening of `sec_gan_beyond`; the published conditional scores and the AC-GAN remark are repeated in `further-reading.md` almost word for word. One owner each.
- The saved-machinery style of the chapter makes this section's experiment a single call to `d2l.train_r3gan`. That is compact, but a reader who opens the chapter here cannot see a training step. A one-line reminder of what the loop does per step is present and sufficient.
- Suspected chapter-wide patterns: a "We can now ..." paragraph before every Summary (also at the end of `relativistic.md` and `dcgan.md`); cells whose printed numbers are fixed by construction; diversity claimed from a proxy (here variance; check what `convergence.md` does on the 25 Gaussians, where coverage can be counted exactly, and consider reusing that count here per class on a toy).
- The chapter never measures coverage on images. If one place should, it is this section, because within-class collapse is its announced twist.
