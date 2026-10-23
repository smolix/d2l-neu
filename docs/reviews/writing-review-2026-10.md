# Writing Review of Basics, Advanced, and the Mathematics Appendix

Date: 2026-10-23. Source: `main` at `732c1431`.
Standard: `docs/writing-guide.md` (cited below as "guide §n").

## Method

Eight read-only reviews covered the 23 chapters in scope. Each review read
every chapter introduction, the opening of every section, and at least three
sections per chapter in full; the State Space Models chapter was read in
full. Each review tagged its findings with one of seven defect classes:
motivation, density, reference burden, language, focus, mathematics, and
code. Separately, I read the four model chapters (Strang ch. 5, MacKay ch. 20,
Cover and Thomas ch. 2, Schölkopf and Smola ch. 1) next to the SSM chapter,
and computed the metrics in the table below from the sources. Line numbers
refer to the source `.md` files at the commit above. I checked about 30
quoted findings against the sources. Every quote was found; two were a few
lines off and are corrected here.

## Main Findings

1. **Sections teach a field, not an idea.** Most chapters cover far more
   than their central idea needs. They carry surveys of named models, dated
   lists of production systems, paper errata, version histories of
   frameworks, and research-level asides inside foundational material
   (pseudospectra in the eigenvalue section, conservative fields in
   single-variable calculus). The reviews found roughly 200 such passages.
   The appendix and the Advanced chapters are the worst affected.
2. **Openings list mechanisms instead of posing a problem.** In the
   mathematics appendix nearly every section opening is a list of topics. In
   ch. 12 and 13, about ten of fourteen sections open with a "This section..."
   roadmap. The model chapters open with a problem the reader can feel:
   Strang's two coupled equations with initial values 8 and 5, MacKay's
   biologist sorting brown things that run away from green things that
   don't.
3. **Arguments depend on distant text.** Ch. 12 makes 3 references per
   thousand words to other chapters and ch. 15 makes 5. Many of these carry
   an argument the reader cannot follow without flipping back ("the
   eigenvalue analysis of :numref:`subsec_bptt-gradient-pathologies` applies
   with the eigenvalues in plain sight"). Coined terms travel between
   chapters as if they were standard: "census", "testbed", "seam", "ladder",
   "rung", "scoreboard", "spine". The Preliminaries point into the appendix
   34 times, and ch. 16 and 17 rely on it for about 80 proofs and facts.
4. **The same fact is taught several times.** The reviews found more than 40
   clusters of repeated material. Examples are the descent lemma (ch. 25 and
   ch. 26), backprop (three times in ch. 25, plus the main text), the
   probability chain rule (four times in ch. 8), the NCCL fallback bug (four
   times in ch. 13), and the $1/b$ variance law (four times in ch. 9).
5. **Symbols change meaning.** Within single chapters, $\alpha$, $\sigma$,
   $\tau$, $\lambda$, $\kappa$, $C$, $T$, and $\mathbf{W}$ each carry two to
   four meanings. Several sections add a glossary box or a "not the $\alpha$
   of earlier" sentence to patch a clash instead of renaming.
6. **The language is mannered.** The scope has 1,700 em dashes, about 1,000
   uses of "exactly", and over 600 of "rather than". There are 44 wh-clause
   headings, plus sentence headings and riddles ("The One That Kept the
   Normalizer"). The prose relies on stagecraft ("The prosecution calls two
   witnesses"), economic metaphor ("will finally cash"), and stacks of hedges
   that sometimes contradict the conclusion a few lines later. Editing has
   left sentences that do not parse in at least a dozen places.
7. **Experiments are over-audited.** Many experiments come with paragraphs on
   what they do not show, confounder tables, per-framework narration, and
   end-of-section scope disclaimers. In several cases the experiment does not
   show the claimed effect at all (MoE without a gain, a GAN showcase in
   which both arms reach the maximum score), or the PyTorch and JAX tabs draw
   opposite conclusions.

The chapters that work best share the model chapters' method. Ch. 8
(Sequence Models) and ch. 10 (Attention) are problem-led and read their
experiments out. Ch. 16's Dirac-GAN passage and ch. 13's performance-model
opening are as good as anything in the model chapters.

## The Model Chapters Compared with D2L

| Technique | Model | D2L, typical case |
|---|---|---|
| Place the chapter in the course in two sentences | Strang 5.1: "The first half was about $Ax=b$." | SSM introduction: 1,840 words, two tables, a results preview |
| Problem with numbers before definitions | Strang 5.1: coupled equations, $v(0)=8$, $w(0)=5$ | eigendecomposition opens with the definition and a list of uses |
| Scalar or $2\times2$ case solved to the end | Strang 5.3: Fibonacci, the California Markov chain | eigendecomposition states $A^n=W\Lambda^nW^{-1}$ with no worked recurrence; the expansion $\sum c_i\lambda_i^k w_i$ first appears 930 lines later |
| Failure motivates the next method | MacKay 20.1–20.2: K-means fails on unequal and elongated clusters, so soft K-means | ResNet never states the degradation problem it solves |
| New method as a difference from the old one | MacKay: "The update step is identical; the only difference is..." | variants arrive as catalogues (25 architectures in ch. 7) |
| One plain sentence after every identity | Cover and Thomas: "Thus $X$ says as much about $Y$ as $Y$ says about $X$." | displays followed by further displays, or by hedges |
| Numbers that settle a property | Cover and Thomas: $D(p\|q)=0.2075$, $D(q\|p)=0.1887$ bits; the ½, ¼, ⅛, ⅛ source needs 1.75 yes/no questions | ch. 28 shows the ½, ¼, ⅛, ⅛ source only in a figure caption (`mdl-information-theory.md:475`), and gives the coding interpretation of KL 400 lines after its definition (`:629`) |
| Definitions in dependency order | Cover and Thomas: entropy, joint and conditional entropy, relative entropy, mutual information, then their identities | ch. 28 uses mutual information for rate–distortion and capacity two sections before defining it (`mdl-information-theory.md:844, 925`) |
| A simple method that contains the general form | Schölkopf and Smola 1.2: the class-means classifier is already a kernel expansion | ch. 12 reaches linear recurrence through a cited simplification of the GRU |
| References add depth, never missing steps | Schölkopf and Smola: later chapters "fill in the gaps ... without losing sight of the main ideas" | ch. 16 "Recall from :eqref:`eq_mdl-f-div-def`"; ch. 9 hands a demonstration to the appendix |
| Scope decisions stated | Strang: defective matrices "will not allow them to take over the book" | Jordan form, pseudospectra, Lyapunov exponents, and random-matrix laws take 3,500 words of ch. 24 |
| Noun headings | "Entropy", "Diagonalization of a Matrix", "Soft K-Means Clustering" | "Why the Gap Lives Where It Does", "Same Logits, Measured" |
| Brief, true wit in asides | MacKay: the "K" in K-means is "a silly name, but we are stuck with it" | "ConvNeXt is a controlled experiment wearing an architecture's name" |

## Ratings

Severity runs from 1 (polish) to 5 (rewrite). "Story" is what a reader
would say the chapter is about now, and what it should be about.

| Ch. | Chapter | Sev. | Story now | Story it should tell |
|---|---|:-:|---|---|
| 1 | Preliminaries | 3 | a tour of tensor APIs, pandas, linear algebra, calculus, autograd, and probability, each pointing to the appendix | the four tools needed to fit $y \approx Xw$ to a table and judge the fit, each introduced by the step that needs it |
| 2 | Linear Regression | 3 | regression plus a training framework, learning theory, and weight decay with SVD, MAP, and AdamW detours | the whole supervised loop on the simplest model: least squares, SGD in code, overfitting, a penalty |
| 3 | Linear Classification | 4 | softmax regression with asides on physics, information theory, and exponential families; a theory survey | categories need probabilities; maximum likelihood gives softmax and cross-entropy; we measure correctly and learn when measurement fails |
| 4 | Multilayer Perceptron | 3 | definition, activation plots, backprop, initialization, a survey of generalization theory, dropout | linear models fail on XOR; a hidden layer fixes it; backprop trains it; initialization and dropout keep it usable |
| 5 | Computation | 3 | a tour of module mechanics and production hygiene | a model is a tree of registered state, and initializing, saving, moving, casting, and inspecting are walks over that tree |
| 6 | CNNs | 3 | convolution from equivariance and locality, shape mechanics, LeNet | images need a layer whose cost does not grow with the image; equivariance and locality force convolution; LeNet beats the MLP |
| 7 | Modern ConvNets | 4 | a chronological museum of 25 architectures | what makes a deep convnet trainable and efficient, and how much each ingredient matters with the recipe held fixed |
| 8 | Sequence Models | 2 | tokens, n-grams, perplexity, RNNs, BPTT, decoding | the same, shorter, with one notation |
| 9 | Optimization | 3 | optimizers from GD to Muon raced on a tiny transformer, with protocol caveats | training is slow because curvature varies across directions and gradients are noisy; each method fixes one of the two |
| 10 | Attention | 2 | attention as soft lookup, made learnable, multi-headed, position-aware, cheaper, interpretable | a fixed state loses information, so attention retrieves by content; what that costs and what it enables |
| 11 | Transformers | 3 | build a configurable GPT, then tour the 2023–2025 model zoo | one block on a residual stream, and the few choices that make it trainable and affordable at predictable cost |
| 12 | State Space Models | 5 | a tour of every recent recurrent architecture, each with experiment, caveats, and back-links | a fixed state gives constant memory; train it in parallel, make it selective, make it editable, learn what it cannot hold |
| 13 | Computational Performance | 3 | measure, classify, fix, re-measure on one machine, mixed with spec sheets and bug forensics | speed is bound by arithmetic, bandwidth, or overhead; intensity says which; each has one standard fix |
| 14 | Reinforcement Learning | 3 | algorithms on FrozenLake with pointers to later chapters and to LLMs | learning to act: the objective, dynamic programming, learning from samples, generalizing with networks |
| 15 | Deep RL | 4 | seven audited algorithms, each followed by diagnostics and disclaimers | neural RL is unstable because targets, data, and policy move together; three controls tame it |
| 16 | GANs | 4 | closed-form evaluations of adversarial objectives, then experiments and a survey | a classifier learns a density ratio; the ratio vanishes on disjoint supports and the game rotates; penalties and pairing fix both |
| 17 | Diffusion Models | 3 | a faithful walk through a lecture course, each step checked on a 2D mixture | generation is regression: denoising gives the score at every noise level, and integrating from noise gives samples |
| 24 | Linear Algebra | 3 | every linear-algebra fact a researcher might meet, each proved and tied to a use | in the right basis a matrix acts as independent scalings; that explains powers, stability, low rank, and conditioning |
| 25 | Calculus | 3 | first-year calculus with proofs, interleaved with optimization and backprop told three times | a derivative is a local linear model; composing them multiplies Jacobians; integrals define expectations |
| 26 | Optimization | 4 | a graduate survey from the descent lemma to KKT and floating point | a race between step size and curvature; the quadratic model explains it, noise sets the floor, convexity and constraints bound it |
| 27 | Probability and Statistics | 3 | densities, eleven distributions, MLE, MAP, EM, posterior approximations, tests, concentration, naive Bayes, each hedged | learning from data is fitting a probabilistic model by likelihood, or averaging over a posterior, and knowing how far a finite sample can be trusted |
| 28 | Information Theory | 4 | a list of quantities and a dozen divergences, each paired with a deep-learning use | log-loss measures surprise and code length; KL is the cost of the wrong model; MI is the KL from independence; training objectives are discrepancies estimated from samples |
| 29 | Dynamics | 3 | ODE solvers, Itô calculus, Fokker–Planck, then a survey of diffusion and flow-matching variants | turning noise into data needs a path, a field, and a solver; the only unknown is a conditional expectation, which regression estimates |

## Metrics

Body words exclude code, slides, and exercises. "Other-chapter refs" counts
`:numref:`/`:eqref:` targets outside the chapter. Sentence length is computed
on prose with math collapsed. Headings counts wh-clause and sentence-form
headings.

| Ch. | Body words | Exercise words | Longest section (body words) | Other-chapter refs / 1k words | Sentences > 35 words | Em dashes / 1k words | Bad headings |
|---|---:|---:|---|---:|---:|---:|---:|
| 1 Preliminaries | 19,939 | 6,853 | probability (6,215) | 2.3 | 5% | 1.4 | 1 |
| 2 Linear Regression | 16,622 | 5,517 | linear-regression (4,170) | 1.5 | 6% | 1.4 | 1 |
| 3 Linear Classification | 15,590 | 5,630 | environment-and-distribution-shift (4,331) | 1.5 | 10% | 0.8 | 2 |
| 4 Multilayer Perceptron | 16,535 | 7,514 | mlp (3,422) | 2.3 | 8% | 2.8 | 0 |
| 5 Computation | 42,358 | 11,136 | gpus-devices-memory (7,442) | 1.3 | 13% | 0.9 | 1 |
| 6 CNNs | 10,964 | 4,913 | conv-layer (2,253) | 1.6 | 4% | 2.6 | 0 |
| 7 Modern ConvNets | 21,658 | 7,341 | batch-norm (3,240) | 1.2 | 9% | 1.6 | 1 |
| 8 Sequence Models | 20,287 | 5,251 | decoding (3,733) | 1.2 | 14% | 2.4 | 2 |
| 9 Optimization | 31,194 | 9,700 | muon (3,782) | 2.1 | 13% | 4.1 | 14 |
| 10 Attention | 15,364 | 13,974 | attention-at-scale (3,633) | 1.2 | 8% | 4.1 | 3 |
| 11 Transformers | 19,234 | 14,755 | kv-cache (3,798) | 3.1 | 14% | 6.5 | 3 |
| 12 State Space Models | 37,166 | 15,449 | ssm (6,943) | 3.0 | 23% | 4.6 | 6 |
| 13 Computational Performance | 22,858 | 16,310 | fast-transformer (5,202) | 1.7 | 24% | 17.6 | 8 |
| 14 Reinforcement Learning | 20,916 | 16,102 | qlearning (3,473) | 3.1 | 2% | 0.0 | 0 |
| 15 Deep RL | 24,135 | 18,599 | ppo (4,127) | 5.3 | 4% | 0.1 | 4 |
| 16 GANs | 30,393 | 17,315 | dcgan (5,065) | 1.4 | 1% | 0.0 | 7 |
| 17 Diffusion Models | 32,470 | 24,824 | discrete-diffusion (4,100) | 2.2 | 22% | 0.0 | 4 |
| 24 Linear Algebra | 23,536 | 8,126 | geometry-linear-algebraic-ops (9,768) | 0.8 | 9% | 1.2 | 3 |
| 25 Calculus | 23,381 | 9,694 | single-variable-calculus (6,571) | 1.9 | 10% | 1.3 | 3 |
| 26 Optimization | 33,403 | 15,761 | convexity (8,000) | 2.5 | 12% | 0.7 | 4 |
| 27 Probability and Statistics | 37,608 | 18,364 | maximum-likelihood (6,608) | 2.0 | 10% | 0.6 | 2 |
| 28 Information Theory | 23,767 | 9,857 | divergences-distances (8,034) | 1.3 | 12% | 1.5 | 5 |
| 29 Dynamics | 29,281 | 11,565 | score-matching-diffusion-flow (9,149) | 1.6 | 13% | 1.4 | 2 |

The chapters with no em dashes (14–17) were revised earlier against the
previous guides. They show that mechanical rules remove dashes but not
density, hedging, or survey material.

## Chapter Notes

Each note lists the main problems, what to cut, and passages worth keeping
as models. Findings cite `file:line`.

### Ch. 1 Preliminaries

- Variance is used about 750 lines before it is defined
  (`probability.md:305`). Steepest descent rests on two facts that are sent
  to the appendix (`calculus.md:284–295`).
- A sentence does not parse: "This limit gives the value approached by the
  ratio ... converges to as we shrink $h$ to zero" (`calculus.md:95`).
- Eight notations are given for one partial derivative (`calculus.md:255`),
  and $\mathbf{A}$ is reused for a different matrix two paragraphs later.
- The linear-algebra opening is a contents list (`linear-algebra.md:12`).
- **Cut:** plotting utilities (move to Tools), Markov/Chebyshev (to
  appendix), the utility and loss-aversion aside, etymology, measurable
  sets.
- **Keep:** `probability.md:9–11` (unequal counts from a fair coin), the
  HIV example (`probability.md:836–880`), the broadcasting rule
  (`ndarray.md:678`).

### Ch. 2 Linear Regression

- The `oo-design` section interrupts regression with tracing, graph
  breaks, and asynchronous devices (`oo-design.md:150`).
- AdamW appears before Adam (`weight-decay.md:224`). The SVD and
  effective-degrees-of-freedom material relies on the appendix.
- `generalization.md:203–209` says the same thing three times.
- The scratch and concise sections disagree on the learning rate without
  comment (`linear-regression.md:208` vs `linear-regression-concise.md:186`).
- **Cut:** the `ProgressBoard` scheduling discussion (to ch. 5), AdamW, SVD
  degrees of freedom, double descent (kept once, in ch. 4), density-plotting
  code.
- **Keep:** the house-price opening (`linear-regression.md:9–15`), least
  squares as projection (`:342–350`), "the gradient is the error-weighted
  input" (`linear-regression-scratch.md:179`).

### Ch. 3 Linear Classification

- The softmax section carries five tangents before and after the
  definition: probit and Fechner, Gumbel, Boltzmann, tie-point genericity,
  exponential families (`softmax-regression.md:147–253`, `:378`).
- One example encodes its labels two ways (`softmax-regression.md:52` vs
  `:75`). $\mathbf{W}$ is transposed silently (`:268`). Source and target
  densities switch from $p_S, p_T$ to $q, p$, and the domain label from
  $\pm1$ to $0/1$ (`environment-and-distribution-shift.md:50, 299, 365, 453`).
- The VC bound appears before the VC dimension is defined
  (`generalization-classification.md:417` vs `:429`).
- **Cut:** the tangents above, Radon's theorem, "Beyond Static Supervised
  Learning", the tank anecdote.
- **Keep:** the always-healthy classifier at 99% (`classification.md:208`),
  0.51 versus 0.99 (`:122`), the footwear example and Goodhart's law
  (`environment-and-distribution-shift.md:14–34`).

### Ch. 4 Multilayer Perceptron

- `generalization-deep.md` is 4,000 words of qualified survey with no
  experiment, and it repeats double descent from ch. 2.
- `dropout.md` derives inverted dropout twice (`:54`, `:106`) and then
  disowns its own intuitions (`:65`, `:80`).
- Six code cells only plot activation functions (`mlp.md:401–648`), against
  the rule that code teaches and does not draw.
- `backprop.md` introduces a third layout for $\mathbf{W}$ (`:68`) and an
  undefined operator, $\textrm{prod}$, that carries every equation (`:170`).
- **Cut:** reduce `generalization-deep` to about 1,200 words around one
  experiment; citation catalogues (`mlp.md:126–134, 294–300, 688–693`).
- **Keep:** the XOR re-representation (`mlp.md:252–267`), the worked
  example $e = (a+b)c$ (`backprop.md:274–288`), the derivation of He
  initialization (`numerical-stability-and-init.md:395–413`).

### Ch. 5 Computation

- About 28 passages use Transformer and LLM concepts six chapters early:
  KV caches (`parameters-state-memory.md:398`), RMSNorm via LayerNorm
  (`custom-layers.md:231`), GPT-2 initialization (`init.md:583`).
- Version histories: "PyTorch 1.7 through 1.11 enabled tf32 *by default*"
  (`numerics.md:231`).
- A duplicated clause left from editing (`gpus-devices-memory.md:685`). The
  index roadmap disagrees with the order in `_quarto.yml`.
- **Cut:** TF32 history, fp8, ZeRO byte conventions, device survey, cudnn
  minutiae, mmap and meta-device loading.
- **Keep:** "The model computes, yet it owns zero parameters"
  (`model-construction.md:609`), loss scaling (`numerics.md:965`), the NaN
  at step 40,000 (`reproducibility-inspection.md:850`).

### Ch. 6 CNNs

- Much of the prose is unrevised legacy text with grammar errors
  (`why-conv.md:109`, `conv-layer.md:241`, `lenet.md:209`).
- The key step of the derivation is asserted, and the tensor $\mathsf V$
  changes order three times (`why-conv.md:139`).
- Padding and stride are used one section before they are defined
  (`conv-layer.md:573`).
- LeNet's training run is never read out or compared with the MLP
  (`lenet.md:348`). The MNIST "trimming" aside (`:279`) is likely wrong.
- **Keep:** `why-conv.md:307`, three $3\times3$ versus one $7\times7$
  (`conv-layer.md:624`), exponential receptive-field growth
  (`padding-and-strides.md:374`).

### Ch. 7 Modern ConvNets

- A catalogue of about 25 architectures with no organizing question.
- ResNet never states the problem it solves: deeper plain networks have
  higher training error (`resnet.md:9`).
- Batch norm uses $\epsilon$ before defining it (`batch-norm.md:93`),
  explains train and test modes three times, and carries a 500-word essay on
  internal covariate shift.
- ConvNeXt is explained through Transformers, four chapters ahead
  (`convnext.md:92`).
- Training runs without a readout (`batch-norm.md:653`,
  `cnn-design.md:427`).
- **Keep:** `training-recipes.md:9` (76.1% to 80.4% from the recipe alone),
  `efficient-convnets.md:62`, "Which of these differences actually carried
  the improvement?" (`convnext.md:13`).

### Ch. 8 Sequence Models

- The best-organized chapter in Basics. Its problems are repetition (the
  chain rule four times) and notation: $T$ is both sequence length and
  temperature, $k$ both beam size and top-$k$, $n$ three things, and
  $\mathbf{W}_\textrm{hx}$ flips layout (`bptt.md:145`).
- "Tokenizers in the Wild" (`text-sequence.md:630–797`) belongs in the
  Language Models part.
- **Keep:** `sequence.md:112`, "the trigram model ... has by far the *worst*
  held-out perplexity" (`language-model.md:412`), the BPE trace
  (`text-sequence.md:201–228`).

### Ch. 9 Optimization

- The test model `TinyLM` is built inside the Adam section from ch. 11
  material (`adam.md:368`); four later sections depend on it.
- The first section calls helpers that the next section defines
  (`optimization-intro.md:204`).
- $\lambda$ is curvature, then a decay rate, then weight decay. The
  $1/b$ law appears four times and the test valley three times.
- Hedges contradict advice: "These four reports are case studies, not a
  representative survey" against "take the consensus column ... as given"
  (`practice.md:54`, `:510`).
- 14 wh-clause headings, the most in scope.
- **Cut:** cache tutorial (to ch. 13), "Large-Scale Use", "The Current
  Frontier", "Configurations Reported", per-framework result paragraphs.
- **Keep:** `momentum.md:63–65`, "One clock cannot serve both populations"
  (`adam.md:54`), the river valley (`lr-scheduler.md:702`).

### Ch. 10 Attention

- Strong. The multi-head opening motivates something the standard layer
  does not fix (`multihead-attention.md:158`).
- It leans on ch. 12 through six forward references
  (`attention-at-scale.md:959`).
- The PyTorch and JAX tabs draw opposite conclusions about the speed of
  linear attention (`attention-at-scale.md:912`, `:930`).
- Exercises are 91% of the body length.
- **Keep:** the scalar database (`queries-keys-values.md:13`), binary
  position bits (`positional-information.md:151`).

### Ch. 11 Transformers

- Hedges that contradict the conclusion (`scaling-laws.md:475` vs `:499`;
  `index.md:39` vs `scaling-laws.md:529`).
- The MoE experiment shows no gain and is left uninterpreted
  (`moe.md:740`).
- "roofline" is used before ch. 13 defines it (`kv-cache.md:543`).
- **Cut:** configuration tables and "Reported Extensions" (about 1,200
  words), Perceiver IO and descendants, the duplicate ViT block.
- **Keep:** `moe.md:4–6`, the rich-get-richer routing loop (`moe.md:298`),
  reversal as a task with known alignment (`encoders-decoders.md:373`).

### Ch. 12 State Space Models

The worst chapter, and the clearest case for the guide.

- **Introduction** (`index.md`, 1,840 words): results before experiments
  ("the expected squared read error is $(n-1)/d_k$"), a five-row glossary of
  "state", a table of confounders, a paragraph on what the chapter is not,
  and more than a dozen model names. Guide §12.1 shows a 300-word
  replacement.
- **Reference burden:** about 430 references in 37,000 body words, 113 of
  them to other chapters. Each section rebuilds its premise from elsewhere:
  `matrix-state.md` from ch. 10's linear attention, `test-time-regression.md`
  from ch. 10's Nadaraya–Watson (185 lines spent learning one bandwidth
  before the section's idea starts at `:235`).
- **Private vocabulary:** "ladder" and "rung" (38 uses, two meanings),
  "crowding", "scoreboard", "classical spine". A callback refers to a
  "pendulum question from this chapter's introduction" that the
  introduction does not contain (`hybrids.md:1008`).
- **Focus:** 170 lines of control theory and the HiPPO, NPLR, and DPLR
  bridge (`ssm.md:751–924`), whose only downstream use is one
  initialization. Also RWKV-7, DeltaProduct, and the formal-language survey
  in `deltanet.md`, the mLSTM stabilizer, and the design rules and recipe
  table of shipped hybrids (`hybrids.md:776–918`).
- **Notation:** $\Delta_t$ is Mamba's step and Longhorn's gate; $\theta$ is
  outer parameters and an inner learning rate; $C$ is readout and chunk
  length.
- **Language:** "The prosecution calls two witnesses" (`mamba.md:135`),
  "load-bearing" (`ssm.md:83`), an answer to a deleted question
  (`matrix-state.md:447`), a sentence with no subject
  (`test-time-regression.md:239`).
- **Proposed sequence:** (1) a KV cache grows and a fixed state does not:
  what can constant memory hold? (2) gates are learned decay; (3) a linear
  update $h_t = a_t h_{t-1} + b_t$ and the scan, taught from the scalar case
  (guide §5.1); (4) input-dependent $a_t$ is selectivity, with
  discretization as one derivation; (5) a matrix state is linear attention
  with decay, its capacity, and chunked training; (6) the delta rule as one
  gradient step on a recall loss, absorbing test-time regression; (7) why a
  fixed budget of bits forces a few attention layers.
- **Keep:** "A useful memory must make decisions" (`lstm.md:76`), "A
  composition of nonlinear maps has no compact closed form" (`ssm.md:71`),
  the comma ten steps ago (`mamba.md:66`), the interference proposition
  (`matrix-state.md:98–132`), the delta rule as prediction error
  (`deltanet.md:395`).

### Ch. 13 Computational Performance

- The method is sound and `performance-model.md` is a model opening. The
  problems are the NCCL bug told four times, draft history ("each corrupted
  an early draft of this section", `fast-transformer.md:150`), hardcoded
  section numbers ("13.3's whole-step lesson", `:394`), and broken
  sentences left by a mechanical rename ("a *negative* configuration for
  the fp32 program", `:744`).
- 17.6 em dashes per thousand words, the highest in scope.
- **Keep:** two matrix sizes on one GPU (`performance-model.md:4`), the
  shoreline argument (`hardware.md:137`), "sixteen bytes per parameter"
  (`memory-precision.md:68`).

### Ch. 14 Reinforcement Learning

- The index opens on an abstract contrast and then gives a 21-row map of
  algorithms, including methods the book never implements
  (`index.md:4`, `:23–45`).
- 64 of about 131 section references point forward. A hardcoded "introduced
  in Chapter 15" (`deep-rl.md:134`).
- `qlearning.md` carries TD learning, Robbins–Monro, bandits, regret, UCB,
  Thompson sampling, and maximization bias. It defines an objective and then
  disowns it (`:32–46`).
- LLM loss-normalization debates sit inside a FrozenLake section
  (`baselines.md:322`).
- **Keep:** `value-iter.md:363`, correlation 0.9 gives a fivefold variance
  reduction (`baselines.md:135`), the first deviation from the expert
  (`imitation.md:306`).

### Ch. 15 Deep RL

- Six symbols are overloaded ($\varepsilon$, $\alpha$, $\tau$, $\rho$,
  $\delta$, and visitation measures).
- Stagecraft ("First the casualties, counted rather than asserted",
  `ppo.md:408`) and JIT and padding prose inside algorithm explanations
  (`actor-critic.md:153`, `dqn.md:226`).
- About eight measurements in `actor-critic.md`; the text disowns one of
  them as "not a standard stability metric" (`:306`).
- SAC's motivating result (REINFORCE never solves Pendulum, SAC does in
  10,000 steps) appears at the end (`sac.md:551`) instead of the start.
- **Keep:** `ppo.md:39`, "So stop waiting for the sample and substitute the
  prediction" (`actor-critic.md:54`), `regularized.md:137`.

### Ch. 16 GANs

- Sentences are short and clean. The two teachable ideas (density ratio,
  rotation of the game) are buried under divergence algebra, a lifting
  lemma, paper errata, and benchmark tables.
- 31 appendix references; "Recall from :eqref:`eq_mdl-f-div-def`"
  (`objectives.md:83`).
- Experiments whose claims hold only under one initialization
  (`dcgan.md:731`) or do not separate the arms (`convergence.md:555`).
- About 18 passages narrate per-framework runs.
- **Keep:** "99 percent of the mixture's samples are noise" (`gan.md:42`),
  the Dirac-GAN passage (`convergence.md:37–85`), the blurry thumbnail
  (`adversarial-losses.md:39`).

### Ch. 17 Diffusion Models

- The backbone is right: each section opens with the problem the previous
  one left. But DDPM first appears in section 7, about 15,000 words in.
- 22% of sentences exceed 35 words; one runs to about 140
  (`flow-matching.md:227`).
- $\alpha$ has three meanings, $\sigma$ four, and $x_0$ flips from data to
  noise (`flow-matching.md:23`, `:163`).
- About 50 appendix references, and 17 citations of the lecture course on
  which the chapter is modeled ("The lecture notation writes ...",
  `limits.md:156`).
- **Keep:** "A learned score is a vector field that points toward the data.
  To turn it into samples, follow it." (`langevin.md:4`), the denoiser
  identity (`denoising.md:158`), "it only erases"
  (`discrete-diffusion.md:141`).

### Ch. 24 Linear Algebra

- The eigenvalue section states $A^n = W\Lambda^nW^{-1}$ (`eig:229`) with
  no starting vector, worked recurrence, or picture. Strang solves
  Fibonacci and a Markov chain with this formula. The clearest
  "each eigendirection evolves on its own" example in the appendix is in
  ch. 26 (`mdl-gradient-based-optimization.md:362`).
- About 3,500 words of pathologies (Jordan form, pseudospectra, Lyapunov
  exponents, circular law, Marchenko–Pastur) against 1,500 for the
  diagonalizable case.
- Singular vectors are used in the section that defines eigenvectors, one
  section before the SVD.
- **Keep:** the dimmed image (`mdl-geometry-linear-algebraic-ops.md:507`),
  "The rotation has *decoupled* the problem" (`mdl-svd-low-rank.md:661`),
  $J(0.99)$ climbing to 37 (`mdl-eigendecomposition.md:455`).

### Ch. 25 Calculus

- The integral section states its motivation 650 lines in
  (`mdl-integral-calculus.md:657`).
- Real-analysis hygiene throughout: measure zero, Dirichlet functions,
  Fubini counterexamples, conservative fields, Rademacher's theorem.
- The descent lemma, Newton's method, and the subgradient also appear in
  ch. 26; backprop is explained three times.
- **Keep:** forward and reverse mode as two parenthesizations
  (`mdl-matrix-calculus-autodiff.md:219`), the sum over paths
  (`mdl-multivariable-calculus.md:437`).

### Ch. 26 Optimization

- Every opening lists questions instead of posing a problem.
- $\kappa$ is defined as $\sigma_1/\sigma_r$, $\lambda_{\max}/\lambda_{\min}$,
  and $L/\mu$; the Lagrange multiplier has opposite signs in ch. 25 and
  ch. 26.
- The narrow-valley figure appears three times and stable log-sum-exp three
  times across ch. 25 and 26.
- **Cut:** BFGS and trust regions, edge of stability, SVRG and SAGA,
  K-FAC/Shampoo/Muon (pointers only), coordinate descent, most conjugates,
  the LP/QP/SOCP/SDP table, fp8.
- **Keep:** "a direction and a distance"
  (`mdl-gradient-based-optimization.md:4`), $n$ uncoupled recursions
  (`:373`), the momentum motivation (`:647`), water-filling
  (`mdl-constrained-optimization-duality.md:1069`).

### Ch. 27 Probability and Statistics

- The section order contradicts the dependencies. Maximum likelihood uses
  consistency, the law of large numbers, and Cramér–Rao two sections before
  the Statistics section defines them
  (`mdl-maximum-likelihood.md:511`). It also calls posterior averaging
  "beyond our scope here" (`:880`), although the next section covers it.
  Put Statistics before maximum likelihood.
- One 26-line paragraph carries regularity conditions, rates, Cramér–Rao,
  efficiency, finite-sample bias, and nonregular exceptions
  (`mdl-maximum-likelihood.md:528–553`).
- Laplace's rule of succession is derived three times, and
  near-orthogonality is taught three times across ch. 24 and ch. 27.
- The concentration section undercuts its own argument with about nine
  hedges (`mdl-concentration-generalization.md:921`), and has a broken
  sentence at `:884`.
- **Cut:** reduce the distribution gallery to a table and three derivations;
  drop the initialization and nearest-neighbor material, Bernstein, and the
  benign-overfitting survey from the concentration section.
- **Keep:** "We have only pushed the problem one digit down"
  (`mdl-random-variables.md:58`), the underflowing product of a billion
  probabilities (`mdl-maximum-likelihood.md:141`), the EM motivation
  (`:924–936`), "How do you average over a distribution you can only
  evaluate pointwise, up to a constant?" (`mdl-bayesian-computation.md:23`).

### Ch. 28 Information Theory

The chapter most directly comparable with Cover and Thomas ch. 2, which
covers eleven topics in about 8,000 words. Ch. 28 has 23,800 body words.

- **Order.** Cover and Thomas go from entropy to joint and conditional
  entropy, then relative entropy (motivated by its coding cost before the
  definition), then mutual information, then identities, Jensen, data
  processing, and Fano. Differential entropy waits until their ch. 8. Ch. 28
  goes from entropy straight to differential entropy and Jacobians, using a
  KL "defined below" (`mdl-information-theory.md:136`). It gives the coding
  interpretation 400 lines after KL (`:629`) and uses mutual information
  for rate–distortion and capacity before defining it (`:844`, `:925`).
  Then come about 8,000 words of divergences, and only then joint entropy
  and mutual information.
- **Catalogue.** `mdl-divergences-distances.md` announces three families and
  appends a fourth. α-divergences, Rényi differential privacy, kernel Stein
  discrepancy, and SVGD are never used again. Its summary table opens by
  disowning itself (`:1310`).
- **Language.** Punchlines ("What destroys information is not corruption
  but *unpredictability*", `mdl-information-theory.md:984`), and rebuttals
  to an imagined reviewer (`mdl-mutual-information.md:684`).
- Binary entropy has three notations ($H(p)$, $h_2$, $H_b$).
- **Restructure:** move joint and conditional entropy, mutual information,
  the chain rule, and data processing into the first section in Cover and
  Thomas order, with the ½, ¼, ⅛, ⅛ example in the text. Cut Rényi, Stein
  methods, quasimetrics, Brier scores, and MDL.
- **Keep:** the three of spades (`mdl-information-theory.md:57`), mutual
  information as the KL from a hypothetical independent world
  (`mdl-mutual-information.md:155`), point masses at distance 0.01 or 100
  motivating $W_1$ (`mdl-divergences-distances.md:840–850`).

### Ch. 29 Dynamics

- The index, ODE, SDE, and Fokker–Planck sections teach well after
  trimming. `mdl-score-matching-diffusion-flow.md` (9,149 body words, 99
  cross-references, an eight-label prerequisite list) is a survey with
  about 40 hedges and opens with a table of contents; its problem appears at
  `:78`.
- Notation collides throughout the score-matching section: $\alpha_t$ twice,
  $\gamma$ twice, $\lambda$ twice, and $\mathbf x_0$ means data or noise
  depending on the convention. The score is defined three times, in ch. 28
  and twice in ch. 29.
- The matrix exponential is defined again in `mdl-odes-solvers.md` after
  saying that ch. 24 already defined it (`:24`, `:274`).
- **Cut:** matrix diffusion, boundary conditions, and JKO in the
  Fokker–Planck section; EDM, log-SNR, v-prediction, stochastic
  interpolants, and attention as interacting particles in the score-matching
  section. Consider splitting that section into diffusion and flow matching,
  and coordinate it with ch. 17, which covers the same ground.
- **Keep:** the three sources of error (`index.md:4–10`), forward Euler on
  $\dot x = -50x$ (`mdl-odes-solvers.md:9`), "a diverging learning rate is a
  solver instability" (`:783`), the deterministic evolution of a random
  path's density (`mdl-fokker-planck.md:4`).

## Order of Work

1. **Ch. 12 State Space Models.** Rewrite to the seven-step sequence above,
   at about half the current length.
2. **Chapter introductions, all 23.** Apply the template of guide §12.1.
   This is a small change that improves every chapter at once.
3. **Ch. 28 and 24–26.** Restructure ch. 28 in Cover and Thomas order.
   Rebuild each opening in ch. 24–26 around a problem, teach the
   diagonalizable case through Strang-style worked recurrences, and remove
   duplication between these chapters and with ch. 9.
4. **Ch. 15, 16, 7, 3.** Remove surveys, audits, and per-framework
   narration; fix notation.
5. **Ch. 17 and ch. 29 together.** Both cover score matching, diffusion, and
   flow matching. Decide which owns each derivation, so that ch. 17 states
   results with intuition and ch. 29 proves them once.
6. **Headings, notation, and banned patterns across the scope.** These can
   be done mechanically once the guide's checks are in `lint_source.py`.
7. **Remaining chapters,** in order of severity.

## Decisions

Decided by Alex on 2026-10-23:

- **Old guides.** `style-guide.md` and `language-guide.md` are moved to
  `docs/attic/` and marked as abandoned.
- **Further reading.** A chapter may end with one optional Further Reading
  notebook, used sparingly and never heavier than one ordinary section
  (guide §12.3). Material marked "Cut" above may go there if a reader would
  still want it; everything else is deleted.
- **Shared test models.** Reuse them. Each is defined once in a place of its
  own and restated in a sentence or two where it is reused (guide §7.5).
- **Exercise volume.** Exercises at 25–90% of the body length are excessive.
  Keep problems that are good and add value; drop filler. About 30% of the
  section's body prose is a guide for balance, not a cap (guide §12.2;
  measured by `tools/exercise_budget.py`). Depth and multi-part problems are
  welcome when the problems are good. Before trimming, the scope had 273,900
  exercise words against 548,800 body words (50%), and 126 of 154 sections
  exceeded 30%.
- **Section length.** Longer sections are acceptable when the length is
  content. Balance is judged per chapter too: Information Theory has three
  sections where other chapters have eight (guide §4.4).
- **Plain NumPy.** Sections whose point needs no framework may be plain
  NumPy. This is intended.

Open:

- **Role of the appendix.** The guide makes it optional reading for the
  main text (§7.6). This moves some statements, with their intuition, back
  into Basics and Advanced; proofs can stay in the appendix.

### Exercise Trimming, 2026-10-23

Every section's problems were reviewed against guide §12.2. Filler was
removed (drills, restatements, sweeps without a question, API trivia,
near-duplicates); good problems were kept even where a section stays above
30%. Hard-coded exercise numbers in the text and slides became `:exref:`
references, and `tools/gen_slides.py` now renders them. Overall, exercise
text fell from 273,900 to 250,600 words (from 50% to 46% of the body).

| Chapter | Before | After | Chapter | Before | After |
|---|--:|--:|---|--:|--:|
| 1 Preliminaries | 35% | 29% | 13 Comp. Performance | 76% | 72% |
| 2 Linear Regression | 34% | 28% | 14 Reinforcement Learning | 82% | 75% |
| 3 Linear Classification | 37% | 29% | 15 Deep RL | 82% | 73% |
| 4 Multilayer Perceptron | 47% | 37% | 16 GANs | 59% | 57% |
| 5 Computation | 27% | 25% | 17 Diffusion Models | 81% | 77% |
| 6 CNNs | 48% | 38% | 24 Linear Algebra | 35% | 34% |
| 7 Modern ConvNets | 35% | 32% | 25 Calculus | 42% | 36% |
| 8 Sequence Models | 27% | 27% | 26 Optimization | 46% | 43% |
| 9 Optimization | 32% | 31% | 27 Probability and Statistics | 49% | 48% |
| 10 Attention | 95% | 67% | 28 Information Theory | 42% | 38% |
| 11 Transformers | 80% | 74% | 29 Dynamics | 41% | 40% |
| 12 State Space Models | 44% | 43% | | | |

The Advanced chapters remain heavy because most of their problems are
substantial rather than filler. A further cut there means choosing among good
problems, which is a judgment for the author.

### Exercise Volume: Example

`chapter_attention/queries-keys-values.md`, the first section of ch. 10,
has 1,215 words of body prose and 1,599 words of exercises (132%). Its ten
exercises have 34 parts, average 160 words each, and seven of them require
code. Two are small projects: a retrieval experiment over random databases
with a noise sweep, and learning a diagonal metric by leave-one-out on
five-dimensional data. Exercise 7, learning the Nadaraya–Watson bandwidth,
repeats what `test-time-regression.md` does in ch. 12. Ch. 10 as a whole has
13,974 exercise words against 15,364 body words (91%).

Volume alone does not separate D2L from the model books. Strang's §5.1 has
40 exercises of about 50 words each, roughly 55% of its text. The
difference is weight: each of Strang's exercises asks for one computation,
whereas each of D2L's is a multi-part assignment, most of them with code.
