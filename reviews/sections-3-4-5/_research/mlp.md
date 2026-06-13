# MLP Chapter Research Digest

*Prepared 2026-06-12. Covers Chapter 5: Multilayer Perceptrons and its seven sections.*

---

## A. How Top Courses and Texts Structure and Teach This Material

### The jump from linear models to hidden layers

Every strong treatment motivates hidden layers the same way: linear models collapse under composition. **d2l.ai** shows this algebraically — composing affine maps produces only affine maps — then pivots to practical failures: loan repayment and image classification where linear separability does not hold. **CS231n** (cs231n.github.io/neural-networks-1/) takes the same algebra but frames it as extending `s = Wx` to `s = W₂ max(0, W₁x)`, making the non-linearity insertion viscerally concrete.

**Nielsen** (*Neural Networks and Deep Learning*, free at neuralnetworksanddeeplearning.com) opens with the perceptron, then steps through the sigmoid neuron, making the biological analogy and then immediately de-emphasizing it. His approach builds reader intuition before abstraction.

**The XOR example** is the canonical motivation in many treatments but is conspicuously *absent* from d2l.ai's current MLP section, which substitutes practical examples. Both **CMU 11-785** (Lecture 2: "Neural Nets As Universal Approximators", deeplearning.cs.cmu.edu/F24) and **Goodfellow et al.** (*Deep Learning*, Ch. 6, deeplearningbook.org) use XOR explicitly as the opening pedagogical wedge — it is simple, unambiguous, and shows a linearly inseparable function that a two-layer net solves cleanly.

**Simon Prince** (*Understanding Deep Learning*, 2023, udlbook.github.io/udlbook/) and **Bishop & Bishop** (*Deep Learning: Foundations and Concepts*, 2024, bishopbook.com — free online version available) both give compact, modern introductions that start from scalar functions and build to the general MLP.

### How activation functions are introduced and compared

**d2l.ai** (d2l.ai/chapter_multilayer-perceptrons/mlp.html) plots ReLU, sigmoid, and tanh with their derivatives side by side — the clearest single-page activation reference in any introductory text. Key annotations: sigmoid "saturates and kills gradients" at extremes; tanh is zero-centered and preferred to sigmoid for that reason; ReLU helps with the vanishing-gradient problem but can produce "dead neurons."

**CS231n** (cs231n.github.io/neural-networks-1/) echoes this hierarchy and makes a practical recommendation: "Use ReLU. Be careful with your learning rates. If this concerns you, use Leaky ReLU or Maxout. Never use sigmoid. Try tanh but expect it to work worse than ReLU/Maxout." This opinionated tier list is memorable and often reproduced.

**Goodfellow et al.** (Ch. 6.3) catalog activations in more depth and are one of the few introductory treatments to flag that the universal-approximation theorem holds for *any* bounded, non-constant squashing function — decoupling the theorem from the specific choice of activation.

Modern activations (GELU, SiLU/Swish, SwiGLU) are absent or marginal in all pre-2022 treatments; see Section E.

### Universal approximation: statement and honest caveats

**CMU 11-785** dedicates Lecture 2 ("Neural Nets As Universal Approximators") entirely to this. The course follows the classical *width* version (Cybenko 1989: single hidden layer, sigmoid, arbitrary width approximates any continuous function on a compact domain; Hornik 1991: generalized to any bounded non-constant activation).

**Goodfellow et al.** (Ch. 6.4.1) give the clearest written statement and add the critical caveat: the theorem guarantees *existence* of a solution, not *learnability* from data, and says nothing about generalization. They also note the depth extension (Leshno et al. 1993): "a single hidden layer is sufficient, but it may be exponentially wide."

**d2l.ai** (Ch. 5.1) handles this briefly — one paragraph citing Cybenko and noting "coming up with the program is the hard part" — which undersells both the result and its limitations.

**The depth-vs-width question** (why depth helps more than width in practice) is largely left implicit in introductory treatments. CS231n advises using deep thin networks over shallow fat ones, citing empirical results, but does not derive why.

### How backpropagation is taught

**Nielsen** (neuralnetworksanddeeplearning.com/chap2.html — note: the site's TLS cert is invalid for the www subdomain; the raw http:// URL resolves) provides the most pedagogically careful treatment via **four equations**: (BP1) error at the output layer, (BP2) error propagated backwards, (BP3) rate of change of cost with bias, (BP4) rate of change of cost with weight. He derives each equation carefully and emphasizes "backprop as a fast algorithm for computing all partial derivatives simultaneously." Free access, no TLS issues at the http version.

**CS231n** (cs231n.github.io/optimization-2/) teaches backprop as a **circuit of gates** with local gradients, chain rule, and forward/backward pass. The key insight stated explicitly: "Each gate independently computes its local gradient, then multiplies it into the gradient flowing backward." Gate semantics — add gates distribute equally, max gates route — give learners a mental model that generalizes to arbitrary graphs. Worked numerical examples with f(x,y,z) = (x+y)z make it concrete. A strong supplement to the equation-based approach.

**Goodfellow et al.** (Ch. 6.5) treats backprop as **reverse-mode automatic differentiation**, framing it as the most general setting and noting the forward/backward pass as a special case. This is the most mathematically complete introductory treatment.

**d2l.ai** (Ch. 5.3) uses a computational graph with squares for variables and circles for operators, a one-hidden-layer example with L₂ regularization, and discusses memory cost ("intermediate values must persist until backpropagation completes"). Serviceable but lacks a fully worked numerical example.

**CMU 11-785** spreads backprop across three lectures (3, 4, 5) — Training Parts I, II, III — with "Calculus of Backpropagation" in Lec 5. The course is known for detailed derivations. Slides are free on deeplearning.cs.cmu.edu/F24/.

**CS230 Stanford** (Lecture 3, cs230.stanford.edu/files/lecture-notes-3.pdf — binary PDF, not easily parseable) covers backprop, initialization, and regularization together. The **shervine cheatsheets** (stanford.edu/~shervine/teaching/cs-230/) provide a compact reference: Xavier init definition, dropout as "dropping neurons with probability p > 0."

### Initialization and vanishing/exploding gradients

**d2l.ai** (Ch. 5.4) gives a clear treatment: demonstrates exploding gradients empirically (100 random matrices → magnitude ~10²⁴), shows sigmoid saturation causing vanishing gradients, derives the Xavier/Glorot formula from variance preservation:
- Forward pass requires `n_in · σ² = 1`
- Backward pass requires `n_out · σ² = 1`
- Compromise: `σ = √(2 / (n_in + n_out))`

For uniform distributions this gives `U(−√(6/(n_in + n_out)), √(6/(n_in + n_out)))`.

**CS231n** (cs231n.github.io/neural-networks-2/) presents the same variance-preservation argument and explicitly recommends **He initialization** for ReLU networks: variance `2.0/n` (i.e., `w ~ randn * sqrt(2/n_in)`), citing "He et al. 2015" as "the current recommendation for use in practice." This is absent from d2l.ai's current chapter.

**Goodfellow et al.** (Ch. 8.4) treat initialization in the context of optimization and emphasize that "we almost always initialize all the weights in the model to values drawn randomly from a Gaussian or uniform distribution."

### Modern generalization: double descent, implicit bias, failure of classical bounds

**d2l.ai** (Ch. 5.5) is among the more current treatments on this topic: mentions double descent explicitly (citing Nakkiran et al. 2021), discusses that traditional complexity-based bounds "cannot explain why neural networks generalize," and frames the section as active research rather than settled science.

**Belkin et al.** (arXiv:1812.11118, 2019) established the "double descent" curve as a formal reconciliation of the interpolation threshold with classical bias-variance theory. **Nakkiran et al.** (ICLR 2020/2021) extended this to epoch-wise and sample-wise double descent.

**MLU-Explain** (mlu-explain.github.io/double-descent/) provides an outstanding interactive visual essay on double descent — free, browser-based, shows the phenomena with an interactive slider. One of the best pedagogical resources on the topic.

No introductory course treatment as of 2024 gives a fully satisfying account of **why** deep nets generalize: the empirical picture is clearer than the theory. **Grokking** (Power et al., arXiv:2201.02177, 2022) added a further wrinkle: networks can generalize long after overfitting when trained long enough, suggesting optimization dynamics matter as much as architecture.

### Dropout: ensemble and co-adaptation views

**d2l.ai** (Ch. 5.6) presents two views:
1. **Smoothness (Bishop's view):** training with noise is equivalent to Tikhonov regularization
2. **Co-adaptation view:** dropout prevents neurons from "co-adapting" to the specific activation pattern of their neighbors, analogous to how sexual reproduction disrupts co-adapted genes

**Srivastava et al.** (JMLR 2014, jmlr.org/papers/v15/srivastava14a.html, open access) present a third view in the original paper: **ensemble interpretation** — dropout trains an exponential number of thinned networks sharing weights, and test-time evaluation with the full network is an approximate geometric mean of those ensembles.

**d2l.ai's inverted dropout** treatment is correct and clear: scaling by `1/(1-p)` during training so test-time code needs no changes. CS231n (neural-networks-2/) gives the same inverted-dropout formulation and recommends `p=0.5` as default.

### Use of a capstone / practical project

**d2l.ai** uses the Kaggle House Prices competition (Ch. 5.7) as the chapter capstone — an end-to-end pipeline from raw data through feature engineering to submission. This is a strong structural choice: it integrates all chapter concepts and connects to a real leaderboard learners can submit to.

**CMU 11-785** uses homework assignments (HW1: backprop from scratch) as the capstone equivalent. CS231n uses assignment 1 (a two-layer net on CIFAR-10). Neither substitutes for a full data-to-leaderboard pipeline.

---

## B. Coverage Gaps and Trims to Consider

### Likely gaps in the current chapter

**He/Kaiming initialization for ReLU nets.** Xavier was derived for tanh/sigmoid activations (symmetric around zero); for ReLU, which kills half the neurons, the variance preservation argument gives `σ = √(2/n_in)`. Every modern introductory treatment (CS231n, Goodfellow, CS182) recommends He init as the default for ReLU networks. The current chapter covers Xavier but does not appear to address this. Since the chapter's own implementation uses ReLU as the primary activation, this is a direct omission.

**GELU, SiLU/Swish, SwiGLU.** These are the activations in virtually every production LLM and vision transformer since ~2019 (BERT uses GELU; LLaMA/PaLM/GPT-4 use SwiGLU). The chapter covers only ReLU, sigmoid, tanh. At minimum, a paragraph noting GELU (Hendrycks & Gimpel, arXiv:1606.08415) and SwiGLU (Shazeer, arXiv:2002.05202) as successors for modern architectures would bring the activation survey current.

**A clean worked backprop example.** d2l.ai's backprop section (5.3) describes the algorithm with a computational graph diagram but lacks a fully worked numerical example showing specific numbers flowing forward and backward. Nielsen's Chapter 2 four-equations derivation and CS231n's f(x,y,z) = (x+y)z circuit example are the gold standard here. A simple 2-input, 1-hidden-neuron worked example would strengthen the pedagogical value.

**Batch normalization pointer.** BatchNorm is not part of the MLP chapter, but given that CS231n notes it "alleviates headaches with initialization" and is now ubiquitous, a brief forward pointer (possibly in the initialization section) would orient learners.

**XOR motivating example.** Both Goodfellow et al. and CMU 11-785 use XOR as the opening wedge for why hidden layers are needed. The current d2l.ai treatment relies on more practical but less crisp examples. The XOR example has pedagogical advantages: it is mathematically minimal, provably not linearly separable, and solved by a specific constructible two-layer network. Worth adding (or referencing) in the motivation.

**Depth vs. width: why depth helps.** The chapter discusses universal approximation (any continuous function, single layer, sufficient width) but does not address why depth is more efficient than width in practice. Goodfellow et al. cite exponential separation results; CS231n recommends "as deep as budget allows." A brief treatment here would strengthen the section.

**Double descent: the visual.** The double-descent curve (MLU-Explain or a custom figure) is the single most clarifying figure for modern generalization. The chapter should include it. The interactive MLU-Explain essay (mlu-explain.github.io/double-descent/) is a citable resource for students.

**Implicit regularization of SGD.** The generalization section could note that SGD finds solutions with smaller norm than GD (flat minima hypothesis), which partially explains the implicit regularization that makes overparameterized nets generalize. This is well-documented but not settled theory.

**Dropout at test time / uncertainty estimation.** MC Dropout (Gal & Ghahramani 2016) as a Bayesian approximation is a useful extension note — not core material but worth a brief mention as a pointer.

**Label smoothing.** A regularization technique commonly paired with dropout in modern training pipelines; one sentence suffices.

**Kaggle capstone context.** The Kaggle House Prices capstone (5.7) does not include a note that gradient-boosted trees (XGBoost, LightGBM) routinely outperform deep nets on structured/tabular data. Grinsztajn et al. 2022 (arXiv:2207.08815) and Shwartz-Ziv & Armon (Information Fusion, 2022) both document this clearly. The capstone is excellent; it should acknowledge the practical baseline honestly so students know what they are competing against.

### Trims to consider

- The **MXNet framework tabs** should be removed or tombstoned. Apache MXNet was retired by the ASF on September 20, 2023, repository archived November 2023, transferred to Apache Attic February 2024. Showing MXNet code as one of four equal framework choices is misleading to students in 2026.
- Any **dated Flax Linen API** (pre-Keras-3 TF, old-style JAX stax/random-key passing) should be updated to current idiomatic usage.
- If torch is pinned to 2.11.0, code examples should at minimum note this and flag any idioms that changed in the 2.x series (e.g., `torch.optim` constructors, `nn.Module` subclassing style).

---

## C. Pedagogical Gems Worth Emulating

**CS231n computational-graph backprop** (cs231n.github.io/optimization-2/). The circuit metaphor — add gates distribute, max gates route, multiply gates swap gradients — gives every gate a memorable interpretation. The worked numerical example `f(x,y,z) = (x+y)z` with green forward values and red backward gradients is the most reusable single-figure teaching device for backprop. Emulate by adding a similar figure to section 5.3.

**Nielsen's four-equation derivation** (neuralnetworksanddeeplearning.com/chap2.html). Deriving BP1–BP4 from first principles, naming each equation, and then showing they compose into an algorithm gives students equations they can verify rather than code they must trust. The labeling convention is worth adopting.

**CS231n's opinionated activation function hierarchy** (cs231n.github.io/neural-networks-1/). The direct recommendation — "Use ReLU, try Leaky ReLU, never use sigmoid" — is memorable and accurate. The explanation of *why* (sigmoid squashes gradients, is not zero-centered) alongside the recommendation is stronger than a neutral survey.

**d2l.ai's activation function + derivative plots** (d2l.ai/chapter_multilayer-perceptrons/mlp.html). Side-by-side ReLU/sigmoid/tanh with derivatives is the clearest single-page activation reference. The derivative plots make the saturation problem immediate and visual.

**CMU 11-785 Lecture 2 XOR opening** (deeplearning.cs.cmu.edu/F24/document/slides/Lec02_f24.pdf). Constructing the solution to XOR by hand — as a two-neuron hidden layer with specific weight values — is the strongest single demonstration of why hidden layers are necessary. Students see that linear models are provably insufficient and that a small non-linear network solves the problem exactly.

**TensorFlow Playground** (playground.tensorflow.org). Created by Daniel Smilkov and Shan Carter (building on Karpathy / Olah). A zero-install browser demo where students tune activations, layer depth, regularization, and watch the decision boundary train in real time. Ideal for the chapter cover page and in-class use. Free.

**MLU-Explain double descent visualization** (mlu-explain.github.io/double-descent/). An interactive scrollytelling essay that walks the reader from the classical U-curve to the interpolation threshold and second descent. Mechanistic animations show why smooth interpolation in the over-parameterized regime generalizes. Free, browser-based.

**Goodfellow et al.'s universal approximation caveat** (deeplearningbook.org, Ch. 6.4). The explicit separation of *existence* (UAT guarantees a solution exists) from *learnability* (gradient descent may not find it) and *generalization* (it may not work on new data) is a three-sentence clarification that most introductory treatments omit.

**He init derivation in CS231n** (cs231n.github.io/neural-networks-2/). The one-line derivation `w = np.random.randn(n) * sqrt(2.0/n)` with the note "this is the current recommendation for use in practice" pairs a formula with a verdict — exactly the format that sticks.

---

## D. Curated Resources for the Chapter Cover Page

*Following the house style from d2l.smola.org: entries as `[Title — Author/Org](url)`, grouped under Books / Courses and video lectures / Foundational papers / Tutorials, notes, and documentation/interactive. Each entry has an em-dash annotation noting relevance and access.*

---

### Books

- [**Deep Learning** — Ian Goodfellow, Yoshua Bengio & Aaron Courville (MIT Press, 2016)](https://www.deeplearningbook.org/) — Ch. 6 ("Deep Feedforward Networks") covers hidden layers, activation functions, universal approximation, and backprop as reverse-mode autodiff; Ch. 7 covers regularization including dropout; free HTML version at deeplearningbook.org.

- [**Neural Networks and Deep Learning** — Michael Nielsen (2015)](http://neuralnetworksanddeeplearning.com/) — Chapter 2 derives the four fundamental equations of backpropagation from first principles; Chapter 3 covers dropout and regularization; free online.

- [**Understanding Deep Learning** — Simon J. D. Prince (MIT Press, 2023)](https://udlbook.github.io/udlbook/) — Compact modern treatment; Chapters 3–7 cover shallow/deep networks, training, initialization, and regularization; full PDF free at udlbook.github.io.

- [**Deep Learning: Foundations and Concepts** — Christopher M. Bishop & Hugh Bishop (Springer, 2024)](https://www.bishopbook.com/) — Comprehensive, Bayesian-flavored treatment of MLPs, backprop, dropout, and initialization; free online version available at bishopbook.com.

---

### Courses and Video Lectures

- [**11-785 Introduction to Deep Learning** — Philipose Bhiksha Raj et al. (CMU, 2024)](https://deeplearning.cs.cmu.edu/F24/index.html) — Lectures 2–8 cover universal approximation, backpropagation (three dedicated lectures), optimization, regularization, and dropout; slides and YouTube recordings freely available.

- [**CS231n: Deep Learning for Computer Vision** — (Stanford, 2024)](https://cs231n.github.io/) — Course notes on neural networks (Modules 1–2) are the most-cited online reference for the MLP computational-graph view of backprop and the activation function comparison; notes free at cs231n.github.io.

- [**6.S191: Introduction to Deep Learning** — Alexander Amini & Ava Amini (MIT, 2026)](https://introtodeeplearning.com/) — Lecture 1 covers deep learning foundations including MLPs; all slides and videos freely available; updated annually.

- [**NYU Deep Learning** — Yann LeCun & Alfredo Canziani (NYU, 2021)](https://atcold.github.io/NYU-DLSP21/) — Lectures on gradient descent, backpropagation, and neural network training by two leading researchers; notebooks and videos freely available.

- [**CS 182: Deep Neural Networks** — (UC Berkeley, 2021)](https://cs182sp21.github.io/) — Lectures 4–7 cover optimization, backpropagation, and training; slides and YouTube recordings freely available.

---

### Foundational Papers

- [**Learning Representations by Back-propagating Errors** — Rumelhart, Hinton & Williams (*Nature*, 1986)](https://www.nature.com/articles/323533a0) — The paper that introduced backpropagation to a broad audience and established the hidden-layer representation-learning framework (paywalled; widely reproduced).

- [**Approximation by Superpositions of a Sigmoidal Function** — George Cybenko (*Math. Control Signals Systems*, 1989)](https://doi.org/10.1007/BF02551274) — First formal proof of the universal approximation theorem for sigmoid networks; single-hidden-layer result (paywalled, noted).

- [**Approximation Capabilities of Multilayer Feedforward Networks** — Kurt Hornik (*Neural Networks*, 1991)](https://doi.org/10.1016/0893-6080(91)90009-T) — Extends Cybenko's UAT to arbitrary bounded non-constant activations and Lp(μ) criteria (paywalled, noted).

- [**Understanding the Difficulty of Training Deep Feedforward Neural Networks** — Glorot & Bengio (AISTATS 2010)](https://proceedings.mlr.press/v9/glorot10a.html) — Derives Xavier/Glorot initialization from the variance-preservation argument; also shows sigmoid saturation causes vanishing gradients; free PDF.

- [**Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification** — He, Zhang, Ren & Sun (arXiv, 2015)](https://arxiv.org/abs/1502.01852) — Derives He/Kaiming initialization for ReLU networks; the current recommended default for ReLU architectures; free PDF.

- [**Rectified Linear Units Improve Restricted Boltzmann Machines** — Nair & Hinton (ICML 2010)](https://www.semanticscholar.org/paper/Rectified-Linear-Units-Improve-Restricted-Boltzmann-Nair-Hinton/a538b05ebb01a40323997629e171c91aa28b8e2f) — First systematic study of ReLU as a neural network activation function; free via Semantic Scholar.

- [**Dropout: A Simple Way to Prevent Neural Networks from Overfitting** — Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (*JMLR*, 2014)](https://jmlr.org/papers/v15/srivastava14a.html) — The dropout paper; introduces the ensemble interpretation and co-adaptation view, inverted dropout, and training/inference protocol; free open access.

- [**Reconciling Modern Machine Learning Practice and the Bias–Variance Trade-off** — Belkin, Hsu, Ma & Mandal (arXiv, 2018/PNAS 2019)](https://arxiv.org/abs/1812.11118) — Introduces and formally characterizes the double-descent phenomenon; explains why overparameterized models can generalize; free PDF.

---

### Tutorials, Notes, and Documentation / Interactive

- [**TensorFlow Playground** — Daniel Smilkov & Shan Carter (Google Brain)](https://playground.tensorflow.org/) — Interactive browser demo: tune network depth, width, activation functions, and regularization while watching the decision boundary train in real time; zero-install, free; ideal for exploring hidden-layer intuition.

- [**Backpropagation, Intuitions** — CS231n course notes (Stanford)](https://cs231n.github.io/optimization-2/) — The computational-graph / gate-semantics treatment of backprop; worked numerical examples with visual circuit diagrams; free online.

- [**Neural Networks, Part 1: Setting Up the Architecture** — CS231n course notes (Stanford)](https://cs231n.github.io/neural-networks-1/) — Comprehensive activation function comparison with practical recommendations; free online.

- [**Double Descent: Visual Essay** — MLU-Explain (Amazon)](https://mlu-explain.github.io/double-descent/) — Interactive scrollytelling that animates the interpolation threshold and the second descent; the best free visual resource on modern generalization; browser-based.

- [**KAN: Kolmogorov-Arnold Networks** — Liu et al. (arXiv:2404.19756, 2024)](https://arxiv.org/abs/2404.19756) — Proposes learnable activation functions on edges (splines) as an alternative to fixed activations on nodes in MLPs; competitive accuracy with better interpretability claims; free PDF; context for the chapter's placement of MLPs relative to alternatives.

---

## E. Developments Since ~2021 — Currency Check

### (i) Modern activations: GELU, SiLU/Swish, SwiGLU

**What to add:** A subsection (or at minimum two paragraphs) in the activation functions section covering GELU and SwiGLU.

**GELU** (Gaussian Error Linear Units; Hendrycks & Gimpel, arXiv:1606.08415, 2016; updated 2023). GELU weights inputs by their magnitude (`x Φ(x)`) rather than gating by sign (ReLU) or squashing (sigmoid). Adopted in BERT, GPT-2/3/4, ViT, and most large transformers. Smooth everywhere, outperforms ReLU on NLP tasks.

**SiLU/Swish** (Ramachandran et al. 2017; arXiv:1710.05941). `x · sigmoid(βx)`. A smooth, non-monotone activation that consistently outperforms ReLU on deep networks. Now the default in EfficientNet, MobileNetV3.

**SwiGLU** (Shazeer, arXiv:2002.05202, 2020). `SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)`. Adopted in PaLM, LLaMA 1/2/3, Mistral, DeepSeek. Standard in the feedforward sublayer of modern transformers. The chapter's activation survey is stale by 2026 without at least acknowledging GELU and SwiGLU.

**Why it matters in 2026:** Any student reading this chapter and then opening a LLaMA or GPT codebase will encounter GELU/SwiGLU immediately. Not mentioning them creates a jarring gap.

**Citable sources:** arXiv:1606.08415 (GELU), arXiv:2002.05202 (SwiGLU).

---

### (ii) He/Kaiming init as default for ReLU nets; LayerNorm/RMSNorm

**What to add/fix:** The initialization section covers Xavier but not He init. This is an error of omission given that the chapter's own code uses ReLU.

**He initialization** (He et al., arXiv:1502.01852, 2015): `σ = √(2/n_in)` for weight initialization in ReLU networks. The factor of 2 accounts for the fact that ReLU zeroes half of inputs, halving the effective variance. This is the default in PyTorch (`nn.init.kaiming_normal_`) and is "the current recommendation" per CS231n.

**LayerNorm / RMSNorm displacing BatchNorm.** BatchNorm (Ioffe & Szegedy, 2015) is the traditional normalization reference but is now rarely used in transformer architectures. LayerNorm (Ba et al. 2016) and RMSNorm (Zhang & Sennrich 2019) are the current standard. At minimum a pointer is needed; optionally a forward reference to batch norm in a later chapter.

**Why it matters in 2026:** Students initializing PyTorch models with `nn.Linear` and ReLU who follow the chapter's Xavier advice are using a sub-optimal initializer. He init is one-line fix with material practical impact.

---

### (iii) Grokking (Power et al. 2022) and double descent

**What to add:** Expand the generalization section to include a brief discussion of grokking and connect to the double-descent curve.

**Grokking** (Power, Burda, Edwards, Babuschkin & Misra, arXiv:2201.02177, 2022): Networks can generalize *well after* overfitting training data, if trained long enough with weight decay. On algorithmic tasks (modular arithmetic, permutations), generalization accuracy jumps from near-chance to near-perfect suddenly, well past the point of training-set memorization. This challenges the simple narrative that "overfitting = bad generalization."

**Deep double descent** (Nakkiran et al., ICLR 2020/J. Stat. Mech. 2021): Extends Belkin et al.'s model-wise double descent to epoch-wise and sample-wise, showing the phenomenon is ubiquitous across architectures and training regimes.

**Why it matters in 2026:** These results are now cited in essentially every serious treatment of generalization. Students who learn only classical bias-variance theory leave with an incorrect mental model of how modern networks generalize.

**Citable sources:** arXiv:2201.02177 (grokking), arXiv:1812.11118 (Belkin double descent), Nakkiran et al. ICLR 2020 (deep double descent).

---

### (iv) MLP renaissance: MLP-Mixer, gMLP, KAN

**What to add:** A brief note (one paragraph) in a section on context or applications noting that MLPs have experienced a renaissance.

**MLP-Mixer** (Tolstikhin et al., arXiv:2105.01601, NeurIPS 2021): An all-MLP vision architecture that matches ViT-like performance on large datasets, demonstrating that neither convolutions nor attention are *necessary* for competitive image recognition — MLPs suffice with enough data and the right structure. Cited 1100+ times.

**KAN: Kolmogorov-Arnold Networks** (Liu et al., arXiv:2404.19756, 2024): Replaces fixed activation functions with learnable univariate splines on edges. Claims superior accuracy on small scientific datasets and interpretability advantages. Follow-up: KAN 2.0 (arXiv:2408.10205). Significant community interest in 2024.

**Why it matters in 2026:** Students should know MLPs are not a stepping stone to be abandoned — they remain active research subjects. The KAN paper specifically reframes the architectural choice the MLP chapter makes (fixed activations on nodes vs. learnable activations on edges), which is a natural pedagogical hook.

---

### (v) Kaggle capstone: trees outperform deep nets on tabular data

**What to add:** A frank note in the Kaggle House Prices capstone (5.7) acknowledging that for structured/tabular data, gradient-boosted tree ensembles are typically the strongest baseline.

**Grinsztajn et al.** ("Why do tree-based models still outperform deep learning on tabular data?", arXiv:2207.08815, NeurIPS 2022 Datasets & Benchmarks): Extensive benchmark across 45 datasets; XGBoost and Random Forests outperform deep nets on medium-sized tabular data (~10K samples). Three identified challenges for NNs: robustness to uninformative features, data orientation invariance, and irregular function fitting.

**Shwartz-Ziv & Armon** ("Tabular Data: Deep Learning is Not All You Need", *Information Fusion* 81, 2022): Similar conclusion; XGBoost outperforms deep models on tabular data; harder to optimize NNs vs. XGBoost.

**Why it matters in 2026:** The Kaggle House Prices leaderboard is dominated by XGBoost-based entries. Teaching students that deep nets are the go-to for all data types is actively misleading. Acknowledging the tabular-data exception and explaining why (invariance to feature permutation, irregular decision boundaries) is honest and useful.

**Citable sources:** arXiv:2207.08815 (Grinsztajn), DOI:10.1016/j.inffus.2021.11.011 (Shwartz-Ziv & Armon).

---

### (vi) Dated tooling

**MXNet archived.** Apache MXNet was retired by the ASF on September 20, 2023; repository archived November 17, 2023; transferred to Apache Attic February 2024. The MXNet framework tabs in all chapter sections should be removed or clearly tombstoned. Showing MXNet as a peer framework in 2026 is misleading.

**Old Flax Linen API.** The Flax Linen API (`flax.linen`) has been superseded by Flax NNX (stable since Flax 0.8, 2024). JAX code examples using `nn.Module` subclassing in the Linen style should be reviewed and updated if they use Linen-specific patterns that differ from NNX.

**Pre-Keras-3 TensorFlow.** TensorFlow's Keras was decoupled in Keras 3 (2023), which supports multiple backends including JAX and PyTorch. Any TF code examples using `tf.keras` should note the Keras 3 context.

**PyTorch idioms.** `torch.optim` optimizer API, `DataLoader` usage, and module subclassing style have been stable, but the `torch.compile()` path introduced in PyTorch 2.0 (2023) is now widely used. Not a blocking issue, but worth noting in build instructions.

---

*End of research digest.*
