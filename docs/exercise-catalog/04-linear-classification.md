# Chapter overview — chapter_linear-classification (7 sections)

This chapter's existing exercises are unusually strong for a bare-numbered,
untagged style: `softmax-regression.md` (9 ex.) is the densest, most
mathematically complete set found in the whole book so far, and
`generalization-classification.md` / `environment-and-distribution-shift.md`
(6 ex. each) were already flagged defect-free by the prior style review. The
best external matches came from **Stanford CS229 PS1** (logistic
regression/GDA Hessian and MLE derivations), **Kevin Murphy's *Probabilistic
ML*** solution manual (multiclass-softmax gradient/Hessian, and a decision-
theory "reject option" exercise that fills a real gap in `classification.md`),
**ISL 2nd ed. Ch. 4** (the binary-logit vs. two-class-softmax coefficient
translation, Ex. 12, maps almost exactly onto our redundant-DOF exercise), and
**Shalev-Shwartz & Ben-David's *Understanding ML*** Ch. 6 (VC-dimension
exercises that generalize our rectangle/shattering proofs to R^d). Google's
**Machine Learning Crash Course** supplied two concrete, verified
check-your-understanding items (ROC/AUC comparison; training-serving skew)
that fit `classification.md` and `environment-and-distribution-shift.md`
respectively. Two sections have **no good external exercise tradition**:
`image-classification-dataset.md` (pure dataloader/tensor-layout engineering —
no course treats this as gradable homework) and
`softmax-regression-concise.md`'s numeric-format question (a systems/numerics
topic absent from ML-course problem sets). `sec_softmax`'s Bradley–Terry,
RealSoftMax, and Gumbel-max exercises likewise have no external homework
analog — they read as this book's own original contributions. Net stance:
keep-heavy across the chapter (41 current exercises total: 32 kept as-is, 9
lightly rewritten for tone/formatting/success-criteria, 0 dropped), with
external material used only as additions/upgrades, per the strong-chapter
guidance.

---

## chapter_linear-classification/softmax-regression.md — Softmax Regression

**Topic:** Deriving the softmax/cross-entropy pair from a noisy-argmax model, its exponential-family structure and gradient, information-theoretic and calibration interpretations.
**Current exercises:** 9; disposition: keep 7, rewrite 2 (drop 0) — this is the chapter's most rigorous, best-scaffolded set (exemplary nested-list formatting; the only defect is a "Can you design a better code?" tone violation in ex. 3(b)). External material adds one concrete numeric variant; everything else is original and worth keeping as-is.

**External sources found:**
- Stanford CS229 (Ng/Ma), Problem Set #1 Q1, Summer 2019 — derives that the logistic-regression cross-entropy Hessian is PSD (hence convex) and that the GDA posterior is a sigmoid of an affine function; close in spirit to this section's log-partition-convexity exercise. — https://cs229.stanford.edu/summer2019/ps1.pdf
- Kevin Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022), Exercise 10.1 "Gradient and Hessian of log-likelihood for multinomial logistic regression" — derives the softmax Jacobian $\partial\mu_k/\partial\eta_j=\mu_k(\delta_{kj}-\mu_j)$ and the multiclass score-function gradient, i.e., exactly the general-$q$ version of this section's exercise 1. — https://probml.github.io/pml-book/solns-public.pdf
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (2nd ed.), Ch. 4 Exercise 12 — given a fitted single-logit binary model, derive the equivalent two-class softmax coefficients (and vice versa), then predict how often the two parametrizations' hard decisions would disagree on new data. — https://www.statlearning.com/
- No good external exercise tradition found for the Bradley–Terry proof, the RealSoftMax/log-sum-exp identity, or the Gumbel-max trick: none of CS229, ISL, CMU 10-601, Berkeley CS189, or Murphy's exercise set pose these as graded homework — they appear only as worked textbook examples or reference material, making this section's treatment comparatively original.

**Proposed problem set** (our reference format):
1. [conceptual] **Exponential-family variance and the softmax Hessian.** Compute the second derivative of the cross-entropy loss for softmax and show it equals the variance of $\mathrm{softmax}(\mathbf{o})$, i.e., derive the general-$q$ Jacobian $\partial_j\mathrm{softmax}(\mathbf{o})_k=\mathrm{softmax}(\mathbf{o})_k(\delta_{jk}-\mathrm{softmax}(\mathbf{o})_j)$.
   *Provenance:* original (parallels Murphy PML Ex. 10.1, overlap med — cite on adoption).
2. [conceptual] **Binary reduction and coefficient translation.** Verify $\hat y_1=\sigma(o_1-o_2)$ for $q=2$ and show adding a constant to every logit leaves $\hat{\mathbf y}$ unchanged (one redundant degree of freedom). Then, given a fitted binary-logit model with coefficients $\hat\beta_0=2,\hat\beta_1=-1$, write down the corresponding two-class softmax coefficients explicitly, and vice versa for a second numeric example.
   *Provenance:* adapted from ISL Ch. 4 Ex. 12 (overlap med — cite on adoption).
3. [conceptual] **Optimal coding under equiprobable classes.** For three equiprobable classes, show why no binary code below $\log_2 3$ bits/symbol is possible for a single observation, then construct a code (e.g., over blocks of $n$ observations) whose per-symbol rate approaches the entropy bound, stating the achieved rate for $n=2$ explicitly.
   *Provenance:* original (rewritten from ex. 3 to remove "Can you design a better code?" and add an explicit success criterion).
4. [conceptual] **PAM-3 ternary signal budget.** Determine the minimum number of ternary (PAM-3) symbols needed to transmit an integer in $\{0,\ldots,7\}$, and argue why ternary signaling can be preferable to binary in the underlying electronics.
   *Provenance:* original.
5. [conceptual] **Bradley–Terry choice model.** Prove that softmax over two item scores satisfies the Bradley–Terry monotonicity requirement, then extend the model to allow a "choose neither" option with a third score.
   *Provenance:* original.
6. [conceptual] **RealSoftMax bounds and limits.** Prove $\mathrm{RealSoftMax}(a,b) > \max(a,b)$, bound the gap, show the $\lambda\to\infty$ limit recovers $\max$, and construct the analogous softmin and its $n$-ary generalization.
   *Provenance:* original.
7. [conceptual] **Log-partition convexity and numerical stability.** Using exercise 1's Hessian, show $g(\mathbf{o})=\log\sum_k\exp(o_k)$ is convex and translation-equivariant, and show that subtracting $\max_k o_k$ gives a numerically stable evaluation.
   *Provenance:* original.
8. [short-code] **Temperature scaling and the Gumbel-max trick.** (i) For $Q(i)\propto P(i)^\alpha$, identify the $\alpha$ that doubles/halves the temperature and the $T\to 0,\infty$ limits; argue temperature scaling changes cross-entropy but not accuracy. (ii) Numerically simulate the noisy-argmax model with i.i.d. Gumbel noise at several temperatures $T$, and verify empirically that the resulting label frequencies match $\mathrm{softmax}(\mathbf{o}/T)$ to within Monte Carlo error for $10^5$ draws.
   *Provenance:* original (merges and reframes existing exercises 8–9 as a code exercise).

---

## chapter_linear-classification/image-classification-dataset.md — The Image Classification Dataset

**Topic:** Loading, batching, resizing, and benchmarking Fashion-MNIST across four framework backends; channel-first vs. channel-last tensor layout.
**Current exercises:** 3; disposition: keep 3, rewrite 0, drop 0 — no defects found, and the three existing items are the section's only concrete, checkable engineering tasks.

**External sources found:**
- **No good external exercise tradition found.** This is a pure data-engineering/framework topic (dataloader throughput vs. batch size, `num_workers` ablation, channel-axis layout), and none of the courses/texts checked — CS229, ISL, CMU 10-601, Berkeley CS189, MIT 6.390 (homework gated behind registration), Murphy's PML — pose it as a graded exercise; they treat data loading as invisible infrastructure. This is a genuine gap in the academic exercise tradition, not a search failure, and matches the section's own profile (shortest exercise list in the chapter, zero math/citations).

**Proposed problem set:**
1. [short-code] **Throughput vs. batch size.** Time one training epoch at `batch_size` 1, 16, 64, 256, 1024; plot throughput (images/sec) against batch size and explain the rise-then-plateau shape.
   *Provenance:* original.
2. [short-code] **Worker-count ablation.** Compare `num_workers=0` against the framework default; state the batch size and hardware at which added workers stop helping and why.
   *Provenance:* original.
3. [conceptual] **Where the channel axis is born.** Trace all four frameworks' `get_dataloader` implementations to identify the exact line that fixes channel-first vs. channel-last layout, and explain why PyTorch/MXNet differ from TensorFlow/JAX here.
   *Provenance:* original.
4. [short-code] **Cost of the resize.** Measure the wall-clock and per-image memory cost of the `resize=(32,32)` step versus using native $28\times28$ images; report whether a simple linear classifier's validation accuracy changes measurably between the two.
   *Provenance:* original.
5. [conceptual] **Why shuffle only the training loader.** `get_dataloader` sets `shuffle=train`. Argue what would go wrong — for reported validation metrics and for run-to-run comparability — if the validation loader were also shuffled every epoch.
   *Provenance:* original.
6. [extended] **Loader-vs-compute profiler.** Sweep a `num_workers` x `batch_size` grid, measure loader throughput, and compare it against the forward-pass throughput of the linear classifier from :numref:`sec_softmax_scratch`; produce a heatmap marking the region where loading rather than compute is the bottleneck, and state the crossover point on your hardware.
   *Provenance:* original.

---

## chapter_linear-classification/classification.md — The Base Classification Model

**Topic:** The shared `Classifier` base class (loss + accuracy reporting), why accuracy can't train a model, and precision/recall/F1/confusion-matrix diagnostics for imbalanced classes.
**Current exercises:** 7; disposition: keep 3, rewrite 4 (drop 0) — content is already strong (decision theory, weighted-ERM forward-reference, ROC curve); four items need only a formatting fix (inline "(i)/(ii)/(iii)" crammed into one paragraph → proper nested lists) plus small enrichments from external material.

**External sources found:**
- Kevin Murphy, *Probabilistic Machine Learning: An Introduction*, Exercise 5.1 "Reject option in classifiers" — derives the Bayes-risk threshold rule $p(\hat y=j_{\max}\mid\mathbf x)\geq 1-\lambda_r/\lambda_s$ for accepting the top class vs. abstaining under a reject cost $\lambda_r$. — https://probml.github.io/pml-book/solns-public.pdf
- Kevin Murphy, *Probabilistic Machine Learning: An Introduction*, Exercise 5.3 "Bayes factors and ROC curves" — the exercise's existence and topic are confirmed by the solution manual's table of contents (its solution text is not populated in the public partial manual, so treat only as a topic pointer, not a quotable prompt). — https://probml.github.io/pml-book/solns-public.pdf
- Google, Machine Learning Crash Course, "Classification: ROC Curve and AUC" — three verified check-your-understanding items: rank four labeled ROC/AUC curves, identify a worse-than-chance model, and choose the preferable threshold point on an ROC curve for a spam filter where false negatives are cheaper than false positives. — https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
- James, Witten, Hastie & Tibshirani, *ISL* (2nd ed.), Ch. 4 Exercise 13(c) — compute a confusion matrix on the `Weekly` stock-direction dataset and explain what it reveals about the classifier's specific error pattern (not just its overall accuracy). — https://www.statlearning.com/

**Proposed problem set:**
1. [conceptual] **Exact vs. batch-mean validation loss.** Express the exact validation loss $L_v$ in terms of the batch-mean estimate $L_v^q$, the last minibatch's loss $l_v^b$, and the sample/batch sizes.
   *Provenance:* original.
2. [conceptual] **Unbiasedness of the batch-mean estimator.** Prove $E[L_v]=E[L_v^q]$, then explain why the exact $L_v$ is still preferable in practice.
   *Provenance:* original.
3. [conceptual] **Bayes-optimal decision rule, with a reject option.**
   1. Given a general loss $l(y,y')$ and posterior $p(y\mid x)$, derive the decision rule minimizing expected loss.
   2. Now suppose the classifier may also abstain at a fixed cost $\lambda_r$ instead of guessing, versus a cost $\lambda_s$ per wrong guess. Derive the confidence threshold on $p(\hat y=j_{\max}\mid x)$ above which guessing beats abstaining.
   *Provenance:* part 1 original; part 2 adapted from Murphy PML Ex. 5.1 (overlap med — cite on adoption).
4. [short-code] **Two classifiers, one confidence.** Two 90%-accurate classifiers assign average correct-class probability $0.91$ (A) vs. $0.51$ (B).
   1. Compute each classifier's average cross-entropy on its correct predictions.
   2. Explain why this alone can't decide which is "safer" — state what you'd need to know about incorrect-prediction probabilities, subgroup calibration, and error costs.
   3. Construct a temperature that sharpens B's probabilities without changing any $\arg\max$ decision, and confirm accuracy is unchanged.
   *Provenance:* original (reformatted to nested list from the existing exercise).
5. [short-code] **Top-$k$ accuracy.** Extend `accuracy` to top-$k$ by replacing the single `argmax` with the indices of the $k$ largest scores; state what top-$q$ accuracy always equals on a $q$-class problem, and why top-5 is a standard companion metric to top-1 on many-class benchmarks.
   *Provenance:* original.
6. [short-code] **Class-weighted loss for imbalance.**
   1. Modify cross-entropy so each true class $j$ carries weight $w_j$.
   2. Show this is weighted ERM :eqref:`eq_weighted-empirical-risk-min` with $\beta_i=w_{y_i}$, and explain why upweighting the rare class raises recall.
   3. Give a data-side alternative with the same effect.
   *Provenance:* original (reformatted to nested list); framing of "explain the error pattern, not just the rate" echoes ISL Ch. 4 Ex. 13(c) (inspired, overlap low).
7. [short-code] **ROC curve by threshold sweep.**
   1. Sweep $\tau\in(0,1)$ on a binary classifier (e.g., sneaker vs. sandal from :numref:`sec_softmax_scratch`), plot the resulting ROC curve, and state what $\tau=0,1$ correspond to.
   2. Argue a random-scoring classifier traces the diagonal in expectation, making AUC a threshold-free ranking-quality summary.
   *Provenance:* original (reformatted to nested list); check-question style adapted from Google MLCC's ROC/AUC page (overlap med — cite on adoption).

---

## chapter_linear-classification/softmax-regression-scratch.md — Softmax Regression Implementation from Scratch

**Topic:** Implementing softmax, cross-entropy, and the forward pass from raw tensor ops; training on Fashion-MNIST; reading a confusion matrix.
**Current exercises:** 6; disposition: keep 4, rewrite 2 (drop 0) — ex. 3 ("is it always good to return the most likely label?") and ex. 4 (large-vocabulary softmax) were flagged as open discussion prompts lacking a deliverable; both get a concrete success criterion.

**External sources found:**
- Kevin Murphy, *Probabilistic Machine Learning: An Introduction*, Exercise 5.1 "Reject option in classifiers" (see full citation above) — supplies exactly the missing deliverable for "is it always good to return the most likely label," namely an explicit accept/abstain threshold rule.
- No focused external exercise found for "large-vocabulary softmax problems." The issue (partition-function cost scaling with vocabulary size, motivating hierarchical softmax/negative sampling) is well known in the NLP/language-modeling literature but appears in papers and lecture prose, not as a graded homework question, in CS229/ISL/CMU/Berkeley/Murphy.

**Proposed problem set:**
1. [short-code] **Overflow and underflow of the naive softmax.** Test the section's `softmax` on an input containing $100$, then on an input whose largest entry is below $-100$; implement a fix that shifts by the largest entry before exponentiating.
   *Provenance:* original.
2. [short-code] **A from-definition cross-entropy.** Implement $-\sum_i y_i\log\hat y_i$ directly (rather than by indexing the true-class probability), compare its runtime to the indexing version, and state precisely when the direct form is and isn't safe to use (hint: the domain of the logarithm).
   *Provenance:* original.
3. [conceptual] **When the top label isn't the right answer to report.** Using the reject-option threshold $p(\hat y=j_{\max}\mid x)\geq 1-\lambda_r/\lambda_s$, decide for a medical-diagnosis scenario with $\lambda_r\ll\lambda_s$ whether the model should report its top label, its full distribution, or abstain, and state the threshold at which behavior switches.
   *Provenance:* adapted from Murphy PML Ex. 5.1 (overlap med — cite on adoption); rewritten from ex. 3 to add a concrete deliverable.
4. [conceptual] **Softmax with a large output space.** For a vocabulary of size $V=10^5$–$10^6$ (predicting the next word from features), identify which steps of softmax and cross-entropy scale with $V$, estimate the resulting per-step cost relative to $V=10$ (Fashion-MNIST), and name one architectural change that avoids paying it in full.
   *Provenance:* inspired by common NLP practice (hierarchical softmax / negative sampling); no single citable external exercise found (overlap low).
5. [short-code] **Hyperparameter sweep.** Plot validation loss against learning rate; determine how large or small the minibatch size must become before validation/training loss visibly change.
   *Provenance:* original.
6. [short-code] **Reading the confusion matrix.** From the column-normalized confusion matrix computed in this section, identify the hardest class and the pair(s) of classes responsible for most errors; explain why a *linear* model struggles on exactly those pairs and why a model sensitive to localized shape cues should help.
   *Provenance:* original.

---

## chapter_linear-classification/softmax-regression-concise.md — Concise Implementation of Softmax Regression

**Topic:** Framework-fused log-softmax + cross-entropy via the log-sum-exp trick; why the model must emit logits, not probabilities.
**Current exercises:** 4; disposition: keep 3, rewrite 1 (drop 0) — ex. 1's numeric-format question was flagged as ambiguous about whether to answer once or per-format; the rewrite requires an explicit table.

**External sources found:**
- **No good external exercise tradition found.** Mixed-precision overflow/underflow ranges (FP64/FP32/BF16/FP16/TF32/INT8) and INT8 dynamic-range tricks are systems/numerics topics covered in vendor documentation and deep-learning-systems courses, not in the classical ML problem sets checked (CS229, ISL, CMU 10-601, Berkeley CS189, Murphy PML) — a genuine gap, not a search failure.
- The RealSoftMax / log-sum-exp bound already established in this chapter (`softmax-regression.md` ex. 6) is the closest available "external-to-this-section" material; it is internal to the book, so problem 5 below is tagged original with an internal cross-reference rather than an external adoption.

**Proposed problem set:**
1. [conceptual] **Exp() range by number format.** For each of FP64, FP32, BF16, FP16, TF32, and INT8, state the smallest and largest argument to $\exp$ that avoids underflow/overflow, as an explicit six-row table.
   *Provenance:* original (rewritten from ex. 1 to require a table for all six formats rather than one implicit answer).
2. [conceptual] **Extending INT8's dynamic range.** Propose a way to extend INT8's $[-128,127]$ dynamic range without using more bits (e.g., a scale factor), and state whether ordinary multiplication and addition still work unmodified on the result.
   *Provenance:* original.
3. [short-code] **Naive vs. fused loss at the extremes.** Feed the from-scratch `softmax` the logits $(1000,0,0)$ and report what happens and why; compute the loss for the same logits via the framework's fused `cross_entropy` and explain why it stays finite; verify the two routes agree to floating-point precision on benign logits $(2,1,0)$.
   *Provenance:* original.
4. [conceptual] **Loss invariance to a logit shift.** Using $\ell=\log\sum_k\exp(o_k)-o_y$, show adding a constant $c$ to every logit leaves $\ell$ unchanged, and explain why this makes subtracting $\bar o=\max_k o_k$ free and safe.
   *Provenance:* original.
5. [short-code] **Verify the log-sum-exp gap bound.** Implement the stable log-sum-exp $\bar o+\log\sum_k\exp(o_k-\bar o)$ from scratch, and empirically confirm over random logit vectors of increasing dimension $q$ that the gap between log-sum-exp and $\max_k o_k$ never exceeds $\log q$ (the bound proved in `softmax-regression.md` ex. 6b).
   *Provenance:* original (internal cross-reference to this chapter's own proof, not an external adoption).

---

## chapter_linear-classification/generalization-classification.md — Generalization in Classification

**Topic:** Test-set statistics (CLT/Hoeffding confidence intervals), adaptive overfitting from test-set reuse, and VC-dimension-based uniform-convergence bounds.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — this is the chapter's "most textbook-classic" set (per the prior style review) and already defect-free; external material is added, not substituted.

**External sources found:**
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning: From Theory to Algorithms* (Cambridge, 2014), Ch. 6 Exercise 5 — proves $\mathrm{VCdim}$ of axis-aligned rectangles in $\mathbb R^d$ equals $2d$, generalizing the book's 2D case. — https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf
- Same book, Ch. 6 Exercise 1 — proves VC-dimension monotonicity: $H'\subseteq H \Rightarrow \mathrm{VCdim}(H')\leq\mathrm{VCdim}(H)$.
- Same book, Ch. 6 Exercise 11 — bounds the VC dimension of a union of $r$ hypothesis classes in terms of $\max_i \mathrm{VCdim}(H_i)$ and $r$.
- CMU 10-601 (Balcan, Spring 2015), Homework 5 Problem 1 — bounds the shattering coefficient of an intersection of two hypothesis classes, $H^*[n]\leq H_1[n]\cdot H_2[n]$, derives $\mathrm{VCdim}$ of half-spaces in $\mathbb R$ and $\mathbb R^2$ and of axis-aligned squares in $\mathbb R^2$, all via Sauer's lemma. — https://www.cs.cmu.edu/~ninamf/courses/601sp15/hw/homework5.pdf

**Proposed problem set:**
1. [conceptual] **Sample size for a tight estimate.** How many samples are needed to estimate a fixed classifier's error to within $0.0001$ with probability $>99.9\%$?
   *Provenance:* original.
2. [conceptual] **Leaking a test set through queries alone.** If you can only observe $\epsilon_\mathcal D(f)$ for models $f$ you choose, how many models must you evaluate before you could appear to reach error $0$ regardless of true error?
   *Provenance:* original.
3. [conceptual] **VC dimension of degree-5 polynomials.** State and justify the VC dimension of the class of fifth-order polynomial classifiers.
   *Provenance:* original.
4. [conceptual] **VC dimension of axis-aligned rectangles, in general $d$.** Find the VC dimension of axis-aligned rectangles in $\mathbb R^2$, then prove the general-$d$ result $\mathrm{VCdim}=2d$.
   *Provenance:* part 1 original; part 2 adapted from Shalev-Shwartz & Ben-David Ch. 6 Ex. 5 (overlap med — cite on adoption).
5. [conceptual] **Shattering the standard basis.** Prove $\mathrm{VC}\geq d+1$ for linear classifiers on $\mathbb R^d$ by shattering $\{\mathbf 0,\mathbf e_1,\ldots,\mathbf e_d\}$ with the given weight construction; combined with the Radon argument in the text, conclude $\mathrm{VC}=d+1$ exactly.
   *Provenance:* original.
6. [conceptual] **Collinear points can't be shattered.** Show three collinear points cannot be shattered by halfplanes, and explain why this does not contradict $\mathrm{VC}=3$ for lines in the plane.
   *Provenance:* original.
7. [conceptual] **Composing hypothesis classes.** Given two hypothesis classes $H_1,H_2$ with shattering coefficients $H_1[n],H_2[n]$, show the shattering coefficient of their intersection satisfies $H^*[n]\leq H_1[n]\cdot H_2[n]$, and use it to bound the VC dimension of an intersection (or union) of two linear-classifier classes.
   *Provenance:* adapted from CMU 10-601 (Balcan, Sp15) HW5 Problem 1(a–c) (overlap med — cite on adoption).
8. [short-code] **Does the finite-sample gap close as predicted?** Extend this section's own CLT-vs-Hoeffding simulation to a second true error rate (e.g., $\epsilon=0.3$) and a finer grid of $n$; report whether the simulated spread still tracks the CLT curve and by what multiple the Hoeffding radius still exceeds it.
   *Provenance:* original (fills the section's only gap: zero short-code exercises despite the text containing a working numpy simulation).

---

## chapter_linear-classification/environment-and-distribution-shift.md — Environment and Distribution Shift

**Topic:** Covariate/label/concept shift; importance-weighted correction via a domain-discriminating classifier; confusion-matrix-based label-shift correction; deployment feedback loops.
**Current exercises:** 6; disposition: keep 6, rewrite 0, drop 0 — previously reviewed as defect-free; external material is added, not substituted.

**External sources found:**
- Google, Machine Learning Crash Course, "Production ML systems: Monitoring pipelines" — a verified check-your-understanding example: a revenue-prediction model uses "number of customers" as a feature, which is unavailable at serving time before the day's sales complete — a concrete instance of training-serving skew. — https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring
- Stanford CS329D, "Machine Learning Under Distributional Shifts" (Hashimoto) — confirmed via its course site that assessment is paper-led discussion plus a semester project, with no traditional problem sets; there is no citable homework exercise to adopt from this course, which is itself a finding (graduate distribution-shift teaching favors open research engagement over structured problem sets). — https://thashim.github.io/cs329D/
- Quiñonero-Candela, Sugiyama, Schwaighofer & Lawrence (eds.), *Dataset Shift in Machine Learning* (MIT Press) — an edited research monograph with no end-of-chapter exercises; correctly used already as this section's cited reference, not as an exercise source.
- Koh, Sagawa, Marklund et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts" (ICML 2021) — already cited in this section's own Summary; its published per-domain in-distribution/out-of-distribution accuracy gaps are a ready-made resource for a written analysis problem (below) rather than a new citation.

**Proposed problem set:**
1. [conceptual] **Search-engine feedback loop.** If a search engine's ranking changes, how might users and advertisers respond, and why is this an instance of the loan/footwear feedback loop from the start of the section?
   *Provenance:* original.
2. [conceptual] **Deriving the covariate-shift identity.** Starting from the risk under $p(\mathbf x,y)$, derive the reweighting identity :eqref:`eq_covariate-shift-identity`, and state precisely the support condition on $p(\mathbf x)$ and $q(\mathbf x)$ under which $\beta_i=p(\mathbf x_i)/q(\mathbf x_i)$ stays finite.
   *Provenance:* original.
3. [short-code] **Build a covariate-shift detector.** Create a shifted copy of a labeled dataset's features (e.g., additive Gaussian noise), train a logistic-regression discriminator to tell original from shifted inputs, and report its accuracy; relate that accuracy to how detectable the shift is.
   *Provenance:* original.
4. [short-code] **Build a covariate-shift corrector.** Using the detector above, compute weights $\beta_i=\exp(h(\mathbf x_i))$, retrain with weighted ERM :eqref:`eq_weighted-empirical-risk-min`, and compare target-domain accuracy with and without reweighting; describe how the variance of $\beta_i$ grows with the shift and how clipping $\beta_i\leftarrow\min(\beta_i,c)$ helps.
   *Provenance:* original.
5. [conceptual] **The label-shift linear system.** Show $\mathbf C\,p(\mathbf y)=\mu(\hat{\mathbf y})$ follows from the law of total probability under the label-shift assumption, and explain why $\mathbf C$ must be invertible for $p(\mathbf y)=\mathbf C^{-1}\mu(\hat{\mathbf y})$ to be usable.
   *Provenance:* original.
6. [conceptual] **Beyond distribution shift.** Besides distribution shift, what else can make empirical risk a poor approximation of risk? (Consider dependence between examples and mismatch between the loss and the deployment objective.)
   *Provenance:* original.
7. [conceptual] **A feature that doesn't exist at serving time.** A model predicts daily revenue using "number of customers so far today" as a feature and performs well in evaluation. Identify why this feature cannot be used for real predictions, name the general failure mode, and propose a training-time fix that avoids it.
   *Provenance:* adapted from Google MLCC's "Monitoring pipelines" check-your-understanding example (overlap med — cite on adoption).
8. [extended] **Classifying real-world shifts.** From the WILDS benchmark's (Koh et al., 2021, already cited in this section) published task descriptions, pick two of its distribution-shift tasks (e.g., a hospital-to-hospital or camera-trap-to-camera-trap shift) and classify each as closer to covariate shift, label shift, or neither, justifying from how the data was collected. If a small WILDS dataset is available locally, additionally train a linear baseline and report its in-distribution vs. out-of-distribution accuracy gap.
   *Provenance:* inspired by Koh et al., WILDS (already cited in-book; overlap low — no new citation required beyond the book's existing one).
