# Leading pages (index.md) — Resources and Further Reading for chapters 6, 7, 8

Ten chapters (preliminaries, linear-regression, linear-classification,
multilayer-perceptrons, and all six `chapter_mdl-*`) end their leading page
with a curated `## Resources and Further Reading {.unnumbered}` section.
Chapters 6 (builders-guide), 7, and 8 are missing theirs. This spec provides
the curated content; Alex asked for chapter 6 to be fixed as part of this
rewrite even though it is otherwise out of scope.

## Format rules (match the existing pages exactly)

- Heading: `## Resources and Further Reading {.unnumbered}` at the end of
  `index.md`, after the `toc` block.
- Opening paragraph: one short paragraph tying the list to the chapter's
  topics, ending with "All are freely accessible online except where noted."
- Groups in bold, in this order (omit a group if empty): **Books**,
  **Courses and video lectures**, **Tutorials, notes, and interactive** (or
  "…and surveys"), **Foundational papers**.
- Each item: `[Title — Authors](url) — annotation.` The annotation must be
  opinionated and chapter-specific (say *which chapter/module of the
  resource* maps to *which section of ours*), one to two lines. Mark
  paywalled items "(paywalled, noted)".
- Dedup rule: a resource already listed on another chapter's leading page may
  be re-listed **only** with a different, chapter-specific pointer (e.g. UDL
  appears on both the linear-regression and MLP pages with different chapter
  annotations). Check `grep -l "Resources and Further Reading"
  chapter_*/index.md` targets before finalizing.
- **Verify every URL resolves (WebFetch or curl) before committing**; replace
  dead links with a live equivalent rather than deleting the item. Two flagged
  as verify-first below: the LeCun 1998 PDF (lecun.com has had outages) and
  the Hubel & Wiesel PMC id.
- These lists are curated drafts: keep the selections unless a link is dead
  or you find a strictly better edition of the *same* resource. Do not grow
  the lists; not encyclopedic.

Leading pages have no code cells, so no notebook execution or output capture
is needed: edit `index.md`, regenerate `.qmd` via the preprocessor, render.

---

## Chapter 6 — `chapter_builders-guide/index.md`

Opening paragraph: the references go deeper on the machinery this chapter
opens up: how frameworks represent models and state, automatic
differentiation, numerics and mixed precision, devices and memory, and
reproducible training.

**Books**

- [Deep Learning with PyTorch — Stevens, Antiga & Viehmann](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf) — free PDF from the PyTorch team; Part 1 walks tensors, storage, autograd, and `nn.Module` mechanics at exactly this chapter's level of "open the machinery".
- [Machine Learning Systems — Vijay Janapa Reddi](https://mlsysbook.ai/) — free online; the systems view around this chapter: frameworks, data pipelines, training infrastructure, and efficient deployment.

**Courses and video lectures**

- [CMU 10-414/714: Deep Learning Systems — Chen & Kolter](https://dlsyscourse.org/) — free lectures and assignments; you build "needle", a miniature framework with autograd, modules, initialization, and GPU support — this entire chapter from the implementor's side.
- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) — free video series; builds autograd, modules, and training loops from scratch in plain Python, making every abstraction in this chapter concrete before you rely on the library's version.
- [fast.ai Part 2: Deep Learning Foundations — Howard et al.](https://course.fast.ai/Lessons/part2.html) — free; rebuilds a training framework from tensor ops upward (modules, initialization, mixed precision, accelerated training), the practitioner's companion to this chapter.

**Tutorials, notes, and interactive**

- [PyTorch internals — Edward Yang](http://blog.ezyang.com/2019/05/pytorch-internals/) — free; the classic guided tour of tensors, strides, dispatch, and autograd inside PyTorch, one level below :numref:`sec_model_construction`.
- [JAX — The Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) — free; pure functions, explicit PRNG keys, and jit constraints — the functional worldview behind this book's JAX tab, stated as a list of gotchas.
- [Train With Mixed Precision — NVIDIA](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — free; the vendor guide to fp16/bf16 arithmetic and loss scaling that :numref:`sec_numerics` distills.
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic — David Goldberg](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — free; the standard reference beneath every rounding and overflow issue in :numref:`sec_numerics`.
- [Making Deep Learning Go Brrrr From First Principles — Horace He](https://horace.io/brrr_intro.html) — free; the compute-, memory-, and overhead-bound mental model that :numref:`sec_use_gpu` builds on.
- [Reproducibility — PyTorch notes](https://pytorch.org/docs/stable/notes/randomness.html) — free; the determinism flags, their costs, and their limits — the fine print behind :numref:`sec_repro`.

**Foundational papers**

- [Automatic Differentiation in Machine Learning: a Survey — Baydin, Pearlmutter, Radul & Siskind (2018), *JMLR*](https://jmlr.org/papers/v18/17-468.html) — free; the definitive account of forward- and reverse-mode autodiff, the algorithm every framework in this book implements.
- [Mixed Precision Training — Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) — free; the origin of the fp16 + master-weights + loss-scaling recipe in :numref:`sec_numerics`.
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library — Paszke et al. (2019)](https://arxiv.org/abs/1912.01703) — free; the design rationale (eager execution, autograd, memory allocator) for the imperative style this chapter teaches.

---

## Chapter 7 — `chapter_convolutional-neural-networks/index.md`

Opening paragraph: the references develop the fundamentals of this chapter:
why convolutions, the mechanics of the operation (padding, stride, dilation,
channels), receptive fields, pooling, and the first convolutional networks.

**Books**

- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 9 (Convolutional Networks) is the canonical treatment of sparse interactions, parameter sharing, and equivariance — the same three-step argument as :numref:`sec_why-conv`, developed at length.
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF; Chapter 10 covers convolutions, stride, dilation, and receptive fields with unusually good figures.
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap6.html) — free online; Chapter 6 introduces convolutional layers, shared weights, and pooling from first principles, a gentle second telling of this chapter.

**Courses and video lectures**

- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/convolutional-networks/) — free notes; the "Convolutional Networks" module (layer arithmetic, parameter counting, layer patterns) is the most widely used online companion to this chapter.
- [Michigan EECS 498-007: Deep Learning for Computer Vision — Justin Johnson](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) — free lecture videos; the convolution and pooling lectures work through kernel/stride/padding arithmetic on the board at exactly this chapter's pace.

**Tutorials, notes, and interactive**

- [A guide to convolution arithmetic for deep learning — Dumoulin & Visin (2016)](https://arxiv.org/abs/1603.07285) — free, with the famous [animations](https://github.com/vdumoulin/conv_arithmetic); the definitive visual reference for padding, stride, dilation, and transposed convolutions.
- [CNN Explainer — Wang et al. (Polo Club)](https://poloclub.github.io/cnn-explainer/) — free, zero-install; an interactive convnet running in the browser where you can inspect every activation, kernel, and receptive field.
- [Computing Receptive Fields of Convolutional Neural Networks — Araujo, Norris & Sim (2019), *Distill*](https://distill.pub/2019/computing-receptive-fields/) — free; derives the closed-form receptive-field arithmetic that :numref:`sec_conv_layer` introduces, including strided and multi-path cases.
- [Conv Nets: A Modular Perspective — Chris Olah](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) — free; a short classic building the "convolution as structured weight sharing" intuition.
- [Image Kernels Explained Visually — Victor Powell](https://setosa.io/ev/image-kernels/) — free, interactive; slide hand-designed kernels over an image, a perfect warm-up for the edge-detection example in :numref:`sec_conv_layer`.
- [Feature Visualization — Olah, Mordvintsev & Schubert (2017), *Distill*](https://distill.pub/2017/feature-visualization/) — free; what convnet units actually respond to, a companion to the activation-visualization exercise in :numref:`sec_lenet`.

**Foundational papers**

- [Gradient-Based Learning Applied to Document Recognition — LeCun, Bottou, Bengio & Haffner (1998), *Proc. IEEE*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) — free PDF (verify link is up); the LeNet-5 paper behind :numref:`sec_lenet`, still worth reading for its systems-level completeness.
- [Backpropagation Applied to Handwritten Zip Code Recognition — LeCun et al. (1989), *Neural Computation*](https://doi.org/10.1162/neco.1989.1.4.541) — the first convnet trained end-to-end with backprop (paywalled, noted; widely reproduced online).
- [Neocognitron — Fukushima (1980), *Biological Cybernetics*](https://doi.org/10.1007/BF00344251) — the pre-backprop ancestor of alternating convolution/pooling stages (paywalled, noted).
- [Receptive fields, binocular interaction and functional architecture in the cat's visual cortex — Hubel & Wiesel (1962), *J. Physiology*](https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/) — free (PMC; verify id); the biological root of local receptive fields and hierarchical feature detection.

---

## Chapter 8 — `chapter_convolutional-modern/index.md`

Opening paragraph: the references trace the architecture race this chapter
narrates and the modern practice it lands on: landmark networks from AlexNet
to ConvNeXt, the training recipes that confound naive comparisons between
them, and efficient networks for deployment.

**Books**

- [Deep Learning for Coders with fastai and PyTorch — Howard & Gugger](https://github.com/fastai/fastbook) — free notebooks; builds ResNets from scratch and applies the modern training tricks (augmentation, schedules, mixed precision) of :numref:`sec_training_recipes` in working code.
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF; Chapter 11 (Residual networks) analyzes *why* residual connections ease optimization — loss-surface and gradient-propagation arguments complementing :numref:`sec_resnet`.

**Courses and video lectures**

- [Michigan EECS 498-007, Lecture: CNN Architectures — Justin Johnson](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) — free videos; walks AlexNet → VGG → GoogLeNet → ResNet with parameter/FLOP accounting, the same tour as this chapter's first half.
- [MIT 6.5940: TinyML and Efficient Deep Learning — Song Han](https://efficientml.ai/) — free lectures; depthwise separability, quantization, pruning, and edge deployment — the systems side of :numref:`sec_efficient_cnns`.
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course) — free; modern practice with pretrained backbones (including ConvNeXt) and transfer learning, the "what you actually do in 2026" companion to this chapter.

**Tutorials, notes, and surveys**

- [timm (pytorch-image-models) — Ross Wightman et al.](https://huggingface.co/docs/timm) — free; the reference implementation zoo where every architecture in this chapter lives, with trained weights and [results tables](https://github.com/huggingface/pytorch-image-models/tree/main/results) comparing them under consistent evaluation.
- [A Recipe for Training Neural Networks — Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/) — free; the debugging-first training discipline that :numref:`sec_training_recipes` systematizes.
- [Zoom In: An Introduction to Circuits — Olah et al. (2020), *Distill*](https://distill.pub/2020/circuits/zoom-in/) — free; opens trained vision models to inspect the features and circuits they learn, useful perspective once you can train the architectures in this chapter.

**Foundational papers**

All free on arXiv or the proceedings site; these are the primary sources this
chapter retells, worth reading in the original:

- [ImageNet Classification with Deep Convolutional Neural Networks — Krizhevsky, Sutskever & Hinton (2012), *NeurIPS*](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — AlexNet (:numref:`sec_alexnet`).
- [Batch Normalization — Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167) — read together with its critique in :numref:`sec_batch_norm`.
- [Deep Residual Learning for Image Recognition — He, Zhang, Ren & Sun (2015)](https://arxiv.org/abs/1512.03385) — ResNet (:numref:`sec_resnet`), the most-cited paper in deep learning.
- [Bag of Tricks for Image Classification with CNNs — He et al. (2019)](https://arxiv.org/abs/1812.01187) — the first systematic demonstration that recipe details rival architecture changes.
- [ResNet Strikes Back — Wightman, Touvron & Jégou (2021)](https://arxiv.org/abs/2110.00476) — the definitive recipe-vs-architecture accounting behind :numref:`sec_training_recipes`.
- [A ConvNet for the 2020s — Liu et al. (2022)](https://arxiv.org/abs/2201.03545) — ConvNeXt (:numref:`sec_convnext`), a controlled ablation worth studying as method, not just result.
- [ConvNets Match Vision Transformers at Scale — Smith et al. (2023)](https://arxiv.org/abs/2310.16764) — the scaling-law resolution of the convnet-vs-transformer debate closing this chapter.
