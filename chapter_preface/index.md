# Preface

## Preface to the Second Edition

Looking back, the first edition feels like it was written in a different world. Text, images, and code were produced by hand, style guides were zealously enforced by coauthors, and the addition of a new framework required the talented hands of experts, in this case Anirudh Dagar (PyTorch and JAX) and Yuan Tang (TensorFlow). As the book grew beyond 1,000 pages and attracted hundreds of GitHub issues, PRs, and Discourse comments, keeping it all up to date became a full-time job. Consequently, when all of the authors moved on to different places, the book sat there for half a decade. Many of its premises, such as accessible code, teaching by example, and the notebook-based format, have stood the test of time. They are as fresh as they were in 2020. Approximately half of its content has aged well, while other parts have since been superseded by much more exciting problems, techniques, and applications. Very little of the code, beyond the D2L library format, survived particularly well, since we have seen tremendous progress in hardware and the associated software tooling.

Two things happened in 2026 that compelled me to pick up this project again and see whether D2L could be "refurbished". First, I received a request from a Greek publisher asking whether I would consent to the book being translated and printed in Greek. This told me that, even five years after we last improved it, the book still had value to the point of someone wanting to print a genuinely new copy. Second, as we have all seen with the rapid progress of agentic AI, the tools from Anthropic, OpenAI, and Google have made it dramatically easier to perform broad updates and repairs that would have been unaffordable previously. They allowed me to bring the code up to date relative to modern frameworks, incorporate the GitHub issues and PRs that were pending, review the bug reports and comments in the Discourse threads, and check and fix the consistency of the code, language, experiments, and math. The build system was revamped: the book now uses [Quarto](https://quarto.org) and [uv](https://docs.astral.sh/uv/), the fora are hosted on [Discourse](https://d2l.discourse.group)'s own servers, and the content format was improved to separate code execution from editing. Having agents also allowed me to ensure that all frameworks perform similarly on the same problems, i.e., no more performance regressions due to poorly used JIT compilation, differences in initialization, optimization schedules, or the use of "magic" random seeds. This is a luxury we could not have afforded before, yet I believe that it significantly improves the teaching quality, as it demonstrates best practices for each framework throughout.

Besides all the necessary plumbing improvements, the content has gone through a number of changes. The book is too long to be viewed as one piece (it weighs in at over 3,000 pages as a PDF). The solution was to break it up into multiple pieces. [Basics](../chapter_basics/index.qmd) covers, well, the basics of deep learning and is a great companion for an introductory course. [Advanced](../chapter_advanced/index.qmd) takes you close to the current state of the art in many areas, including [optimization](../chapter_optimization/index.qmd), [state space models](../chapter_recurrent-modern/index.qmd), [transformers](../chapter_transformers/index.qmd), [reinforcement learning](../chapter_reinforcement-learning/index.qmd), [GANs](../chapter_generative-adversarial-networks/index.qmd), and [diffusion models](../chapter_diffusion-models/index.qmd), without going into too much depth in any one of them. Going into depth would be impossible, given the information explosion in each of these fields. [Mathematics for Deep Learning](../chapter_mathematics-for-deep-learning/index.qmd) has grown significantly to cover more about [diffusion and flows](../chapter_mdl-dynamics/index.qmd), [information theory](../chapter_mdl-information-theory/index.qmd), and [optimization](../chapter_mdl-optimization/index.qmd). It could serve well as a textbook for an introductory math course in this field, and it helps cover the necessary background that is often taken for granted when reading papers (or some of the chapters of this book). I am grateful to Rasool Fakoor for contributing the reinforcement learning and diffusion chapters.

A lot of work remains to be done. The [Language Models](../chapter_language-models/index.qmd) part is still unwritten, and the book does not yet cover computer vision, graphics, audio, speech, or semi-supervised learning, nor does it provide a detailed treatment of agents. With a bit of time, plenty of tokens, and generous amounts of curation, the latter (the agents) will aid me in completing the former (the book). This should also help keep the book up to date whenever new scientific discoveries happen and new techniques are invented.

*Alex Smola, November 2026*

## Preface to the First Edition

Just a few years ago, there were no legions of deep learning scientists
developing intelligent products and services at major companies and startups.
When we entered the field, machine learning
did not command headlines in daily newspapers.
Our parents had no idea what machine learning was,
let alone why we might prefer it to a career in medicine or law.
Machine learning was a blue skies academic discipline
whose industrial significance was limited
to a narrow set of real-world applications,
including speech recognition and computer vision.
Moreover, many of these applications
required so much domain knowledge
that they were often regarded as entirely separate areas
for which machine learning was one small component.
At that time, neural networks---the
predecessors of the deep learning methods
that we focus on in this book---were
generally regarded as outmoded.


Yet in just a few years, deep learning has taken the world by surprise,
driving rapid progress in such diverse fields
as computer vision, natural language processing,
automatic speech recognition, reinforcement learning,
and biomedical informatics.
Moreover, the success of deep learning
in so many tasks of practical interest
has even catalyzed developments
in theoretical machine learning and statistics.
With these advances in hand,
we can now build cars that drive themselves
with more autonomy than ever before
(though less autonomy than some companies might have you believe),
dialogue systems that debug code by asking clarifying questions,
and software agents beating the best human players in the world at board games such as Go, a feat once thought to be decades away.
Already, these tools exert ever-wider influence on industry and society,
changing the way movies are made, diseases are diagnosed,
and playing a growing role in basic sciences---from astrophysics, to climate modeling, to weather prediction, to biomedicine.

*Aston Zhang, Zach Lipton, Mu Li, and Alex Smola, 2021*


## About This Book

This book is our attempt to make deep learning approachable by teaching you the *concepts*, the *context*, and the *code* in one place.

### One Medium Combining Code, Math, and HTML

For any computing technology to reach its full impact, it must be well understood, well documented, and supported by mature, well-maintained tools. The key ideas should be clearly distilled, minimizing the time it takes to bring new practitioners up to date. Mature libraries should automate common tasks, and exemplar code should make it easy for practitioners to modify, apply, and extend standard models to suit their needs.

Dynamic web applications offer a clear parallel. Although companies such as Amazon developed successful database-driven web applications in the 1990s, the broader potential of this technology to aid creative entrepreneurs was unlocked only when powerful, well-documented web frameworks emerged over the following decade.

Realizing the potential of deep learning presents unique challenges because any single application brings together several disciplines. Applying deep learning requires simultaneously understanding (i) the motivation for casting a problem in a particular way; (ii) the mathematical form of a given model; (iii) the optimization algorithms that fit the model to data; (iv) the statistical principles that tell us when we should expect our models to generalize to unseen data, along with practical methods for certifying that they have, in fact, generalized; and (v) the engineering techniques required to train models efficiently, navigating the pitfalls of numerical computing and getting the most out of available hardware. Teaching the critical thinking skills required to formulate problems, the mathematics to solve them, and the software tools to implement those solutions all in one place is a formidable challenge. Our goal in this book is to provide a unified resource that brings aspiring practitioners up to speed.

When we started this book project, no single resource simultaneously (i) remained up to date, (ii) covered the breadth of modern machine learning with sufficient technical depth, and (iii) interleaved the rigorous exposition expected of a textbook with the clean, runnable code expected of a hands-on tutorial. We found plenty of code examples illustrating how to use a particular deep learning framework (for instance, basic matrix computations in TensorFlow) or how to implement specific architectures (such as LeNet, AlexNet, or ResNet) scattered across blog posts and GitHub repositories. However, these examples typically showed *how* to implement a method while leaving out *why* certain algorithmic design choices were made. Interactive articles, such as those published on [Distill](http://distill.pub) or on personal blogs, explored selected topics visually, but covered only isolated corners of the field and often lacked accompanying code. Meanwhile, the deep learning textbooks that had appeared---such as :citet:`Goodfellow.Bengio.Courville.2016`, which offers a comprehensive survey of the foundations of deep learning---did not connect their mathematical descriptions to working implementations, often leaving readers unsure how to bridge the gap from theory to code. Moreover, too many learning resources remained locked behind the paywalls of commercial course providers.

We set out to create a resource that would (i) be freely available to everyone; (ii) offer enough technical depth to serve as a springboard toward becoming an applied machine learning scientist; (iii) include runnable code showing readers *how* to solve problems in practice; (iv) allow rapid updates, both by us and by the community at large; and (v) be complemented by a [discussion forum](https://d2l.discourse.group/c/5) for interactive exchange on technical details and open questions.

These goals were often in conflict. Equations, theorems, and citations are best written and typeset in LaTeX. Code is most naturally expressed in Python. And web pages live natively in HTML and JavaScript. Furthermore, we wanted the content to be equally accessible as executable notebooks, a printed book, a downloadable PDF, and an interactive website. Because no existing publishing workflow met all of these demands, we assembled our own (:numref:`sec_how_to_contribute`). We settled on GitHub to host the source and coordinate community contributions; Markdown as the single source format mixing prose, equations, and code, from which Jupyter notebooks are generated automatically; Quarto as our rendering engine; and Discourse as our discussion platform. While no system is perfect, these choices strike an effective balance among competing demands. We believe that *Dive into Deep Learning* was the first book published using such an integrated workflow.


### Learning by Doing

Many textbooks present concepts in strict succession, covering each topic in exhaustive detail before moving on. For example, the classic textbook by :citet:`Bishop.2006` treats each subject so thoroughly that simply reaching the chapter on linear regression requires a nontrivial investment of effort. While experts prize that book precisely for its thoroughness, this structure limits its accessibility as an introductory text for beginners.

In this book, we teach most concepts *just in time*. In other words, you encounter new ideas at the very moment they are needed to accomplish a concrete, practical goal. While we devote a brief opening chapter to essential preliminaries such as linear algebra and probability---and provide a comprehensive mathematical appendix for deeper study---we want you to experience the satisfaction of training your first working model before worrying about more abstract theory.

Aside from a few preliminary notebooks that provide a crash course in the basic mathematical background, each subsequent chapter introduces a manageable set of new concepts alongside self-contained working examples on real datasets. This approach presented an organizational challenge: some models might logically be grouped together in a single notebook, while other ideas are best taught by running several models in succession. Even so, adhering to a policy of *one working example, one notebook* offers a major practical advantage: it makes it as easy as possible for you to launch your own research or engineering projects by building directly on our code. Simply copy a notebook and start modifying it.

Throughout the book, we interleave runnable code with theoretical background as needed. As a rule, we err on the side of putting a tool in your hands before explaining every detail of how it works, filling in the deeper background once you have seen it in action. For instance, we use *stochastic gradient descent* to train early models well before we analyze why it works so effectively. This approach gives practitioners the tools they need to solve problems quickly, at the modest cost of asking you to trust a few of our pedagogical choices along the way.

We also teach core deep learning concepts from scratch. Often, we look under the hood at fine details of models that modern deep learning frameworks normally hide from users. This comes up especially in the foundational chapters, where we want you to understand everything that happens inside a layer, a loss function, or an optimizer. In these cases, we regularly present two implementations side by side: one built from scratch relying only on basic tensor operations and automatic differentiation, and a second, concise version written with the high-level APIs of a deep learning framework. Once we have explained how a component works from the ground up, we rely on its high-level framework implementation in subsequent chapters.


### Content and Structure

The book opens with three short introductory chapters: :numref:`chap_installation` walks through setting up an environment to run the code, :numref:`chap_notation` summarizes our mathematical conventions, and :numref:`chap_introduction` surveys the landscape of deep learning. The remainder of the book is organized into six parts, with the two core parts---Basics and Advanced---mapped out in :numref:`fig_book_org`.

![Structure of the book. An arrow from one chapter to another means that the second builds on the first; the arrow from the Basics to the Advanced part means that every Advanced chapter builds on the Basics. The Language Models, Image Models, and Attic parts are not shown.](../img/front-book-org.svg)
:label:`fig_book_org`


* **Basics**.
In :numref:`chap_preliminaries`, we quickly bring you up to speed on the practical prerequisites for hands-on deep learning: storing and manipulating data as tensors, computing derivatives automatically, and reasoning about probability and uncertainty. Next, :numref:`chap_regression` and :numref:`chap_classification` fit linear models to data and introduce the three basic ingredients of nearly every method in this book: a parameterized model, a loss function that measures its errors, and an optimization algorithm that reduces that loss. :numref:`chap_perceptrons` adds hidden layers to build deep networks and asks why models with so many parameters can still generalize to new data. :numref:`chap_computation` then turns to the mechanics of deep learning software: how models are assembled from reusable modules, initialized, saved, and accelerated on GPUs. The final three chapters of the part build domain structure directly into the architecture: the convolutional neural networks (CNNs) of :numref:`chap_cnn` and :numref:`chap_modern_cnn` exploit the spatial symmetries of images, while the recurrent sequence models of :numref:`chap_rnn` capture the sequential order of text and time series. The first five chapters are best read in order; after that, the chapters on images and sequences can be read independently of one another.

* **Advanced**.
These chapters build on the foundations of the Basics part to develop the architectures and training methods used at the frontier today. They can be read in several orders, with the arrows in :numref:`fig_book_org` indicating dependencies between chapters. Where the Basics part treats optimizers, learning rate schedules, and batch sizes largely as working recipes, :numref:`chap_optimization` examines why they work and how they must adapt as models scale. :numref:`chap_attention` introduces *attention* mechanisms, which allow each position in a sequence to retrieve information directly from any other position, and :numref:`chap_transformers` stacks these layers into the transformer architectures that power modern large language models. :numref:`chap_modern_rnn` revisits recurrence through state space models, which maintain a fixed-size state while still training across all positions of a sequence in parallel. :numref:`chap_performance` investigates what limits the speed of a training step on a single GPU and across multiple GPUs, using a transformer as its running example. :numref:`chap_reinforcement_learning` and :numref:`chap_deep_rl` shift from supervised learning to agents that learn from rewards through interaction with an environment, starting with tabular problems small enough to solve exactly before scaling up with neural networks. Finally, :numref:`chap_gans` and :numref:`chap_diffusion` tackle generative modeling: synthesizing new data either through a minimax game between a generator and a discriminator, or by learning to iteratively reverse a corruption process.

* **Language Models and Image Models**.
The Language Models part, currently in preparation, extends the methods of the Advanced part to modern large language models: encoders, embeddings, and retrieval (:numref:`chap_lm_encoders`), pretraining (:numref:`chap_lm_pretraining`), fine-tuning (:numref:`chap_lm_finetuning`), alignment via preferences and verifiable rewards (:numref:`chap_lm_rl`), agents (:numref:`chap_lm_agents`), and efficient serving (:numref:`chap_lm_serving`). The Image Models part (:numref:`chap_cv`) covers object detection, semantic segmentation, and neural style transfer; it is carried over from the first edition and is slated for an upcoming rewrite.

* **Mathematics for Deep Learning**.
This appendix develops the mathematical foundations underlying the main text. While the main chapters state each mathematical result alongside its intuition, they point here for complete proofs, derivations, and deeper context. :numref:`chap_mdl-linear-algebra` covers geometric projections, eigendecompositions, and the singular value decomposition; :numref:`chap_mdl-calculus` develops multivariable derivatives, gradients, and the mechanics of automatic differentiation. :numref:`chap_mdl-optimization` provides the theoretical guarantees behind the empirical behaviors observed in :numref:`chap_optimization`, such as why curvature places an upper bound on the learning rate. :numref:`chap_mdl-probability-statistics` covers statistical estimation, Bayesian inference, and generalization bounds; :numref:`chap_mdl-information-theory` explains what cross-entropy measures and how probability distributions can be compared via divergences and optimal transport; and :numref:`chap_mdl-dynamics` develops the differential equations and stochastic processes behind diffusion and flow-based generative models. Readers already comfortable with a topic can skip its chapter or use it as a reference; for everyone else, the appendix is designed to serve as a self-contained introductory textbook.

* **Tools for Deep Learning**.
:numref:`chap_appendix_tools` covers practical engineering workflows: running the book's notebooks locally or in hosted cloud environments, selecting and provisioning GPU hardware, scaling distributed training, serving models in production, and contributing to this book. It also provides the API reference for the `d2l` library (:numref:`sec_d2l`).

* **Attic**.
The final part preserves chapters from the first edition that are no longer actively expanded in this edition, covering Gaussian processes, hyperparameter optimization, and recommender systems (:numref:`chap_gp` through :numref:`chap_recsys`).


### Code
:label:`sec_code`

Most sections of this book feature executable code. We believe that many key intuitions in deep learning are best developed through hands-on experimentation: tweaking the code in small ways and observing how the results change. Ideally, an overarching mathematical theory would tell us ahead of time how every architectural or hyperparameter adjustment behaves. In practice, however, deep learning practitioners must often operate ahead of formal theory. Despite rapid progress, complete theoretical explanations for the effectiveness of many techniques remain elusive: the mathematics required to characterize highly nonlinear networks is challenging, generalization often hinges on structural properties of real-world data that resist simple formalization, and systematic theoretical inquiry into modern architectures is still catching up with empirical discovery. As the theory of deep learning matures, we look forward to incorporating deeper formal insights into future editions of this book.

Our code is provided in up to four deep learning frameworks, each presented in its own interactive tab. The Basics part and the Mathematics for Deep Learning appendix are implemented in PyTorch, JAX, TensorFlow, and MXNet, while the Advanced part is implemented in PyTorch and JAX. :numref:`chap_installation` helps you choose a framework and set up a working environment. Every notebook begins with a standard cell of shared imports, explained in :numref:`sec_preamble`. To keep the text focused, we display that preamble cell only once there, allowing subsequent sections to jump straight into the code relevant to the topic at hand.

To avoid repeating boilerplate across chapters, we collect reusable functions and classes into the lightweight `d2l` library. Throughout the book, code blocks marked with `#@save` (such as helper functions, base classes, or standard imports) are defined and explained once in the text, and simultaneously saved into the `d2l` package so that later chapters can call them directly via `d2l`. A complete listing of these functions and classes appears in :numref:`sec_d2l`. The framework-independent core of the `d2l` library requires only the following standard Python imports:

```{.python .input #preface-code-1}
#@save
import inspect
import collections
from collections import defaultdict
from dataclasses import asdict
from IPython import display
import json
import math
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import gzip
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
You are reading the MXNet edition of this book. MXNet implementations are available for the Basics part and the Mathematics for Deep Learning appendix. The Advanced part, beginning with :numref:`chap_optimization`, is implemented in PyTorch and JAX, so you can switch to either of those tabs when reading the advanced chapters. We execute the MXNet code using the MXNet 2.0 build pinned in the environment configuration of :numref:`chap_installation`; if you encounter version incompatibilities, please use that pinned environment. The MXNet backend of the `d2l` library adds the following imports:
:end_tab:

:begin_tab:`pytorch`
You are reading the PyTorch edition of this book, which provides implementations across every chapter. We execute the code using the PyTorch version pinned in the environment configuration of :numref:`chap_installation`. Because deep learning frameworks evolve rapidly, we recommend using that pinned environment if you encounter any incompatibilities with a newer release. The PyTorch backend of the `d2l` library adds the following imports:
:end_tab:

:begin_tab:`tensorflow`
You are reading the TensorFlow edition of this book. TensorFlow implementations are available for the Basics part and the Mathematics for Deep Learning appendix. The Advanced part, beginning with :numref:`chap_optimization`, is implemented in PyTorch and JAX, so you can switch to either of those tabs when reading the advanced chapters. We execute the code using the TensorFlow version pinned in the environment configuration of :numref:`chap_installation`; because deep learning frameworks evolve rapidly, we recommend using that pinned environment if you encounter any incompatibilities with a newer release. The TensorFlow backend of the `d2l` library adds the following imports:
:end_tab:

:begin_tab:`jax`
You are reading the JAX edition of this book, which provides implementations for the Basics part, the Advanced part, and the Mathematics for Deep Learning appendix. JAX operates by transforming pure numerical Python functions---automatically differentiating, compiling, and vectorizing them. Throughout the book, we define neural network models using Flax's `nnx` module and train them with optimizers from Optax. We execute the code using the JAX ecosystem versions pinned in the environment configuration of :numref:`chap_installation`; because these libraries evolve rapidly, we recommend using that pinned environment if you encounter any incompatibilities with a newer release. The JAX backend of the `d2l` library adds the following imports:
:end_tab:

```{.python .input #preface-code-2}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input #preface-code-2}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input #preface-code-2}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input #preface-code-2}
#@tab jax
#@save
from functools import partial
from flax import nnx
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
from typing import Any, Callable
```

### Target Audience

This book is written for undergraduate and graduate students, software engineers, and researchers who want a solid, hands-on grasp of modern deep learning. Because we develop every concept from the ground up, no prior background in deep learning or machine learning is required. Explaining the methods of deep learning thoroughly requires both mathematics and programming, though we assume only a modest foundation in linear algebra, calculus, probability, and Python. Should you need a refresher on any of the mathematical prerequisites---or want to explore the theory in greater depth---the [Mathematics for Deep Learning](../chapter_mathematics-for-deep-learning/index.qmd) appendix covers nearly all of the mathematics used in this book. In the main chapters, we consistently prioritize clear intuition and core ideas over formal mathematical generality.

If you would like to go even further into the mathematical foundations, we warmly recommend several classic texts: *Linear Analysis* by :citet:`Bollobas.1999` covers linear algebra and functional analysis in depth; *All of Statistics* :cite:`Wasserman.2013` offers a concise, lucid introduction to statistical inference; and Joe Blitzstein's [textbook](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918) and [course](https://projects.iq.harvard.edu/stat110/home) on probability are pedagogical gems. Finally, if you are new to Python, we recommend working through an interactive [Python tutorial](http://learnpython.org/) before diving into the code.


### Notebooks, Website, GitHub, and Forum

All of the executable notebooks for this book can be downloaded from the [D2L.ai website](https://d2l.ai) and from our [GitHub repository](https://github.com/smolix/d2l-neu). We also host an active community discussion forum at [d2l.discourse.group](https://d2l.discourse.group/c/5); whenever you have questions about any section of the book, you will find a direct link to the corresponding discussion thread at the bottom of the notebook.



## Acknowledgments

We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
This book was originally implemented with MXNet as the primary framework.
We thank Anirudh Dagar and Yuan Tang for adapting a majority part of earlier MXNet code into PyTorch and TensorFlow implementations, respectively.
Since July 2021, we have redesigned and reimplemented this book in PyTorch, MXNet, and TensorFlow, choosing PyTorch as the primary framework.
We thank Anirudh Dagar for adapting a majority part of more recent PyTorch code into JAX implementations.
We thank Gaosheng Wu, Liujun Hu, Ge Zhang, and Jiehang Xie from Baidu for adapting a majority part of more recent PyTorch code into PaddlePaddle implementations in the Chinese draft.
We thank Shuai Zhang for integrating the LaTeX style from the press into the PDF building.

On GitHub, we thank every contributor of this English draft
for making it better for everyone.
Their GitHub IDs or names are (in no particular order):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi, and Likun Lei.

We thank Amazon Web Services, especially Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky,
and Andrew Jassy for their generous support in writing this book.
Without the available time, resources, discussions with colleagues,
and continuous encouragement, this book would not have happened.
During the preparation of the book for publication,
Cambridge University Press has offered excellent support.
We thank our commissioning editor David Tranah
for his help and professionalism.


### Second Edition Contributors

We are also grateful to the many community members who reported errata, suggested mathematical and pedagogical clarifications, fixed broken links and code bugs, or contributed pull requests across GitHub and the discussion forum over the past four years, including Abhishek Gupta (\@abhishek-gupta2707), Akshaya Bali (\@akshayabali), Alankrit (\@alankrit02), Alexhead (\@Alexhead), Amirhossein (\@amirhosein-mesbah), Andrey (\@v4lproik), Andrii Oriekhov (\@andriyor), Ankit Kumar Singh (\@ankitkumarsingh19091995), Ashwinpathak20 (\@ashwinpathak20), Avinash (\@avinashscs), Baptiste (\@baptiste-pasquier), Basti564 (\@Basti564), Berke (\@berkeoral), Bhavesh Laddagiri (\@bhaveshladdagiri), Bjoernenz (\@bjoernenz), Brian Bartoldson (\@bartoldson), Catalin (\@catalin-sipos), Chaitanya (\@chaitanyab2001), Cuong V. Nguyen (\@ncv), Daniel (\@dakenf), Daniel (\@dani2442), Danishjeet Singh (\@danishjeet), dark_master (\@Dark_Master), Dd3607 (\@Dd3607), Denis Kazakov (\@Denis-Kazakov), Dilip (\@DilipSCU), doc_karl (\@Doc_Karl), DomonkosNagy (\@DomonkosNagy), Dongryeol Lee (\@dongryeol-lee), Dr.rer.nat (\@Dr.rer.nat), Duc (\@nguyenduc1102), Ebad Sayed (\@ebadsayed), Elvis (\@elvisesp), Emmett (\@emmett9001), EnesBerkSak (\@EnesBerkSak), Eurus (\@Eurus-Holmes), Evan (\@evanmiller), Evgenii (\@kaiser179), Fardin (\@fardinabbasi), Fenilgandhi (\@fenilgandhi), Fredrick (\@FriskyFred), Fzyzcjy (\@fzyzcjy), Georgy (\@goshik92), Glenn (\@glennmatlin), GoodMan (\@GoodMan666), Hannes (\@hannesdornbusch), Haocong Wang (\@HaocongWang), HappyB1rd (\@HappyB1rd), Harsh (\@Harsh8604), HenryJia (\@HenryJia), Himanshu (\@himanshurawlani), HyeongjinKim (\@HyeongjinKim), Inaki (\@InakiLPP), James (\@james-Howard), Jayanth (\@jayanthjayanth), Jcbl1 (\@jcbl1), Jiawei He (\@h-jiawei), Jinyang (\@jinyangyuan), Joe (\@joekakone), Jon (\@jontonsoup4), Jonathan (\@JonathanRayner), Juanlu (\@Juanlu001), Juha (\@kamiju), kishan_kc (\@Kishan_KC), L. Jia (\@L.Jia), Legin (\@Legin2022), Lieven (\@lievengovaerts), Lin (\@Lin13579), Lockon (\@Lockon007), Lorenzoh (\@lorenzoh), Luohao (\@luohao-brian), Mahmood (\@mahmood-taghavi), Manimaran (\@Manimaran-T), Marc (\@MarcBresson), Mark (\@Mark1001), Martin (\@mjirik), Matthieu (\@matthieubulte), Mhamed (\@mhamedbenhadj), Michael (\@michael-fonder), Mieszko (\@Mieszko4), Minato (\@Minato-namikaze), Moises (\@moises-b2), MrDiver (\@MrDiver), Noam (\@noamgot), Notplus (\@notplus), OvO (\@OvO), Pan (\@pan-x-c), Phumtawan (\@Phumtawan), Piotrek (\@piotrek-k), prajwal_pitlehra (\@Prajwal_Pitlehra), Qaq (\@qaq), Ranis (\@ranisalt), Robert (\@robert-abc), Saketh (\@sakethram), Sam (\@SamFancher), Satvik (\@Satvik-Garg), Sepehr (\@sepehr-gh), Sepehr (\@sepehrnoey), Shichao (\@shichaoy), Shiva (\@shiva-sankaran), Smsxgz (\@smsxgz), Sravan (\@sravan-s), Stest2012 (\@stest2012), Tirth (\@Tirth27), Tushar (\@tusharmaurya), Verify (\@verify-code), Victor (\@victor-iyi), Wj (\@wj-Mcat), Xinyu (\@xinyu-dev), and Yibo (\@yibotongxue).

We also thank the following readers for their active engagement, feedback, and technical discussions on the forum and repository:

Amritansh (\@Amritansh), Angad (\@Angad_Sandhu), anh_tuan_bui (\@Anh_Tuan_Bui), arijit_das (\@Arijit_Das), Artyom (\@artyompal), carlos_arriaga (\@Carlos_Arriaga), Daniel, David (\@d_c), Fan (\@Fan_Gu), Hanker (\@Hanker), Igor (\@igormorgado), Jeet (\@Jeet), Jonathan (\@jonathan-code-r), mahmoud_taha (\@Mahmoud_Taha), Marios (\@marios_g), Mark (\@Mark_Hall), max_ah (\@Max_Ah), Maxe (\@Maxe), Mugen (\@Mugen), Nishant (\@Nishant_Thakur), Porter (\@Porter), Raj (\@Raj), Rling (\@Rling), and Sansui (\@Sansui).



:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/186)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17963)
:end_tab:

<!-- slides -->

::: {.slide title="The D2L Code Foundation"}
D2L notebooks use shared imports and helper functions from
the `d2l` package. Blocks marked with `#@save` are exported
there so later notebooks can focus on models instead of
repeating setup code. The header of the library is nothing more than
imports, with the module bound to the name `d2l`:

~~~python
#@save
import math
import sys
from matplotlib import pyplot as plt
...
d2l = sys.modules[__name__]
~~~
:::

::: {.slide title="Framework-Specific Imports"}
Each backend tab imports the tensor library, neural-network
layers, data transforms, and utilities used throughout the
book. The public `d2l` namespace stays consistent while the
implementation switches by tab.

@preface-code-2

:::

::: {.slide title="How to Read Later Notebooks"}
- `#@save` means the code becomes part of the reusable `d2l`
  library.
- `#@tab` marks backend-specific code for MXNet, PyTorch,
  TensorFlow, or JAX.
- Most chapters teach algorithms; this preface explains where
  the shared names and imports come from.
:::
