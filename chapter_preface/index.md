# Preface

## Preface to the Second Edition

Looking back at the first edition, it feels like a different world when it was written. Text, images and code were produced by hand, style guides were zealously enforced by coauthors, and the addition of a new framework required the talented hand of an expert, in this case Anirudh Dagar (PyTorch and JAX) and Yuan Tang (TensorFlow). As the book grew beyond 1,000 pages, hundreds of GitHub issues, PRs and Discourse comments, keeping it all up to date became a full time job. As such, when all of the authors moved on to different places, the book sat there for half a decade. Many of its premises, such as accessible code, teaching by example, and the notebook-based format, have stood the test of time. They are as fresh as in 2020. Approximately half of its content has aged well, while other parts have since been superseded by much more exciting problems, techniques and applications. Very little of the code, beyond the D2L library format, survived particularly well, since we have seen tremendous progress in hardware and the associated software tooling.

Two things happened in 2026 that compelled me to pick up this project again and see whether D2L could be 'refurbished' again. Firstly, I received a request from a Greek publisher, whether I would consent to the book being translated and printed in Greek. This told me that, even 5 years after we last improved it, the book still had value to the point of someone wanting to print a genuinely new copy. Secondly, as we have all seen with the rapid progress of agentic AI, the tools from Anthropic, OpenAI, and Google have made it dramatically easier to perform broad updates and repairs that would have been unaffordable previously. It allowed me to bring the code up to date relative to modern frameworks, incorporate the GitHub issues and PRs that were pending, review the bug reports and comments in the Discourse threads, and perform a separate code, language, experiment and math consistency analysis and fix. The build system was revamped, the book now uses [Quarto](https://quarto.org) and [uv](https://docs.astral.sh/uv/), the fora are on [Discourse](https://d2l.discourse.group)'s own servers, and the content format was improved to allow for the separation between code execution and editing. Having agents also allowed me to ensure that all frameworks perform similarly on the same problems, i.e. no more performance regressions due to poorly used JIT compilation, differences in initialization, optimization schedules, or the use of 'magic' random seeds. This is a luxury that would have been unaffordable previously, yet I believe that it significantly improves the teaching quality, as it shows best practice of how to use a framework throughout.

Besides all the necessary plumbing improvements, the content has gone through a number of changes. Firstly, the book is too long to be viewed as one piece (it weighs in at over 3,000 pages as a PDF). The solution was to break it up into multiple pieces. [Basics](../chapter_basics/index.qmd) covers, well, the basics of deep learning and is a great companion for an introductory course. [Advanced](../chapter_advanced/index.qmd) takes you close to the current state of the art in many areas, including optimization (:numref:`chap_optimization`), state space models (:numref:`chap_modern_rnn`), transformers (:numref:`chap_transformers`), reinforcement learning (:numref:`chap_reinforcement_learning` and :numref:`chap_deep_rl`), GANs (:numref:`chap_gans`) and diffusion models (:numref:`chap_diffusion`), without going into too much depth in any one of them. The latter is impossible, due to the information explosion in each of the fields. [Mathematics for Deep Learning](../chapter_mathematics-for-deep-learning/index.qmd) has grown significantly to cover more about diffusion and flows (:numref:`chap_mdl-dynamics`), information theory (:numref:`chap_mdl-information-theory`) and optimization (:numref:`chap_mdl-optimization`). As such, it would well serve as a textbook for an introductory math course in this field and it helps cover the necessary background that is often taken for granted when reading papers (or some of the chapters of this book). I am grateful to Rasool Fakoor for contributing the reinforcement learning and diffusion chapters (:numref:`chap_reinforcement_learning`, :numref:`chap_deep_rl` and :numref:`chap_diffusion`).

A lot of work remains to be done. The [Language Models](../chapter_language-models/index.qmd) part is still unwritten, no content covers computer vision, graphics, audio, speech, semi-supervised learning, or provides a detailed treatment of agents. With a bit of time, plenty of tokens and generous amounts of curation, the latter (the agents) will aid me in completing the former (the book). This should also help keep the book up to date, whenever new scientific discoveries happen and new techniques are invented.

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

This book represents our attempt to make deep learning approachable,
teaching you the *concepts*, the *context*, and the *code*.

### One Medium Combining Code, Math, and HTML

For any computing technology to reach its full impact,
it must be well understood, well documented, and supported by
mature, well-maintained tools.
The key ideas should be clearly distilled,
minimizing the onboarding time needed
to bring new practitioners up to date.
Mature libraries should automate common tasks,
and exemplar code should make it easy for practitioners
to modify, apply, and extend common applications to suit their needs.


As an example, take dynamic web applications.
Despite a large number of companies, such as Amazon,
developing successful database-driven web applications in the 1990s,
the potential of this technology to aid creative entrepreneurs
was realized to a far greater degree only in the past ten years,
owing in part to the development of powerful, well-documented frameworks.


Testing the potential of deep learning presents unique challenges
because any single application brings together various disciplines.
Applying deep learning requires simultaneously understanding
(i) the motivations for casting a problem in a particular way;
(ii) the mathematical form of a given model;
(iii) the optimization algorithms for fitting the models to data;
(iv) the statistical principles that tell us
when we should expect our models
to generalize to unseen data
and practical methods for certifying
that they have, in fact, generalized;
and (v) the engineering techniques
required to train models efficiently,
navigating the pitfalls of numerical computing
and getting the most out of available hardware.
Teaching the critical thinking skills
required to formulate problems,
the mathematics to solve them,
and the software tools to implement those solutions
all in one place presents formidable challenges.
Our goal in this book is to present a unified resource
to bring would-be practitioners up to speed.

When we started this book project,
there were no resources that simultaneously
(i) remained up to date;
(ii) covered the breadth of modern machine learning practices
with sufficient technical depth;
and (iii) interleaved exposition of
the quality one expects of a textbook
with the clean runnable code
that one expects of a hands-on tutorial.
We found plenty of code examples illustrating
how to use a given deep learning framework
(e.g., how to do basic numerical computing with matrices in TensorFlow)
or for implementing particular techniques
(e.g., code snippets for LeNet, AlexNet, ResNet, etc.)
scattered across various blog posts and GitHub repositories.
However, these examples typically focused on
*how* to implement a given approach,
but left out the discussion of
*why* certain algorithmic decisions are made.
While some interactive resources
have popped up sporadically
to address a particular topic,
e.g., the engaging blog posts
published on the website [Distill](http://distill.pub), or personal blogs,
they only covered selected topics in deep learning,
and often lacked associated code.
On the other hand, while several deep learning textbooks
have emerged---e.g., :citet:`Goodfellow.Bengio.Courville.2016`,
which offers a comprehensive survey
on the basics of deep learning---these
resources do not marry the descriptions
to realizations of the concepts in code,
sometimes leaving readers clueless
as to how to implement them.
Moreover, too many resources
are hidden behind the paywalls
of commercial course providers.

We set out to create a resource that could
(i) be freely available for everyone;
(ii) offer sufficient technical depth
to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(iii) include runnable code, showing readers
*how* to solve problems in practice;
(iv) allow for rapid updates, both by us
and also by the community at large;
and (v) be complemented by a [forum](https://d2l.discourse.group/c/5)
for interactive discussion of technical details and to answer questions.

These goals were often in conflict.
Equations, theorems, and citations
are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the Internet as a website.
No workflows seemed suited to these demands,
so we decided to assemble our own (:numref:`sec_how_to_contribute`).
We settled on GitHub to share the source
and to facilitate community contributions;
Markdown sources, from which we generate Jupyter notebooks,
for mixing code, equations and text;
Quarto as a rendering engine;
and Discourse as a discussion platform.
While our system is not perfect,
these choices strike a compromise
among the competing concerns.
We believe that *Dive into Deep Learning*
might be the first book published
using such an integrated workflow.


### Learning by Doing

Many textbooks present concepts in succession,
covering each in exhaustive detail.
For example,
the excellent textbook of
:citet:`Bishop.2006`
teaches each topic so thoroughly
that getting to the chapter
on linear regression requires
a nontrivial amount of work.
While experts love this book
precisely for its thoroughness,
for true beginners, this property limits
its usefulness as an introductory text.

In this book, we teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric concepts.

Aside from a few preliminary notebooks that provide a crash course
in the basic mathematical background,
each subsequent chapter both introduces a reasonable number of new concepts
and provides several self-contained working examples, using real datasets.
This presented an organizational challenge.
Some models might logically be grouped together in a single notebook.
And some ideas might be best taught
by executing several models in succession.
By contrast, there is a big advantage to adhering
to a policy of *one working example, one notebook*:
This makes it as easy as possible for you to
start your own research projects by leveraging our code.
Just copy a notebook and start modifying it.

Throughout, we interleave the runnable code
with background material as needed.
In general, we err on the side of making tools
available before explaining them fully
(often filling in the background later).
For instance, we might use *stochastic gradient descent*
before explaining why it is useful
or offering some intuition for why it works.
This helps to give practitioners the necessary
ammunition to solve problems quickly,
at the expense of requiring the reader
to trust us with some curatorial decisions.

This book teaches deep learning concepts from scratch.
Sometimes, we delve into fine details about models
that would typically be hidden from users
by modern deep learning frameworks.
This comes up especially in the basic tutorials,
where we want you to understand everything
that happens in a given layer or optimizer.
In these cases, we often present
two versions of the example:
one where we implement everything from scratch,
relying only on NumPy-like functionality
and automatic differentiation,
and a more practical example,
where we write succinct code
using the high-level APIs of deep learning frameworks.
After explaining how some component works,
we rely on the high-level API in subsequent tutorials.


### Content and Structure

Three short chapters come first.
:numref:`chap_installation` sets up an environment
for running the code,
:numref:`chap_notation` collects the notation,
and :numref:`chap_introduction`
gives an overview of deep learning.
The rest of the book is organized in parts,
the two largest being Basics and Advanced
(:numref:`fig_book_org`).

![Structure of the book. An arrow from one chapter to another means that the second builds on the first; the arrow from the Basics to the Advanced part means that every Advanced chapter builds on the Basics. The Language Models, Image Models, and Attic parts are not shown.](../img/front-book-org.svg)
:label:`fig_book_org`


* **Basics**.
In :numref:`chap_preliminaries`,
we quickly bring you up to speed
on the prerequisites for hands-on deep learning:
how to store and manipulate data as tensors,
how to differentiate with them,
and how to reason about uncertainty.
:numref:`chap_regression` and :numref:`chap_classification`
fit linear models to data
and introduce the ingredients
of nearly every model in this book:
a model with parameters,
a loss that measures its errors,
and an algorithm that reduces the loss.
:numref:`chap_perceptrons` adds hidden layers
and asks why a network with many parameters
can still generalize.
:numref:`chap_computation` shows how models
are assembled from modules,
initialized, saved, and run on a GPU.
The last three chapters build what we know about the data into the model.
The convolutional neural networks (CNNs)
of :numref:`chap_cnn` and :numref:`chap_modern_cnn`
exploit the spatial structure of images,
and the sequence models of :numref:`chap_rnn`
the order of the words in a text.
The first five chapters are best read in order.
After them, the chapters on images and on sequences
do not depend on each other.

* **Advanced**.
These chapters take the tools of the Basics
to the models in use today.
They can be read in several orders;
the arrows in :numref:`fig_book_org`
show which chapters build on which.
The Basics use optimizers, learning rate schedules,
and batch sizes as recipes.
:numref:`chap_optimization` explains why they work
and what must change as models grow.
:numref:`chap_attention` introduces *attention*,
which lets each position in a sequence
retrieve what it needs from any other,
and :numref:`chap_transformers` stacks attention layers
into the transformers behind most large language models.
:numref:`chap_modern_rnn` returns to recurrence
with state space models,
which keep a state of fixed size
yet train on all positions of a sequence at once.
:numref:`chap_performance` asks what limits the speed
of a training step on one GPU and on several,
using a transformer as its running example.
:numref:`chap_reinforcement_learning` and :numref:`chap_deep_rl`
turn to agents that learn from rewards rather than labels,
first on problems small enough to solve exactly,
then with neural networks.
Finally, :numref:`chap_gans` and :numref:`chap_diffusion`
generate new data:
by a game between a generator and a critic,
and by learning to remove noise.

* **Language Models, Image Models, and Attic**.
Three shorter parts follow.
Language Models (:numref:`chap_nlp_pretrain` and :numref:`chap_nlp_app`)
pretrains text representations and fine-tunes them for downstream tasks,
Image Models (:numref:`chap_cv`)
covers object detection, segmentation, and style transfer,
and the Attic collects Gaussian processes,
hyperparameter optimization, and recommender systems
(:numref:`chap_gp` through :numref:`chap_recsys`).
We plan to rewrite the first two.

* **Mathematics for Deep Learning**.
This appendix collects the mathematics behind the main text.
The main text states each result it uses, with its intuition,
and points to the appendix for proofs and derivations.
:numref:`chap_mdl-linear-algebra` asks what a matrix does
to the vectors it acts on:
projections, eigendecompositions,
and the singular value decomposition that splits any matrix
into rotations and stretches.
:numref:`chap_mdl-calculus` explains how far a gradient can be trusted
and how automatic differentiation computes every slope
of a network for the cost of a few forward passes.
:numref:`chap_mdl-optimization` proves what :numref:`chap_optimization`
shows by experiment:
why the learning rate has a ceiling,
when convexity guarantees a global optimum,
and what noisy gradients cost.
:numref:`chap_mdl-probability-statistics` asks how far a measured
test accuracy can be from the truth,
and covers distributions, maximum likelihood, Bayesian inference,
and generalization bounds.
:numref:`chap_mdl-information-theory` explains what a cross-entropy measures
and how else two distributions can be compared,
from the KL divergence to optimal transport.
:numref:`chap_mdl-dynamics` treats differential equations,
deterministic and stochastic,
and shows how running a noising process backward
turns noise into data,
the mathematics of diffusion and flow models.
If you already know a topic, you can skip its chapter;
if you do not, the appendix serves as a first textbook.

* **Tools for Deep Learning**.
:numref:`chap_appendix_tools` covers the practical side:
running the notebooks locally or in a hosted environment,
renting or buying hardware,
scaling training, serving models,
and contributing to this book.
It also documents the `d2l` library (:numref:`sec_d2l`).


### Code
:label:`sec_code`

Most sections of this book feature executable code.
We believe that some intuitions are best developed
via trial and error,
tweaking the code in small ways and observing the results.
Ideally, an elegant mathematical theory might tell us
precisely how to tweak our code to achieve a desired result.
However, deep learning practitioners today
must often tread where no solid theory provides guidance.
Despite our best attempts, formal explanations
for the efficacy of various techniques are
still lacking, for a variety of reasons: the mathematics to characterize these models
can be so difficult;
the explanation likely depends on properties
of the data that currently lack clear definitions;
and serious inquiry on these topics
has only recently kicked into high gear.
We are hopeful that as the theory of deep learning progresses,
each future edition of this book will provide insights
that eclipse those presently available.

The code comes in up to four frameworks,
each shown in its own tab.
The Basics part and the mathematics appendix
are implemented in PyTorch, JAX, TensorFlow, and MXNet,
and the Advanced part in PyTorch and JAX.
:numref:`chap_installation` helps you choose a framework
and set up an environment for it.
Every notebook begins with the same cell of standard imports,
which :numref:`sec_preamble` explains.
The book shows that cell only there,
so each section can start with the code that is particular to it.

To avoid unnecessary repetition, we collect
the functions and classes that later sections reuse
in the `d2l` library.
Throughout, we mark blocks of code
(such as functions, classes,
or collections of import statements) with `#@save`.
Such a block is defined and explained once,
and later sections call it through `d2l`
instead of defining it again.
We document these classes and functions in :numref:`sec_d2l`.
The `d2l` library is lightweight.
Its framework-independent part needs only the following imports:

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
You are reading the MXNet version of this book.
MXNet code is available for the Basics part
and the mathematics appendix.
The Advanced part, beginning with :numref:`chap_optimization`,
is implemented in PyTorch and JAX,
so switch to one of their tabs there.
We run the code with the MXNet 2.0 build
pinned in the environments of :numref:`chap_installation`.
If some code fails with another version,
please use that environment.
The MXNet version of the `d2l` library adds the following imports.
:end_tab:

:begin_tab:`pytorch`
You are reading the PyTorch version of this book,
which covers every chapter.
We run the code with the PyTorch version
pinned in the environments of :numref:`chap_installation`.
Deep learning frameworks change quickly,
so if some code fails with a newer version,
please use that environment.
The PyTorch version of the `d2l` library adds the following imports.
:end_tab:

:begin_tab:`tensorflow`
You are reading the TensorFlow version of this book.
TensorFlow code is available for the Basics part
and the mathematics appendix.
The Advanced part, beginning with :numref:`chap_optimization`,
is implemented in PyTorch and JAX,
so switch to one of their tabs there.
We run the code with the TensorFlow version
pinned in the environments of :numref:`chap_installation`.
Deep learning frameworks change quickly,
so if some code fails with a newer version,
please use that environment.
The TensorFlow version of the `d2l` library adds the following imports.
:end_tab:

:begin_tab:`jax`
You are reading the JAX version of this book,
which covers the Basics part, the Advanced part,
and the mathematics appendix.
JAX transforms ordinary numerical Python functions:
it differentiates them, compiles them, and vectorizes them.
We define models with the `nnx` module of Flax
and train them with the optimizers of Optax.
We run the code with the JAX version
pinned in the environments of :numref:`chap_installation`.
Deep learning frameworks change quickly,
so if some code fails with a newer version,
please use that environment.
The JAX version of the `d2l` library adds the following imports.
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

This book is for students (undergraduate or graduate),
engineers, and researchers, who seek a solid grasp
of the practical techniques of deep learning.
Because we explain every concept from scratch,
no previous background in deep learning or machine learning is required.
Fully explaining the methods of deep learning
requires some mathematics and programming,
but we will only assume that you enter with some basics,
including modest amounts of linear algebra,
calculus, probability, and Python programming.
Just in case you have forgotten anything,
the [Mathematics for Deep Learning](../chapter_mathematics-for-deep-learning/index.qmd) appendix provides a refresher
on most of the mathematics
you will find in this book.
Usually, we will prioritize
intuition and ideas
over mathematical rigor.
If you would like to extend these foundations
beyond the prerequisites to understand our book,
we happily recommend some other terrific resources:
*Linear Analysis* by :citet:`Bollobas.1999`
covers linear algebra and functional analysis in great depth.
*All of Statistics* :cite:`Wasserman.2013`
provides a marvelous introduction to statistics.
Joe Blitzstein's [books](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918)
and [courses](https://projects.iq.harvard.edu/stat110/home)
on probability and inference are pedagogical gems.
And if you have not used Python before,
you may want to peruse this [Python tutorial](http://learnpython.org/).


### Notebooks, Website, GitHub, and Forum

All our notebooks can be downloaded
from the [D2L.ai website](https://d2l.ai)
and from [GitHub](https://github.com/smolix/d2l-neu).
Associated with this book we have launched a discussion forum
at [d2l.discourse.group](https://d2l.discourse.group/c/5).
Whenever you have questions on any section of the book,
you can find a link to the associated discussion page
at the end of each notebook.



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

For the second edition, we are deeply grateful to the community members who
reported errata, proposed mathematical refinements, submitted pull requests,
and contributed bug fixes on the discussion forum and in the GitHub repository.
We thank every contributor whose pull requests, issue reports, and forum
suggestions were incorporated into this edition
(GitHub or forum handles in parentheses):

Aaron_L, abhiqaz18, Abhishek (\@abhi-cser), Adam Johnson (\@Calcu-dev), Amir Taj (\@ahttajdini), Amit Kumar (\@Amit_Kumar), Ang (\@kianmeng), Anirudh Dagar (\@anirudh), Archit Singhal (\@singhalarchit), Bailey Lissington (\@llamington), Bcghost, Bharath Valaboju (\@Bharath314), Bálint Mucsányi (\@MucsanyiBalint), caydenwei, Changyu (\@gcy), Chen Li (Charry) (\@CharryLee0426), Daniel Sinkin (\@Daniel-Sinkin), Davit Gogolashvili (\@Davit1123), Denis (\@Denis_Kazakov), DEVANSH GAJJAR (\@DEVANSH-GAJJAR), dingcurie, djholty, dr-neil, ducatyb, Ethan (\@lyconghk), euwyngoh, Fabián Giana (\@washiloo), Faerryn, flyingmars (\@2892510130), Gabriel Godefroy, GFlow, Gianpi Colonna (\@gcolonna12), gui1223, Haricharan0311, Huy (\@huyndao), hymalaya, Irma Ravkic (\@Irma_Ravkic), ishaanharry, Jingguo Yao (\@yaojingguo), jkp (\@jkpjkpjkp), Jona Mars, Jose Robledo (\@jorobledo), jueljust, kamesh, Krishnnan V (\@skrishnanv), leis, Leo Wu (\@Leo_Wu), Leslie (\@wsehjk), Liam, Liming Wang (\@LimingW), Ludger Starke (\@Ludger_Starke), Michael Mickelson (\@mickelsonmichael), Morey, Morteza, ms (\@MarianSlassi), Naimesh Chaudhari (\@nchaudh03), Nathan-McJames, Nick McGreivy, Nicolas Corthorn (\@nico-corthorn), ning ke (\@ning_ke), Omri Berman (\@Omri_Berman), Orkunozturk, peng, Peng Bai (\@baip), Pratik Srivastav (\@Pratik_Srivastav), Raj Kane (\@rajrkane), Rajan Adhikari (\@jugalraj), rdong8, rezahabibi96, Ronaldpan, rrrrp, Ruiyang Sun (Ryan Suen, Ryan Soon) (\@ruiyangsun-z), SangHyun Park (\@hy38), Shishir Singh (\@Shishir), Steven Jokes, sysy (\@AAtomical), Taksh Kothari (\@chessing234), Tario You (\@tario-you), Tianqi Zhu (\@Tianqi_Zhu), Timur713, Todd Morrill, Veryyoung957, vlad-ulmeanu01, wahab. (\@000wahab000), WEN, wsuchy, Xiaojinghu93, Yuan Tang (\@terrytangyuan), Yuntian Xiao (\@OneCoin123), zainabtareen, zemelee, Zhiwen Zhou (\@zhouzhiwen2000), and 牟良威 (\@whiteode).

We also thank the readers whose questions and suggestions on the discussion
forum helped improve the second edition:

Aaron (\@aaronsl), Aayush Shah (\@AayushSameerShah), Abhijeet Gulati (\@AbhijeetSG), Chao Wang (\@Chao_Wang), chris elgoog (\@chris_elgoog), Codebat, dagarwal, Ilya (\@Def255), Inzamam Anwar, Jian Chen (\@Jian_Chen), Kupo Kupu, Le Zhang (\@lezhang-thu), Libor Burian (\@burny), Luis (\@lregistros), Michael Ma (\@OptimusT), Nick (\@NickYi1990), nick t (\@six), R8a, S_X, Seungwoo Kim (\@seungwoos), Shengwu Yang (\@ysw), ss007k, Vinay Kumar (\@imflash217), Zhangjiekui, 王 阳 (\@LilNeo), and 立青-Lam (\@JH.Lam).


## Summary

Deep learning has revolutionized pattern recognition,
introducing technology that now powers a wide range of  technologies,
in such diverse fields as computer vision,
natural language processing,
and automatic speech recognition.
To successfully apply deep learning,
you must understand how to cast a problem,
the basic mathematics of modeling,
the algorithms for fitting your models to data,
and the engineering techniques to implement it all.
This book presents a comprehensive resource,
including prose, figures, mathematics, and code, all in one place.



## Exercises

1. Register an account on the discussion forum of this book [d2l.discourse.group](https://d2l.discourse.group/).
1. Install Python on your computer.
1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community.

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
