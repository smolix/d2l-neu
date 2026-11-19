# Installation
:label:`chap_installation`

To get the most out of this book, you should run the code, modify it, and experiment with it first-hand. Every chapter is available as an executable Jupyter notebook for PyTorch, JAX, TensorFlow, and MXNet. To make local setup as frictionless as possible, we distribute a self-contained `.zip` bundle for each framework (available from the **Notebooks** menu at the top of the page). Each bundle includes all of the notebooks---already populated with their saved outputs and figures so you can browse them immediately---along with the matching `d2l` library source code and pinned [uv](https://docs.astral.sh/uv/) lock files for both CPU and GPU environments.

While you can read the pre-computed outputs in the notebooks without installing any deep learning libraries, executing and modifying the code requires Python, JupyterLab, and your chosen framework. The included `uv` lock files install a complete, tested environment in a couple of commands.

## Choosing a Framework

If you do not already have a strong preference, we recommend starting with **PyTorch**, which is the one framework that covers every chapter of the book. The Basics part (:numref:`chap_preliminaries` through :numref:`chap_rnn`) and the Mathematics for Deep Learning appendix are implemented in all four frameworks (PyTorch, JAX, TensorFlow, and MXNet), while the Advanced part (beginning with :numref:`chap_optimization`) is implemented in PyTorch and JAX.

You do not need specialized hardware to begin: the first several chapters run comfortably on any modern laptop CPU. A GPU becomes valuable later on when we train larger convolutional networks, transformers, and generative models.

The downloadable notebook bundles match the published edition of the book on this website. If you want to explore the bleeding-edge source files, build scripts, or development environments, you can also clone the repository directly from [GitHub](https://github.com/smolix/d2l-neu):

```bash
git clone https://github.com/smolix/d2l-neu.git
cd d2l-neu
```

For simply reading and running the notebooks, however, the framework-specific `.zip` bundle is much smaller and requires far fewer build dependencies.

## Installing `uv`

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage both the Python interpreter and the package dependencies for the notebooks, so you do not need Conda or a pre-existing Python installation. On macOS or Linux, install `uv` using its standalone installer (or your system package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell, run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If the installer updates your `PATH`, open a fresh terminal window and verify that `uv` is available:

```bash
uv --version
```

Our notebook environments target Python 3.12; if Python 3.12 is not already present on your machine, `uv` will download and manage an isolated interpreter automatically.

## Downloading and Unpacking the Notebooks

Download the archive for your chosen framework from the **Notebooks** menu at the top of the page and extract it. For example, to unpack the PyTorch bundle on macOS or Linux, run the following commands in your download directory:

```bash
unzip d2l-pytorch.zip
cd d2l-pytorch
```

On Windows, you can extract the `.zip` archive using File Explorer or PowerShell. Run all subsequent commands from inside the extracted folder, which contains `pyproject.toml`, the `d2l/` package source, and the `uv` lock files.

## Setting Up a CPU Environment

We recommend starting with the CPU environment to verify your setup quickly. Inside the extracted notebook directory, create an isolated Python 3.12 virtual environment and synchronize it with the pinned CPU lock file:

```bash
uv venv --python 3.12
uv pip sync pylock.cpu.toml
```

If you are running MXNet on an Apple Silicon Mac, synchronize against its dedicated macOS lock file instead:

```bash
uv pip sync pylock.cpu-macos.toml
```

For MXNet on Linux, `pylock.cpu.toml` uses the Linux `x86-64` wheel distributed with the [smolix/mxnet release](https://github.com/smolix/mxnet/releases/tag/mxnet-2.0.0.zombie.1). Note that MXNet builds are available only for Linux (`x86-64`) and macOS (Apple Silicon); on Windows, please use PyTorch, JAX, or TensorFlow, or run the notebooks in a hosted cloud environment. For PyTorch, JAX, and TensorFlow, `pylock.cpu.toml` automatically selects the appropriate packages for your operating system and architecture.

Once the environment is synchronized, launch JupyterLab:

```bash
uv run --no-sync jupyter lab
```

JupyterLab will normally open a tab in your default web browser automatically; if it does not, copy and paste the local `http://localhost:...` URL printed in your terminal. Passing `--no-sync` tells `uv` to run directly inside the exact CPU or GPU environment you just synchronized with `uv pip sync`.

## Setting Up a GPU Environment

The included GPU lock files reproduce the hardware-accelerated environments used to execute this book. They target Linux (`x86-64`) machines equipped with an NVIDIA GPU. Before installing the GPU environment, confirm that your NVIDIA driver is active and visible:

```bash
nvidia-smi
```

From the extracted notebook directory, create the virtual environment (if you have not already) and synchronize it with the GPU lock file:

```bash
uv venv --python 3.12
uv pip sync pylock.gpu.toml
uv run --no-sync jupyter lab
```

You do not need a separate virtual environment to switch from CPU to GPU: running `uv pip sync pylock.gpu.toml` in an existing environment cleanly replaces the CPU packages with their GPU counterparts. The GPU lock files pin the following accelerator stacks:

* **PyTorch:** CUDA 12.8 wheels.
* **JAX:** the CUDA 12 pip distribution.
* **TensorFlow:** the `and-cuda` distribution with its bundled CUDA libraries.
* **MXNet:** the CUDA 13.3 Blackwell build from `smolix/mxnet`.

Because GPU software stacks depend on your operating system, kernel driver, and GPU architecture, the pinned lock file may occasionally differ from what your specific machine requires. If `pylock.gpu.toml` does not match your hardware or OS, consult the official installation guide for [PyTorch](https://pytorch.org/get-started/locally/), [JAX](https://docs.jax.dev/en/latest/installation.html), or [TensorFlow](https://www.tensorflow.org/install/pip). You can always use the CPU lock file for the early chapters and switch to a hosted notebook (:numref:`sec_hosted_notebooks`) or a cloud GPU instance (:numref:`sec_cloud_instances`) when you reach the more compute-intensive chapters later in the book.

## Verifying the Installation

You can verify that your framework is installed properly---and check whether it detects a GPU accelerator---with a one-line command:

:begin_tab:`pytorch`

```bash
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

:end_tab:

:begin_tab:`jax`

```bash
uv run --no-sync python -c "import jax; print(jax.__version__); print(jax.devices())"
```

:end_tab:

:begin_tab:`tensorflow`

```bash
uv run --no-sync python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

:end_tab:

:begin_tab:`mxnet`

```bash
uv run --no-sync python -c "import mxnet as mx; print(mx.__version__); print(mx.context.num_gpus())"
```

:end_tab:

Note that a CPU-only installation will report `False`, a CPU device, `[]`, or `0` GPUs; this is completely normal when running without an NVIDIA accelerator.

## The Standard Imports
:label:`sec_preamble`

Every notebook in this book begins with a standard preamble cell that imports the core libraries used throughout the text. Across all four tabs, this cell imports Python's built-in `math` module for scalar functions and constants, NumPy under its standard alias `np` for CPU arrays, and the framework-specific backend of our companion library as `d2l`. Only the deep learning framework imports differ from one tab to the next:

:begin_tab:`pytorch`

```py
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

Here `torch` is the core PyTorch package. Its `nn` module provides neural network layers and loss functions as stateful classes, while `F` (`torch.nn.functional`) exposes the corresponding operations as stateless functions such as `F.relu` and `F.cross_entropy`.
:end_tab:

:begin_tab:`jax`

```py
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

Here `jax` provides function transformations (such as `jax.grad` and `jax.jit`) and explicit PRNG key management, while `jnp` exposes its accelerator-backed, NumPy-compatible array interface. We build neural network layers and models with Flax's `nnx` module and train them using optimizers and schedules from `optax`.
:end_tab:

:begin_tab:`tensorflow`

```py
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

Here TensorFlow is imported under its customary alias `tf`, with its high-level neural network layers and model abstractions residing in `tf.keras`.
:end_tab:

:begin_tab:`mxnet`

```py
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

Because MXNet's array module mirrors the NumPy API, we import it as `mnp` so that `np` unambiguously refers to standard NumPy across every framework tab. The `npx` module adds deep learning extensions, `autograd` records computation graphs for automatic differentiation, `gluon` and `nn` provide neural network layers, and `init` supplies parameter initializers. Calling `npx.set_np()` activates MXNet's NumPy-compatible array semantics.
:end_tab:

The final import binds the framework-specific backend of the book's companion package (`d2l.torch`, `d2l.jax`, `d2l.tensorflow`, or `d2l.mxnet`) to the uniform name `d2l`. Whenever a section defines and explains a reusable helper function or class marked with `#@save`, it is exported into `d2l` so later chapters can call it cleanly without repeating the definition.

To keep the book uncluttered, we print this standard import cell only once, right here. The downloadable and hosted Jupyter notebooks include it at the top of every file so that each notebook runs standalone, whereas the web and PDF editions begin each section directly with the code specific to that topic. Whenever a notebook requires an additional library (such as Gymnasium in the reinforcement learning chapters) or needs to configure compiler flags before initializing the framework, we display that extra setup cell explicitly. Plotting requires no special boilerplate: Jupyter renders figures inline by default.

## Data, Models, and Disk Space

While the notebook `.zip` archive includes all of the notebooks, pre-computed cell outputs, and figures, it does not bundle large training datasets or pretrained model weights. Instead, notebooks download those assets automatically on first use and store them in a local cache directory. As a result, you can read the notebooks and inspect their saved outputs completely offline, but executing data-dependent notebooks requires an active internet connection and additional disk space.

Finally, keep in mind that some advanced chapters later in the book require more memory or compute than a typical laptop provides. Start with the CPU environment for the foundational chapters, and transition to a local GPU or a cloud machine once the workloads demand it. If you are considering purchasing dedicated hardware for deep learning, :numref:`sec_hardware_buyers` offers practical guidance on what to look for.
