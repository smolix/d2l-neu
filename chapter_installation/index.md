# Installation
:label:`chap_installation`

The examples in this book are available as Jupyter notebooks for PyTorch, JAX,
TensorFlow, and MXNet. Each framework download contains the notebooks, their
saved outputs, the figures they reference, the matching `d2l` source code, and
pinned CPU and GPU environments. This makes a notebook bundle the shortest path
to running the book locally.

The downloads are ZIP archives rather than tar-gzipped files. Saved outputs can
be read without installing a framework. Running the notebooks requires Python,
Jupyter, and the selected framework; the included [uv](https://docs.astral.sh/uv/)
lock files install these together.

## Choose a Framework

We recommend PyTorch if you do not already have a framework preference. The
first several chapters run comfortably on a CPU. A GPU becomes useful for larger
convolutional networks, transformers, and the application chapters.

* [PyTorch notebooks](/notebooks/d2l-pytorch.zip)
* [JAX notebooks](/notebooks/d2l-jax.zip)
* [TensorFlow notebooks](/notebooks/d2l-tensorflow.zip)
* [MXNet notebooks](/notebooks/d2l-mxnet.zip)

The published bundles correspond to the version of the book on this website.
The latest source and development version are available from
[smolix/d2l-neu on GitHub](https://github.com/smolix/d2l-neu):

```bash
git clone https://github.com/smolix/d2l-neu.git
cd d2l-neu
```

The repository contains the book sources, build tools, framework environments,
and the current `d2l` package. For reading and executing notebooks, the
framework-specific ZIP above is smaller and requires fewer build tools.

## Install uv

[uv](https://docs.astral.sh/uv/getting-started/installation/) manages both the
Python interpreter and the packages used by the notebooks. Conda is not needed.
On macOS or Linux, install uv with its standalone installer or a system package
manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell, use:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal if the installer updates your `PATH`, then check the
installation:

```bash
uv --version
```

The notebook environments use Python 3.12. uv downloads a suitable interpreter
when one is not already available.

## Download and Unpack the Notebooks

Download one framework archive from the links above and unpack it. For example,
for PyTorch on macOS or Linux:

```bash
curl -LO https://d2l.smola.org/notebooks/d2l-pytorch.zip
unzip d2l-pytorch.zip
cd d2l-pytorch
```

Windows can extract the ZIP from File Explorer or PowerShell. Run the remaining
commands from the extracted directory, which contains `pyproject.toml`, the
`d2l/` package, and the uv lock files.

## CPU Environment

The CPU environment is the recommended starting point. Create an isolated
Python 3.12 environment and synchronize it with the CPU lock:

```bash
uv venv --python 3.12
uv pip sync pylock.cpu.toml
```

For MXNet on Apple Silicon, use its separate macOS lock:

```bash
uv pip sync pylock.cpu-macos.toml
```

The MXNet Linux CPU lock uses the Linux/x86-64 wheel distributed with the
[smolix/mxnet release](https://github.com/smolix/mxnet/releases/tag/mxnet-2.0.0.zombie.1).
The other CPU locks select platform-appropriate packages where their frameworks
provide them.

Start JupyterLab in the environment selected by the lock file:

```bash
uv run --no-sync jupyter lab
```

JupyterLab normally opens a browser automatically. Otherwise, open the local URL
printed in the terminal. The `--no-sync` option tells uv to preserve the exact
CPU or GPU environment selected with `uv pip sync`.

## GPU Environment

The included GPU locks reproduce the accelerator environments used for the
book. They target Linux on x86-64 with an NVIDIA GPU. Before installing one,
check that the NVIDIA driver is visible:

```bash
nvidia-smi
```

From the extracted notebook directory, create the environment if necessary and
synchronize the GPU lock:

```bash
uv venv --python 3.12
uv pip sync pylock.gpu.toml
uv run --no-sync jupyter lab
```

Synchronizing a GPU lock after a CPU lock replaces packages as needed; a second
virtual environment is optional. The locks currently select the following
accelerator stacks:

* PyTorch: CUDA 12.8 wheels.
* JAX: the CUDA 12 pip distribution.
* TensorFlow: the `and-cuda` distribution and its CUDA libraries.
* MXNet: the CUDA 13.3 Blackwell build from `smolix/mxnet`.

GPU software is sensitive to the operating system, driver, and device. If the
provided lock does not match your machine, use the framework's current
installation guide for [PyTorch](https://pytorch.org/get-started/locally/),
[JAX](https://docs.jax.dev/en/latest/installation.html), or
[TensorFlow](https://www.tensorflow.org/install/pip). The CPU lock remains useful
for the early chapters. A hosted notebook or GPU server is often more convenient
for later, compute-intensive chapters.

## Verify the Installation

The following commands report the installed framework version and whether it
can see an accelerator.

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

A CPU installation normally reports no GPU; that is not an error.

## Data, Models, and Disk Space

The notebook ZIP contains the notebooks, saved cell outputs, and book figures.
It does not contain every training dataset or pretrained model. Notebooks fetch
these assets on first use and cache them in a data or framework cache directory.
Consequently, executing data-dependent notebooks requires an internet
connection and additional disk space even though reading their saved outputs
does not.

Some later examples also require more memory or execution time than a typical
laptop provides. Begin with the CPU environment and the early chapters, then
move to the GPU environment or a remote machine when the workload warrants it.
