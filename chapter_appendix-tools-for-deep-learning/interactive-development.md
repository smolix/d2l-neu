# Notebooks
:label:`sec_interactive_development`

Every section of this book is an executable notebook in which prose,
mathematics, code, and output share one document. This format supports the
empirical study of deep learning: readers can train a model, vary the learning
rate, layer width, or amount of data, and inspect the result beside the
explanation. Notebooks are also widely used for data inspection and research
prototypes.

The notebooks are designed to be edited and rerun, including with variants
mentioned only in the text. This section explains local execution and
effective notebook use, especially *restart and run all*, which detects hidden
state and order-dependent results. If you
would rather not install anything, :numref:`sec_hosted_notebooks` shows how
to run the same notebooks free of charge on Colab or Kaggle; if you need more
hardware than your laptop offers, see :numref:`sec_cloud_instances`.

## Why Notebooks?

A plain Python script re-runs from the top every time, so a slow step — say,
loading and preprocessing a dataset — repeats on every experiment. A notebook
keeps a live Python process (the *kernel*) between executions: load the data
once, then iterate on the model as often as you like while the data stays in
memory. Inline plots and interleaved explanations make this execution model useful
for both teaching and research.

This convenience introduces persistent *state*, whose location and lifetime
must be understood.

### The Document and the Kernel

A notebook is really two things. The *document* stores cells and saved
outputs — it is what you read, share, and commit. The *kernel* holds the live
state: imported modules, variables, random-number generators, open files,
compiled programs, and accelerator memory. The document orders cells top to
bottom; the kernel only knows the order in which you *executed* them.
Confusing these two orders causes many notebook bugs.

![The document orders cells top to bottom; the kernel accumulates state in execution order. Restart and run all replays document order — the only order a reader can reproduce — and exposes the mismatch.](../img/tools-kernel-state.svg)
:label:`fig_tools_kernel_state`

:numref:`fig_tools_kernel_state` shows the classic failure. During an
editing session you executed a data-loading cell first, then wrote a
model-definition cell *above* it, and everything worked because the kernel
already held `data`. The document now tells a story that never happened. A
reader who runs the cells in the order shown — or you, tomorrow, after a
restart — hits a `NameError`.

The execution counters (`In [3]`, `In [1]`, …) number the cell executions
since the kernel started; they are historical clues, not a dependency
graph. The following pair of cells demonstrates benign, document-ordered
state: the second cell works only because the first ran.

```{.python .input #interactive-development-hidden-state-a}
import numpy as np

rng = np.random.default_rng(7)
samples = rng.normal(size=5)
```

```{.python .input #interactive-development-hidden-state-b}
float(samples.mean())
```

That dependency is fine, because it follows the reading order. A cell that
depends on an assignment *below* it, or on a cell you have since deleted, is
not. Keep cells small, keep function definitions separate from experiments,
and pass data explicitly rather than mutating globals across many cells.

### Restart and Run All

**Restart kernel and run all cells** discards the kernel's accumulated state
and replays the document from a blank slate, in reading order. It catches
hidden kernel state and order dependencies before a notebook is
shared. Run it before saving, sharing, or committing a notebook.
Every notebook in this book is built and tested with this procedure — the
outputs you see on the website are produced by a clean top-to-bottom run.

## Running the Book Locally

### Setting Up

Download the notebook archive described in :ref:`chap_installation`, or
clone the current sources from [GitHub](https://github.com/smolix/d2l-neu).
The archive includes CPU and GPU `uv` environment files. From the extracted
directory, create the environment once and launch JupyterLab through it:

```bash
uv sync --locked
uv run jupyter lab
```

Use the GPU lock file on a compatible NVIDIA system when the examples require
it. The *kernel* selected inside a
notebook must belong to the environment where the packages were installed. A
Jupyter server can see many kernels; its own Python process does not
determine which one executes your notebook.

### A Quick Sanity Check

Before changing code, record a compact identity check:

```{.python .input #interactive-development-identity}
import os
import platform
import sys

{
    "python": sys.executable.replace(os.path.expanduser("~"), "~"),
    "version": platform.python_version(),
    "working_directory": os.path.basename(os.getcwd()),
}
```

This catches the two most common setup mistakes in one cell: a kernel from
the wrong environment (`sys.executable` points somewhere unexpected) and a
notebook opened from a directory where relative data paths no longer resolve.
The cell abbreviates the home directory to `~` and prints only the last
component of the working directory, so that its saved output reveals no
personal path.

## Working in an Editor

### JupyterLab

[JupyterLab](https://jupyterlab.readthedocs.io/) combines a file browser,
notebook editor, terminals, a text editor, debugger support, and a view of
running kernels. The essentials are stable even as the interface evolves:

* Run a cell with `Shift+Enter`; use a terminal tab for `uv`, Git, and
  inspecting files.
* Select the kernel by environment name, then verify `sys.executable`.
* Interrupt a long computation before restarting. Restarting releases Python
  state and, normally, the accelerator memory owned by that process.
* Use **Restart Kernel and Run All Cells** before saving a result for others.
* Inspect the **Running** panel and stop kernels you no longer use — closing
  a browser tab does not necessarily stop its kernel, and an orphaned kernel
  can hold gigabytes of GPU memory.

### VS Code

[Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
edits and runs `.ipynb` files with kernel selection, a variable inspector,
cell-level debugging, and notebook-aware diffs. Open the repository as a
folder, choose the interpreter created by `uv`, and select that interpreter
as the notebook kernel.

When a notebook grows into a project, VS Code can navigate definitions, run
tests, format code, and review Git diffs. Keep testable logic in `.py` modules
and use the notebook for explanation and experiment records. This
repository also includes a VS Code extension for the book's
authoring workflow — switching framework views, previewing slides, syncing
edits back to the source — described in :numref:`sec_developers_guide`.

### Debugging and Timing

Use the notebook debugger when the kernel supports it, or move a small
failing call into a module and debug it in a terminal. Resist broad
`try/except` blocks that print "failed": they discard the exception type and
location that diagnosis needs. For quick measurements, IPython's magics are
built in:

```{.python .input #interactive-development-timing}
values = np.arange(100_000, dtype=np.float64)
%timeit values @ values
```

One caveat when timing accelerators: GPU operations are asynchronous, so a
host-side timer may measure only the dispatch of work, not its completion.
Synchronize explicitly (or use the framework's benchmarking utilities), warm
up compiled kernels first, and record the shapes, precision, device, and
library versions alongside the number.

## Remote Machines

Your editor, the notebook server, the kernel, and the accelerator need not
be on the same machine. A common setup runs JupyterLab next to the GPU while
you edit from a laptop.

![A local editor reaches a remote Jupyter server and kernel through an SSH tunnel; the accelerator stays attached to the remote kernel.](../img/tools-remote-layers.svg)
:label:`fig_tools_remote_layers`

Bind the remote server to loopback and forward it over SSH:

```bash
# Remote machine
uv run jupyter lab --no-browser --ip 127.0.0.1 --port 8888
```

```bash
# Local machine
ssh -N -L 8888:127.0.0.1:8888 myserver
```

Then open the tokenized `http://127.0.0.1:8888/...` URL locally. Never bind
an unauthenticated Jupyter server to all network interfaces: anyone who can
reach the port can execute code as you. The SSH tunnel provides encryption
and access control; Jupyter's token remains a useful second boundary.

VS Code's Remote SSH installs a small server-side
component so the editor UI runs locally while terminals, extensions, files,
and kernels all live on the remote machine. Confirm the status bar shows
**SSH** before selecting the Python environment — a local kernel cannot use
the remote GPU. In either setup, treat remote compute as disposable: commit code and copy
checkpoints to durable storage before
it is rebooted, preempted, or deleted (:numref:`sec_cloud_instances`).

Before sharing or committing a notebook, perform a final reproducibility and
privacy check:
restart and run all; check that setup does not rely on personal paths or
unrecorded downloads; remove secrets and private data from outputs; keep the
outputs that teach or verify something and delete noisy progress logs; and
note the environment and hardware needed to interpret any measurement.

## Summary

* Notebooks interleave prose, code, and results, and retain expensive state
  between experiments. Run and edit them rather than treating them as static
  documents.
* The document and the kernel hold different state in different orders;
  restart and run all is the test that they agree.
* Launch JupyterLab from the book's `uv` environment, and verify the kernel
  with `sys.executable` before trusting any result.
* JupyterLab integrates notebooks, terminals, and kernels; VS Code adds
  modules, tests, diffs, and first-class remote development.
* Use SSH tunneling or VS Code Remote SSH instead of exposing a notebook
  server to the network.

## Exercises

1. [code] **Out-of-order execution.** Reproduce the failure of
   :numref:`fig_tools_kernel_state` in a new notebook: run a cell that
   defines `data`, insert a cell above it that uses `data`, and run the new
   cell. Restart and run all then stops with a `NameError`.
    1. Construct a pair of cells whose out-of-order execution changes a value
       instead of raising an error, for example a learning rate that a cell
       further down sets again before you rerun the training cell. What does
       restart and run all report for this notebook? Which of the two
       failures is more dangerous, and why?
    1. A saved notebook shows the execution counters $1, 2, \ldots, n$ on its
       $n$ code cells in document order. What does this guarantee about how
       its outputs were produced? :citet:`Pimentel.Murta.Braganholo.ea.2019`
       distinguish three kinds of hidden state: a re-executed cell, a cell
       edited after it ran, and a removed cell. Which of them can such a
       notebook still contain? A kernel restart resets the counter to $1$;
       take restarts into account.
    1. Another notebook shows the counters $3, 8, 9$ on its three code
       cells. What do the gaps tell you about executions that the document
       no longer shows? Do they prove that the outputs depend on hidden
       state?
   :exlabel:`ex_interactive-development-out-of-order`

1. [code] **Cell reruns.** The cell
   `#interactive-development-hidden-state-a` creates the generator `rng`
   and draws `samples` in one step, so rerunning it and then
   `#interactive-development-hidden-state-b` reproduces the printed mean
   $-0.2639$.
    1. Split the cell into one that creates `rng` and one that draws
       `samples`. Rerun the drawing cell twice and then
       `#interactive-development-hidden-state-b`. Predict whether the mean
       changes on each rerun, then check. Which value does restart and run
       all print?
    1. A cell rescales images with `pixels = pixels / 255`. What happens
       when a reader runs it twice? Rewrite the cell so that running it again
       leaves the kernel in the same state.
    1. An experiment cell trains a model from random initial weights that it
       draws from a generator created in an earlier cell. Restructure the
       cell so that every rerun of that cell alone reproduces its result,
       whichever other cells have run before it.
   :exlabel:`ex_interactive-development-rerun`

1. **Limits of a clean run.** Restart and run all replays the
   document in a fresh kernel, but on the same machine. Consider five
   notebooks that pass this test on their author's machine:
   (i) a cell loads `features.npy` if the file exists and otherwise
   computes it, the computing branch contains a bug, and `features.npy` is
   not in the repository;
   (ii) a package was installed from a cell that has since been deleted;
   (iii) a cell reads `/home/alice/data/train.csv`;
   (iv) a cell reads an environment variable that is set in the shell that
   launched JupyterLab;
   (v) a paragraph quotes an accuracy that a cell computes with an unseeded
   random generator.
    1. For each notebook, describe what a reader who downloads it
       experiences.
    1. Suppose the notebook is instead executed top to bottom on a fresh
       machine, in a new environment built from the lock file, from a fresh
       clone of the repository. Which of the five failures does this
       catch? For the failures that remain, name a check that catches them.

1. [code] **Broad exception handlers.** The following loop whitens the
   batches of a dataset with 1001 examples, and it prints `failed` once.

   ```python
   import numpy as np

   data = np.random.default_rng(0).normal(size=(1001, 4))

   def whiten(batch):
       cov = np.cov(batch, rowvar=False)
       return batch @ np.linalg.inv(np.linalg.cholesky(cov)).T

   for i in range(0, len(data), 100):
       try:
           out = whiten(data[i:i + 100])
       except Exception:
           print("failed")
   ```

    1. Remove the `try` statement, run the loop, and run `%debug` in the next
       cell. The debugger starts in the innermost frame, inside NumPy; move up
       to the frame of `whiten` and inspect `batch` and `cov`. Which batch
       fails, and why?
    1. Rewrite the handler so that the loop still processes the remaining
       batches but a failure keeps the information that the diagnosis in
       (a) needed.
    1. A training pipeline shuffles the 1001 examples in every epoch and
       splits them into batches of 100 as the loop does. Name two ways to
       handle the batch that fails in (a), and state what each one changes
       about the data the model sees.

1. [code] **Use-before-define detector.** ● Restart and run all finds an
   out-of-order dependency only by executing the notebook. A static check
   reads the document without running it.
    1. Write a function that reads a `.ipynb` file and reports every name
       that a code cell reads, that no earlier cell and no built-in binds,
       but that a later cell binds. Convert magics such as `%timeit` into
       Python with `IPython.core.inputtransformer2.TransformerManager`
       before you parse a cell with `ast`. Test the function on the notebook
       of exercise :exref:`ex_interactive-development-out-of-order`.
    1. The body of a function runs when the function is called, not when it
       is defined. Construct a notebook on which your detector raises a
       false alarm for this reason, and one on which it misses a real
       failure. Fix the false alarm.
    1. Add two warnings: a name that a cell binds again after an earlier
       cell has read it, and a cell that reads and rebinds the same name.
       Which failure of exercise
       :exref:`ex_interactive-development-out-of-order`a and of exercise
       :exref:`ex_interactive-development-rerun`b does each warning target?
       Can either warning tell a harmless case from a harmful one?
   :exlabel:`ex_interactive-development-use-before-define`

1. [code] **Worker processes.** ● Python's `multiprocessing` starts workers
   by one of three methods: `fork` copies the parent process, `spawn` starts
   each worker in a fresh interpreter, and `forkserver` forks each worker
   from a server process that was itself started fresh. Under every method,
   a pool sends each task, including the function to call, to its workers
   with `pickle`. Data loaders with worker processes rely on the same
   mechanism.
    1. Define `square(i)` returning `i * i` in a notebook cell. In another
       cell, run `p.map(square, range(4))` inside
       `with multiprocessing.get_context(method).Pool(2) as p:` for every
       method that your system offers. If a cell does not return, interrupt
       the kernel. What happens for each method, and where do the error
       messages appear?
    1. Explain the behavior from what `pickle` stores for a function and
       from the module in which a notebook defines its functions.
    1. Python 3.12, which the book's environments use, starts workers with
       `fork` on Linux and with `spawn` on macOS and Windows. Python 3.14
       changed the default on Linux to `forkserver`. Restructure the notebook
       so that the pool works under every start method.

1. [code] **Asynchronous accelerator timing.** ● Consider the product of two
   $8192 \times 8192$ matrices in a 16-bit format on a GPU: bf16 where the
   GPU supports it, and fp16 on the T4 and P100 of the free Colab and Kaggle
   tiers (:numref:`sec_hosted_notebooks`), which have no bf16 units. Use one
   of these free GPUs if your machine has none.
    1. Compute the time the product needs at the dense throughput that your
       GPU's datasheet lists for this format, or, for an RTX 4090, at its
       entry of :numref:`tab_gpu_specs`.
    1. Time one call with `time.perf_counter()`, once with and once without
       a device synchronization before the timer stops. In both
       measurements, synchronize before the timer starts and exclude the
       first call. Which measurement is consistent with (a)? What throughput
       would the other imply?
    1. Time the first call on its own and explain why it differs from later
       calls.
    1. Time the unsynchronized product with `%timeit`, which runs it many
       times in a row. Where does the result fall relative to the two
       measurements of (b), and why?

1. **A shared GPU server.** ● Several users of one remote server each
   start JupyterLab with
   `uv run jupyter lab --no-browser --ip 127.0.0.1 --port 8888` and connect
   from their laptops with `ssh -N -L 8888:127.0.0.1:8888 myserver`.
    1. Who can connect to a server that listens on `127.0.0.1` of that
       machine? What protects your kernel from them? Find where your
       running server records its token on disk, and which users can read
       that file.
    1. Your server holds port 8888 when a second user runs the same
       command. Jupyter then tries up to 50 other ports nearby by
       default. Where does the second user's tunnel lead, and what does that
       user see in the browser? How does the second user find the port on
       which their own server listens?
    1. On your laptop, a local JupyterLab that you use already listens on
       port 8888 when you open the tunnel. By default `ssh` only warns when
       it cannot bind the local port, and keeps running. What does the
       browser at `http://127.0.0.1:8888` reach? The cell
       `#interactive-development-identity` prints `~` for the home directory
       on the laptop and on the server alike. Which check tells the two
       machines apart?
    1. Write a connection procedure that turns each collision of (b) and (c)
       into an error you notice.
