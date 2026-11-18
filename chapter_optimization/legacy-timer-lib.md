# Legacy Timer Library (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook. `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It carries all four framework variants of `Timer` (the
file name predates the arrival of the current PyTorch/JAX one):
  * `%%tab pytorch, jax` (`time.perf_counter`, cell id
    `minibatch-sgd-vectorization-and-caches-2`): moved here byte for byte, id
    included, from `chapter_optimization/sgd.md` in the 2026-11 pass, which
    cut the timing helpers from that section.
  * `%%tab tensorflow, mxnet` (`time.time`, no cell id): the class is byte
    for byte the one that `sgd.md` held in an untagged cell until 2026-11;
    only the tab line is new.
The hidden label below keeps the "Defined in :numref:`sec_minibatch_sgd`"
docstring, so the extracted d2l/_blocks shards, and with them the freshness
fingerprints of every consumer, are unchanged. No section of the
Optimization chapter mentions `d2l.Timer` (it times with `d2l.Benchmark`);
the code that calls it lives in the Image Models part, the Attic and the
Tools appendix:
  * `chapter_computer-vision/{image-augmentation,kaggle-cifar10,kaggle-dog,ssd}.md`
  * `chapter_recommender-systems/{mf,autorec,neumf}.md`
  * `chapter_appendix-tools-for-deep-learning/utils.md` (training helpers)

Two things remain to be done, neither in a chapter-9 pass:
  * The docstring now points a reader to a section that does not show the
    class: `help(d2l.Timer)` names `sec_minibatch_sgd`, and no rendered page
    defines `Timer`. Correcting the pointer changes the shard and marks every
    consumer above stale, so it waits for the next full re-execution of the
    book. At that point move the pytorch,jax cell, with its id, to a rendered
    home (`utils.md`, `sec_utils`, where the training helpers first call it),
    tagged `pytorch, jax` and never untagged.
  * The tensorflow/mxnet cell can be removed when the consumers above drop
    their TensorFlow and MXNet tabs.
-->

<!-- This label is NOT rendered (this file is build-only). It exists only so
`build_lib.py`'s `find_section_label` writes the same "Defined in
:numref:`…`" docstring suffix that the classes carried at their original
home. -->
:label:`sec_minibatch_sgd`

```{.python .input}
%%tab tensorflow, mxnet
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

```{.python .input #minibatch-sgd-vectorization-and-caches-2}
%%tab pytorch, jax
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.perf_counter()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.perf_counter() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```
