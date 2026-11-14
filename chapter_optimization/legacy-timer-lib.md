# Legacy Timer Library (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook — `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It carries all four framework variants of `Timer`:
  * `%%tab pytorch, jax` (`time.perf_counter`, cell id
    `minibatch-sgd-vectorization-and-caches-2`): moved here byte for byte, id
    included, from `chapter_optimization/sgd.md` in the 2026-11 pass, which
    cut the timing helpers from that section.
  * `%%tab tensorflow, mxnet` (`time.time`): byte for byte the cell that
    `sgd.md` held untagged until 2026-11.
The hidden label below keeps the "Defined in :numref:`sec_minibatch_sgd`"
docstring, so the extracted d2l/_blocks shards, and with them the freshness
fingerprints of every consumer, are unchanged. In the Optimization chapter
only an exercise of `batch-size.md` mentions `d2l.Timer`; the code that calls
it lives in the Image Models part, the Attic and the Tools appendix:
  * `chapter_computer-vision/{image-augmentation,kaggle-cifar10,kaggle-dog,ssd}.md`
  * `chapter_natural-language-processing-pretraining/{word2vec,bert}-pretraining.md`
  * `chapter_natural-language-processing-applications/natural-language-inference-attention.md`
  * `chapter_recommender-systems/{mf,autorec,neumf}.md`
  * `chapter_appendix-tools-for-deep-learning/utils.md` (training helpers)
The tensorflow/mxnet cell can be removed when those chapters drop their
TensorFlow and MXNet tabs.
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
