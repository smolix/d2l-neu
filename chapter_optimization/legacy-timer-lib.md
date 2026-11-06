# Legacy Timer Library (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook — `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It carries the tensorflow and mxnet variants of `Timer`.
The PyTorch/JAX `Timer` lives in `chapter_optimization/sgd.md`, whose
Advanced-part cell is tagged `%%tab pytorch, jax`. The still-four-framework
notebooks of the Language Models and Image Models parts and the Tools
appendix call `d2l.Timer` in their tensorflow and mxnet tabs too:
  * `chapter_computer-vision/{image-augmentation,kaggle-cifar10,kaggle-dog,ssd}.md`
  * `chapter_natural-language-processing-pretraining/word2vec-pretraining.md`
  * `chapter_appendix-tools-for-deep-learning/utils.md` (training helpers)
The two classes are byte-for-byte the cell that `sgd.md` held untagged until
2026-11, so the extracted d2l/_blocks shards, and with them the freshness
fingerprints of those tensorflow and mxnet consumers, are unchanged. This
file can be removed when those chapters drop their TensorFlow and MXNet
tabs.
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
