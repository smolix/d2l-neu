# Legacy Multi-GPU Library (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook — `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It carries the tensorflow and mxnet variants of
`split_batch` and `resnet18`, whose PyTorch/JAX homes now live in the rebuilt
`multiple-gpus.md` (§13.5) and `multi-gpu-practice.md` (§13.6). The rebuilt
Advanced-part chapter carries PyTorch and JAX tabs only, but the still-four-
framework notebooks in the Language Models and Image Models parts import these
symbols:
  * `d2l.split_batch` (mxnet): `chapter_natural-language-processing-applications/natural-language-inference-attention.md`, `chapter_computer-vision/{image-augmentation,kaggle-cifar10,kaggle-dog}.md`
  * `d2l.resnet18` (tensorflow, mxnet): `chapter_computer-vision/image-augmentation.md`, `chapter_computer-vision/kaggle-cifar10.md`
  * `d2l.evaluate_accuracy_gpus` (mxnet): `chapter_computer-vision/{image-augmentation,kaggle-cifar10}.md`
Semantics are frozen byte-for-byte as they were in the pre-rebuild
`multiple-gpus.md` / `multiple-gpus-concise.md`, so those chapters' committed
outputs stay reproducible from source without a re-capture. This file is
deleted, and its symbols removed, when ch. 17–19 (the Language Models / Image
Models parts) drop their tensorflow/mxnet tabs — a removal already expected.
-->

<!-- These labels are NOT rendered (this file is build-only, absent from
_quarto.yml/CHAPTER_NUMBERING). They exist only so `build_lib.py`'s
`find_section_label` regenerates the same "Defined in :numref:`…`" docstring
suffix the tf/mxnet blocks carried at their original home, keeping the
extracted d2l/_blocks shards — and hence the freshness fingerprints of the
ch. 18/19 tf/mxnet consumers — byte-identical after the move. -->
:label:`sec_multi_gpu`

**`split_batch` (tensorflow, mxnet only — PyTorch/JAX version is
`chapter_computational-performance/multiple-gpus.md`'s `#@save`).**

```{.python .input}
%%tab mxnet
def split_batch(X, y, devices):  #@save
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
%%tab tensorflow
def split_batch(X, y, devices):  #@save
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (tf.split(X, len(devices)), tf.split(y, len(devices)))
```

:label:`sec_multi_gpu_concise`

**`resnet18` (tensorflow, mxnet only — PyTorch/JAX version is
`chapter_computational-performance/multi-gpu-practice.md`'s `#@save`).**

```{.python .input}
%%tab mxnet
def resnet18(num_classes):  #@save
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
%%tab tensorflow
def resnet18(num_classes, in_channels=1):  #@save
    """A slightly modified ResNet-18 model built with Keras."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = tf.keras.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(num_channels, use_1x1conv=True,
                                     strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    # Smaller conv, no max-pool (same as the PT version)
    net = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes),
    ])
    return net
```

**`evaluate_accuracy_gpus` (mxnet only — the multi-GPU accuracy helper the
ch. 19 mxnet notebooks `image-augmentation.md` and `kaggle-cifar10.md` call
via `d2l.evaluate_accuracy_gpus`; PyTorch/TF/JAX evaluate inline).**

```{.python .input}
%%tab mxnet
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):  #@save
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```
