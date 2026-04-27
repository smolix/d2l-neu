#!/usr/bin/env python3
"""Debug TF cnn-design slowness.

Root causes identified:
1. No @tf.function — entire training loop runs in eager mode
2. Trainer asserts num_gpus == 0 (GPU support blocked)
3. Data pipeline: shuffle after batch, no prefetch, no parallel map
4. Single-threaded data loading

This script reproduces the problem and tests fixes incrementally.
"""

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

# Verify GPU is available
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
print(f"TF version: {tf.__version__}")

# ── Import d2l ──
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d2l import tensorflow as d2l


# ── Model (unchanged from d2l source) ──
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])

    def stage(self, depth, num_channels, groups, bot_mul):
        net = tf.keras.models.Sequential()
        for i in range(depth):
            if i == 0:
                net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                    use_1x1conv=True, strides=2))
            else:
                net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
        return net

    def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
        super(AnyNet, self).__init__()
        self.save_hyperparameters()
        self.net = self.stem(stem_channels)
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))


class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)


# ── Fixed data pipeline ──
class FashionMNISTFixed(d2l.FashionMNIST):
    """FashionMNIST with fixed data pipeline."""
    def get_dataloader(self, train):
        data = self.train if train else self.val
        process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                                tf.cast(y, dtype='int32'))
        resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
        shuffle_buf = len(data[0]) if train else 1
        ds = tf.data.Dataset.from_tensor_slices(process(*data))
        if train:
            ds = ds.shuffle(shuffle_buf)
        ds = (ds
              .batch(self.batch_size)
              .map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))
        return ds


# ── Fixed trainer with @tf.function and GPU support ──
class TrainerFixed(d2l.Trainer):
    def __init__(self, max_epochs, num_gpus=1, gradient_clip_val=0):
        # Skip the assert num_gpus == 0
        d2l.HyperParameters.__init__(self)
        self.save_hyperparameters()
        self.gpu = None
        if num_gpus > 0:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu = gpus[0].name.replace('physical_device:', '')
                print(f"Using GPU: {self.gpu}")

    def fit_epoch(self):
        self.model.training = True
        for batch in self.train_dataloader:
            if self.gpu:
                batch = tuple(tf.identity(t) for t in batch)
            self._train_step(batch)
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.training = False
        for batch in self.val_dataloader:
            if self.gpu:
                batch = tuple(tf.identity(t) for t in batch)
            self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    @tf.function
    def _train_step(self, batch):
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, params))


# ── Benchmark ──
def benchmark(name, trainer_cls, data_cls, epochs=3):
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    data = data_cls(batch_size=128, resize=(96, 96))

    if trainer_cls is TrainerFixed:
        trainer = trainer_cls(max_epochs=epochs, num_gpus=1)
        with tf.device('/GPU:0'):
            model = RegNetX32(lr=0.05)
    else:
        trainer = trainer_cls(max_epochs=epochs)
        with d2l.try_gpu():
            model = RegNetX32(lr=0.01)

    t0 = time.time()
    trainer.fit(model, data)
    elapsed = time.time() - t0

    print(f"\n{name}: {epochs} epochs in {elapsed:.1f}s ({elapsed/epochs:.1f}s/epoch)")
    return elapsed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['original', 'fixed', 'both'],
                        default='both')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    if args.mode in ('original', 'both'):
        t_orig = benchmark("ORIGINAL (eager, no GPU, bad pipeline)",
                           d2l.Trainer, d2l.FashionMNIST, args.epochs)

    if args.mode in ('fixed', 'both'):
        t_fixed = benchmark("FIXED (@tf.function, GPU, good pipeline)",
                            TrainerFixed, FashionMNISTFixed, args.epochs)

    if args.mode == 'both':
        print(f"\nSpeedup: {t_orig/t_fixed:.1f}x")
