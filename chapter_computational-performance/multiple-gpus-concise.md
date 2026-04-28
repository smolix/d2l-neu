# Concise Implementation for Multiple GPUs
:label:`sec_multi_gpu_concise`

Implementing parallelism from scratch for every new model is no fun. Moreover, there is significant benefit in optimizing synchronization tools for high performance. In the following we will show how to do this using high-level APIs of deep learning frameworks.
The mathematics and the algorithms are the same as in :numref:`sec_multi_gpu`.
Quite unsurprisingly you will need at least two GPUs to run code of this section.

```{.python .input #multiple-gpus-concise-concise-implementation-for-multiple-gpus}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #multiple-gpus-concise-concise-implementation-for-multiple-gpus}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #multiple-gpus-concise-concise-implementation-for-multiple-gpus}
#@tab jax
from d2l import jax as d2l
import functools
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
import flax
import numpy as np
```

```{.python .input #multiple-gpus-concise-concise-implementation-for-multiple-gpus}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import keras
```

## [**A Toy Network**]

Let's use a slightly more meaningful network than LeNet from :numref:`sec_multi_gpu` that is still sufficiently easy and quick to train.
We pick a ResNet-18 variant :cite:`He.Zhang.Ren.ea.2016`. Since the input images are tiny we modify it slightly. In particular, the difference from :numref:`sec_resnet` is that we use a smaller convolution kernel, stride, and padding at the beginning.
Moreover, we remove the max-pooling layer.

```{.python .input #multiple-gpus-concise-a-toy-network}
#@tab mxnet
#@save
def resnet18(num_classes):
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

```{.python .input #multiple-gpus-concise-a-toy-network}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

```{.python .input #multiple-gpus-concise-a-toy-network}
#@tab jax
#@save
class ResNet18(nn.Module):
    """A slightly modified ResNet-18 model."""
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            # ResNet blocks
            d2l.Residual(64, training=self.training),
            d2l.Residual(64, training=self.training),
            d2l.Residual(128, use_1x1conv=True, strides=(2, 2),
                         training=self.training),
            d2l.Residual(128, training=self.training),
            d2l.Residual(256, use_1x1conv=True, strides=(2, 2),
                         training=self.training),
            d2l.Residual(256, training=self.training),
            d2l.Residual(512, use_1x1conv=True, strides=(2, 2),
                         training=self.training),
            d2l.Residual(512, training=self.training),
            # Global average pooling and classifier
            lambda x: x.mean(axis=(1, 2)),
            nn.Dense(self.num_classes),
        ])

    def __call__(self, x):
        return self.net(x)
```

```{.python .input #multiple-gpus-concise-a-toy-network}
#@tab tensorflow
#@save
def resnet18(num_classes, in_channels=1):
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

## Network Initialization

:begin_tab:`mxnet`
The `initialize` function allows us to initialize parameters on a device of our choice.
For a refresher on initialization methods see :numref:`sec_numerical_stability`. What is particularly convenient is that it also allows us to initialize the network on *multiple* devices simultaneously. Let's try how this works in practice.
:end_tab:

:begin_tab:`pytorch`
We will initialize the network inside the training loop.
For a refresher on initialization methods see :numref:`sec_numerical_stability`.
:end_tab:

:begin_tab:`jax`
In JAX, we initialize the model parameters and create a `TrainState` that bundles the parameters with the optimizer. For multi-GPU training, we replicate the state across all devices using `jax.tree.map`.
:end_tab:

:begin_tab:`tensorflow`
With `tf.distribute.MirroredStrategy`, the model is built inside `strategy.scope()`. All variables created within the scope are automatically mirrored across all GPUs. We will initialize the network and optimizer inside the training function.
:end_tab:

```{.python .input #multiple-gpus-concise-network-initialization-1}
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input #multiple-gpus-concise-network-initialization-1}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

```{.python .input #multiple-gpus-concise-network-initialization-1}
#@tab jax
net = ResNet18(num_classes=10)
# Count available devices (GPUs/TPUs)
num_devices = jax.local_device_count()
print(f'Using {num_devices} devices: {jax.devices()}')
# We will initialize the network inside the training loop
```

```{.python .input #multiple-gpus-concise-network-initialization-1}
#@tab tensorflow
# MirroredStrategy distributes training across all available GPUs
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')
# The model will be created inside strategy.scope() in the training function
```

:begin_tab:`mxnet`
Using the `split_and_load` function introduced in :numref:`sec_multi_gpu` we can divide a minibatch of data and copy portions to the list of devices provided by the `devices` variable. The network instance *automatically* uses the appropriate GPU to compute the value of the forward propagation. Here we generate 4 observations and split them over the GPUs.
:end_tab:

```{.python .input #multiple-gpus-concise-network-initialization-2}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Once data passes through the network, the corresponding parameters are initialized *on the device the data passed through*.
This means that initialization happens on a per-device basis. Since we picked GPU 0 and GPU 1 for initialization, the network is initialized only there, and not on the CPU. In fact, the parameters do not even exist on the CPU. We can verify this by printing out the parameters and observing any errors that might arise.
:end_tab:

```{.python .input #multiple-gpus-concise-network-initialization-3}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Next, let's replace the code to [**evaluate the accuracy**] by one that works (**in parallel across multiple devices**). This serves as a replacement of the `evaluate_accuracy_gpu` function from :numref:`sec_lenet`. The main difference is that we split a minibatch before invoking the network. All else is essentially identical.
:end_tab:

```{.python .input #multiple-gpus-concise-network-initialization-4}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
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

## [**Training**]

As before, the training code needs to perform several basic functions for efficient parallelism:

* Network parameters need to be initialized across all devices.
* While iterating over the dataset minibatches are to be divided across all devices.
* We compute the loss and its gradient in parallel across devices.
* Gradients are aggregated and parameters are updated accordingly.

In the end we compute the accuracy (again in parallel) to report the final performance of the network. The training routine is quite similar to implementations in previous chapters, except that we need to split and aggregate data.

```{.python .input #multiple-gpus-concise-training-1}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input #multiple-gpus-concise-training-1}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs. Note: `nn.DataParallel` is
    # convenient for a single-process demo, but PyTorch recommends
    # `nn.parallel.DistributedDataParallel` for production training
    # because it scales better and avoids GIL contention.
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input #multiple-gpus-concise-training-1}
#@tab jax
def train(num_devices, batch_size, lr):
    data = d2l.FashionMNIST(batch_size=batch_size)
    train_iter = data.get_dataloader(train=True)
    test_iter = data.get_dataloader(train=False)
    net = ResNet18(num_classes=10, training=True)
    # Initialize parameters
    dummy_input = jnp.ones((1, 28, 28, 1))
    key = jax.random.PRNGKey(0)
    variables = net.init(key, dummy_input)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    # Create optimizer and training state
    tx = optax.sgd(lr)

    class TrainState(train_state.TrainState):
        batch_stats: dict

    state = TrainState.create(apply_fn=net.apply, params=params,
                              tx=tx, batch_stats=batch_stats)
    # Replicate state across devices
    num_devices = jax.local_device_count()
    state = jax.tree.map(
        lambda x: jnp.stack([x] * num_devices), state)

    @functools.partial(jax.pmap, axis_name='batch')
    def train_step(state, images, labels):
        """A single training step on one device."""
        def loss_fn(params):
            logits, updates = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                images, mutable=['batch_stats'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels).mean()
            return loss, updates
        (loss, updates), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        # Average gradients across devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        state = state.apply_gradients(grads=grads)
        state = state.replace(
            batch_stats=updates['batch_stats'])
        return state, loss

    @functools.partial(jax.pmap, axis_name='batch')
    def eval_step(state, images, labels):
        """Evaluate accuracy on one device."""
        logits, _ = state.apply_fn(
            {'params': state.params,
             'batch_stats': state.batch_stats},
            images, mutable=['batch_stats'])
        return (logits.argmax(axis=-1) == labels).sum(), labels.shape[0]

    def reshape_batch(X, y, num_devices):
        """Reshape a batch for pmap: (batch, ...) -> (num_devices, per_device, ...)."""
        per_device = X.shape[0] // num_devices
        X = X[:per_device * num_devices].reshape(
            num_devices, per_device, *X.shape[1:])
        y = y[:per_device * num_devices].reshape(num_devices, per_device)
        return X, y

    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            X, y = np.array(X), np.array(y)
            X, y = reshape_batch(X, y, num_devices)
            state, loss = train_step(state, X, y)
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        timer.stop()
        # Evaluate accuracy
        correct, total = 0, 0
        for X, y in test_iter:
            X, y = np.array(X), np.array(y)
            X, y = reshape_batch(X, y, num_devices)
            c, t = eval_step(state, X, y)
            correct += int(c.sum())
            total += int(t.sum())
        test_acc = correct / total
        animator.add(epoch + 1, (test_acc,))
    print(f'test acc: {test_acc:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {num_devices} devices')
```

```{.python .input #multiple-gpus-concise-training-1}
#@tab tensorflow
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # Restrict to the requested number of GPUs
    gpus = tf.config.list_logical_devices('GPU')[:num_gpus]
    strategy = tf.distribute.MirroredStrategy(
        devices=[g.name for g in gpus])
    # Build and compile the model inside strategy.scope() so that
    # all variables are automatically mirrored across GPUs and
    # gradients are all-reduced automatically on every step.
    with strategy.scope():
        net = resnet18(10)
        net.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy'])
    num_epochs = 10
    timer = d2l.Timer()
    timer.start()
    history = net.fit(train_iter, epochs=num_epochs, verbose=0)
    timer.stop()
    test_acc = net.evaluate(test_iter, verbose=0)[1]
    print(f'test acc: {test_acc:.2f}, {timer.sum():.1f} sec total '
          f'on {str([g.name for g in gpus])}')
```

Let's see how this works in practice. As a warm-up we [**train the network on a single GPU.**]

```{.python .input #multiple-gpus-concise-training-2}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input #multiple-gpus-concise-training-2}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input #multiple-gpus-concise-training-2}
#@tab jax
train(num_devices=1, batch_size=256, lr=0.1)
```

```{.python .input #multiple-gpus-concise-training-2}
#@tab tensorflow
train(num_gpus=1, batch_size=256, lr=0.1)
```

Next we [**use 2 GPUs for training**]. Compared with LeNet
evaluated in :numref:`sec_multi_gpu`,
the model for ResNet-18 is considerably more complex. This is where parallelization shows its advantage. The time for computation is meaningfully larger than the time for synchronizing parameters. This improves scalability since the overhead for parallelization is less relevant.

```{.python .input #multiple-gpus-concise-training-3}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input #multiple-gpus-concise-training-3}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input #multiple-gpus-concise-training-3}
#@tab jax
train(num_devices=2, batch_size=512, lr=0.2)
```

```{.python .input #multiple-gpus-concise-training-3}
#@tab tensorflow
train(num_gpus=2, batch_size=512, lr=0.2)
```

## Summary

:begin_tab:`mxnet`
* Gluon provides primitives for model initialization across multiple devices by providing a context list.
:end_tab:

:begin_tab:`jax`
* JAX provides `jax.pmap` for data-parallel training across multiple devices with automatic gradient aggregation via `jax.lax.pmean`.
* `jax.tree.map` handles distributing state across devices for `pmap`-based multi-GPU training.
:end_tab:

:begin_tab:`tensorflow`
* `tf.distribute.MirroredStrategy` provides the simplest path to multi-GPU training in TensorFlow/Keras 3. Build the model and optimizer inside `strategy.scope()` and the framework mirrors all variables and aggregates gradients automatically.
:end_tab:

* Data is automatically evaluated on the devices where the data can be found.
* Take care to initialize the networks on each device before trying to access the parameters on that device. Otherwise you will encounter an error.
* The optimization algorithms automatically aggregate over multiple GPUs.



## Exercises

:begin_tab:`mxnet`
1. This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation. What happens if you try this with 16 GPUs (e.g., on an AWS p2.16xlarge instance)?
1. Sometimes, different devices provide different computing power. We could use the GPUs and the CPU at the same time. How should we divide the work? Is it worth the effort? Why? Why not?
1. What happens if we drop `npx.waitall()`? How would you modify training such that you have an overlap of up to two steps for parallelism?
:end_tab:

:begin_tab:`pytorch`
1. This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation. What happens if you try this with 16 GPUs (e.g., on an AWS p2.16xlarge instance)?
1. Sometimes, different devices provide different computing power. We could use the GPUs and the CPU at the same time. How should we divide the work? Is it worth the effort? Why? Why not?
:end_tab:

:begin_tab:`jax`
1. This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation. What happens if you try this with 16 GPUs (e.g., on an AWS p2.16xlarge instance) or with TPUs?
1. Sometimes, different devices provide different computing power. We could use the GPUs and the CPU at the same time. How should we divide the work? Is it worth the effort? Why? Why not?
1. What happens if we replace `jax.pmap` with `jax.vmap`? How does the behavior differ?
:end_tab:

:begin_tab:`tensorflow`
1. This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation. What happens if you try this with 4 GPUs?
1. Try replacing `MirroredStrategy` with `tf.distribute.MultiWorkerMirroredStrategy`. What changes are needed for multi-machine training?
1. What happens if you move `net.compile` outside `strategy.scope()`? Does training still work correctly?
:end_tab:



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
