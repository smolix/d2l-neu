# Learning Rate Scheduling
:label:`sec_scheduler`

So far we primarily focused on optimization *algorithms* for how to update the weight vectors rather than on the *rate* at which they are being updated. Nonetheless, adjusting the learning rate is often just as important as the actual algorithm. There are a number of aspects to consider:

* Most obviously the *magnitude* of the learning rate matters. If it is too large, optimization diverges, if it is too small, it takes too long to train or we end up with a suboptimal result. We saw previously that the condition number of the problem matters (see e.g., :numref:`sec_momentum` for details). Intuitively it is the ratio of the amount of change in the least sensitive direction vs. the most sensitive one.
* Secondly, the rate of decay is just as important. If the learning rate remains large we may simply end up bouncing around the minimum and thus not reach optimality. :numref:`sec_minibatch_sgd` discussed this in some detail and we analyzed performance guarantees in :numref:`sec_sgd`. In short, we want the rate to decay, but probably more slowly than $\mathcal{O}(t^{-\frac{1}{2}})$ which would be a good choice for convex problems.
* Another aspect that is equally important is *initialization*. This pertains both to how the parameters are set initially (review :numref:`sec_numerical_stability` for details) and also how they evolve initially. This goes under the moniker of *warmup*, i.e., how rapidly we start moving towards the solution initially. Large steps in the beginning might not be beneficial, in particular since the initial set of parameters is random. The initial update directions might be quite meaningless, too.
* Lastly, there are a number of optimization variants that perform cyclical learning rate adjustment. This is beyond the scope of the current chapter. We recommend the reader to review details in :citet:`Izmailov.Podoprikhin.Garipov.ea.2018`, e.g., how to obtain better solutions by averaging over an entire *path* of parameters.

Given the fact that there is a lot of detail needed to manage learning rates, most deep learning frameworks have tools to deal with this automatically. In the current chapter we will review the effects that different schedules have on accuracy and also show how this can be managed efficiently via a *learning rate scheduler*.

## Toy Problem

We begin with a toy problem that is cheap enough to compute easily, yet sufficiently nontrivial to illustrate some of the key aspects. For that we pick a slightly modernized version of LeNet (`relu` instead of `sigmoid` activation, MaxPooling rather than AveragePooling), as applied to Fashion-MNIST. Moreover, we hybridize the network for performance. Since most of the code is standard we just introduce the basics without further detailed discussion. See :numref:`chap_cnn` for a refresher as needed.

```{.python .input #lr-scheduler-toy-problem-1}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input #lr-scheduler-toy-problem-1}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input #lr-scheduler-toy-problem-1}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

```{.python .input #lr-scheduler-toy-problem-1}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
import math
import numpy as np

class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def net_fn():
    return Net()

loss_fn = optax.softmax_cross_entropy_with_integer_labels

batch_size = 256
data = d2l.FashionMNIST(batch_size=batch_size)
train_iter = data.get_dataloader(train=True)
test_iter = data.get_dataloader(train=False)

def evaluate_accuracy_jax(params, apply_fn, data_iter):
    metric = d2l.Accumulator(2)  # num_correct, num_examples
    for X, y in data_iter:
        X, y = jnp.array(X), jnp.array(y)
        y_hat = apply_fn({'params': params}, X)
        metric.add(float(jnp.sum(jnp.argmax(y_hat, axis=1) == y)),
                   float(y.shape[0]))
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, num_epochs, lr, scheduler=None):
    model = net if not callable(net) else net()
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 28, 28, 1))
    params = model.init(key, dummy)['params']
    if scheduler is not None:
        # Use inject_hyperparams so we can update the LR each epoch
        tx = optax.inject_hyperparams(optax.sgd)(learning_rate=lr)
    else:
        tx = optax.sgd(lr)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, X, y):
        def compute_loss(params):
            logits = model.apply({'params': params}, X)
            return jnp.mean(loss_fn(logits, y))
        l, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state_new = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        logits = model.apply({'params': params}, X)
        acc = jnp.mean(jnp.argmax(logits, axis=1) == y)
        return params, opt_state_new, l, acc

    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = jnp.array(X), jnp.array(y)
            params, opt_state, l, acc = train_step(
                params, opt_state, X, y)
            metric.add(float(l) * X.shape[0], float(acc) * X.shape[0],
                       X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / num_batches,
                             (train_loss, train_acc, None))

        test_acc = evaluate_accuracy_jax(params, model.apply, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

        if scheduler:
            new_lr = scheduler(epoch)
            opt_state.hyperparams['learning_rate'] = jnp.array(new_lr)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

Let's have a look at what happens if we invoke this algorithm with default settings, such as a learning rate of $0.3$ and train for $30$ iterations. Note how the training accuracy keeps on increasing while progress in terms of test accuracy stalls beyond a point. The gap between both curves indicates overfitting.

```{.python .input #lr-scheduler-toy-problem-2}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input #lr-scheduler-toy-problem-2}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input #lr-scheduler-toy-problem-2}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

```{.python .input #lr-scheduler-toy-problem-2}
#@tab jax
lr, num_epochs = 0.3, 30
train(net_fn, train_iter, test_iter, num_epochs, lr)
```

## Schedulers

One way of adjusting the learning rate is to set it explicitly at each step. This is conveniently achieved by the `set_learning_rate` method. We could adjust it downward after every epoch (or even after every minibatch), e.g., in a dynamic manner in response to how optimization is progressing.

```{.python .input #lr-scheduler-schedulers-1}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input #lr-scheduler-schedulers-1}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input #lr-scheduler-schedulers-1}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.learning_rate.numpy())
```

```{.python .input #lr-scheduler-schedulers-1}
#@tab jax
lr = 0.1
tx = optax.inject_hyperparams(optax.sgd)(learning_rate=lr)
dummy_params = {'w': jnp.zeros(10)}
opt_state = tx.init(dummy_params)
print(f'learning rate is now {opt_state.hyperparams["learning_rate"]:.2f}')
```

More generally we want to define a scheduler. When invoked with the number of updates it returns the appropriate value of the learning rate. Let's define a simple one that sets the learning rate to $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.

```{.python .input #lr-scheduler-schedulers-2}
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr
        # Gluon 2.0 optimizers read `base_lr` off the scheduler at init.
        self.base_lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Let's plot its behavior over a range of values.

```{.python .input #lr-scheduler-schedulers-3}
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Now let's see how this plays out for training on Fashion-MNIST. We simply provide the scheduler as an additional argument to the training algorithm.

```{.python .input #lr-scheduler-schedulers-4}
#@tab mxnet
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input #lr-scheduler-schedulers-4}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input #lr-scheduler-schedulers-4}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input #lr-scheduler-schedulers-4}
#@tab jax
train(net_fn, train_iter, test_iter, num_epochs, lr, scheduler)
```

This worked quite a bit better than previously. Two things stand out: the curve was rather more smooth than previously. Secondly, there was less overfitting. Unfortunately it is not a well-resolved question as to why certain strategies lead to less overfitting in *theory*. There is some argument that a smaller stepsize will lead to parameters that are closer to zero and thus simpler. However, this does not explain the phenomenon entirely since we do not really stop early but simply reduce the learning rate gently.

## Policies

While we cannot possibly cover the entire variety of learning rate schedulers, we attempt to give a brief overview of popular policies below. Common choices are polynomial decay and piecewise constant schedules. Beyond that, cosine learning rate schedules have been found to work well empirically on some problems. Lastly, on some problems it is beneficial to warm up the optimizer prior to using large learning rates.

### Factor Scheduler

One alternative to a polynomial decay would be a multiplicative one, that is $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ for $\alpha \in (0, 1)$. To prevent the learning rate from decaying beyond a reasonable lower bound the update equation is often modified to $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```{.python .input #lr-scheduler-factor-scheduler}
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

This can also be accomplished by a built-in scheduler in MXNet via the `lr_scheduler.FactorScheduler` object. It takes a few more parameters, such as warmup period, warmup mode (linear or constant), the maximum number of desired updates, etc.; Going forward we will use the built-in schedulers as appropriate and only explain their functionality here. As illustrated, it is fairly straightforward to build your own scheduler if needed.

### Multi Factor Scheduler

A common strategy for training deep networks is to keep the learning rate piecewise constant and to decrease it by a given amount every so often. That is, given a set of times when to decrease the rate, such as $s = \{5, 10, 20\}$ decrease $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ whenever $t \in s$. Assuming that the values are halved at each step we can implement this as follows.

```{.python .input #lr-scheduler-multi-factor-scheduler-1}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5,
                                              epoch_size=len(train_iter))
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input #lr-scheduler-multi-factor-scheduler-1}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input #lr-scheduler-multi-factor-scheduler-1}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input #lr-scheduler-multi-factor-scheduler-1}
#@tab jax
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

The intuition behind this piecewise constant learning rate schedule is that one lets optimization proceed until a stationary point has been reached in terms of the distribution of weight vectors. Then (and only then) do we decrease the rate such as to obtain a higher quality proxy to a good local minimum. The example below shows how this can produce ever slightly better solutions.

```{.python .input #lr-scheduler-multi-factor-scheduler-2}
#@tab mxnet
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input #lr-scheduler-multi-factor-scheduler-2}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input #lr-scheduler-multi-factor-scheduler-2}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input #lr-scheduler-multi-factor-scheduler-2}
#@tab jax
train(net_fn, train_iter, test_iter, num_epochs, 0.5, scheduler)
```

### Cosine Scheduler

A rather perplexing heuristic was proposed by :citet:`Loshchilov.Hutter.2016`. It relies on the observation that we might not want to decrease the learning rate too drastically in the beginning and moreover, that we might want to "refine" the solution in the end using a very small learning rate. This results in a cosine-like schedule with the following functional form for learning rates in the range $t \in [0, T]$.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


Here $\eta_0$ is the initial learning rate, $\eta_T$ is the target rate at time $T$. Furthermore, for $t > T$ we simply pin the value to $\eta_T$ without increasing it again. In the following example, we set the max update step $T = 20$.

```{.python .input #lr-scheduler-cosine-scheduler-1}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01,
                                         epoch_size=len(train_iter))
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input #lr-scheduler-cosine-scheduler-1}
#@tab pytorch, tensorflow, jax
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

In the context of computer vision this schedule *can* lead to improved results. Note, though, that such improvements are not guaranteed (as can be seen below).

```{.python .input #lr-scheduler-cosine-scheduler-2}
#@tab mxnet
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input #lr-scheduler-cosine-scheduler-2}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input #lr-scheduler-cosine-scheduler-2}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input #lr-scheduler-cosine-scheduler-2}
#@tab jax
train(net_fn, train_iter, test_iter, num_epochs, 0.3, scheduler)
```

### Warmup

In some cases initializing the parameters is not sufficient to guarantee a good solution. This is particularly a problem for some advanced network designs that may lead to unstable optimization problems. We could address this by choosing a sufficiently small learning rate to prevent divergence in the beginning. Unfortunately this means that progress is slow. Conversely, a large learning rate initially leads to divergence.

A rather simple fix for this dilemma is to use a warmup period during which the learning rate *increases* to its initial maximum and to cool down the rate until the end of the optimization process. For simplicity one typically uses a linear increase for this purpose. This leads to a schedule of the form indicated below.

```{.python .input #lr-scheduler-warmup-1}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20,
                                         warmup_steps=5 * len(train_iter),
                                         base_lr=0.3, final_lr=0.01,
                                         epoch_size=len(train_iter))
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input #lr-scheduler-warmup-1}
#@tab pytorch, tensorflow, jax
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Note that the network converges better initially (in particular observe the performance during the first 5 epochs).

```{.python .input #lr-scheduler-warmup-2}
#@tab mxnet
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input #lr-scheduler-warmup-2}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input #lr-scheduler-warmup-2}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

```{.python .input #lr-scheduler-warmup-2}
#@tab jax
train(net_fn, train_iter, test_iter, num_epochs, 0.3, scheduler)
```

Warmup can be applied to any scheduler (not just cosine). For a more detailed discussion of learning rate schedules and many more experiments see also :cite:`Gotmare.Keskar.Xiong.ea.2018`. In particular they find that a warmup phase limits the amount of divergence of parameters in very deep networks. This makes intuitively sense since we would expect significant divergence due to random initialization in those parts of the network that take the most time to make progress in the beginning.

## Summary

* Decreasing the learning rate during training can lead to improved accuracy and (most perplexingly) reduced overfitting of the model.
* A piecewise decrease of the learning rate whenever progress has plateaued is effective in practice. Essentially this ensures that we converge efficiently to a suitable solution and only then reduce the inherent variance of the parameters by reducing the learning rate.
* Cosine schedulers are popular for some computer vision problems. See e.g., [GluonCV](http://gluon-cv.mxnet.io) for details of such a scheduler.
* A warmup period before optimization can prevent divergence.
* Optimization serves multiple purposes in deep learning. Besides minimizing the training objective, different choices of optimization algorithms and learning rate scheduling can lead to rather different amounts of generalization and overfitting on the test set (for the same amount of training error).

## Exercises

1. Experiment with the optimization behavior for a given fixed learning rate. What is the best model you can obtain this way?
1. How does convergence change if you change the exponent of the decrease in the learning rate? Use `PolyScheduler` for your convenience in the experiments.
1. Apply the cosine scheduler to large computer vision problems, e.g., training ImageNet. How does it affect performance relative to other schedulers?
1. How long should warmup last?
1. Can you connect optimization and sampling? Start by using results from :citet:`Welling.Teh.2011` on Stochastic Gradient Langevin Dynamics.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:

<!-- slides -->

::: {.slide title="Learning Rate Schedules"}
The optimizer matters; the **learning rate schedule**
often matters more. With a constant $\eta$ you trade off
fast-but-unstable vs. slow-but-converged. A good schedule
gets both: aggressive early, careful late.
:::

::: {.slide title="What a schedule manages"}
- Initial $\eta$ — too large diverges, too small wastes time.
- Decay over training — final fine-tuning needs small $\eta$
  for noise to settle.
- Early instability — Transformers and adaptive optimizers
  can blow up in the first few hundred steps without warmup.
:::

::: {.slide title="Common schedules"}
- **Step / multi-step decay** — drop $\eta$ by 10× at
  preset epochs.
- **Cosine annealing** — smooth decay following a half
  cosine; popular in vision and Transformers.
- **Warmup** — linearly grow $\eta$ for the first
  ~$T_w$ steps, then decay. Standard for Transformers.
:::

::: {.slide title="Toy training loop"}
LeNet on Fashion-MNIST as the experimental harness:

- same model and data for every schedule;
- only the learning-rate policy changes;
- compare training curves, not isolated final numbers.

The implementation is deliberately ordinary so that the schedule
effect is the only teaching variable.
:::

::: {.slide title="Toy baseline"}
Constant $\eta=0.3$ is the baseline. Watch for the usual
pattern: fast early movement, then noisy late progress as
the step size stays too large for fine tuning.

@!lr-scheduler-toy-problem-2
:::

::: {.slide title="Constant-LR baselines"}
Fixed learning rates expose the central tradeoff:

- large $\eta$: quick early progress, noisy late convergence;
- small $\eta$: stable late convergence, slow early progress.

@!lr-scheduler-schedulers-1
:::

::: {.slide title="Constant-LR baselines (cont.)"}
The first custom scheduler uses
$\eta_t = \eta_0(t+1)^{-1/2}$.

@!lr-scheduler-schedulers-3

It is simple and monotone, but modern practice usually prefers
multi-step or cosine schedules.
:::

::: {.slide title="Square-root schedule training"}
Apply the same $\eta_t = \eta_0(t+1)^{-1/2}$ policy during
training. The curve should smooth out because late updates
are smaller:

@!lr-scheduler-schedulers-4
:::

::: {.slide title="Polynomial / factor decay"}
$\eta_t = \eta_0 \cdot (1 + \beta t)^{-\alpha}$ — gradual
decay. The ML classic before step decay took over:

@!lr-scheduler-factor-scheduler
:::

::: {.slide title="Multi-step decay"}
Drop $\eta$ by a fixed factor at preset epochs (e.g. 30, 60,
90). Standard for ImageNet ResNet training:

@!lr-scheduler-multi-factor-scheduler-1
:::

::: {.slide title="Multi-step training"}
@!lr-scheduler-multi-factor-scheduler-2

Notice the loss curve changes slope after each scheduled
drop: high $\eta$ explores quickly, then lower $\eta$
settles into a narrower basin.
:::

::: {.slide title="Cosine annealing"}
$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$.
Smooth decay over $T$ steps; small $\eta$ at the end yields
clean fine-tuning. Often paired with warmup and warm
restarts:

@!lr-scheduler-cosine-scheduler-1
:::

::: {.slide title="Cosine training"}
@!lr-scheduler-cosine-scheduler-2

Cosine avoids abrupt jumps. The tail becomes increasingly
conservative, which often improves final accuracy without
manual milestone tuning.
:::

::: {.slide title="Warmup"}
Adam-trained Transformers diverge if $\eta$ starts at the
target value — preconditioner $\hat{\mathbf{s}}_t$ hasn't
stabilized yet. Linear warmup from 0 to $\eta_0$ over the
first ~1k steps fixes it:

@!lr-scheduler-warmup-1
:::

::: {.slide title="Warmup training"}
@!lr-scheduler-warmup-2

Warmup spends early epochs ramping up instead of taking a
full-size first step. This protects unstable initial
statistics, then hands off to the cosine decay.
:::

::: {.slide title="Recap"}
- Schedule beats fixed $\eta$ — aggressive early, gentle
  late.
- **Multi-step** is the vision standard; **cosine** is
  smoother and often slightly better with the same budget.
- **Warmup** is mandatory for Transformer / large-Adam
  training: prevents early divergence as the second-moment
  EMA stabilizes.
- "Cosine + warmup" is the modern default. Most LLM
  training does exactly this.
:::
