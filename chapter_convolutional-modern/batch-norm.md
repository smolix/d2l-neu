```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Normalization Layers
:label:`sec_batch_norm`

Training deep neural networks is difficult.
Getting them to converge in a reasonable amount of time can be tricky.
In this section, we describe *batch normalization*, a popular and effective technique
that consistently accelerates the convergence of deep networks :cite:`Ioffe.Szegedy.2015`.
Together with residual blocks---covered later in :numref:`sec_resnet`---batch normalization
has made it possible for practitioners to routinely train networks with over 100 layers.
A secondary (serendipitous) benefit of batch normalization lies in its inherent regularization.
Batch statistics are not the only option, though: we also cover layer normalization
and group normalization, close cousins that normalize each example on its own.

```{.python .input #batch-norm-batch-normalization}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #batch-norm-batch-normalization}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #batch-norm-batch-normalization}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #batch-norm-batch-normalization}
%%tab jax
from d2l import jax as d2l
from flax import nnx
from jax import numpy as jnp
import jax
```

## Training Deep Networks

When working with data, we often preprocess before training.
Choices regarding data preprocessing often make an enormous difference in the final results.
Recall our application of MLPs to predicting house prices (:numref:`sec_kaggle_house`).
Our first step when working with real data
was to standardize each input feature to have zero mean and unit marginal
variance across observations. In vector notation, the mean is
$\boldsymbol{\mu}=\boldsymbol{0}$ and the covariance has unit diagonal,
$\Sigma_{ii}=1$; its off-diagonal entries need not vanish
:cite:`friedman1987exploratory`.
Yet another strategy is to rescale vectors to unit length, possibly zero mean *per observation*.
This can work well, e.g., for spatial sensor data. These preprocessing techniques and many others, are
beneficial for keeping the estimation problem well controlled. 
For a review of feature selection and extraction see the article of :citet:`guyon2008feature`, for example.
Standardizing vectors also has the nice side-effect of constraining the function complexity of functions that act upon it. For instance, the celebrated radius-margin bound :cite:`Vapnik95` in support vector machines and the Perceptron Convergence Theorem :cite:`Novikoff62` rely on inputs of bounded norm. 

Intuitively, this standardization plays nicely with our optimizers
since it puts the parameters *a priori* on a similar scale.
As such, it is only natural to ask whether a corresponding normalization step *inside* a deep network
might not be beneficial. While this is not quite the reasoning that led to the invention of batch normalization :cite:`Ioffe.Szegedy.2015`, it is a useful way of understanding it and its cousin, layer normalization :cite:`Ba.Kiros.Hinton.2016`, within a unified framework.

Second, for a typical MLP or CNN, as we train,
the variables 
in intermediate layers (e.g., affine transformation outputs in MLP)
may take values with widely varying magnitudes:
whether along the layers from input to output, across units in the same layer,
and over time due to our updates to the model parameters.
The inventors of batch normalization postulated informally
that this drift in the distribution of such variables could hamper the convergence of the network.
Intuitively, we might conjecture that if one
layer has variable activations that are 100 times that of another layer,
this might necessitate compensatory adjustments in the learning rates. Adaptive solvers
such as AdaGrad :cite:`Duchi.Hazan.Singer.2011`, Adam :cite:`Kingma.Ba.2014`, Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`, or Distributed Shampoo :cite:`anil2020scalable` aim to address this from the viewpoint of optimization, e.g., by adding aspects of second-order methods. 
The alternative is to prevent the problem from occurring, simply by adaptive normalization.

Third, deeper networks are complex and tend to be more liable to overfitting.
This means that regularization becomes more critical. A common technique for regularization is noise
injection. This has been known for a long time, e.g., with regard to noise injection for the
inputs :cite:`Bishop.1995`. It also forms the basis of dropout in :numref:`sec_dropout`. As it turns out, quite serendipitously, batch normalization conveys all three benefits: preprocessing, numerical stability, and regularization.

Batch normalization is applied to individual layers, or optionally, to all of them:
In each training iteration,
we first normalize the inputs (of batch normalization)
by subtracting their mean and
dividing by their standard deviation,
where both are estimated based on the statistics of the current minibatch.
Next, we apply a scale coefficient and an offset to recover the lost degrees
of freedom. It is precisely due to this *normalization* based on *batch* statistics
that *batch normalization* derives its name.

For a fully connected activation normalized only across examples, a minibatch
of size 1 removes all input-dependent signal: after subtracting the mean, every
normalized feature is 0. The shift parameter and later biases can still learn,
but the normalized feature cannot convey how that example differs from another.
As you might guess, since we are devoting a whole section to batch normalization,
with large enough minibatches the approach proves effective and stable.
One takeaway here is that when applying batch normalization,
the choice of batch size affects both the noise in the statistics and the
resulting optimization behavior.

Denote by $\mathcal{B}$ a minibatch and let $\mathbf{x} \in \mathcal{B}$ be an input to 
batch normalization ($\textrm{BN}$). In this case the batch normalization is defined as follows:

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

In :eqref:`eq_batchnorm`,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ is the  sample mean
and $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ is the sample standard deviation of the minibatch $\mathcal{B}$.
After applying standardization,
the resulting minibatch has zero mean and variance close to one; the added
$\epsilon$ makes the variance slightly smaller than one.
The choice of unit variance
(rather than some other magic number) is arbitrary. We recover this degree of freedom
by including an elementwise
*scale parameter* $\boldsymbol{\gamma}$ and *shift parameter* $\boldsymbol{\beta}$,
with one entry per normalized feature (or channel) and broadcasting over the
other axes. Both are parameters that
need to be learned as part of model training.

Before the learned affine transformation, batch normalization keeps the
normalized activations on a controlled scale. The final activations are not
bounded, since the learned scale $\boldsymbol{\gamma}$ can itself grow.
Practical experience confirms that, as alluded to when discussing feature rescaling, batch normalization seems to allow for more aggressive learning rates.
We calculate $\hat{\boldsymbol{\mu}}_\mathcal{B}$ and ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ in :eqref:`eq_batchnorm` as follows:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Note that we add a small constant $\epsilon > 0$
to the variance estimate
to ensure that we never attempt division by zero,
even in cases where the empirical variance estimate might be very small or vanish.
The estimates $\hat{\boldsymbol{\mu}}_\mathcal{B}$ and ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ counteract the scaling issue
by using noisy estimates of mean and variance.
You might think that this noisiness should be a problem.
On the contrary, it is actually beneficial.

This turns out to be a recurring theme in deep learning.
For reasons that are not yet well-characterized theoretically,
various sources of noise in optimization
often lead to faster training and less overfitting:
this variation appears to act as a form of regularization.
:citet:`Teye.Azizpour.Smith.2018` and :citet:`Luo.Wang.Shao.ea.2018`
related the properties of batch normalization to Bayesian priors and penalties, respectively. 
In particular, this sheds some light on the puzzle
of why moderate minibatches can regularize more than very large ones. There is
no universal best range: the effective sample count also includes spatial
positions in a convolution, and distributed implementations may synchronize
statistics across devices. Very large batches reduce this source of noise,
whereas very small sample counts make the estimates unreliable.

Fixing a trained model, you might think
that we would prefer using the entire dataset
to estimate the mean and variance.
Once training is complete, why would we want
the same image to be classified differently,
depending on the batch in which it happens to reside?
During training, such exact calculation is infeasible
because the intermediate variables
for all data examples
change every time we update our model.
During training, implementations therefore maintain exponential moving
averages of each layer's mean and variance. Those running estimates, rather
than exact statistics over the entire dataset, are the standard default for
prediction; a separate calibration pass is sometimes used after fine-tuning.
Thus batch normalization layers function differently
in *training mode* (normalizing by minibatch statistics)
than in *prediction mode* (normalizing by running statistics).
In this form they closely resemble the behavior of dropout regularization of :numref:`sec_dropout`,
where noise is only injected during training.


## Batch Normalization Layers

Batch normalization implementations for fully connected layers
and convolutional layers are slightly different.
One key difference between batch normalization and other layers
is that because the former operates on a full minibatch at a time,
we cannot just ignore the batch dimension
as we did before when introducing other layers.

### Fully Connected Layers

When applying batch normalization to fully connected layers,
:citet:`Ioffe.Szegedy.2015`, in their original paper inserted batch normalization after the affine transformation
and *before* the nonlinear activation function. Later applications experimented with
inserting batch normalization right *after* activation functions.
Denoting the input to the fully connected layer by $\mathbf{x}$,
the affine transformation
by $\mathbf{W}\mathbf{x} + \mathbf{b}$ (with the weight parameter $\mathbf{W}$ and the bias parameter $\mathbf{b}$),
and the activation function by $\phi$,
we can express the computation of a batch-normalization-enabled,
fully connected layer output $\mathbf{h}$ as follows:

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Recall that mean and variance are computed
on the *same* minibatch
on which the transformation is applied.

### Convolutional Layers

Similarly, with convolutional layers,
we can apply batch normalization after the convolution
but before the nonlinear activation function. The key difference from batch normalization
in fully connected layers is that we apply the operation on a per-channel basis
*across all locations*. Applying the same affine normalization at every
location preserves the translation-equivariant form of the convolutional
feature map, apart from the boundary and sampling qualifications discussed in
:numref:`sec_padding`.

Assume that our minibatches contain $m$ examples
and that for each channel,
the output of the convolution has height $p$ and width $q$.
For convolutional layers, we carry out each batch normalization
over the $m \cdot p \cdot q$ elements per output channel simultaneously.
Thus, we collect the values over all spatial locations
when computing the mean and variance
and consequently
apply the same mean and variance
within a given channel
to normalize the value at each spatial location.
Each channel has its own scale and shift parameters,
both of which are scalars.

### Layer Normalization
:label:`subsec_layer-normalization-in-bn`

For convolutional batch normalization, a minibatch of size 1 still supplies
spatial values, so its statistics are defined. They may nevertheless be poor
estimates, and they couple distant positions in the same image. *Layer
normalization* :cite:`Ba.Kiros.Hinton.2016` removes the batch dependence by
normalizing features within each example. Which features are grouped depends
on the model: transformer and ConvNeXt blocks normalize the channel vector at
each position, whereas other uses may include spatial axes. For an
$n$-dimensional vector $\mathbf{x}$,

$$
\textrm{LN}(\mathbf{x})_i =
\gamma_i \frac{x_i - \hat{\mu}}{\hat\sigma} + \beta_i,
$$

where each coordinate has its own learned gain $\gamma_i$ and bias $\beta_i$,
and

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

As before, $\epsilon > 0$ prevents division by zero. Before the learned affine
map and ignoring $\epsilon$, the normalized vector is invariant to positive
rescaling and flips sign under negative rescaling. Layer normalization does not
depend on the minibatch size and computes the same function in training and at
test time. It controls the scale entering the affine map but does not by itself
guarantee that optimization cannot diverge.

### Batch Normalization During Prediction

As we mentioned earlier, batch normalization typically behaves differently
in training mode than in prediction mode.
First, the noise in the sample mean and the sample variance
arising from estimating each on minibatches
is no longer desirable once we have trained the model.
Second, we might not have the luxury
of computing per-batch normalization statistics.
For example,
we might need to apply our model to make one prediction at a time.

Typically, after training, we use the entire dataset
to compute stable estimates of the variable statistics
and then fix them at prediction time.
Hence, batch normalization behaves differently during training than at test time.
Recall that dropout also exhibits this characteristic.

## Implementation from Scratch

To see how batch normalization works in practice, we implement one from scratch below.

```{.python .input #batch-norm-implementation-from-scratch-1}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether we are in training mode
    if not autograd.is_training():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input #batch-norm-implementation-from-scratch-1}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum,
               training):
    if not training:
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
            running_var = X.var(dim=0, unbiased=True)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            running_var = X.var(dim=(0, 2, 3), unbiased=True, keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = ((1.0 - momentum) * moving_var
                      + momentum * running_var)
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.detach(), moving_var.detach()
```

```{.python .input #batch-norm-implementation-from-scratch-1}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input #batch-norm-implementation-from-scratch-1}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    if deterministic:
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean[...]) / jnp.sqrt(moving_var[...] + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance over batch and spatial dimensions. Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 1, 2), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 1, 2), keepdims=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean[...] = momentum * moving_mean[...] + (1.0 - momentum) * mean
        moving_var[...] = momentum * moving_var[...] + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y
```

We can now create a proper `BatchNorm` layer.
Our layer will maintain proper parameters
for scale `gamma` and shift `beta`,
both of which will be updated in the course of training.
Additionally, our layer will maintain
moving averages of the means and variances
for subsequent use during model prediction.

Putting aside the algorithmic details,
note the design pattern underlying our implementation of the layer.
Typically, we define the mathematics in a separate function, say `batch_norm`.
We then integrate this functionality into a custom layer,
whose code mostly addresses bookkeeping matters,
such as moving data to the right device context,
allocating and initializing any required variables,
keeping track of moving averages (here for mean and variance), and so on.
This pattern enables a clean separation of mathematics from boilerplate code.
Also note that for the sake of convenience
we did not worry about automatically inferring the input shape here;
thus we need to specify the number of features throughout.
By now all modern deep learning frameworks offer automatic detection of size and shape in the
high-level batch normalization APIs (in practice we will use this instead).

```{.python .input #batch-norm-implementation-from-scratch-2}
%%tab mxnet
from mxnet import gluon

class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = gluon.Parameter('gamma', shape=shape, init=init.One())
        self.beta = gluon.Parameter('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input #batch-norm-implementation-from-scratch-2}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.ones(shape))

    def forward(self, X):
        Y, moving_mean, moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1,
            training=self.training)
        self.moving_mean.copy_(moving_mean)
        self.moving_var.copy_(moving_var)
        return Y
```

```{.python .input #batch-norm-implementation-from-scratch-2}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    def call(self, inputs, training=False):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            self.assign_moving_average(
                self.moving_mean, batch_mean)
            self.assign_moving_average(
                self.moving_variance, batch_variance)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input #batch-norm-implementation-from-scratch-2}
%%tab jax
class BatchNorm(nnx.Module):
    deterministic: bool

    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer.
    # `num_dims`: 2 for a fully connected layer and 4 for a convolutional layer
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    def __init__(self, num_features, num_dims, deterministic=False):
        self.deterministic = deterministic
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, 1, 1, num_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nnx.Param(jnp.ones(shape))
        self.beta = nnx.Param(jnp.zeros(shape))

        # The variables that are not model parameters are initialized to 0 and
        # 1. Save them to the 'batch_stats' collection
        self.moving_mean = nnx.BatchStat(jnp.zeros(shape))
        self.moving_var = nnx.BatchStat(jnp.ones(shape))

    def set_view(self, *, deterministic):
        self.deterministic = deterministic

    def __call__(self, X):
        return batch_norm(X, self.deterministic, self.gamma, self.beta,
                          self.moving_mean, self.moving_var,
                          eps=1e-5, momentum=0.9)
```

We used `momentum` to govern the aggregation over past mean and variance estimates. This is somewhat of a misnomer as it has nothing whatsoever to do with the *momentum* term of optimization. Nonetheless, it is the commonly adopted name for this term and in deference to API naming convention we use the same variable name in our code.

## LeNet with Batch Normalization

To see how to apply `BatchNorm` in context,
below we apply it to a traditional LeNet model (:numref:`sec_lenet`).
Recall that batch normalization is applied
after the convolutional layers or fully connected layers
but before the corresponding activation functions.

```{.python .input #batch-norm-lenet-with-batch-normalization-1}
%%tab pytorch
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120),
            BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
            BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
```

```{.python .input #batch-norm-lenet-with-batch-normalization-1}
%%tab mxnet
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
            nn.Activation('sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2),
            nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
            nn.Activation('sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
            BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
            nn.Dense(84), BatchNorm(84, num_dims=2),
            nn.Activation('sigmoid'), nn.Dense(num_classes))
        self.initialize()
```

```{.python .input #batch-norm-lenet-with-batch-normalization-1}
%%tab tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=6, kernel_size=5),
            BatchNorm(), tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5),
            BatchNorm(), tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
            BatchNorm(), tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(84), BatchNorm(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(num_classes)])
```

```{.python .input #batch-norm-lenet-with-batch-normalization-1}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Sequential(
            nnx.Conv(1, 6, kernel_size=(5, 5), rngs=rngs),
            BatchNorm(6, num_dims=4),
            nnx.sigmoid,
            lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nnx.Conv(6, 16, kernel_size=(5, 5), rngs=rngs),
            BatchNorm(16, num_dims=4),
            nnx.sigmoid,
            lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nnx.Linear(7 * 7 * 16, 120, rngs=rngs),
            BatchNorm(120, num_dims=2), nnx.sigmoid,
            nnx.Linear(120, 84, rngs=rngs),
            BatchNorm(84, num_dims=2), nnx.sigmoid,
            nnx.Linear(84, num_classes, rngs=rngs))
```

:begin_tab:`jax`
NNX stores running means and variances as `BatchStat` variables inside each
layer. The training view updates them in place, while the evaluation view sets
`deterministic=True` recursively and reads the accumulated statistics.
:end_tab:

As before, we will train our network on the Fashion-MNIST dataset.
This code is virtually identical to that when we first trained LeNet.

```{.python .input #batch-norm-lenet-with-batch-normalization-3}
%%tab mxnet
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
trainer.fit(model, data)
```

```{.python .input #batch-norm-lenet-with-batch-normalization-3}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input #batch-norm-lenet-with-batch-normalization-3}
%%tab jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
trainer.fit(model, data)
```

```{.python .input #batch-norm-lenet-with-batch-normalization-3}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

Let's have a look at the scale parameter `gamma`
and the shift parameter `beta` learned
from the first batch normalization layer.

```{.python .input #batch-norm-lenet-with-batch-normalization-4}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input #batch-norm-lenet-with-batch-normalization-4}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input #batch-norm-lenet-with-batch-normalization-4}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input #batch-norm-lenet-with-batch-normalization-4}
%%tab jax
model.net.layers[1].gamma[...].reshape((-1,)), \
model.net.layers[1].beta[...].reshape((-1,))
```

## Concise Implementation

Compared with the `BatchNorm` class,
which we just defined ourselves,
we can use the `BatchNorm` class defined in high-level APIs from the deep learning framework directly.
The code looks virtually identical
to our implementation above, except that we no longer need to provide additional arguments for it to get the dimensions right.

```{.python .input #batch-norm-concise-implementation-1}
%%tab pytorch
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(num_classes))
```

```{.python .input #batch-norm-concise-implementation-1}
%%tab tensorflow
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=6, kernel_size=5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(84),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(num_classes)])
```

```{.python .input #batch-norm-concise-implementation-1}
%%tab mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2),
            nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2),
            nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
            nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
            nn.Dense(num_classes))
        self.initialize()
```

```{.python .input #batch-norm-concise-implementation-1}
%%tab jax
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        # Flax's default momentum=0.99 decays the OLD running stats; PT/MX use
        # momentum=0.1 on the NEW stats, i.e. decay-of-OLD = 0.9. Pass 0.9 to
        # match the other tabs.
        self.net = nnx.Sequential(
            nnx.Conv(1, 6, kernel_size=(5, 5), rngs=rngs),
            nnx.BatchNorm(6, momentum=0.9, rngs=rngs), nnx.sigmoid,
            lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nnx.Conv(6, 16, kernel_size=(5, 5), rngs=rngs),
            nnx.BatchNorm(16, momentum=0.9, rngs=rngs), nnx.sigmoid,
            lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nnx.Linear(7 * 7 * 16, 120, rngs=rngs),
            nnx.BatchNorm(120, momentum=0.9, rngs=rngs), nnx.sigmoid,
            nnx.Linear(120, 84, rngs=rngs),
            nnx.BatchNorm(84, momentum=0.9, rngs=rngs), nnx.sigmoid,
            nnx.Linear(84, num_classes, rngs=rngs))
```

Below, we use the same hyperparameters to train our model.
Note that as usual, the high-level API variant runs much faster
because its code has been compiled to C++ or CUDA
while our custom implementation must be interpreted by Python.

```{.python .input #batch-norm-concise-implementation-2}
%%tab mxnet
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
trainer.fit(model, data)
```

```{.python .input #batch-norm-concise-implementation-2}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input #batch-norm-concise-implementation-2}
%%tab jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
trainer.fit(model, data)
```

```{.python .input #batch-norm-concise-implementation-2}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## Beyond Batch Normalization
:label:`subsec_beyond_bn`

Batch normalization is unusual among the layers in this book: its output for one
example depends on the other examples in the minibatch. That coupling is where
the regularizing noise comes from, and it is also where the practical trouble
comes from. Two problems recur.

The first is *minibatch coupling*. The estimates $\hat{\boldsymbol{\mu}}_\mathcal{B}$
and $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ stand and fall with the batch size:
we saw above that a fully connected layer learns nothing at batch size 1
and that moderate minibatches in the 50--100 range work best.
This bites hardest in dense prediction. Object detection and semantic
segmentation train on high-resolution images, so memory often limits the batch
to one or two images per device, exactly where batch statistics are noisiest.

The second is the *train/serve discrepancy*. As discussed above, the layer
applies minibatch statistics during training but moving averages at prediction
time, so the network computes two different functions depending on the mode.
Whenever the moving averages summarize the served data poorly, for instance
after fine-tuning on a new domain or when a bug freezes them, accuracy drops in
ways that are hard to trace back to a normalization layer.

### Group Normalization

*Group normalization* :cite:`wu2018groupnorm` removes the batch from the
statistics. It divides the $c$ channels into $G$ groups of $c/G$ channels and
standardizes each example separately over each group, i.e., over
$(c/G) \cdot p \cdot q$ elements, again followed by a per-channel scale
$\boldsymbol{\gamma}$ and shift $\boldsymbol{\beta}$ as in :eqref:`eq_batchnorm`.
Setting $G=1$ normalizes each example over all channels and positions at once;
setting $G=c$ normalizes every channel separately (a variant known as
*instance normalization*). Intermediate group counts such as $G=32$ tend to
work best: enough elements per group for stable statistics, without forcing
unrelated channels to share them.

Because no other example enters the statistics, group normalization computes
the same function at batch size 1 as at batch size 32, and the same function
during training as at prediction time; there are no moving averages to
maintain. The code below checks both claims: after group normalization, every
(example, group) pair has mean 0 and variance 1, and normalizing an example
alone gives the same output as normalizing it inside a batch.

```{.python .input #batch-norm-group-normalization}
%%tab pytorch
X = torch.randn(32, 16, 8, 8)
gn = nn.GroupNorm(4, 16)  # 4 groups of 4 channels each
Y = gn(X)
G = Y.reshape(32, 4, -1)  # Collect each (example, group) pair's elements
(G.mean(dim=2).abs().max(), G.var(dim=2, unbiased=False).mean(),
 torch.allclose(gn(X[:1]), Y[:1], atol=1e-6))
```

```{.python .input #batch-norm-group-normalization}
%%tab mxnet
X = np.random.normal(size=(32, 16, 8, 8))
gn = nn.GroupNorm(num_groups=4)  # 4 groups of 4 channels each
gn.initialize()
Y = gn(X)
G = Y.reshape(32, 4, -1)  # Collect each (example, group) pair's elements
(np.abs(G.mean(axis=2)).max(), G.var(axis=2).mean(),
 bool(np.abs(gn(X[:1]) - Y[:1]).max() < 1e-5))
```

```{.python .input #batch-norm-group-normalization}
%%tab tensorflow
X = tf.random.normal((32, 8, 8, 16))  # Channels last, as Keras expects
gn = tf.keras.layers.GroupNormalization(groups=4)  # Groups the last axis
Y = gn(X)
# Collect each (example, group) pair's elements
G = tf.reshape(tf.transpose(Y, (0, 3, 1, 2)), (32, 4, -1))
(tf.reduce_max(tf.abs(tf.reduce_mean(G, axis=2))),
 tf.reduce_mean(tf.math.reduce_variance(G, axis=2)),
 bool(tf.reduce_all(tf.abs(gn(X[:1]) - Y[:1]) < 1e-5)))
```

```{.python .input #batch-norm-group-normalization}
%%tab jax
X = jax.random.normal(jax.random.PRNGKey(0), (32, 8, 8, 16))  # Channels last
gn = nnx.GroupNorm(16, num_groups=4, rngs=nnx.Rngs(1))
Y = gn(X)
# Collect each (example, group) pair's elements
G = jnp.transpose(Y, (0, 3, 1, 2)).reshape(32, 4, -1)
(jnp.abs(G.mean(axis=2)).max(), G.var(axis=2).mean(),
 jnp.allclose(gn(X[:1]), Y[:1], atol=1e-6))
```

The trade-off runs in the other direction too: with large batches, batch
normalization averages over more elements and keeps a small accuracy edge on
ImageNet classification, but as the batch shrinks below roughly eight examples
per device its accuracy degrades quickly while group normalization is
unaffected :cite:`wu2018groupnorm`. Group normalization is therefore the
default in detection and segmentation heads and in diffusion U-Nets, settings
where large per-device batches are unaffordable.

### Layer Normalization in Convolutional Networks

Layer normalization from :numref:`subsec_layer-normalization-in-bn` is the
other batch-free option, and modern convolutional networks use it in a specific
form: at every spatial position, normalize the $c$ channel values at that
position, just as a transformer normalizes each token's embedding. This
per-position, channels-last layer normalization is the choice made by ConvNeXt,
a convolutional architecture we will meet later in this chapter: it replaces
every batch normalization in a ResNet-style network with layer normalization,
uses fewer normalization layers overall, and loses no accuracy in the process
(:numref:`sec_convnext`).

### Normalizer-Free Networks

One can push further and ask whether deep networks need normalization layers at
all. Normalizer-free networks (NFNets) :cite:`brock2021nfnet` answer no:
combining *weight standardization* (standardizing each convolution's weights
rather than its activations) with *adaptive gradient clipping* (clipping a
unit's gradient when its norm grows large relative to the corresponding weight
norm) trains ResNet-style networks that held the ImageNet state of the art at
publication, with no normalization anywhere. Dropping batch normalization
eliminates minibatch coupling and the train/serve discrepancy in one stroke,
and it removes the cost of computing batch statistics on large activations. We
return to normalizer-free networks when we discuss scaling up convolutional
networks in :numref:`sec_cnn-design`.

## Discussion

Intuitively, batch normalization is thought
to make the optimization landscape smoother.
However, we must be careful to distinguish between
speculative intuitions and true explanations
for the phenomena that we observe when training deep models.
Recall that we do not even know why simpler
deep neural networks (MLPs and conventional CNNs)
generalize well in the first place.
Even with dropout and weight decay,
they remain so flexible that their ability to generalize to unseen data
likely needs significantly more refined learning-theoretic generalization guarantees.

The original paper proposing batch normalization :cite:`Ioffe.Szegedy.2015`, in addition to introducing a powerful and useful tool,
offered an explanation for why it works:
by reducing *internal covariate shift*.
Presumably by *internal covariate shift* they
meant something like the intuition expressed above---the
notion that the distribution of variable values changes
over the course of training.
However, there were two problems with this explanation:
i) This drift is very different from *covariate shift*,
rendering the name a misnomer. If anything, it is closer to concept drift. 
ii) The explanation offers an under-specified intuition
but leaves the question of *why precisely this technique works*
an open question wanting for a rigorous explanation.
Throughout this book, we aim to convey the intuitions that practitioners
use to guide their development of deep neural networks.
However, we believe that it is important
to separate these guiding intuitions
from established scientific fact.
Eventually, when you master this material
and start writing your own research papers
you will want to be clear to delineate
between technical claims and hunches.

Following the success of batch normalization,
its explanation in terms of *internal covariate shift*
has repeatedly surfaced in debates in the technical literature
and broader discourse about how to present machine learning research.
In a memorable speech given while accepting a Test of Time Award
at the 2017 NeurIPS conference,
Ali Rahimi used *internal covariate shift*
as a focal point in an argument likening
the modern practice of deep learning to alchemy.
Subsequently, the example was revisited in detail
in a position paper outlining
troubling trends in machine learning :cite:`Lipton.Steinhardt.2018`.
Other authors
have proposed alternative explanations for the success of batch normalization,
some :cite:`Santurkar.Tsipras.Ilyas.ea.2018`
claiming that batch normalization's success comes despite exhibiting behavior
that is in some ways opposite to those claimed in the original paper.


We note that the *internal covariate shift*
is no more worthy of criticism than any of
thousands of similarly vague claims
made every year in the technical machine learning literature.
Likely, its resonance as a focal point of these debates
owes to its broad recognizability for the target audience.
Batch normalization became a standard component of convolutional classifiers,
while GroupNorm, LayerNorm, and normalizer-free networks now cover regimes in
which batch statistics are undesirable. The competing explanations remain a
useful case study in separating an empirical result from its proposed mechanism.

The practical points are:

* During model training, batch normalization continuously adjusts the intermediate output of
  the network by utilizing the mean and standard deviation of the minibatch, so that the
  values of the intermediate output in each layer throughout the neural network are more stable.
* Batch normalization is slightly different for fully connected layers than for convolutional layers. In fact,
  for convolutional layers, layer normalization can sometimes be used as an alternative.
* Like a dropout layer, batch normalization layers have different behaviors
  in training mode than in prediction mode.
* Batch normalization is useful for regularization and improving convergence in optimization. By contrast,
  the original motivation of reducing internal covariate shift seems not to be a valid explanation.
* For more robust models that are less sensitive to input perturbations, consider removing batch normalization :cite:`wang2022removing`.

## Exercises

1. Should we remove the bias parameter from the fully connected layer or the convolutional layer before the batch normalization? Why?
1. Compare the learning rates for LeNet with and without batch normalization.
    1. Plot the increase in validation accuracy.
    1. How large can you make the learning rate before the optimization fails in both cases?
1. Do we need batch normalization in every layer? Experiment with it.
1. Implement a "lite" version of batch normalization that only removes the mean, or alternatively one that
   only removes the variance. How does it behave?
1. Fix the parameters `beta` and `gamma`. Observe and analyze the results.
1. Can you replace dropout by batch normalization? How does the behavior change?
1. Research ideas: think of other normalization transforms that you can apply:
    1. Can you apply the probability integral transform?
    1. Can you use a full-rank covariance estimate? Why should you probably not do that? 
    1. Can you use other compact matrix variants (block-diagonal, low-displacement rank, Monarch, etc.)?
    1. Does a sparsification compression act as a regularizer?
    1. Are there other projections (e.g., convex cone, symmetry group-specific transforms) that you can use?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/330)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18005)
:end_tab:

<!-- slides -->

::: {.slide title="BatchNorm stabilizes deep nets"}
**Batch Normalization** (Ioffe & Szegedy, 2015) is the
single-biggest stability win in modern deep learning.

At each layer, **normalize** activations within the
minibatch to zero mean / unit variance, then **rescale**
with learned $\gamma$ and $\beta$:

$$\text{BN}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \hat\mu_\mathcal{B}}{\sqrt{\hat\sigma_\mathcal{B}^2 + \epsilon}} + \beta.$$
:::

::: {.slide title="Why it works"}
- Lets you train **much deeper** nets — gradients stay
  well-conditioned through the depth.
- Allows **higher learning rates**; mildly regularizing.
- **Test time** uses running estimates of mean / variance
  (no minibatch then).
- Spawned a family — **LayerNorm** (per-example, used in
  Transformers), **GroupNorm**, **InstanceNorm**.
:::

::: {.slide title="From scratch"}
Compute per-channel mean and variance over the minibatch (and
spatial dims, for conv); normalize, then scale + shift:

@batch-norm-batch-normalization

@batch-norm-implementation-from-scratch-1
:::

::: {.slide title="Wrapping as a `Module`"}
Buffers for `moving_mean` / `moving_var` (updated only during
training); learnable `gamma` / `beta` parameters:

@batch-norm-implementation-from-scratch-2
:::

::: {.slide title="LeNet + BatchNorm"}
Drop a `BatchNorm` layer between each conv/linear and its
activation:

@batch-norm-lenet-with-batch-normalization-1
:::

::: {.slide title="Train"}
Trains noticeably **faster** than vanilla LeNet — same accuracy in
fewer epochs:

@batch-norm-lenet-with-batch-normalization-3

. . .

After training, `gamma` and `beta` are non-trivial — the layer
**learned** the scale/shift it wants:

@batch-norm-lenet-with-batch-normalization-4
:::

::: {.slide title="The framework version"}
`nn.BatchNorm2d` for conv layers, `nn.BatchNorm1d` for linear
layers — same idea, much faster, handles the eval/training mode
switch automatically:

@batch-norm-concise-implementation-1

. . .

@batch-norm-concise-implementation-2
:::

::: {.slide title="The two real problems"}
BatchNorm couples examples through the batch statistics:

- **Minibatch coupling**: the estimates degrade as batches shrink.
  Detection / segmentation train at 1--2 images per device.
- **Train/serve discrepancy**: minibatch statistics in training,
  moving averages at prediction. One layer, two functions.

. . .

**GroupNorm** (Wu & He, 2018): standardize each example within
groups of channels. No batch in the statistics: batch size 1
works, and training equals serving.
:::

::: {.slide title="GroupNorm in code"}
Per-(example, group) mean 0 / variance 1, at **any** batch size:

@batch-norm-group-normalization

. . .

The default in detection / segmentation heads and diffusion
U-Nets, where per-device batches are small.
:::

::: {.slide title="Recap"}
- BatchNorm normalizes activations to zero mean / unit variance
  within each minibatch, then **rescales** with learned $\gamma,
  \beta$.
- Track running statistics during training; **use them** at
  inference (no minibatch at test time).
- Enables much **deeper networks**, **higher LRs**, faster
  convergence; mildly regularizing.
- Spawned a family — **LayerNorm** (per-example, used in
  Transformers), **GroupNorm**, **InstanceNorm**.
:::
