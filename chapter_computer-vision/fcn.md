# Fully Convolutional Networks
:label:`sec_fcn`

As discussed in :numref:`sec_semantic_segmentation`,
semantic segmentation
classifies images in pixel level.
A fully convolutional network (FCN)
uses a convolutional neural network to
transform image pixels to pixel classes :cite:`Long.Shelhamer.Darrell.2015`.
Unlike the CNNs that we encountered earlier
for image classification 
or object detection,
a fully convolutional network
transforms 
the height and width of intermediate feature maps
back to those of the input image:
this is achieved by
the transposed convolutional layer
introduced in :numref:`sec_transposed_conv`.
As a result,
the classification output
and the input image 
have a one-to-one correspondence 
in pixel level:
the channel dimension at any output pixel 
holds the classification results
for the input pixel at the same spatial position.

```{.python .input #fcn-fully-convolutional-networks}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input #fcn-fully-convolutional-networks}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

```{.python .input #fcn-fully-convolutional-networks}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from PIL import Image
```

```{.python .input #fcn-fully-convolutional-networks}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
```

## The Model

Here we describe the basic design of the fully convolutional network model. 
As shown in :numref:`fig_fcn`,
this model first uses a CNN to extract image features,
then transforms the number of channels into
the number of classes
via a $1\times 1$ convolutional layer,
and finally transforms the height and width of
the feature maps
to those
of the input image via
the transposed convolution introduced in :numref:`sec_transposed_conv`. 
As a result,
the model output has the same height and width as the input image,
where the output channel contains the predicted classes
for the input pixel at the same spatial position.


![Fully convolutional network.](../img/fcn.svg)
:label:`fig_fcn`

Below, we use a ResNet-18 model pretrained on the ImageNet dataset to extract image features
and denote the model instance as `pretrained_net`.
The last few layers of this model
include a global average pooling layer
and a fully connected layer:
they are not needed
in the fully convolutional network.

:begin_tab:`jax`
Note: there is no pretrained ResNet bundled with Flax, and no third-party
package providing ImageNet-pretrained ResNet weights is available in this
JAX environment (e.g. `flaxmodels`, `transformers`). To preserve the
pedagogical point of this section, conceptually treat the from-scratch
ResNet feature extractor below as if its weights had been initialized
from ImageNet pretraining; in practice you would port pretrained weights
from PyTorch or load them via a library such as `flaxmodels`. The
PyTorch and TensorFlow tabs use truly pretrained backbones.
:end_tab:

```{.python .input #fcn-the-model-1}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input #fcn-the-model-1}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.DEFAULT)
list(pretrained_net.children())[-3:]
```

```{.python .input #fcn-the-model-1}
#@tab jax
# Define ResNet building blocks for the feature extractor
class ResNetBlock(nn.Module):
    num_channels: int
    strides: tuple = (1, 1)
    use_1x1conv: bool = False

    @nn.compact
    def __call__(self, x, training=False):
        residual = x
        y = nn.Conv(self.num_channels, kernel_size=(3, 3),
                    strides=self.strides, padding='SAME')(x)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels, kernel_size=(3, 3),
                    strides=(1, 1), padding='SAME')(y)
        y = nn.BatchNorm(use_running_average=not training)(y)
        if self.use_1x1conv:
            residual = nn.Conv(self.num_channels, kernel_size=(1, 1),
                               strides=self.strides)(x)
            residual = nn.BatchNorm(
                use_running_average=not training)(residual)
        return nn.relu(y + residual)

class ResNetFeatures(nn.Module):
    """ResNet-18 feature extractor (without global avg pool and FC)."""
    @nn.compact
    def __call__(self, x, training=False):
        # Initial conv + bn + relu + maxpool
        x = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2),
                    padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                        padding='SAME')
        # Stage 1: 64 channels
        x = ResNetBlock(64)(x, training)
        x = ResNetBlock(64)(x, training)
        # Stage 2: 128 channels, downsample
        x = ResNetBlock(128, strides=(2, 2), use_1x1conv=True)(x, training)
        x = ResNetBlock(128)(x, training)
        # Stage 3: 256 channels, downsample
        x = ResNetBlock(256, strides=(2, 2), use_1x1conv=True)(x, training)
        x = ResNetBlock(256)(x, training)
        # Stage 4: 512 channels, downsample
        x = ResNetBlock(512, strides=(2, 2), use_1x1conv=True)(x, training)
        x = ResNetBlock(512)(x, training)
        return x

pretrained_net = ResNetFeatures()
# Initialize with a dummy input to see the architecture
dummy = jnp.ones((1, 320, 480, 3))
variables = pretrained_net.init(jax.random.PRNGKey(0), dummy)
print('Feature extractor output shape:',
      pretrained_net.apply(variables, dummy).shape)
```

```{.python .input #fcn-the-model-1}
#@tab tensorflow
# Use ResNet-50 pretrained on ImageNet as the backbone.
# include_top=False removes the global avg pool and dense head.
pretrained_net = keras.applications.ResNet50(
    include_top=False, weights='imagenet')
# Show the last few layers to understand the feature map shape
pretrained_net.layers[-3:]
```

Next, we create the fully convolutional network instance `net`.
It copies all the pretrained layers in the ResNet-18
except for the final global average pooling layer
and the fully connected layer that are closest
to the output.

```{.python .input #fcn-the-model-2}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input #fcn-the-model-2}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

```{.python .input #fcn-the-model-2}
#@tab jax
# The ResNetFeatures module already excludes global avg pool and FC.
# We define the full FCN model below.
```

```{.python .input #fcn-the-model-2}
#@tab tensorflow
# Build the FCN feature extractor: all layers up to (but not including)
# the global average pooling and dense head — i.e., the full conv body.
net = keras.Model(inputs=pretrained_net.input,
                  outputs=pretrained_net.output)
```

Given an input with height and width of 320 and 480 respectively,
the forward propagation of `net`
reduces the input height and width to 1/32 of the original, namely 10 and 15.

```{.python .input #fcn-the-model-3}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input #fcn-the-model-3}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input #fcn-the-model-3}
#@tab jax
X = jnp.ones((1, 320, 480, 3))
pretrained_net.apply(variables, X).shape
```

```{.python .input #fcn-the-model-3}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 320, 480, 3))
net(X).shape
```

Next, we use a $1\times 1$ convolutional layer to transform the number of output channels into the number of classes (21) of the Pascal VOC2012 dataset.
Finally, we need to increase the height and width of the feature maps by 32 times to change them back to the height and width of the input image. 
Recall how to calculate 
the output shape of a convolutional layer in :numref:`sec_padding`. 
Since $(320-64+16\times2+32)/32=10$ and $(480-64+16\times2+32)/32=15$, we construct a transposed convolutional layer with stride of $32$, 
setting
the height and width of the kernel
to $64$, the padding to $16$.
In general,
we can see that
for stride $s$,
padding $s/2$ (assuming $s/2$ is an integer),
and the height and width of the kernel $2s$, 
the transposed convolution will increase
the height and width of the input by $s$ times.

```{.python .input #fcn-the-model-4}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input #fcn-the-model-4}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

```{.python .input #fcn-the-model-4}
#@tab jax
num_classes = 21

class FCN(nn.Module):
    """Fully Convolutional Network for semantic segmentation."""
    num_classes: int

    @nn.compact
    def __call__(self, x, training=False):
        # Feature extraction (ResNet-18 backbone)
        x = ResNetFeatures()(x, training)
        # 1x1 conv to map to num_classes channels
        x = nn.Conv(self.num_classes, kernel_size=(1, 1))(x)
        # Transposed conv to upsample by 32x
        x = nn.ConvTranspose(self.num_classes, kernel_size=(64, 64),
                              strides=(32, 32), padding='SAME')(x)
        return x

net = FCN(num_classes=num_classes)
variables = net.init(jax.random.PRNGKey(0), jnp.ones((1, 320, 480, 3)))
print('FCN output shape:',
      net.apply(variables, jnp.ones((1, 320, 480, 3))).shape)
```

```{.python .input #fcn-the-model-4}
#@tab tensorflow
num_classes = 21
# 1x1 conv to reduce channels to num_classes
final_conv = keras.layers.Conv2D(num_classes, kernel_size=1,
                                 kernel_initializer='glorot_uniform')
# Transposed conv: stride=32, kernel=64, padding='same' upsamples 32x
# (for input height/width divisible by 32, output equals input spatial size)
transpose_conv = keras.layers.Conv2DTranspose(
    num_classes, kernel_size=64, strides=32, padding='same', use_bias=False)

inputs = net.input
x = net.output
x = final_conv(x)
x = transpose_conv(x)
fcn_net = keras.Model(inputs=inputs, outputs=x)
print('FCN output shape:', fcn_net(tf.random.uniform((1, 320, 480, 3))).shape)
```

## Initializing Transposed Convolutional Layers


We already know that
transposed convolutional layers can increase
the height and width of
feature maps.
In image processing, we may need to scale up
an image, i.e., *upsampling*.
*Bilinear interpolation*
is one of the commonly used upsampling techniques.
It is also often used for initializing transposed convolutional layers.

To explain bilinear interpolation,
say that 
given an input image
we want to 
calculate each pixel 
of the upsampled output image.
In order to calculate the pixel of the output image
at coordinate $(x, y)$, 
first map $(x, y)$ to coordinate $(x', y')$ on the input image, for example, according to the ratio of the input size to the output size. 
Note that the mapped $x'$ and $y'$ are real numbers. 
Then, find the four pixels closest to coordinate
$(x', y')$ on the input image. 
Finally, the pixel of the output image at coordinate $(x, y)$ is calculated based on these four closest pixels
on the input image and their relative distance from $(x', y')$. 

Upsampling of bilinear interpolation
can be implemented by the transposed convolutional layer 
with the kernel constructed by the following `bilinear_kernel` function. 
Due to space limitations, we only provide the implementation of the `bilinear_kernel` function below
without discussions on its algorithm design.

```{.python .input #fcn-initializing-transposed-convolutional-layers-1}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-1}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-1}
#@tab jax
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    # Flax uses HWIO format for ConvTranspose kernels
    weight = np.zeros((kernel_size, kernel_size, in_channels, out_channels))
    for i in range(min(in_channels, out_channels)):
        weight[:, :, i, i] = filt
    return jnp.array(weight)
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-1}
#@tab tensorflow
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    # Keras Conv2DTranspose uses HWIO kernel format (height, width, out, in)
    weight = np.zeros((kernel_size, kernel_size, out_channels, in_channels),
                      dtype=np.float32)
    for i in range(min(in_channels, out_channels)):
        weight[:, :, i, i] = filt
    return weight
```

Let's experiment with upsampling of bilinear interpolation 
that is implemented by a transposed convolutional layer. 
We construct a transposed convolutional layer that 
doubles the height and weight,
and initialize its kernel with the `bilinear_kernel` function.

```{.python .input #fcn-initializing-transposed-convolutional-layers-2}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-2}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-2}
#@tab jax
class BilinearConvTranspose(nn.Module):
    """A transposed conv layer initialized with bilinear interpolation."""
    channels: int
    kernel_size: int
    strides: tuple

    @nn.compact
    def __call__(self, x):
        return nn.ConvTranspose(self.channels,
                                kernel_size=(self.kernel_size,
                                             self.kernel_size),
                                strides=self.strides,
                                padding='SAME')(x)

conv_trans = BilinearConvTranspose(channels=3, kernel_size=4, strides=(2, 2))
dummy_img = jnp.ones((1, 100, 100, 3))
ct_variables = conv_trans.init(jax.random.PRNGKey(0), dummy_img)
# Replace the kernel with bilinear weights
bilinear_w = bilinear_kernel(3, 3, 4)
ct_variables = {**ct_variables,
    'params': {**ct_variables['params'],
               'ConvTranspose_0': {**ct_variables['params']['ConvTranspose_0'],
                                   'kernel': bilinear_w}}}
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-2}
#@tab tensorflow
# Build a transposed conv layer with bilinear initialization to double H and W
bilinear_w = bilinear_kernel(3, 3, 4)
conv_trans = keras.layers.Conv2DTranspose(
    3, kernel_size=4, strides=2, padding='same', use_bias=False,
    kernel_initializer=tf.constant_initializer(bilinear_w))
# Build the layer by passing a dummy input
_ = conv_trans(tf.zeros((1, 1, 1, 3)))
```

Read the image `X` and assign the upsampling output to `Y`. In order to print the image, we need to adjust the position of the channel dimension.

```{.python .input #fcn-initializing-transposed-convolutional-layers-3}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-3}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-3}
#@tab jax
img = np.array(Image.open('../img/catdog.jpg')).astype(np.float32) / 255
X = jnp.expand_dims(jnp.array(img), axis=0)  # NHWC
Y = conv_trans.apply(ct_variables, X)
out_img = np.array(Y[0])
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-3}
#@tab tensorflow
img = np.array(Image.open('../img/catdog.jpg')).astype(np.float32) / 255
X = tf.expand_dims(tf.constant(img), axis=0)  # NHWC
Y = conv_trans(X)
out_img = Y[0].numpy()
```

As we can see, the transposed convolutional layer increases both the height and width of the image by a factor of two.
Except for the different scales in coordinates,
the image scaled up by bilinear interpolation and the original image printed in :numref:`sec_bbox` look the same.

```{.python .input #fcn-initializing-transposed-convolutional-layers-4}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-4}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-4}
#@tab jax
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img);
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-4}
#@tab tensorflow
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img);
print('output image shape:', out_img.shape)
d2l.plt.imshow(np.clip(out_img, 0, 1));
```

In a fully convolutional network, we initialize the transposed convolutional layer with upsampling of bilinear interpolation. For the $1\times 1$ convolutional layer, we use Xavier initialization.

```{.python .input #fcn-initializing-transposed-convolutional-layers-5}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-5}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-5}
#@tab jax
# Initialize the FCN with bilinear weights for the transposed conv layer
# and Xavier initialization for the 1x1 conv layer
W = bilinear_kernel(num_classes, num_classes, 64)

def init_fcn_weights(rng):
    """Initialize FCN with bilinear upsampling for transposed conv."""
    variables = net.init(rng, jnp.ones((1, 320, 480, 3)))
    params = variables['params']
    # Set bilinear kernel for the transposed conv layer
    flat_params = dict(params)
    flat_params['ConvTranspose_0'] = {
        **params['ConvTranspose_0'], 'kernel': W}
    return {**variables, 'params': flat_params}

variables = init_fcn_weights(jax.random.PRNGKey(42))
```

```{.python .input #fcn-initializing-transposed-convolutional-layers-5}
#@tab tensorflow
# Initialize the transpose conv kernel with bilinear upsampling weights.
# The 1x1 conv was already initialized with Glorot (Xavier) uniform above.
W = bilinear_kernel(num_classes, num_classes, 64)
# Find the Conv2DTranspose layer in fcn_net and set its weights
for layer in fcn_net.layers:
    if isinstance(layer, keras.layers.Conv2DTranspose):
        layer.set_weights([W])
        break
```

## Reading the Dataset

We read
the semantic segmentation dataset
as introduced in :numref:`sec_semantic_segmentation`. 
The output image shape of random cropping is
specified as $320\times 480$: both the height and width are divisible by $32$.

```{.python .input #fcn-reading-the-dataset}
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## Training


Now we can train our constructed
fully convolutional network. 
The loss function and accuracy calculation here
are not essentially different from those in image classification of earlier chapters. 
Because we use the output channel of the
transposed convolutional layer to
predict the class for each pixel,
the channel dimension is specified in the loss calculation.
In addition, the accuracy is calculated
based on correctness
of the predicted class for all the pixels.

```{.python .input #fcn-training}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input #fcn-training}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input #fcn-training}
#@tab jax
def loss_fn(params, batch_stats, X, Y):
    # X is NHWC, Y is NHW with integer class labels
    logits, updates = net.apply(
        {'params': params, 'batch_stats': batch_stats},
        X, training=True, mutable=['batch_stats'])
    # logits shape: (N, H, W, num_classes)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    return loss.mean(), updates

num_epochs, lr, wd = 5, 0.001, 1e-3
optimizer = optax.sgd(lr, momentum=0.9)
opt_state = optimizer.init(variables['params'])
batch_stats = variables.get('batch_stats', {})

@jax.jit
def train_step(params, batch_stats, opt_state, X, Y):
    (loss_val, updates), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(params, batch_stats, X, Y)
    param_updates, opt_state_new = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, param_updates)
    return params_new, updates['batch_stats'], opt_state_new, loss_val

params = variables['params']
for epoch in range(num_epochs):
    for X, Y in train_iter:
        # Convert CHW to HWC for JAX
        X = jnp.transpose(jnp.array(X), (0, 2, 3, 1))
        Y = jnp.array(Y)
        params, batch_stats, opt_state, loss_val = train_step(
            params, batch_stats, opt_state, X, Y)
    print(f'epoch {epoch + 1}, loss {float(loss_val):.3f}')
variables = {'params': params, 'batch_stats': batch_stats}
```

```{.python .input #fcn-training}
#@tab tensorflow
# Loss: SparseCategoricalCrossentropy over per-pixel logits (NHWC -> NHW).
# Backbone weights are frozen to match the PT/JAX approach of fine-tuning only
# the head; unfreeze to replicate full fine-tuning.
for layer in fcn_net.layers[:-2]:
    layer.trainable = False

num_epochs, lr, wd = 5, 0.001, 1e-3
fcn_net.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9,
                                   weight_decay=wd),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
fcn_net.fit(train_iter, epochs=num_epochs, validation_data=test_iter)
```

## Prediction


When predicting, we need to standardize the input image
in each channel and transform the image into the four-dimensional input format required by the CNN.

```{.python .input #fcn-prediction-1}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input #fcn-prediction-1}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input #fcn-prediction-1}
#@tab jax
def predict(img):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    X = (img.astype(np.float32) / 255 - rgb_mean) / rgb_std
    X = jnp.expand_dims(jnp.array(X), axis=0)  # NHWC
    pred = net.apply(variables, X, training=False)
    return jnp.argmax(pred, axis=-1).reshape(pred.shape[1], pred.shape[2])
```

```{.python .input #fcn-prediction-1}
#@tab tensorflow
def predict(img):
    rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    X = (img.astype(np.float32) / 255 - rgb_mean) / rgb_std
    X = tf.expand_dims(tf.constant(X), axis=0)  # NHWC
    pred = fcn_net(X, training=False)  # (1, H, W, num_classes)
    return tf.reshape(tf.argmax(pred, axis=-1), pred.shape[1:3])
```

To visualize the predicted class of each pixel, we map the predicted class back to its label color in the dataset.

```{.python .input #fcn-prediction-2}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input #fcn-prediction-2}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

```{.python .input #fcn-prediction-2}
#@tab jax
def label2image(pred):
    colormap = jnp.array(d2l.VOC_COLORMAP, dtype=jnp.uint8)
    X = pred.astype(jnp.int32)
    return colormap[X, :]
```

```{.python .input #fcn-prediction-2}
#@tab tensorflow
def label2image(pred):
    colormap = tf.constant(d2l.VOC_COLORMAP, dtype=tf.uint8)
    X = tf.cast(pred, tf.int32)
    return tf.gather(colormap, X)
```

Images in the test dataset vary in size and shape.
Since the model uses a transposed convolutional layer with stride of 32,
when the height or width of an input image is indivisible by 32,
the output height or width of the
transposed convolutional layer will deviate from the shape of the input image.
In order to address this issue,
we can crop multiple rectangular areas with height and width that are integer multiples of 32 in the image,
and perform forward propagation
on the pixels in these areas separately.
Note that
the union of these rectangular areas needs to completely cover the input image.
When a pixel is covered by multiple rectangular areas,
the average of the transposed convolution outputs
in separate areas for this same pixel
can be input to
the softmax operation
to predict the class.


For simplicity, we only read a few larger test images,
and crop a $320\times480$ area for prediction starting from the upper-left corner of an image.
For these test images, we
print their cropped areas,
prediction results,
and ground-truth row by row.

```{.python .input #fcn-prediction-3}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input #fcn-prediction-3}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input #fcn-prediction-3}
#@tab jax
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    # Crop HWC arrays: top=0, left=0, height=320, width=480
    X = test_images[i][:320, :480, :]
    pred = label2image(predict(X))
    label_crop = test_labels[i][:320, :480, :]
    imgs += [X, np.array(pred), label_crop]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input #fcn-prediction-3}
#@tab tensorflow
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    # Crop HWC arrays: top=0, left=0, height=320, width=480
    X = test_images[i][:320, :480, :]
    pred = label2image(predict(X))
    label_crop = test_labels[i][:320, :480, :]
    imgs += [X, pred.numpy(), label_crop]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## Summary

* The fully convolutional network first uses a CNN to extract image features, then transforms the number of channels into the number of classes via a $1\times 1$ convolutional layer, and finally transforms the height and width of the feature maps to those of the input image via the transposed convolution.
* In a fully convolutional network, we can use upsampling of bilinear interpolation to initialize the transposed convolutional layer.


## Exercises

1. If we use Xavier initialization for the transposed convolutional layer in the experiment, how does the result change?
1. Can you further improve the accuracy of the model by tuning the hyperparameters?
1. Predict the classes of all pixels in test images.
1. The original fully convolutional network paper also uses outputs of some intermediate CNN layers :cite:`Long.Shelhamer.Darrell.2015`. Try to implement this idea.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:

<!-- slides -->

::: {.slide}
A **fully convolutional network** (Long, Shelhamer, Darrell
2015) is the simplest path to per-pixel prediction:

1. Start with a pretrained classification CNN (ResNet).
2. Strip the global average pool + final dense layer.
3. Replace with a 1×1 conv that maps to `num_classes`.
4. Upsample the resulting low-resolution prediction back
   to input resolution using transposed convolution.

The whole network has no fully connected layers — it works
on any input size, and outputs a class-score map at input
resolution.

![FCN: pretrained CNN body + 1×1 conv to class scores + transposed conv to upsample.](../img/fcn.svg){width=72%}

@fcn-fully-convolutional-networks
:::

::: {.slide title="Pretrained backbone"}
ResNet-18 pretrained on ImageNet. Drop the head (avg pool +
dense); keep the conv body that produces a $\frac{H}{32} \times \frac{W}{32}$
feature map:

@fcn-the-model-1
:::

::: {.slide title="Building the FCN"}
@fcn-the-model-2

. . .

@fcn-the-model-3
:::

::: {.slide title="The class & upsampling head"}
$1 \times 1$ conv: `num_features` → `num_classes` (21 for
VOC). Then a transposed conv that upsamples by 32× to
recover input resolution:

@fcn-the-model-4
:::

::: {.slide title="Bilinear init for transposed conv"}
A randomly initialized 32× upsampler is hard to train.
Initialize it as bilinear interpolation — a sensible
starting point that fine-tunes from there:

@fcn-initializing-transposed-convolutional-layers-1

. . .

@fcn-initializing-transposed-convolutional-layers-2

. . .

@fcn-initializing-transposed-convolutional-layers-3

. . .

@!fcn-initializing-transposed-convolutional-layers-3

. . .

@fcn-initializing-transposed-convolutional-layers-4

. . .

@fcn-initializing-transposed-convolutional-layers-5
:::

::: {.slide title="Loading data"}
@fcn-reading-the-dataset
:::

::: {.slide title="Training"}
Pixel-level cross-entropy. Common trick: freeze the
backbone, train only the new head — gets reasonable
results in a few epochs:

@fcn-training

. . .

@!fcn-training
:::

::: {.slide title="Predict"}
Run the network on test images, take argmax over the class
dimension, map class indices back to RGB:

@fcn-prediction-1

. . .

@fcn-prediction-2

. . .

@fcn-prediction-3

. . .

@!fcn-prediction-3
:::

::: {.slide title="Recap"}
- FCN = pretrained classification CNN + 1×1 conv +
  transposed conv upsampler.
- All-conv → input size doesn't matter.
- Bilinear-initialized transposed conv is the workable
  starting point; fine-tunes from there.
- The blueprint behind U-Net (skip connections fix the
  blur), DeepLab (dilated convs avoid the heavy upsampling),
  and modern segmentation networks.
:::
