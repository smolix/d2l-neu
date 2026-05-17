# Image Augmentation
:label:`sec_image_augmentation`

In :numref:`sec_alexnet`, 
we mentioned that large datasets 
are a prerequisite
for the success of
deep neural networks
in various applications.
*Image augmentation* 
generates similar but distinct training examples
after a series of random changes to the training images, thereby expanding the size of the training set.
Alternatively,
image augmentation can be motivated
by the fact that 
random tweaks of training examples 
allow models to rely less on
certain attributes, thereby improving their generalization ability.
For example, we can crop an image in different ways to make the object of interest appear in different positions, thereby reducing the dependence of a model on the position of the object. 
We can also adjust factors such as brightness and color to reduce a model's sensitivity to color.
It is probably true
that image augmentation was indispensable
for the success of AlexNet at that time.
In this section we will discuss this widely used technique in computer vision.

```{.python .input #image-augmentation}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input #image-augmentation}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings
import numpy as np
warnings.filterwarnings('ignore', message='.*dtype.*align.*',
                        category=np.exceptions.VisibleDeprecationWarning)
```

```{.python .input #image-augmentation}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
from functools import partial
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state
import flax
import optax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
```

```{.python .input #image-augmentation}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import keras
from PIL import Image
import numpy as np
```

## Common Image Augmentation Methods

In our investigation of common image augmentation methods, we will use the following $400\times 500$ image an example.

```{.python .input #image-augmentation-common-image-augmentation-methods-1}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input #image-augmentation-common-image-augmentation-methods-1}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

```{.python .input #image-augmentation-common-image-augmentation-methods-1}
#@tab jax
from PIL import Image
d2l.set_figsize()
img = Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

```{.python .input #image-augmentation-common-image-augmentation-methods-1}
#@tab tensorflow
d2l.set_figsize()
img = Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

Most image augmentation methods have a certain degree of randomness. To make it easier for us to observe the effect of image augmentation, next we define an auxiliary function `apply`. This function runs the image augmentation method `aug` multiple times on the input image `img` and shows all the results.

```{.python .input #image-augmentation-common-image-augmentation-methods-2}
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Flipping and Cropping

:begin_tab:`mxnet`
Flipping the image left and right usually does not change the category of the object. 
This is one of the earliest and most widely used methods of image augmentation.
Next, we use the `transforms` module to create the `RandomFlipLeftRight` instance, which flips
an image left and right with a 50% chance.
:end_tab:

:begin_tab:`pytorch`
Flipping the image left and right usually does not change the category of the object. 
This is one of the earliest and most widely used methods of image augmentation.
Next, we use the `transforms` module to create the `RandomHorizontalFlip` instance, which flips
an image left and right with a 50% chance.
:end_tab:

:begin_tab:`jax`
Flipping the image left and right usually does not change the category of the object. 
This is one of the earliest and most widely used methods of image augmentation.
Next, we define a `RandomHorizontalFlip` function using `tf.image`, which flips
an image left and right with a 50% chance. We convert between PIL images and TensorFlow tensors as needed.
:end_tab:

:begin_tab:`tensorflow`
Flipping the image left and right usually does not change the category of the object. 
This is one of the earliest and most widely used methods of image augmentation.
Next, we define a `RandomHorizontalFlip` function using `tf.image`, which flips
an image left and right with a 50% chance. We convert between PIL images and TensorFlow tensors as needed.
:end_tab:

```{.python .input #image-augmentation-flipping-and-cropping-1}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input #image-augmentation-flipping-and-cropping-1}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

```{.python .input #image-augmentation-flipping-and-cropping-1}
#@tab jax
def RandomHorizontalFlip():
    def aug(img):
        img_tf = tf.constant(np.array(img))
        img_tf = tf.image.random_flip_left_right(img_tf)
        return Image.fromarray(img_tf.numpy())
    return aug

apply(img, RandomHorizontalFlip())
```

```{.python .input #image-augmentation-flipping-and-cropping-1}
#@tab tensorflow
def RandomHorizontalFlip():
    def aug(img):
        img_tf = tf.constant(np.array(img))
        img_tf = tf.image.random_flip_left_right(img_tf)
        return Image.fromarray(img_tf.numpy())
    return aug

apply(img, RandomHorizontalFlip())
```

:begin_tab:`mxnet`
Flipping up and down is not as common as flipping left and right. But at least for this example image, flipping up and down does not hinder recognition.
Next, we create a `RandomFlipTopBottom` instance to flip
an image up and down with a 50% chance.
:end_tab:

:begin_tab:`pytorch`
Flipping up and down is not as common as flipping left and right. But at least for this example image, flipping up and down does not hinder recognition.
Next, we create a `RandomVerticalFlip` instance to flip
an image up and down with a 50% chance.
:end_tab:

:begin_tab:`jax`
Flipping up and down is not as common as flipping left and right. But at least for this example image, flipping up and down does not hinder recognition.
Next, we create a `RandomVerticalFlip` function to flip
an image up and down with a 50% chance.
:end_tab:

:begin_tab:`tensorflow`
Flipping up and down is not as common as flipping left and right. But at least for this example image, flipping up and down does not hinder recognition.
Next, we create a `RandomVerticalFlip` function to flip
an image up and down with a 50% chance.
:end_tab:

```{.python .input #image-augmentation-flipping-and-cropping-2}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input #image-augmentation-flipping-and-cropping-2}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

```{.python .input #image-augmentation-flipping-and-cropping-2}
#@tab jax
def RandomVerticalFlip():
    def aug(img):
        img_tf = tf.constant(np.array(img))
        img_tf = tf.image.random_flip_up_down(img_tf)
        return Image.fromarray(img_tf.numpy())
    return aug

apply(img, RandomVerticalFlip())
```

```{.python .input #image-augmentation-flipping-and-cropping-2}
#@tab tensorflow
def RandomVerticalFlip():
    def aug(img):
        img_tf = tf.constant(np.array(img))
        img_tf = tf.image.random_flip_up_down(img_tf)
        return Image.fromarray(img_tf.numpy())
    return aug

apply(img, RandomVerticalFlip())
```

In the example image we used, the cat is in the middle of the image, but this may not be the case in general. 
In :numref:`sec_pooling`, we explained that the pooling layer can reduce the sensitivity of a convolutional layer to the target position.
In addition, we can also randomly crop the image to make objects appear in different positions in the image at different scales, which can also reduce the sensitivity of a model to the target position.

In the code below, we randomly crop an area with an area of $10\% \sim 100\%$ of the original area each time, and the ratio of width to height of this area is randomly selected from $0.5 \sim 2$. Then, the width and height of the region are both scaled to 200 pixels. 
Unless otherwise specified, the random number between $a$ and $b$ in this section refers to a continuous value obtained by random and uniform sampling from the interval $[a, b]$.

```{.python .input #image-augmentation-flipping-and-cropping-3}
#@tab mxnet
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input #image-augmentation-flipping-and-cropping-3}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input #image-augmentation-flipping-and-cropping-3}
#@tab jax
def RandomResizedCrop(size, scale=(0.1, 1), ratio=(0.5, 2)):
    target_h, target_w = size
    def aug(img):
        img_tf = tf.constant(np.array(img))
        h, w = tf.shape(img_tf)[0], tf.shape(img_tf)[1]
        area = tf.cast(h * w, tf.float32)
        log_ratio = (tf.math.log(float(ratio[0])), tf.math.log(float(ratio[1])))
        target_area = tf.random.uniform([], scale[0], scale[1]) * area
        aspect = tf.exp(tf.random.uniform([], log_ratio[0], log_ratio[1]))
        crop_h = tf.cast(tf.round(tf.sqrt(target_area / aspect)), tf.int32)
        crop_w = tf.cast(tf.round(tf.sqrt(target_area * aspect)), tf.int32)
        crop_h = tf.minimum(crop_h, h)
        crop_w = tf.minimum(crop_w, w)
        offset_h = tf.random.uniform([], 0, h - crop_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, w - crop_w + 1, dtype=tf.int32)
        img_tf = tf.image.crop_to_bounding_box(img_tf, offset_h, offset_w,
                                                crop_h, crop_w)
        img_tf = tf.cast(img_tf, tf.float32)
        img_tf = tf.image.resize(img_tf, [target_h, target_w])
        img_tf = tf.cast(img_tf, tf.uint8)
        return Image.fromarray(img_tf.numpy())
    return aug

shape_aug = RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input #image-augmentation-flipping-and-cropping-3}
#@tab tensorflow
def RandomResizedCrop(size, scale=(0.1, 1), ratio=(0.5, 2)):
    target_h, target_w = size
    def aug(img):
        img_tf = tf.constant(np.array(img))
        h, w = tf.shape(img_tf)[0], tf.shape(img_tf)[1]
        area = tf.cast(h * w, tf.float32)
        log_ratio = (tf.math.log(float(ratio[0])), tf.math.log(float(ratio[1])))
        target_area = tf.random.uniform([], scale[0], scale[1]) * area
        aspect = tf.exp(tf.random.uniform([], log_ratio[0], log_ratio[1]))
        crop_h = tf.cast(tf.round(tf.sqrt(target_area / aspect)), tf.int32)
        crop_w = tf.cast(tf.round(tf.sqrt(target_area * aspect)), tf.int32)
        crop_h = tf.minimum(crop_h, h)
        crop_w = tf.minimum(crop_w, w)
        offset_h = tf.random.uniform([], 0, h - crop_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, w - crop_w + 1, dtype=tf.int32)
        img_tf = tf.image.crop_to_bounding_box(img_tf, offset_h, offset_w,
                                                crop_h, crop_w)
        img_tf = tf.cast(img_tf, tf.float32)
        img_tf = tf.image.resize(img_tf, [target_h, target_w])
        img_tf = tf.cast(img_tf, tf.uint8)
        return Image.fromarray(img_tf.numpy())
    return aug

shape_aug = RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Changing Colors

Another augmentation method is changing colors. We can change four aspects of the image color: brightness, contrast, saturation, and hue. In the example below, we randomly change the brightness of the image to a value between 50% ($1-0.5$) and 150% ($1+0.5$) of the original image.

```{.python .input #image-augmentation-changing-colors-1}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input #image-augmentation-changing-colors-1}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

```{.python .input #image-augmentation-changing-colors-1}
#@tab jax
def RandomBrightness(max_delta):
    def aug(img):
        img_tf = tf.cast(tf.constant(np.array(img)), tf.float32) / 255.0
        img_tf = tf.image.random_brightness(img_tf, max_delta)
        img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
        return Image.fromarray((img_tf.numpy() * 255).astype(np.uint8))
    return aug

apply(img, RandomBrightness(0.5))
```

```{.python .input #image-augmentation-changing-colors-1}
#@tab tensorflow
def RandomBrightness(max_delta):
    def aug(img):
        img_tf = tf.cast(tf.constant(np.array(img)), tf.float32) / 255.0
        img_tf = tf.image.random_brightness(img_tf, max_delta)
        img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
        return Image.fromarray((img_tf.numpy() * 255).astype(np.uint8))
    return aug

apply(img, RandomBrightness(0.5))
```

Similarly, we can randomly change the hue of the image.

```{.python .input #image-augmentation-changing-colors-2}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input #image-augmentation-changing-colors-2}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

```{.python .input #image-augmentation-changing-colors-2}
#@tab jax
def RandomHue(max_delta):
    def aug(img):
        img_tf = tf.cast(tf.constant(np.array(img)), tf.float32) / 255.0
        img_tf = tf.image.random_hue(img_tf, max_delta)
        img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
        return Image.fromarray((img_tf.numpy() * 255).astype(np.uint8))
    return aug

apply(img, RandomHue(0.5))
```

```{.python .input #image-augmentation-changing-colors-2}
#@tab tensorflow
def RandomHue(max_delta):
    def aug(img):
        img_tf = tf.cast(tf.constant(np.array(img)), tf.float32) / 255.0
        img_tf = tf.image.random_hue(img_tf, max_delta)
        img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
        return Image.fromarray((img_tf.numpy() * 255).astype(np.uint8))
    return aug

apply(img, RandomHue(0.5))
```

We can also create a `RandomColorJitter` instance and set how to randomly change the `brightness`, `contrast`, `saturation`, and `hue` of the image at the same time.

```{.python .input #image-augmentation-changing-colors-3}
#@tab mxnet
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input #image-augmentation-changing-colors-3}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input #image-augmentation-changing-colors-3}
#@tab jax
def RandomColorJitter(brightness=0, contrast=0, saturation=0, hue=0):
    def aug(img):
        img_tf = tf.cast(tf.constant(np.array(img)), tf.float32) / 255.0
        if brightness > 0:
            img_tf = tf.image.random_brightness(img_tf, brightness)
        if contrast > 0:
            img_tf = tf.image.random_contrast(img_tf, 1 - contrast,
                                              1 + contrast)
        if saturation > 0:
            img_tf = tf.image.random_saturation(img_tf, 1 - saturation,
                                                1 + saturation)
        if hue > 0:
            img_tf = tf.image.random_hue(img_tf, hue)
        img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
        return Image.fromarray((img_tf.numpy() * 255).astype(np.uint8))
    return aug

color_aug = RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                              hue=0.5)
apply(img, color_aug)
```

```{.python .input #image-augmentation-changing-colors-3}
#@tab tensorflow
def RandomColorJitter(brightness=0, contrast=0, saturation=0, hue=0):
    def aug(img):
        img_tf = tf.cast(tf.constant(np.array(img)), tf.float32) / 255.0
        if brightness > 0:
            img_tf = tf.image.random_brightness(img_tf, brightness)
        if contrast > 0:
            img_tf = tf.image.random_contrast(img_tf, 1 - contrast,
                                              1 + contrast)
        if saturation > 0:
            img_tf = tf.image.random_saturation(img_tf, 1 - saturation,
                                                1 + saturation)
        if hue > 0:
            img_tf = tf.image.random_hue(img_tf, hue)
        img_tf = tf.clip_by_value(img_tf, 0.0, 1.0)
        return Image.fromarray((img_tf.numpy() * 255).astype(np.uint8))
    return aug

color_aug = RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                              hue=0.5)
apply(img, color_aug)
```

### Combining Multiple Image Augmentation Methods

In practice, we will combine multiple image augmentation methods. 
For example,
we can combine the different image augmentation methods defined above and apply them to each image via a `Compose` instance.

```{.python .input #image-augmentation-combining-multiple-image-augmentation-methods}
#@tab mxnet
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input #image-augmentation-combining-multiple-image-augmentation-methods}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input #image-augmentation-combining-multiple-image-augmentation-methods}
#@tab jax
def Compose(transforms):
    def aug(img):
        for t in transforms:
            img = t(img)
        return img
    return aug

augs = Compose([RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input #image-augmentation-combining-multiple-image-augmentation-methods}
#@tab tensorflow
def Compose(transforms):
    def aug(img):
        for t in transforms:
            img = t(img)
        return img
    return aug

augs = Compose([RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## Training with Image Augmentation

Let's train a model with image augmentation.
Here we use the CIFAR-10 dataset instead of the Fashion-MNIST dataset that we used before. 
This is because the position and size of the objects in the Fashion-MNIST dataset have been normalized, while the color and size of the objects in the CIFAR-10 dataset have more significant differences. 
The first 32 training images in the CIFAR-10 dataset are shown below.

```{.python .input #image-augmentation-training-with-image-augmentation-1}
#@tab mxnet
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[:32][0], 4, 8, scale=0.8);
```

```{.python .input #image-augmentation-training-with-image-augmentation-1}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

```{.python .input #image-augmentation-training-with-image-augmentation-1}
#@tab jax
(train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
d2l.show_images([Image.fromarray(train_images[i]) for i in range(32)],
                4, 8, scale=0.8);
```

```{.python .input #image-augmentation-training-with-image-augmentation-1}
#@tab tensorflow
(train_images, train_labels), _ = keras.datasets.cifar10.load_data()
d2l.show_images([Image.fromarray(train_images[i]) for i in range(32)],
                4, 8, scale=0.8);
```

In order to obtain definitive results during prediction, we usually only apply image augmentation to training examples, and do not use image augmentation with random operations during prediction. 
Here we only use the simplest random left-right flipping method. In addition, we use a `ToTensor` instance to convert a minibatch of images into the format required by the deep learning framework, i.e., 
32-bit floating point numbers between 0 and 1 with the shape of (batch size, number of channels, height, width).

```{.python .input #image-augmentation-training-with-image-augmentation-2}
#@tab mxnet
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input #image-augmentation-training-with-image-augmentation-2}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

```{.python .input #image-augmentation-training-with-image-augmentation-2}
#@tab jax
def train_augs(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_flip_left_right(image)
    return image, label

def test_augs(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
```

```{.python .input #image-augmentation-training-with-image-augmentation-2}
#@tab tensorflow
def train_augs(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_flip_left_right(image)
    return image, label

def test_augs(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
```

:begin_tab:`mxnet`
Next, we define an auxiliary function to facilitate reading the image and
applying image augmentation. 
The `transform_first` function provided by Gluon's
datasets applies image augmentation to the first element of each training
example (image and label), i.e., the image. 
For
a detailed introduction to `DataLoader`, please refer to :numref:`sec_fashion_mnist`.
:end_tab:

:begin_tab:`pytorch`
Next, we define an auxiliary function to facilitate reading the image and
applying image augmentation. 
The `transform` argument provided by PyTorch's
dataset applies augmentation to transform the images.
For
a detailed introduction to `DataLoader`, please refer to :numref:`sec_fashion_mnist`.
:end_tab:

:begin_tab:`jax`
Next, we define an auxiliary function to facilitate reading the image and
applying image augmentation. 
We use `tf.keras.datasets` to load CIFAR-10 and `tf.data.Dataset` for batching,
then convert each batch to NumPy arrays via `tfds.as_numpy()` for use with JAX.
For
a detailed introduction to data loading, please refer to :numref:`sec_fashion_mnist`.
:end_tab:

:begin_tab:`tensorflow`
Next, we define an auxiliary function to facilitate reading the image and
applying image augmentation.
We use `keras.datasets` to load CIFAR-10 and `tf.data.Dataset` for batching
and preprocessing.
For
a detailed introduction to data loading, please refer to :numref:`sec_fashion_mnist`.
:end_tab:

```{.python .input #image-augmentation-training-with-image-augmentation-3}
#@tab mxnet
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input #image-augmentation-training-with-image-augmentation-3}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

```{.python .input #image-augmentation-training-with-image-augmentation-3}
#@tab jax
def load_cifar10(is_train, aug_fn, batch_size):
    (train_imgs, train_lbls), (test_imgs, test_lbls) = (
        tf.keras.datasets.cifar10.load_data())
    if is_train:
        images, labels = train_imgs, train_lbls.squeeze()
    else:
        images, labels = test_imgs, test_lbls.squeeze()
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        ds = ds.shuffle(10000)
    ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
```

```{.python .input #image-augmentation-training-with-image-augmentation-3}
#@tab tensorflow
def load_cifar10(is_train, aug_fn, batch_size):
    (train_imgs, train_lbls), (test_imgs, test_lbls) = (
        keras.datasets.cifar10.load_data())
    if is_train:
        images, labels = train_imgs, train_lbls.squeeze()
    else:
        images, labels = test_imgs, test_lbls.squeeze()
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        ds = ds.shuffle(10000)
    ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
```

### Multi-GPU Training

We train the ResNet-18 model from
:numref:`sec_resnet` on the
CIFAR-10 dataset.
Recall the introduction to
multi-GPU training in :numref:`sec_multi_gpu_concise`.
In the following,
we define a function to train and evaluate the model using multiple GPUs.

```{.python .input #image-augmentation-multi-gpu-training-1}
#@tab mxnet
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input #image-augmentation-multi-gpu-training-1}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum() if l.numel() > 1 else l * y.numel()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input #image-augmentation-multi-gpu-training-1}
#@tab jax
#@save
@partial(jax.jit, static_argnums=(3, 4))  # net, loss_fn are static
def train_batch_ch13(state, X, y, net, loss_fn):
    """Train for a minibatch with JAX (defined in Chapter 13)."""
    def compute_loss(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            X, mutable=['batch_stats'])
        loss = loss_fn(logits, y).mean()
        return loss, (logits, updates)
    (loss, (logits, updates)), grads = jax.value_and_grad(
        compute_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    train_loss_sum = loss * X.shape[0]
    train_acc_sum = (logits.argmax(axis=-1) == y).sum()
    return state, train_loss_sum, train_acc_sum
```

```{.python .input #image-augmentation-multi-gpu-training-1}
#@tab tensorflow
#@save
def train_batch_ch13(net, X, y, loss, optimizer):
    """Train for a minibatch with Keras (defined in Chapter 13)."""
    with tf.GradientTape() as tape:
        pred = net(X, training=True)
        l = loss(y, pred)
    grads = tape.gradient(l, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    train_loss_sum = tf.reduce_sum(l)
    train_acc_sum = tf.reduce_sum(
        tf.cast(tf.argmax(pred, axis=1) == tf.cast(y, tf.int64), tf.float32))
    return train_loss_sum, train_acc_sum
```

```{.python .input #image-augmentation-multi-gpu-training-2}
#@tab mxnet
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input #image-augmentation-multi-gpu-training-2}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input #image-augmentation-multi-gpu-training-2}
#@tab jax
#@save
def train_ch13(net, train_iter, test_iter, loss_fn, state, num_epochs):
    """Train a model with JAX (defined in Chapter 13)."""
    num_batches = sum(1 for _ in tfds.as_numpy(train_iter))
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()

    # Use a separate eval module (training=False) so BatchNorm uses
    # running stats instead of batch stats at test time. Params are shared
    # with the training network; only the `training` flag differs.
    eval_net = net.clone(training=False)

    @jax.jit
    def eval_step(params, batch_stats, X):
        logits = eval_net.apply(
            {'params': params, 'batch_stats': batch_stats}, X)
        return logits

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(tfds.as_numpy(train_iter)):
            timer.start()
            state, l, acc = train_batch_ch13(
                state, jnp.array(features), jnp.array(labels), net, loss_fn)
            n = features.shape[0]
            metric.add(float(l), float(acc), n, n)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        # Evaluate on test set
        correct, total = 0, 0
        for X, y in tfds.as_numpy(test_iter):
            logits = eval_step(state.params, state.batch_stats, jnp.array(X))
            correct += int((logits.argmax(axis=-1) == y).sum())
            total += y.shape[0]
        test_acc = correct / total
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')
    return state
```

```{.python .input #image-augmentation-multi-gpu-training-2}
#@tab tensorflow
#@save
def train_ch13(net, train_iter, test_iter, loss, optimizer, num_epochs):
    """Train a model with Keras (defined in Chapter 13)."""
    num_batches = sum(1 for _ in train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, optimizer)
            n = features.shape[0]
            metric.add(float(l), float(acc), n, n)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        # Evaluate on test set
        correct, total = 0, 0
        for X, y in test_iter:
            logits = net(X, training=False)
            correct += int(tf.reduce_sum(tf.cast(
                tf.argmax(logits, axis=1) == tf.cast(y, tf.int64),
                tf.float32)))
            total += y.shape[0]
        test_acc = correct / total
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')
```

Now we can define the `train_with_data_aug` function to train the model with image augmentation.
This function gets all available GPUs, 
uses Adam as the optimization algorithm,
applies image augmentation to the training dataset,
and finally calls the `train_ch13` function just defined to train and evaluate the model.

```{.python .input #image-augmentation-multi-gpu-training-3}
#@tab mxnet
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input #image-augmentation-multi-gpu-training-3}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
net.apply(d2l.init_cnn)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input #image-augmentation-multi-gpu-training-3}
#@tab jax
batch_size = 256

class ResNet18(nn.Module):
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
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
            lambda x: x.mean(axis=(1, 2)),
            nn.Dense(self.num_classes),
        ])

    def __call__(self, x):
        return self.net(x)

net = ResNet18(num_classes=10, training=True)

def train_with_data_aug(train_aug_fn, test_aug_fn, net, lr=0.001):
    train_iter = load_cifar10(True, train_aug_fn, batch_size)
    test_iter = load_cifar10(False, test_aug_fn, batch_size)
    loss_fn = optax.softmax_cross_entropy_with_integer_labels
    # Initialize model parameters
    dummy_input = jnp.ones((1, 32, 32, 3))
    key = jax.random.PRNGKey(0)
    variables = net.init(key, dummy_input)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    class TrainState(train_state.TrainState):
        batch_stats: dict

    state = TrainState.create(apply_fn=net.apply, params=params,
                              tx=optax.adam(lr), batch_stats=batch_stats)
    state = train_ch13(net, train_iter, test_iter, loss_fn, state, 10)
```

```{.python .input #image-augmentation-multi-gpu-training-3}
#@tab tensorflow
batch_size = 256

def get_net_tf():
    net = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                            input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(512, kernel_size=3, strides=2, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10),
    ])
    return net

def train_with_data_aug(train_aug_fn, test_aug_fn, net, lr=0.001):
    train_iter = load_cifar10(True, train_aug_fn, batch_size)
    test_iter = load_cifar10(False, test_aug_fn, batch_size)
    loss = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    train_ch13(net, train_iter, test_iter, loss, optimizer, 10)

net = get_net_tf()
```

Let's train the model using image augmentation based on random left-right flipping.

```{.python .input #image-augmentation-multi-gpu-training-4}
train_with_data_aug(train_augs, test_augs, net)
```

## Summary

* Image augmentation generates random images based on existing training data to improve the generalization ability of models.
* In order to obtain definitive results during prediction, we usually only apply image augmentation to training examples, and do not use image augmentation with random operations during prediction.
* Deep learning frameworks provide many different image augmentation methods, which can be applied simultaneously.


## Exercises

1. Train the model without using image augmentation: `train_with_data_aug(test_augs, test_augs)`. Compare training and testing accuracy when using and not using image augmentation. Can this comparative experiment support the argument that image augmentation can mitigate overfitting? Why?
1. Combine multiple different image augmentation methods in model training on the CIFAR-10 dataset. Does it improve test accuracy? 
1. Refer to the online documentation of the deep learning framework. What other image augmentation methods does it also provide?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:

<!-- slides -->

::: {.slide}
**Image augmentation** multiplies the dataset's effective
size for free: apply small, label-preserving perturbations
on the fly during training (flips, crops, color jitter).
Each example is seen many times but never the *exact* same
way.

Two effects:

- More data = less overfitting.
- The model learns invariance to whatever you apply.

Modern pipelines: random erasing, mixup, cutmix,
RandAugment — same idea, more aggressive.
:::

::: {.slide title="Setup"}
Load a sample image and a helper to display a grid of
augmented samples:

- start from one image;
- sample a random transform several times;
- visualize a grid so the transform distribution is visible,
  not just one lucky draw.

The important object is the augmentation distribution, not the
helper function used to plot it.
:::

::: {.slide title="Reference image"}
@!image-augmentation-common-image-augmentation-methods-1
:::

::: {.slide title="Flips and crops"}
Random horizontal flip — the cheapest, most-used
augmentation:

@image-augmentation-flipping-and-cropping-1

. . .

Vertical flip — used selectively (faces? probably not):

@image-augmentation-flipping-and-cropping-2
:::

::: {.slide title="Random resized crop"}
Crop a random rectangle, resize back to the input size.
The single most effective augmentation in vision: scale
invariance and translation invariance in one trick:

@image-augmentation-flipping-and-cropping-3
:::

::: {.slide title="Color jitter — brightness"}
@image-augmentation-changing-colors-1
:::

::: {.slide title="Color jitter — hue"}
@image-augmentation-changing-colors-2
:::

::: {.slide title="Combined color jitter"}
Brightness, contrast, saturation, hue — all at once. Tame
the magnitudes; large jitters destroy semantic content:

@image-augmentation-changing-colors-3
:::

::: {.slide title="Composing augmentations"}
`Compose([flip, crop, color, ToTensor])` — a pipeline of
transforms applied in order. Standard recipe:

@image-augmentation-combining-multiple-image-augmentation-methods
:::

::: {.slide title="Training with augmentation"}
Train CIFAR-10 ResNet18 with and without augmentation. Same
model, same hyperparameters — augmentation just transforms
the data loader output:

- training loader: random crop + random horizontal flip;
- test loader: deterministic normalization only;
- model and optimizer stay unchanged.

This separation matters: evaluation should measure the trained
classifier, not randomness in the augmentation pipeline.
:::

::: {.slide title="CIFAR-10 samples"}
@!image-augmentation-training-with-image-augmentation-1
:::

::: {.slide title="Training helper"}
The training helper has no augmentation-specific logic.

It receives already-transformed minibatches from the data loader,
then performs the usual supervised update:

$$
\mathbf{x}' \sim a(\mathbf{x}), \qquad
\min_\theta \ell(f_\theta(\mathbf{x}'), y).
$$

That is the clean abstraction: augment in input pipeline, train in
optimization loop.
:::

::: {.slide title="Train it"}
@!image-augmentation-multi-gpu-training-4
:::

::: {.slide title="Recap"}
- Augmentation = label-preserving random perturbations
  applied each epoch — effectively multiplies the dataset
  size.
- Standard recipe: random horizontal flip + random
  resized crop + light color jitter.
- Modern aggressive variants (RandAugment, mixup, cutmix,
  AutoAugment) push accuracy further with the same data.
- Apply only at training time; eval uses center crop and
  no jitter.
:::
