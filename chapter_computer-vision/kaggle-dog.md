# Dog Breed Identification (ImageNet Dogs) on Kaggle
:label:`sec_kaggle_dog`

In this section, we will practice
the dog breed identification problem on
Kaggle. The web address of this competition is https://www.kaggle.com/c/dog-breed-identification

In this competition,
120 different breeds of dogs will be recognized.
In fact,
the dataset for this competition is
a subset of the ImageNet dataset.
Unlike the images in the CIFAR-10 dataset in :numref:`sec_kaggle_cifar10`,
the images in the ImageNet dataset are both higher and wider in varying dimensions.
:numref:`fig_kaggle_dog` shows the information on the competition's webpage. You need a Kaggle account
to submit your results.


![The dog breed identification competition website. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input #kaggle-dog-dog-breed-identification-imagenet-dogs-on-kaggle}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input #kaggle-dog-dog-breed-identification-imagenet-dogs-on-kaggle}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

```{.python .input #kaggle-dog-dog-breed-identification-imagenet-dogs-on-kaggle}
#@tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import flaxmodels as fm
import tensorflow as tf  # data pipeline only (tf.data); all compute runs in JAX
import os
```

```{.python .input #kaggle-dog-dog-breed-identification-imagenet-dogs-on-kaggle}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import keras
import numpy as np
import os
```

## Obtaining and Organizing the Dataset

The competition dataset is divided into a training set and a test set, which contain 10222 and 10357 JPEG images
of three RGB (color) channels, respectively.
Among the training dataset,
there are 120 breeds of dogs
such as Labradors, Poodles, Dachshunds, Samoyeds, Huskies, Chihuahuas, and Yorkshire Terriers.


### Downloading the Dataset

After logging into Kaggle,
you can click on the "Data" tab on the
competition webpage shown in :numref:`fig_kaggle_dog` and download the dataset by clicking the "Download All" button.
After unzipping the downloaded file in `../data`, you will find the entire dataset in the following paths:

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test

You may have noticed that the above structure is
similar to that of the CIFAR-10 competition in :numref:`sec_kaggle_cifar10`, where folders `train/` and `test/` contain training and testing dog images, respectively, and `labels.csv` contains
the labels for the training images.
Similarly, to make it easier to get started, we provide a small sample of the dataset mentioned above: `train_valid_test_tiny.zip`.
If you are going to use the full dataset for the Kaggle competition, you need to change the `demo` variable below to `False`.

```{.python .input #kaggle-dog-downloading-the-dataset}
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `False`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### Organizing the Dataset

We can organize the dataset similarly to what we did in :numref:`sec_kaggle_cifar10`, namely splitting out
a validation set from the original training set, and moving images into subfolders grouped by labels.

The `reorg_dog_data` function below reads
the training data labels, splits out the validation set, and organizes the training set.

```{.python .input #kaggle-dog-organizing-the-dataset}
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## Image Augmentation

Recall that this dog breed dataset
is a subset of the ImageNet dataset,
whose images
are larger than those of the CIFAR-10 dataset
in :numref:`sec_kaggle_cifar10`.
The following
lists a few image augmentation operations
that might be useful for relatively larger images.

```{.python .input #kaggle-dog-image-augmentation-1}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input #kaggle-dog-image-augmentation-1}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

```{.python .input #kaggle-dog-image-augmentation-1}
#@tab jax
def transform_train_fn(image, label):
    """Training augmentation: random crop, flip, color jitter; scale to [0, 1]
    (the ResNet-34 backbone applies ImageNet mean/std internally)."""
    image = tf.cast(image, tf.float32)
    # Random resized crop to 224x224
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.4 * 255)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image / 255.0, label
```

```{.python .input #kaggle-dog-image-augmentation-1}
#@tab tensorflow
def transform_train_fn(image, label):
    """Training augmentation: random crop, flip, color jitter, normalize."""
    image = tf.cast(image, tf.float32)
    # Random resized crop to 224x224
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.4 * 255)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return tf.keras.applications.resnet50.preprocess_input(image), label
```

During prediction,
we only use image preprocessing operations
without randomness.

```{.python .input #kaggle-dog-image-augmentation-2}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input #kaggle-dog-image-augmentation-2}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

```{.python .input #kaggle-dog-image-augmentation-2}
#@tab jax
def transform_test_fn(image, label):
    """Test preprocessing: resize, center crop; scale to [0, 1] (the ResNet-34
    backbone applies ImageNet mean/std internally)."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    # Center crop to 224x224
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    return image / 255.0, label
```

```{.python .input #kaggle-dog-image-augmentation-2}
#@tab tensorflow
def transform_test_fn(image, label):
    """Test preprocessing: resize, center crop, normalize."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    # Center crop to 224x224
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    return tf.keras.applications.resnet50.preprocess_input(image), label
```

## Reading the Dataset

As in :numref:`sec_kaggle_cifar10`,
we can read the organized dataset
consisting of raw image files.

```{.python .input #kaggle-dog-reading-the-dataset-1}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
```

```{.python .input #kaggle-dog-reading-the-dataset-1}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

```{.python .input #kaggle-dog-reading-the-dataset-1}
#@tab jax
def _load_image_folder_tf(folder_path):
    """Load images from a class-subfolder directory into a tf.data.Dataset."""
    ds = tf.keras.utils.image_dataset_from_directory(
        folder_path, label_mode='int', image_size=(256, 256),
        batch_size=None, shuffle=False)
    return ds

train_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'train'))
train_valid_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'train_valid'))
valid_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'valid'))
test_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'test'))
```

```{.python .input #kaggle-dog-reading-the-dataset-1}
#@tab tensorflow
def _load_image_folder_tf(folder_path):
    """Load images from a class-subfolder directory into a tf.data.Dataset."""
    ds = keras.utils.image_dataset_from_directory(
        folder_path, label_mode='int', image_size=(256, 256),
        batch_size=None, shuffle=False)
    return ds

train_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'train'))
train_valid_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'train_valid'))
valid_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'valid'))
test_ds = _load_image_folder_tf(
    os.path.join(data_dir, 'train_valid_test', 'test'))
```

Below we create data iterator instances
the same way
as in :numref:`sec_kaggle_cifar10`.

```{.python .input #kaggle-dog-reading-the-dataset-2}
#@tab mxnet
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input #kaggle-dog-reading-the-dataset-2}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

```{.python .input #kaggle-dog-reading-the-dataset-2}
#@tab jax
train_iter = (train_ds.map(transform_train_fn, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(10000).batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
train_valid_iter = (train_valid_ds.map(transform_train_fn,
                    num_parallel_calls=tf.data.AUTOTUNE)
                    .shuffle(10000).batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))
valid_iter = (valid_ds.map(transform_test_fn, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
test_iter = (test_ds.map(transform_test_fn, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(batch_size, drop_remainder=False)
             .prefetch(tf.data.AUTOTUNE))
```

```{.python .input #kaggle-dog-reading-the-dataset-2}
#@tab tensorflow
train_iter = (train_ds.map(transform_train_fn, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(10000).batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
train_valid_iter = (train_valid_ds.map(transform_train_fn,
                    num_parallel_calls=tf.data.AUTOTUNE)
                    .shuffle(10000).batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))
valid_iter = (valid_ds.map(transform_test_fn, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
test_iter = (test_ds.map(transform_test_fn, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(batch_size, drop_remainder=False)
             .prefetch(tf.data.AUTOTUNE))
```

## Fine-Tuning a Pretrained Model

Again,
the dataset for this competition is a subset of the ImageNet dataset.
Therefore, we can use the approach discussed in
:numref:`sec_fine_tuning`
to select a model pretrained on the
full ImageNet dataset and use it to extract image features to be fed into a
custom small-scale output network.
High-level APIs of deep learning frameworks
provide a wide range of models
pretrained on the ImageNet dataset.
Here, we choose
a pretrained ResNet-34 model,
where we simply reuse
the input of this model's output layer
(i.e., the extracted
features).
Then we can replace the original output layer with a small custom
output network that can be trained,
such as stacking two
fully connected layers.
Different from the experiment in
:numref:`sec_fine_tuning`,
the following does
not retrain the pretrained model used for feature
extraction. This reduces training time and
memory for storing gradients.

Recall that we
standardized images using
the means and standard deviations of the three RGB channels for the full ImageNet dataset.
In fact,
this is also consistent with the standardization operation
by the pretrained model on ImageNet.

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-1}
#@tab mxnet
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential()
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.reset_ctx(devices)
    return finetune_net
```

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-1}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(
        weights=torchvision.models.ResNet34_Weights.DEFAULT)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-1}
#@tab jax
# Frozen ImageNet-pretrained ResNet-34 (Flax, runs on GPU), matching the
# PyTorch/MXNet tabs: it emits 1000 ImageNet logits, on top of which we train a
# small dog-breed head. `normalize=True` applies the standard ImageNet mean/std
# internally, so the data pipeline only needs images in [0, 1].
class OutputNet(nn.Module):
    """Small output network for fine-tuning."""
    num_classes: int = 120

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

def get_net():
    backbone = fm.ResNet34(output='logits', pretrained='imagenet')
    backbone_vars = backbone.init(jax.random.PRNGKey(0),
                                  jnp.ones((1, 224, 224, 3)))
    output_net = OutputNet(num_classes=120)
    return backbone, backbone_vars, output_net
```

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-1}
#@tab tensorflow
def get_net():
    # Load pretrained ResNet50, freeze backbone, add custom head. Keep the
    # ImageNet logits as frozen features to match the PyTorch tab.
    backbone = keras.applications.ResNet50(
        weights='imagenet', include_top=True, classifier_activation=None,
        input_shape=(224, 224, 3))
    backbone.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = backbone(inputs, training=False)
    x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(120)(x)
    finetune_net = keras.Model(inputs, outputs)
    return finetune_net
```

Before calculating the loss,
we first obtain the input of the pretrained model's output layer, i.e., the extracted feature.
Then we use this feature as input for our small custom output network to calculate the loss.

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-2}
#@tab mxnet
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-2}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-2}
#@tab jax
def loss_fn(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)

def extract_features(backbone, backbone_vars, X_batch):
    """Frozen ResNet-34 forward (on GPU) -> 1000-dim ImageNet logits."""
    return backbone.apply(backbone_vars, jnp.asarray(X_batch), train=False)

def precompute_features(backbone, backbone_vars, data_iter):
    """Run the frozen backbone (on GPU) over the whole dataset and cache the
    (features, labels) tensors as JAX arrays. Subsequent training only
    iterates the small classifier head over these cached features."""
    feats_list, labels_list = [], []
    for features, labels in data_iter:
        f = extract_features(backbone, backbone_vars, features.numpy())
        feats_list.append(np.asarray(f))
        labels_list.append(labels.numpy())
    feats = jnp.array(np.concatenate(feats_list, axis=0))
    labels = jnp.array(np.concatenate(labels_list, axis=0))
    return feats, labels

def evaluate_loss_from_feats(feats, labels, output_net, variables,
                             batch_size):
    l_sum, n = 0.0, 0
    for i in range(0, feats.shape[0], batch_size):
        fb = feats[i:i + batch_size]
        yb = labels[i:i + batch_size]
        logits = output_net.apply(variables, fb, training=False)
        l = loss_fn(logits, yb)
        l_sum += float(l.sum())
        n += int(yb.shape[0])
    return l_sum / n
```

```{.python .input #kaggle-dog-fine-tuning-a-pretrained-model-2}
#@tab tensorflow
loss = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def evaluate_loss(data_iter, net):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        logits = net(features, training=False)
        l = loss(labels, logits)
        l_sum += float(tf.reduce_sum(l))
        n += len(labels)
    return l_sum / n
```

## Defining the Training Function

We will select the model and tune hyperparameters according to the model's performance on the validation set. The model training function `train` only
iterates parameters of the small custom output network.

```{.python .input #kaggle-dog-defining-the-training-function}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input #kaggle-dog-defining-the-training-function}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input #kaggle-dog-defining-the-training-function}
#@tab jax
def train(backbone, backbone_vars, output_net, train_iter, valid_iter,
          num_epochs, lr, wd, lr_period, lr_decay):
    # Only train the small custom output network
    # ResNet50 with include_top=True outputs 1000 ImageNet logits.
    dummy = jnp.ones((1, 1000))
    variables = output_net.init(jax.random.PRNGKey(0), dummy, training=True)
    timer = d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)

    # Run the frozen TF backbone over the training set once to determine
    # n_train and num_batches, which are needed to configure the LR schedule
    # before the epoch loop starts.
    print('Pre-extracting train features...')
    train_feats, train_labels = precompute_features(backbone, backbone_vars,
                                                     train_iter)
    if valid_iter is not None:
        print('Pre-extracting valid features...')
        valid_feats, valid_labels = precompute_features(backbone, backbone_vars,
                                                        valid_iter)
    # Use the same batch size as the data loader (defined globally).
    bs = batch_size
    n_train = int(train_feats.shape[0])
    num_batches = (n_train + bs - 1) // bs

    # `optax.exponential_decay.transition_steps` counts *gradient-update
    # steps*, not epochs — unlike PyTorch's `StepLR(step_size=lr_period)`,
    # which the PT tab steps once per epoch. Scale by `num_batches` so the
    # LR decays every `lr_period` *epochs*, matching PT/MX.
    schedule = optax.exponential_decay(
        init_value=lr, transition_steps=lr_period * num_batches,
        decay_rate=lr_decay, staircase=True)
    tx = optax.chain(optax.add_decayed_weights(wd),
                     optax.sgd(schedule, momentum=0.9))
    opt_state = tx.init(variables['params'])

    @jax.jit
    def train_step(variables, opt_state, feats, y):
        def compute_loss(params):
            logits = output_net.apply({'params': params}, feats,
                                      training=True)
            l = loss_fn(logits, y)
            # Backprop on the per-batch *sum* (not mean) to match the PT/TF
            # tabs, which use reduction='none' + .sum(). Otherwise the
            # effective learning rate here is 1/batch_size smaller and the
            # head barely moves.
            s = l.sum()
            return s, s
        grads, l_sum = jax.grad(
            compute_loss, has_aux=True)(variables['params'])
        updates, new_opt_state = tx.update(grads, opt_state,
                                           variables['params'])
        new_params = optax.apply_updates(variables['params'], updates)
        new_variables = {'params': new_params}
        return new_variables, new_opt_state, l_sum

    rng = np.random.default_rng(0)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        # Re-extract train features each epoch so each epoch sees freshly
        # augmented images (random crop/flip/jitter from the tf.data pipeline).
        # This matches PyTorch, which runs augmentation + backbone forward on
        # every batch in every epoch rather than caching a single augmented pass.
        train_feats, train_labels = precompute_features(backbone, backbone_vars,
                                                        train_iter)
        # Shuffle indices each epoch
        perm = rng.permutation(n_train)
        for i in range(num_batches):
            timer.start()
            idx = perm[i * bs:(i + 1) * bs]
            feats = train_feats[idx]
            y = train_labels[idx]
            variables, opt_state, l = train_step(
                variables, opt_state, feats, y)
            metric.add(float(l), int(y.shape[0]))
            timer.stop()
            if (i + 1) % max(num_batches // 5, 1) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss_from_feats(
                valid_feats, valid_labels, output_net, variables, bs)
            animator.add(epoch + 1, (None, valid_loss))
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec')
    return variables
```

```{.python .input #kaggle-dog-defining-the-training-function}
#@tab tensorflow
def train(net, train_iter, valid_iter, num_epochs, lr, wd, lr_period,
          lr_decay):
    # Only train the custom head; backbone is already frozen in get_net()
    # Keras's `ExponentialDecay.decay_steps` counts *gradient-update
    # steps*, not epochs — unlike PyTorch's `StepLR(step_size=lr_period)`,
    # which the PT tab steps once per epoch. Scale by `num_batches` so the
    # LR decays every `lr_period` *epochs*, matching PT/MX.
    num_batches = sum(1 for _ in train_iter)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=lr_period * num_batches,
        decay_rate=lr_decay,
        staircase=True)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9,
                                     weight_decay=wd)
    timer = d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            with tf.GradientTape() as tape:
                logits = net(features, training=True)
                l = loss(labels, logits)
            grads = tape.gradient(l, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            metric.add(float(tf.reduce_sum(l)), len(labels))
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net)
            animator.add(epoch + 1, (None, valid_loss))
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec')
    return net
```

## Training and Validating the Model

Now we can train and validate the model.
The following hyperparameters are all tunable.
For example, the number of epochs can be increased. Because `lr_period` and `lr_decay` are set to 2 and 0.9, respectively, the learning rate of the optimization algorithm will be multiplied by 0.9 after every 2 epochs.

```{.python .input #kaggle-dog-training-and-validating-the-model}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input #kaggle-dog-training-and-validating-the-model}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input #kaggle-dog-training-and-validating-the-model}
#@tab jax
num_epochs, lr, wd = 10, 1e-4, 1e-4
lr_period, lr_decay = 2, 0.9
backbone, backbone_vars, output_net = get_net()
variables = train(backbone, backbone_vars, output_net, train_iter, valid_iter,
                  num_epochs, lr, wd, lr_period, lr_decay)
```

```{.python .input #kaggle-dog-training-and-validating-the-model}
#@tab tensorflow
num_epochs, lr, wd = 10, 1e-4, 1e-4
lr_period, lr_decay = 2, 0.9
net = get_net()
net = train(net, train_iter, valid_iter, num_epochs, lr, wd, lr_period,
            lr_decay)
```

## Classifying the Testing Set and Submitting Results on Kaggle


Similar to the final step in :numref:`sec_kaggle_cifar10`,
in the end all the labeled data (including the validation set) are used for training the model and classifying the testing set.
We will use the trained custom output network
for classification.

```{.python .input #kaggle-dog-classifying-the-testing-set-and-submitting-results-on-kaggle}
#@tab mxnet
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input #kaggle-dog-classifying-the-testing-set-and-submitting-results-on-kaggle}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input #kaggle-dog-classifying-the-testing-set-and-submitting-results-on-kaggle}
#@tab jax
backbone, backbone_vars, output_net = get_net()
variables = train(backbone, backbone_vars, output_net, train_valid_iter, None,
                  num_epochs, lr, wd, lr_period, lr_decay)

preds = []
for data, label in test_iter:
    feats = extract_features(backbone, backbone_vars, data.numpy())
    logits = output_net.apply(variables, feats, training=False)
    output = jax.nn.softmax(logits, axis=-1)
    preds.extend(np.array(output))
# Get class names from the train_valid dataset directory
class_names = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'train_valid')))
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(class_names) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input #kaggle-dog-classifying-the-testing-set-and-submitting-results-on-kaggle}
#@tab tensorflow
net = get_net()
net = train(net, train_valid_iter, None, num_epochs, lr, wd, lr_period,
            lr_decay)
preds = []
for data, label in test_iter:
    logits = net(data, training=False)
    output = tf.nn.softmax(logits, axis=-1)
    preds.extend(output.numpy())
# Get class names from the train_valid dataset directory
class_names = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'train_valid')))
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(class_names) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

The above code
will generate a `submission.csv` file
to be submitted
to Kaggle in the same way described in :numref:`sec_kaggle_house`.


## Summary


* Images in the ImageNet dataset are larger (with varying dimensions) than CIFAR-10 images. We may modify image augmentation operations for tasks on a different dataset.
* To classify a subset of the ImageNet dataset, we can leverage pre-trained models on the full ImageNet dataset to extract features and only train a custom small-scale output network. This will lead to less computational time and memory cost.


## Exercises

1. When using the full Kaggle competition dataset, what results can you achieve when you increase `batch_size` (batch size) and `num_epochs` (number of epochs) while setting some other hyperparameters as `lr = 0.01`, `lr_period = 10`, and `lr_decay = 0.1`?
1. Do you get better results if you use a deeper pretrained model? How do you tune hyperparameters? Can you further improve the results?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/380)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1481)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1481)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1481)
:end_tab:

<!-- slides -->

::: {.slide title="Kaggle Dog Breed"}
A second Kaggle capstone: ImageNet Dogs (120 fine-grained
breeds). The big difference from CIFAR-10: this is a
*subset* of ImageNet, so a pretrained ResNet already knows
*almost everything* about these classes. Fine-tuning is
the right play.

![Kaggle "Dog Breed Identification" page.](../img/kaggle-dog.jpg){width=72%}

@kaggle-dog-dog-breed-identification-imagenet-dogs-on-kaggle
:::

::: {.slide title="Downloading"}
@kaggle-dog-downloading-the-dataset
:::

::: {.slide title="Organizing the dataset"}
Same idea as CIFAR-10 — reshuffle the Kaggle layout into
`train/<class>/img.jpg` for the standard ImageFolder loader:

@kaggle-dog-organizing-the-dataset
:::

::: {.slide title="Augmentation"}
ImageNet-scale augmentation: random resized crop, random
horizontal flip, color jitter, and the same input
preprocessing convention the pretrained backbone expects:

@kaggle-dog-image-augmentation-1

. . .

@kaggle-dog-image-augmentation-2
:::

::: {.slide title="Data loaders"}
@kaggle-dog-reading-the-dataset-1

. . .

@kaggle-dog-reading-the-dataset-2
:::

::: {.slide title="Frozen ImageNet features"}
This competition is close to ImageNet, so we reuse a
pretrained ResNet as a frozen feature extractor and train
only a small 120-way breed classifier:

@kaggle-dog-fine-tuning-a-pretrained-model-1
:::

::: {.slide title="Head loss and validation"}
Only the custom output network receives gradients. The
validation loss is computed through the same frozen
features, so it measures whether the dog-breed head is
generalizing:

@kaggle-dog-fine-tuning-a-pretrained-model-2
:::

::: {.slide title="Training function"}
The helper is mostly framework bookkeeping. The training
structure is:

- precompute frozen ImageNet features;
- train the 120-way head with cross-entropy;
- report validation loss on held-out breeds;
- repeat on all training data before writing the
  submission file.

That is the practical transfer-learning tradeoff: far less
memory and time, while keeping most ImageNet visual
knowledge.
:::

::: {.slide title="Train"}
Expect validation loss to be the useful curve here; with
120 fine-grained classes, top-line accuracy can be noisy on
the tiny book subset. On the full competition data, train
longer and tune the head/augmentation strength.

@kaggle-dog-training-and-validating-the-model
:::

::: {.slide title="Submit predictions"}
Write one probability vector per test image. The CSV has
image id plus 120 breed probabilities, so the final layer
must stay aligned with the competition's class order:

@kaggle-dog-classifying-the-testing-set-and-submitting-results-on-kaggle
:::

::: {.slide title="Recap"}
- ImageNet Dogs ⊂ ImageNet → fine-tuning a pretrained
  CNN crushes from-scratch training.
- Standard recipe: pretrained backbone, new 120-way head,
  ImageNet-scale augmentation, ImageNet-compatible preprocessing.
- Same shape as the CIFAR-10 deck; only the dataset and
  the choice "train from scratch vs fine-tune" differ.
- The general lesson: when your task is close to the
  pretraining domain, transfer learning beats everything.
:::
