```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Softmax Regression Implementation
:label:`sec_softmax_scratch`

Just as we implemented linear regression both from scratch and concisely in :numref:`sec_linear_scratch`, we now build softmax regression on Fashion-MNIST both ways:

1. **From scratch:** We implement the rowwise softmax, vectorized cross-entropy loss, and `SoftmaxRegressionScratch` parameters ($\mathbf{W} \in \mathbb{R}^{784 \times 10}, \mathbf{b} \in \mathbb{R}^{10}$) by hand, while introducing the reusable `Classifier(d2l.Module)` base class that tracks both cross-entropy loss and classification accuracy across the rest of the book. Along the way, we expose the floating-point overflow of the naive softmax formula, train to $\sim 83\%$ validation accuracy, and dissect what the linear model learned via its $10 \times 10$ confusion matrix and its ten $28 \times 28$ weight templates.
2. **Concisely with a numerically stable fused loss (:numref:`sec_softmax_concise`):** We then replace our manual parameters and clamped loss with a built-in framework linear layer and the shifted *log-sum-exp* cross-entropy loss computed directly from raw logits—eliminating both overflow and underflow while preserving intact gradients for arbitrary logits.

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```


## The Softmax

### Normalizing Each Row

For a minibatch of $n$ examples, the logits form an $n \times q$ matrix $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ (:numref:`subsec_softmax_vectorization`). Softmax normalizes each row independently:

$$\mathrm{softmax}(\mathbf{O})_{ij} = \frac{\exp(O_{ij})}{\sum_{k=1}^q \exp(O_{ik})}.$$

In code, we exponentiate `O`, sum across columns (`axis=1`) with `keepdims=True` so the row sums keep shape `(n, 1)`, and divide via broadcasting:

```{.python .input #softmax-regression-scratch-the-softmax-2}
def softmax(O):
    O_exp = d2l.exp(O)
    return O_exp / d2l.reduce_sum(O_exp, 1, keepdims=True)  # Broadcast
```

Let us sanity-check `softmax` on two rows: the cat/chicken/dog logits
$(1.0, 2.2, 0.3)$ from :numref:`sec_softmax`, and a row of three identical
zeros.

```{.python .input #softmax-regression-scratch-the-softmax-3}
O = d2l.tensor([[1.0, 2.2, 0.3], [0.0, 0.0, 0.0]])
Y_hat = softmax(O)
Y_hat, d2l.reduce_sum(Y_hat, 1)
```


### Overflow

Why should you never deploy this three-line `softmax` formula in production? Because in 32-bit floating point (`float32`), numbers top out near $3.4 \times 10^{38}$, so $\exp(o)$ overflows to `inf` as soon as $o > 88.7$! Feed `softmax` the logits $(1000, 0, 0)$ and $\infty / \infty$ immediately evaluates to `NaN`—even though the true probability distribution is $(1, 0, 0)$ to machine precision.

As we proved in :numref:`sec_softmax`, softmax is invariant to shifting every logit in a row by the same constant. Subtracting the row maximum $o_{\max} = \max_k o_k$ before exponentiating ensures every exponent $o_j - o_{\max}$ is $\leq 0$ (and the largest is $0$), making overflow impossible. Every framework's built-in softmax applies this shift internally:

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab pytorch
O = d2l.tensor([[1000.0, 0.0, 0.0]])
O_max = O.amax(dim=1, keepdim=True)
softmax(O), softmax(O - O_max), torch.softmax(O, dim=1)
```

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab mxnet
O = d2l.tensor([[1000.0, 0.0, 0.0]])
O_max = O.max(axis=1, keepdims=True)
softmax(O), softmax(O - O_max), npx.softmax(O, axis=1)
```

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab tensorflow
O = d2l.tensor([[1000.0, 0.0, 0.0]])
O_max = tf.reduce_max(O, axis=1, keepdims=True)
softmax(O), softmax(O - O_max), tf.nn.softmax(O, axis=1)
```

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab jax
O = d2l.tensor([[1000.0, 0.0, 0.0]])
O_max = O.max(axis=1, keepdims=True)
softmax(O), softmax(O - O_max), jax.nn.softmax(O, axis=1)
```


For our Fashion-MNIST scratch run below, pixel values lie in $[0, 1]$ and weights stay small, so `softmax` does not overflow during training; however, shifting alone still leaves very negative logits vulnerable to *underflow* ($\exp(-200) = 0.0$ in float32, turning $\log 0$ into $-\infty$). In :numref:`sec_softmax_concise` below, we eliminate both hazards permanently by fusing softmax and the logarithm in log space.

## Loss and Accuracy

### Two Uses of the Logits

While gradient descent optimizes the smooth cross-entropy loss, humans judge a classifier by its *accuracy*—the fraction of examples where the top-scoring class matches the true label $y^{(i)}$:

$$\textrm{accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\!\Big(\mathop{\mathrm{argmax}}_j o^{(i)}_j = y^{(i)}\Big).$$
:eqlabel:`eq_accuracy`

Both metrics flow from the exact same logit vector (:numref:`fig_mdl-clf-loss-accuracy`), and tracking both during validation is essential: if a model becomes overconfident on a few mistakes, its validation cross-entropy loss can rise even while its validation accuracy is still climbing.

![Training (top) maps logits through softmax to the smooth cross-entropy loss; evaluation (bottom) compares $\operatorname{argmax}_j o_j$ against the label $y$ to compute discrete accuracy.](../img/mdl-clf-loss-accuracy.svg)
:label:`fig_mdl-clf-loss-accuracy`

### The `Classifier` Class
:label:`subsec_classifier-class`

We package this dual-metric validation behavior into `Classifier(d2l.Module)`, the base class for every classification model in the book. It overrides `validation_step` to return both `'loss'` and `'acc'`, attaches `accuracy(Y_hat, Y, averaged=True)` to compute :eqref:`eq_accuracy`, and installs plain minibatch SGD at learning rate `self.lr` as the default `configure_optimizers` on `d2l.Module`:

```{.python .input #classification-the-classifier-class-1}
%%tab pytorch, mxnet, jax
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        return {'loss': self.loss(Y_hat, batch[-1]),
                'acc': self.accuracy(Y_hat, batch[-1])}
```

```{.python .input #classification-the-classifier-class-1}
%%tab tensorflow
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1], training=False)
        return {'loss': self.loss(Y_hat, batch[-1]),
                'acc': self.accuracy(Y_hat, batch[-1])}
```

```{.python .input #classification-accuracy-1  n=9}
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the fraction of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, (-1,)), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```


```{.python .input #classification-the-classifier-class-2}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return gluon.Trainer(self.collect_params(), 'sgd',
                         {'learning_rate': self.lr})
```

```{.python .input #classification-the-classifier-class-2}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input #classification-the-classifier-class-2}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(float(self.lr))
```

```{.python .input #classification-the-classifier-class-2}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```


## The Model and the Loss

### The Parameters and Forward Pass

Flattening each $28 \times 28$ Fashion-MNIST image yields $d = 784$ input features for $q = 10$ classes, so `SoftmaxRegressionScratch` allocates $\mathbf{W} \in \mathbb{R}^{784 \times 10}$ (initialized from $\mathcal{N}(0, 0.01^2)$) and $\mathbf{b} \in \mathbb{R}^{10}$ (initialized to $0$), totaling $7{,}850$ parameters. Notice a quick sanity check: near-zero initial weights produce near-zero logits, so the untrained model starts close to a uniform prediction ($\hat{y}_j \approx 0.1$) with initial cross-entropy loss near $-\log(0.1) = \log 10 \approx 2.30$. In `forward`, we reshape each minibatch of images to `(-1, 784)` and return `softmax(X @ W + b)`:

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab pytorch
class SoftmaxRegressionScratch(Classifier):
    """Softmax regression implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, lr, std=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = nn.Parameter(d2l.normal(0, std, (num_inputs, num_outputs)))
        self.b = nn.Parameter(d2l.zeros(num_outputs))
```

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab mxnet
class SoftmaxRegressionScratch(Classifier):
    """Softmax regression implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, lr, std=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = gluon.Parameter('W', shape=(num_inputs, num_outputs),
                                 init=init.Normal(std))
        self.b = gluon.Parameter('b', shape=(num_outputs,), init=init.Zero())
        self.initialize()
```

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab tensorflow
class SoftmaxRegressionScratch(Classifier):
    """Softmax regression implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, lr, std=0.01):
        super().__init__()
        self.save_hyperparameters()
        normal = tf.keras.initializers.RandomNormal(mean=0, stddev=std)
        self.W = self.add_weight(name='W', shape=(num_inputs, num_outputs),
                                 initializer=normal)
        self.b = self.add_weight(name='b', shape=(num_outputs,),
                                 initializer='zeros')
```

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab jax
class SoftmaxRegressionScratch(Classifier):
    """Softmax regression implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, lr, std=0.01, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.W = nnx.Param(
            rngs.params.normal((num_inputs, num_outputs)) * std)
        self.b = nnx.Param(jnp.zeros(num_outputs))
```

```{.python .input #softmax-regression-scratch-the-model-2}
%%tab pytorch, tensorflow, jax
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))  # One flattened image per row
    return softmax(d2l.matmul(X, self.W) + self.b)
```

```{.python .input #softmax-regression-scratch-the-model-2}
%%tab mxnet
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))  # One flattened image per row
    return softmax(d2l.matmul(X, self.W.data()) + self.b.data())
```

### The Cross-Entropy Loss

Recall from :eqref:`eq_l_cross_entropy` that $l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_y$. Instead of materializing one-hot matrices and taking inner products, we pluck out the true-class probability $\hat{y}^{(i)}_{y^{(i)}}$ for every row $i$ in one vectorized indexing step (`y_hat[range(len(y_hat)), y]`, or `tf.gather(y_hat, y, batch_dims=1)` in TensorFlow). In our scratch `cross_entropy`, we clamp probabilities below at $10^{-12}$ as a temporary guard against $\log 0$:

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-1}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-1}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.gather(y_hat, y, batch_dims=1)
```

That extracts 0.1 for the first example and 0.5 for the second. Wrapping
this in `cross_entropy`, we average the negative logarithms across the
minibatch. Notice the one defensive clamp to $10^{-12}$ before taking the
logarithm: it stops an underflowed zero probability from turning into
$-\infty$, capping the per-example loss at $27.6$, though—as the comment
notes—clamping also zeroes out the gradient whenever it triggers. In
:numref:`sec_softmax_concise`, we replace this stopgap with a fused loss
computed directly from logits.

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab pytorch
def cross_entropy(y_hat, y):  #@save
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = y_hat[list(range(len(y_hat))), y].clamp(min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab mxnet
def cross_entropy(y_hat, y):  #@save
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = y_hat[list(range(len(y_hat))), y].clip(min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab jax
def cross_entropy(y_hat, y):  #@save
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = jnp.clip(y_hat[jnp.arange(len(y_hat)), y], min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab tensorflow
def cross_entropy(y_hat, y):  #@save
    p = tf.gather(y_hat, y, batch_dims=1)
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    return -tf.reduce_mean(tf.math.log(tf.maximum(p, 1e-12)))

cross_entropy(y_hat, y)
```

That yields the mean of $-\log 0.1 \approx 2.30$ and $-\log 0.5 \approx 0.69$,
or roughly 1.50. Let us attach it as the model's `loss` method:

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-3}
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```


## Training and Evaluation

### The Training Run

Let us train `SoftmaxRegressionScratch` on Fashion-MNIST for $10$ epochs with minibatch size $256$ and learning rate $\eta = 0.1$, and attach a convenient `trainer.evaluate(model, data, train=False)` helper to score the finished model on the full validation and training splits:

```{.python .input #softmax-regression-scratch-training}
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```


```{.python .input #softmax-regression-scratch-the-training-run-1}
@d2l.add_to_class(d2l.Trainer)  #@save
def evaluate(self, model, data, train=False):
    """Metrics of `model` on the validation data of `data` (on its training
    data if `train`), averaged over examples."""
    self.train_dataloader = None
    self.val_dataloader = (data.train_dataloader() if train
                           else data.val_dataloader())
    self.prepare_model(model)
    return self.validate()
```

Averaged across all 10,000 validation images, these metrics match the final
points recorded in `trainer.history['val_loss'][-1]` and
`trainer.history['val_acc'][-1]`.

```{.python .input #softmax-regression-scratch-prediction-accuracy}
trainer.evaluate(model, data)
```

Our classifier gets roughly 83% of the validation images right. How well
does the finished model fit the 60,000 training images?

```{.python .input #softmax-regression-scratch-the-training-run-2}
trainer.evaluate(model, data, train=True)
```

After 10 epochs, validation accuracy settles near $83\%$ (varying by $\pm 1\%$ across random seeds) and training accuracy sits near $84$–$85\%$. That tiny generalization gap of one to two points reflects the restricted capacity of a linear classifier: $7{,}850$ parameters cannot memorize $60{,}000$ images (:numref:`chap_classification_generalization`).

### Misclassified Images

To see *how* the model errs, let us pull the first validation minibatch of $256$ images, count the mistakes, and display the first eight misclassified garments with their true and predicted labels (`true → pred`):

```{.python .input #softmax-regression-scratch-prediction-1}
X, y = next(iter(data.val_dataloader()))
preds = d2l.astype(d2l.argmax(model(X), axis=1), y.dtype)
wrong = preds != y
print(f'{int(d2l.reduce_sum(d2l.astype(wrong, d2l.int32)))} of {len(y)} wrong')
```

Let us display the first eight misclassified images, showing the true label
above the model's prediction.

```{.python .input #softmax-regression-scratch-prediction-2}
labels = [f'{true}\n→ {pred}' for true, pred in zip(
    data.text_labels(y[wrong]), data.text_labels(preds[wrong]))]
data.visualize([X[wrong], y[wrong]], labels=labels)
```


### The Confusion Matrix

To summarize the error pattern across all $10{,}000$ validation images, we tally the $10 \times 10$ *confusion matrix* $\mathbf{N} \in \mathbb{R}^{q \times q}$, where rows index the predicted class $\hat{c}^{(m)} = \operatorname{argmax}_j o^{(m)}_j$ and columns index the true class $y^{(m)}$:

$$N_{ij} = \sum_{m=1}^n \mathbf{1}\big(\hat{c}^{(m)} = i\big)\, \mathbf{1}\big(y^{(m)} = j\big).$$
:eqlabel:`eq_confusion-matrix`

(Placing true classes in the columns and predicted classes in the rows makes column-normalized conditional probabilities multiply cleanly as $\boldsymbol{\mu} = \mathbf{C}\mathbf{p}$ when we study label shift in :numref:`sec_environment-and-distribution-shift`.) Using one-hot matrices $\mathbf{P}, \mathbf{Y} \in \{0, 1\}^{n \times q}$ for a minibatch's predicted and true classes, the product $\mathbf{P}^\top\mathbf{Y}$ accumulates all $100$ counts in a single matrix multiplication per minibatch. Let us compute $\mathbf{N}$, normalize each column to sum to $1$ to obtain $C_{ij} = P(\hat{c} = i \mid y = j)$, plot $\mathbf{C}$ as a heatmap, and print the five most frequent confusions:

```{.python .input #softmax-regression-scratch-the-confusion-matrix-1}
def one_hot(y, q):
    """Rows of the q x q identity matrix, one per class index in y."""
    return d2l.astype(d2l.reshape(y, (-1, 1)) == d2l.arange(q, dtype=y.dtype),
                      d2l.float32)

N = d2l.zeros((10, 10))
for X, y in data.val_dataloader():
    P = one_hot(d2l.argmax(model(X), axis=1), 10)
    N += d2l.matmul(d2l.transpose(P), one_hot(y, 10))
int(d2l.reduce_sum(N))
```

All 10,000 validation images are accounted for. Because raw counts depend
on class frequency, we normalize each column of $\mathbf{N}$ by its sum to
obtain $C_{ij} = N_{ij} / \sum_{i'} N_{i'j}$, an empirical estimate of the
conditional probability $P(\hat{c} = i \mid y = j)$. Each column shows
where examples of true class $j$ land, while the diagonal gives per-class
recognition rates. Let us render $\mathbf{C}$ as a heatmap via
`d2l.show_heatmaps`.

```{.python .input #softmax-regression-scratch-prediction-confusion}
C = N / d2l.reduce_sum(N, 0, keepdims=True)  # Each column sums to 1
names = data.text_labels(range(10))
d2l.show_heatmaps(d2l.reshape(C, (1, 1, 10, 10)), xlabel='true class',
                  ylabel='predicted class', figsize=(4, 4), cmap='Blues',
                  xticklabels=names, yticklabels=names)
```

Let us also print the five largest off-diagonal entries to spotlight the
most frequent confusions:

```{.python .input #softmax-regression-scratch-the-confusion-matrix-2}
mistakes = d2l.numpy(N) * (1 - np.eye(10))  # Zero the diagonal
for k in np.argsort(mistakes, axis=None)[::-1][:5]:
    i, j = divmod(int(k), 10)
    print(f'{names[j]} taken for {names[i]}: {int(mistakes[i, j])}')
```


The confusion matrix confirms what our visual inspection in :numref:`sec_fashion_mnist` predicted: trousers, bags, and footwear are recognized $\sim 90\%$ of the time or better, whereas the five largest off-diagonal confusions all occur among the four look-alike upper-body tops (T-shirt, pullover, coat, and shirt)—with shirts recognized only about half the time because their silhouette sits right in the middle of the other three.

### The Learned Templates

Because each logit $o_j = \mathbf{w}_j^\top\mathbf{x} + b_j$ is an inner product between the flattened image $\mathbf{x}$ and column $j$ of $\mathbf{W}$, reshaping each 784-element column $\mathbf{w}_j$ back to $28 \times 28$ lets us view the entire classifier as ten visual *templates* (red = positive weights, blue = negative weights):

```{.python .input #softmax-regression-scratch-what-the-model-learned}
%%tab pytorch, tensorflow
W = d2l.numpy(model.W)
templates = W.T.reshape(10, 28, 28)  # Column j of W as an image
v = np.abs(templates).max()
d2l.show_images(templates, 2, 5, titles=names, cmap='RdBu_r', vmin=-v, vmax=v);
```

```{.python .input #softmax-regression-scratch-what-the-model-learned}
%%tab mxnet
W = d2l.numpy(model.W.data())
templates = W.T.reshape(10, 28, 28)  # Column j of W as an image
v = np.abs(templates).max()
d2l.show_images(templates, 2, 5, titles=names, cmap='RdBu_r', vmin=-v, vmax=v);
```

```{.python .input #softmax-regression-scratch-what-the-model-learned}
%%tab jax
W = d2l.numpy(model.W[...])
templates = W.T.reshape(10, 28, 28)  # Column j of W as an image
v = np.abs(templates).max()
d2l.show_images(templates, 2, 5, titles=names, cmap='RdBu_r', vmin=-v, vmax=v);
```


Each template is a recognizable ghostly silhouette of its garment—notice the blue negative channel between the trouser legs that penalizes dresses, and the similar torso shapes of the four tops that differ mainly around the collar and sleeves. Because a linear layer can only take a single fixed global dot product per class, subtle local features like collars or buttons cannot overcome overall silhouette overlap; breaking past $\sim 83\%$ requires nonlinear hidden layers (:numref:`chap_perceptrons`) and spatial convolutions (:numref:`chap_cnn`).

## Concise Implementation
:label:`sec_softmax_concise`

Now let us see how deep learning frameworks simplify softmax regression while permanently curing the overflow and underflow hazards of our naive `softmax` and clamped `cross_entropy`.

### Defining the Model

In `SoftmaxRegression(d2l.Classifier)`, a flattening step and a single built-in fully connected layer (`LazyLinear` / `Dense` / `nnx.Linear`) with $10$ outputs compute the logits $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ (initialized with $\mathcal{N}(0, 0.01^2)$ weights and zero biases to match our scratch baseline). Crucially, **`forward` returns the raw logits $\mathbf{O}$ directly, without applying softmax**:

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def weight_init(self, module):
        # The Trainer calls this after the lazy layer has learned its shape
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, X):
        return self.net(X)
```

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab mxnet
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Dense(num_outputs)
        self.net.initialize(init.Normal(sigma=0.01))

    def forward(self, X):
        return self.net(X)
```

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential()
        self.net.add(tf.keras.layers.Flatten())
        self.net.add(tf.keras.layers.Dense(
            num_outputs,
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.01)))

    def forward(self, X):
        return self.net(X)
```

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(num_inputs, num_outputs,
                              kernel_init=nnx.initializers.normal(0.01),
                              rngs=rngs)

    def forward(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        return self.net(X)
```


### A Stable Cross-Entropy Loss
:label:`subsec_softmax-implementation-revisited`

Why does `SoftmaxRegression.forward` stop at the logits instead of returning probabilities? Recall the twin floating-point hazards of computing $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})$ first and $-\log \hat{y}_y$ second in `float32`:
1. **Overflow:** $\exp(o)$ overflows to `inf` for $o > 88.7$. Subtracting $o_{\max} = \max_k o_k$ from every logit fixes overflow because every shifted exponent $o_k - o_{\max}$ is $\leq 0$.
2. **Underflow:** Even after shifting by $o_{\max}$, any class whose logit sits more than $104$ below $o_{\max}$ (such as class 1 in $\mathbf{o} = (0, -200, 0)$) has $\exp(o_y - o_{\max})$ underflow to exact `0.0` in float32! Taking $\log(0.0)$ then produces `-inf` (or, with our scratch `1e-12` clamp, caps the loss at $27.6$ and **zeroes out the gradient**, leaving the model unable to learn from a confident blunder!).

We avoid both overflow and underflow by *never materializing intermediate probabilities at all*. Combining the logarithm and the shifted softmax algebraically—the classic *LogSumExp trick*—turns the division into a subtraction:

$$l(y, \mathbf{o}) = \log \sum_{k=1}^q \exp(o_k) - o_y = o_{\max} + \log \sum_{k=1}^q \exp(o_k - o_{\max}) - o_y.$$
:eqlabel:`eq_softmax-concise-lse-loss`

Because every shifted exponent $o_k - o_{\max}$ is $\leq 0$ and at least one equals $0$, the sum inside the logarithm is trapped strictly inside $[1, q]$, and its logarithm lies strictly inside $[0, \log q]$! Neither overflow nor $\log 0$ can ever occur, and automatic differentiation yields the exact finite gradient $\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$.

Every framework provides this fused log-sum-exp cross-entropy (`F.cross_entropy` in PyTorch, `SoftmaxCrossEntropyLoss` in MXNet, `SparseCategoricalCrossentropy(from_logits=True)` in Keras, and `optax.softmax_cross_entropy_with_integer_labels` in JAX). We install it once on `d2l.Classifier.loss` so every classifier in the rest of the book inherits it, and verify on our extreme logits $(1000, 0, 0)$ with true labels $0$ and $1$ that it returns the exact finite losses $0$ and $1000$:

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')
```

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab mxnet
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = gluon.loss.SoftmaxCrossEntropyLoss()
    l = fn(Y_hat, Y)
    return l.mean() if averaged else l
```

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    reduction = (tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                 if averaged else tf.keras.losses.Reduction.NONE)
    fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=reduction)
    return fn(Y, Y_hat)
```

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return fn(Y_hat, Y).mean() if averaged else fn(Y_hat, Y)
```


```{.python .input #softmax-regression-concise-the-framework-loss}
O = d2l.tensor([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
y = d2l.tensor([0, 1])
SoftmaxRegression(num_outputs=3, lr=0.1).loss(O, y, averaged=False)
```


(Watch out for one classic pitfall: because `Classifier.loss` expects raw logits and applies log-softmax internally, accidentally placing a softmax at the end of `forward` applies softmax *twice*, trapping the training loss above a positive floor; see Exercise 4 below.)

### Training

Training `SoftmaxRegression` with `d2l.Trainer` takes four lines and reproduces our $\sim 83\%$ validation accuracy with rock-solid numerical stability:

```{.python .input #softmax-regression-concise-training}
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```


## Summary

- **From-scratch pipeline and `Classifier`:** Implementing rowwise `softmax`, indexed `cross_entropy`, and `SoftmaxRegressionScratch` alongside the reusable `Classifier` base class (which logs both validation loss and accuracy) reaches $\sim 83\%$ validation accuracy and $84$–$85\%$ training accuracy on Fashion-MNIST in 10 epochs.
- **Error analysis via confusion matrix and templates:** Accumulating $\mathbf{P}^\top\mathbf{Y}$ across minibatches builds the $10 \times 10$ confusion matrix without Python loops over examples, revealing that errors concentrate among the four upper-body tops whose learned $28 \times 28$ weight templates overlap heavily.
- **Fused LogSumExp loss (:numref:`sec_softmax_concise`):** Evaluating $l(y, \mathbf{o}) = o_{\max} + \log\sum_k \exp(o_k - o_{\max}) - o_y$ directly from raw logits prevents both `exp` overflow and `log(0)` underflow while keeping gradients intact across arbitrary logit magnitudes.

## Exercises

1. [code] **Cross-entropy from one-hot labels.** The `cross_entropy` of this
   section selects the correct-class probability by indexing. Implement an
   alternative that follows the definition $-\sum_j y_j \log \hat{y}_j$
   literally, by converting `y` to a one-hot matrix and taking row-wise
   inner products.
    1. Use it as the `loss` of `SoftmaxRegressionScratch` and confirm that
       training is unchanged.
    1. Time both versions on random probability matrices with $256$ rows
       and $q \in \{10, 10^3, 10^5\}$ columns, and state how the memory
       that each version allocates grows with $q$.
    1. Describe a setting in which the inner-product form applies but the
       indexing form does not.

1. **Large vocabularies.** Suppose that softmax regression predicts the
   next word from a feature vector of dimension $d = 1024$. The vocabulary
   has $V = 10^6$ words, and each minibatch holds $n = 256$ examples.
   List the operations of `forward` and `cross_entropy` whose cost grows
   with $V$. Compute the multiply-adds of the matrix product, the float32
   memory of `W`, and the float32 memory of the $n \times V$ matrix of
   probabilities, and compare each number with the Fashion-MNIST model of
   this section.


1. [code] **Templates and class means.** The templates are what minibatch
   SGD made of the pixels it saw.
    1. Train `SoftmaxRegressionScratch` again with `std=0`, so that every
       weight starts at 0 and every class starts with the same logit.
       Explain why training works all the same, and compare the accuracy
       and the templates with those of the text.
    1. Show that after one SGD step from zero weights, column $j$ of
       $\mathbf{W}$ is proportional to
       $\sum_i \big(y^{(i)}_j - \tfrac{1}{10}\big)\, \mathbf{x}^{(i)}$ over
       the first minibatch: the sum of the images of class $j$ minus a
       tenth of the sum of all images.
    1. Compute the mean image of each class over the training set and
       compare it with the trained template of that class, by eye and by
       the correlation of their pixels. Where do they agree, and where does
       a template depart from the mean? Explain why a template that merely
       copied the class mean would confuse the tops even more.


1. [code] **Double softmax.** The `loss` of this section expects logits.
   Suppose that a model's `forward` mistakenly ends with a softmax. The
   loss then receives a probability vector $\mathbf{p}$ in place of the
   logits and evaluates $-\log \mathrm{softmax}(\mathbf{p})_y$.
    1. Find the smallest value this loss can take over all probability
       vectors $\mathbf{p}$ and labels $y$, as a function of the number of
       classes $q$. Evaluate it for $q = 10$.
    1. Copy the from-scratch `softmax` of :numref:`sec_softmax_scratch`,
       which is not part of the `d2l` package. Subclass
       `SoftmaxRegression` so that `forward` applies this `softmax` to the
       logits. Train it with `lr=0.1`, `d2l.FashionMNIST(batch_size=256)`,
       and `d2l.Trainer(max_epochs=10)`. Compare its training loss and
       validation accuracy with those of `SoftmaxRegression`, and relate
       the final training loss to the value from the previous step.


1. [code] **Stable softplus.** For two classes with logits $(x, 0)$, the
   first term of the loss is $\log(1 + e^x)$, the *softplus* of $x$.
    1. Plot $\log(1 + e^x)$ and $\max(x, 0)$ on $[-5, 5]$. Show that the
       first exceeds the second by at most $\log 2$, at $x = 0$, and
       explain in one sentence why log-sum-exp is called a smooth maximum.
    1. Show that $\log(1 + e^x) = \max(x, 0) + \log(1 + e^{-|x|})$.
    1. In single precision, evaluate the naive expression
       `log(1 + exp(x))`, the rewritten expression, and the rewritten
       expression with `log1p` in place of `log(1 + ...)`, for
       $x \in \{-100, -20, -5, 0, 5, 20, 100\}$. Compare each value with a
       double-precision reference, and report where each version
       overflows and where it loses all relative accuracy.
    1. Compute the derivative of the naive expression at $x = -20$ by
       automatic differentiation. Explain why it is accurate although the
       value of the expression is not, and what this implies for training
       and for reporting the loss of confidently classified examples.


1. [code] **The gradient survives.** For the logits $(0, -200, 0)$ with true label 1, the shifted softmax probability of class 1 underflows to $0$ in float32, so the clamped scratch loss `cross_entropy(softmax(O - O_max), y)` hits its `1e-12` floor and has zero gradient with respect to the logits. Pass the same logits and label to the fused `loss` of `SoftmaxRegression` and compute the gradient with respect to the logits by automatic differentiation. Compare it with $\hat{\mathbf{y}} - \mathbf{y}$ from :numref:`sec_softmax`, and explain why the fused loss moves the logit of class 1 while the clamped scratch loss does not.

1. **Number formats.** Half-precision formats store a number in 16 bits:
   FP16 has 5 exponent bits and 10 mantissa bits, BFLOAT16 has 8 and 7, and
   FP32 has 8 and 23. Determine for each format the largest logit $o$ for
   which $\exp(o)$ does not overflow. Then explain why a network whose
   logits reach 20 can be trained in BFLOAT16 but not in FP16 unless its
   softmax is computed in FP32. Finally, consider the shifted sum of
   :eqref:`eq_softmax-concise-lse-loss` for $q = 10^5$ classes, each term
   at most 1, accumulated in FP16: what goes wrong once the running sum
   exceeds 2,048, and what does the 10-bit mantissa have to do with it?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/260)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17983)
:end_tab:

