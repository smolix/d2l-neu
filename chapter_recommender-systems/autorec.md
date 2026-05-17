# AutoRec: Rating Prediction with Autoencoders

Although the matrix factorization model achieves decent performance on the rating prediction task, it is essentially a linear model. Thus, such models are not capable of capturing complex nonlinear and intricate relationships that may be predictive of users' preferences. In this section, we introduce a nonlinear neural network collaborative filtering model, AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`. It identifies collaborative filtering (CF) with an autoencoder architecture and aims to integrate nonlinear transformations into CF on the basis of explicit feedback. Neural networks have been proven to be capable of approximating any continuous function, making it suitable to address the limitation of matrix factorization and enrich the expressiveness of matrix factorization.

On the one hand, AutoRec has the same structure as an autoencoder which consists of an input layer, a hidden layer, and a reconstruction (output) layer.  An autoencoder is a neural network that learns to copy its input to its output in order to code the inputs into the hidden (and usually low-dimensional) representations. In AutoRec, instead of explicitly embedding users/items into low-dimensional space, it uses the column/row of the interaction matrix as input, then reconstructs the interaction matrix in the output layer.

On the other hand, AutoRec differs from a traditional autoencoder: rather than learning the hidden representations, AutoRec focuses on learning/reconstructing the output layer. It uses a partially observed interaction matrix as input, aiming to reconstruct a completed rating matrix. In the meantime, the missing entries of the input are filled in the output layer via reconstruction for the purpose of recommendation.

There are two variants of AutoRec: user-based and item-based. For brevity, here we only introduce the item-based AutoRec. User-based AutoRec can be derived accordingly.


## Model

Let $\mathbf{R}_{*i}$ denote the $i^\textrm{th}$ column of the rating matrix, where unknown ratings are set to zeros by default. The neural architecture is defined as:

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

where $f(\cdot)$ and $g(\cdot)$ represent activation functions, $\mathbf{W}$ and $\mathbf{V}$ are weight matrices, $\mu$ and $b$ are biases. Let $h( \cdot )$ denote the whole network of AutoRec. The output $h(\mathbf{R}_{*i})$ is the reconstruction of the $i^\textrm{th}$ column of the rating matrix.

The following objective function aims to minimize the reconstruction error:

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

where $\| \cdot \|_{\mathcal{O}}$ means only the contribution of observed ratings are considered, that is, only weights that are associated with observed inputs are updated during back-propagation.

```{.python .input #autorec-model  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

```{.python .input #autorec-model  n=3}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
```

## Implementing the Model

A typical autoencoder consists of an encoder and a decoder. The encoder projects the input to hidden representations and the decoder maps the hidden layer to the reconstruction layer. We follow this practice and create the encoder and decoder with fully connected layers. The activation of encoder is set to `sigmoid` by default and no activation is applied for decoder. Dropout is included after the encoding transformation to reduce over-fitting. The gradients of unobserved inputs are masked out to ensure that only observed ratings contribute to the model learning process.

```{.python .input #autorec-implementing-the-model  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

```{.python .input #autorec-implementing-the-model  n=2}
#@tab pytorch
class AutoRec(nn.Module):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super().__init__()
        self.encoder = nn.Linear(num_users, num_hidden)
        self.decoder = nn.Linear(num_hidden, num_users)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(torch.sigmoid(self.encoder(input)))
        pred = self.decoder(hidden)
        if self.training:  # Mask the gradient during training
            return pred * torch.sign(input)
        else:
            return pred
```

## Reimplementing the Evaluator

Since the input and output have been changed, we need to reimplement the evaluation function, while we still use RMSE as the accuracy measure.

```{.python .input #autorec-reimplementing-the-evaluator  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

```{.python .input #autorec-reimplementing-the-evaluator  n=3}
#@tab pytorch
def evaluator(network, inter_matrix, test_data, devices):
    network.eval()
    scores = []
    with torch.no_grad():
        for values in inter_matrix:
            values = values.to(devices[0])
            scores.append(network(values).cpu().numpy())
    recons = np.concatenate(scores, axis=0)
    # Calculate the test RMSE
    rmse = np.sqrt(
        np.sum(np.square(test_data - np.sign(test_data) * recons))
        / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Training and Evaluating the Model

Now, let's train and evaluate AutoRec on the MovieLens dataset. We can clearly see that the test RMSE is lower than the matrix factorization model, confirming the effectiveness of neural networks in the rating prediction task.

```{.python .input #autorec-training-and-evaluating-the-model  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

```{.python .input #autorec-training-and-evaluating-the-model  n=4}
#@tab pytorch
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_inter_mat_t = torch.tensor(train_inter_mat, dtype=torch.float32)
test_inter_mat_np = np.array(test_inter_mat)
train_iter = torch.utils.data.DataLoader(train_inter_mat_t, shuffle=True,
                                         drop_last=True, batch_size=256,
                                         num_workers=d2l.get_dataloader_workers())
test_iter = torch.utils.data.DataLoader(train_inter_mat_t, shuffle=False,
                                        batch_size=1024,
                                        num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
nn.init.normal_(net.encoder.weight, std=0.01)
nn.init.normal_(net.decoder.weight, std=0.01)
net = net.to(devices[0])
lr, num_epochs, wd = 0.002, 25, 1e-5
loss = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
timer = d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                        legend=['train loss', 'test RMSE'])
for epoch in range(num_epochs):
    net.train()
    total_loss, n = 0., 0
    for i, values in enumerate(train_iter):
        timer.start()
        values = values.to(devices[0])
        optimizer.zero_grad()
        preds = net(values)
        l = loss(preds, values * torch.sign(values))
        l.backward()
        optimizer.step()
        total_loss += l.item()
        n += values.shape[0]
        timer.stop()
    test_rmse = evaluator(net, test_iter, test_inter_mat_np, devices)
    train_l = total_loss / n
    animator.add(epoch + 1, (train_l, test_rmse))
print(f'train loss {total_loss / n:.3f}, test RMSE {test_rmse:.3f}')
```

## Summary

* We can frame the matrix factorization algorithm with autoencoders, while integrating non-linear layers and dropout regularization.
* Experiments on the MovieLens 100K dataset show that AutoRec achieves superior performance than matrix factorization.



## Exercises

* Vary the hidden dimension of AutoRec to see its impact on the model performance.
* Try to add more hidden layers. Is it helpful to improve the model performance?
* Can you find a better combination of decoder and encoder activation functions?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:

<!-- slides -->

::: {.slide}
**AutoRec** (Sedhain et al., 2015) — recasts collaborative
filtering as autoencoder reconstruction.

The input is a *partially observed* rating vector for one
item (1 column of the rating matrix, length = #users, with
zeros for unobserved entries). The autoencoder reconstructs
it. Loss is computed only at the *observed* positions —
unobserved entries are ignored.

$$\mathcal{L} = \sum_{(u,i) \in \Omega} (r_{ui} - h(\mathbf{r}_{*i}; \theta)_u)^2 + \lambda \|\theta\|^2.$$

Adds the nonlinearity that pure MF lacks. Two variants:
**user-based** (input = ratings the user gave) and
**item-based** (input = ratings the item received). The
deck implements item-based.
:::

::: {.slide title="The model"}
Encoder: linear → activation → bottleneck. Decoder: linear
→ ratings. Train as an autoencoder over the item vectors:

@autorec-model

. . .

@autorec-implementing-the-model
:::

::: {.slide title="Evaluator with masking"}
RMSE only over observed positions (mask out the zeros):

@autorec-reimplementing-the-evaluator
:::

::: {.slide title="Training"}
Standard SGD; the masked loss is the trick that turns
autoencoder loss into a recommender:

@autorec-training-and-evaluating-the-model
:::

::: {.slide title="Recap"}
- AutoRec = rating-vector autoencoder with masked loss.
- One nonlinearity bridge between matrix factorization and
  full neural CF.
- User-based or item-based; item-based usually performs
  slightly better on MovieLens.
- Dense neural-CF models (NeuMF, next deck) build on the
  same idea with explicit user/item embeddings instead of
  raw rating vectors.
:::
