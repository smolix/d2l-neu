# AutoRec: Rating Prediction with Autoencoders

Matrix factorization assigns each user and item a vector and scores their
interaction with a bilinear function. AutoRec replaces that score with a
nonlinear reconstruction map :cite:`Sedhain.Menon.Sanner.ea.2015`. Its input is
a partially observed row or column of the rating matrix, and its output predicts
ratings for the same user or item. The observation mask, rather than the filler
value used for missing entries, determines which reconstruction errors enter the
loss.

Like an ordinary autoencoder, AutoRec contains an encoder, a hidden
representation, and a decoder. Its distinctive feature is the data it
reconstructs: a sparse vector of ratings rather than a fully observed example.
The hidden representation and decoder are learned jointly from errors on the
observed output coordinates; predictions at unobserved coordinates supply the
recommendations.

An item-based AutoRec reconstructs one item column across users. A user-based
variant applies the same construction to user rows. We develop the item-based
version.


## Model

Let $\mathbf R\in\mathbb R^{m\times n}$ contain ratings from $m$ users for
$n$ items, and let $M_{ui}=1$ when $R_{ui}$ is observed and $0$ otherwise.
For item $i$, the network input is the filled vector
$\mathbf x_i=\mathbf M_{*i}\odot\mathbf R_{*i}\in\mathbb R^m$. Zero is only a
computational filler; $\mathbf M$ determines which entries are missing. With a
hidden width $h$, item-based AutoRec computes

$$
h_\theta(\mathbf x_i)
=f\!\left(\mathbf Wg(\mathbf V\mathbf x_i+\boldsymbol\mu)+\mathbf b\right),
$$

where $\mathbf V\in\mathbb R^{h\times m}$,
$\boldsymbol\mu\in\mathbb R^h$, $\mathbf W\in\mathbb R^{m\times h}$, and
$\mathbf b\in\mathbb R^m$. The output contains one reconstructed rating per
user.

The observation mask is applied to the output error:

$$
\underset{\theta}{\operatorname{minimize}}\quad
\sum_{i=1}^n\sum_{u=1}^m
M_{ui}\big(R_{ui}-h_\theta(\mathbf x_i)_u\big)^2
+\lambda\big(\|\mathbf W\|_F^2+\|\mathbf V\|_F^2\big).
$$

Masking removes loss terms for unobserved *outputs*. It does not restrict an
update to a subset of parameters: every encoder or decoder parameter that
influences an observed reconstruction may receive a gradient.

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

The implementation uses a sigmoid encoder, a linear decoder, and dropout on the
hidden representation. During training, multiplying the predictions by the
binary observation mask makes the squared loss equivalent to the elementwise
objective above. During evaluation, the unmasked output is needed because its
unobserved coordinates are the predictions of interest. This code assumes all
recorded ratings are positive, so `sign(input)` is the binary mask.

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
        if autograd.is_training():  # Mask unobserved output errors
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
        if self.training:  # Mask unobserved output errors
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

We train AutoRec on the MovieLens split defined earlier and report RMSE over held-out observed ratings. In this run its RMSE can be compared with the matrix-factorization run only as a local result: the architectures, tuning budgets, and random variation have not been controlled well enough to attribute the difference to nonlinearity alone.

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
        # The model already zeros out predictions where the input rating is
        # 0 (unobserved). Comparing against `values` directly therefore
        # gives a loss that only penalizes observed entries, matching the
        # masked-loss formulation used in the MXNet branch.
        l = loss(preds, values)
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

* AutoRec learns a nonlinear map from a partially observed rating vector to its
  reconstruction; an observation mask restricts the training error to known
  ratings.
* On the MovieLens split and hyperparameters used here, AutoRec obtains a lower
  test RMSE than the matrix-factorization run. This local comparison is not a
  general ranking of the two model classes.



## Exercises

* Vary the hidden dimension of AutoRec to see its impact on the model performance.
* Try to add more hidden layers. Is it helpful to improve the model performance?
* Can you find a better combination of decoder and encoder activation functions?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/401)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/401)
:end_tab:

<!-- slides -->

::: {.slide title="AutoRec"}
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

::: {.slide title="The observation mask separates missing from zero"}
The setup cell selects the backend-specific `d2l` package and
tensor library. The model itself is the same idea in both tabs:
reconstruct an item rating vector with a masked loss.

@autorec-model
:::

::: {.slide title="AutoRec reconstructs a sparse rating vector"}
Encoder: linear -> activation -> bottleneck. Decoder: linear
-> ratings. During training, the forward pass masks unobserved
entries so gradients come only from known ratings:

@autorec-implementing-the-model
:::

::: {.slide title="Evaluator with masking"}
RMSE only over observed positions (mask out the zeros):

@autorec-reimplementing-the-evaluator
:::

::: {.slide title="Only observed ratings contribute to reconstruction error"}
Standard SGD; the masked loss is the trick that turns
autoencoder loss into a recommender:

@autorec-training-and-evaluating-the-model

Watch the plot for two signals: training loss should fall, and
test RMSE should stabilize rather than diverge. Overfitting shows
up when reconstruction keeps improving but held-out RMSE worsens.
:::

::: {.slide title="AutoRec replaces a bilinear score with nonlinear reconstruction"}
- AutoRec = rating-vector autoencoder with masked loss.
- One nonlinearity bridge between matrix factorization and
  full neural CF.
- User-based or item-based; item-based usually performs
  slightly better on MovieLens.
- Dense neural-CF models (NeuMF, next deck) build on the
  same idea with explicit user/item embeddings instead of
  raw rating vectors.
:::
