```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Predicting House Prices on Kaggle
:label:`sec_kaggle_house`

Now that we have introduced some basic tools
for building and training deep networks
and regularizing them with techniques including
weight decay and dropout,
we are ready to put all this knowledge into practice
by participating in a Kaggle competition.
The house price prediction competition
is a great place to start.
The data is fairly generic and does not exhibit exotic structure
that might require specialized models (as audio or video might).
This dataset, collected by :citet:`De-Cock.2011`,
covers house prices in Ames, Iowa from the period 2006--2010.
It was assembled as a modern, larger alternative to the small
end-of-century teaching datasets that preceded it, boasting both more
examples and more features.


In this section, we will walk you through details of
data preprocessing, model design, and hyperparameter selection.
We hope that through a hands-on approach,
you will gain some intuitions that will guide you
in your career as a data scientist.

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd

npx.set_np()
```

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
```

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
```

## Kaggle

[Kaggle](https://www.kaggle.com) is a popular platform
that hosts machine learning competitions.
Each competition centers on a dataset and many
are sponsored by stakeholders who offer prizes
to the winning solutions.
The platform helps users to interact
via forums and shared code,
fostering both collaboration and competition.
While leaderboard chasing often spirals out of control,
with researchers focusing myopically on preprocessing steps
rather than asking fundamental questions,
there is also tremendous value in the objectivity of a platform
that facilitates direct quantitative comparisons
among competing approaches as well as code sharing
so that everyone can learn what did and did not work.
If you want to participate in a Kaggle competition,
you will first need to register for an account
(see :numref:`fig_kaggle`).

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

On the house price prediction competition page, as illustrated
in :numref:`fig_house_pricing`,
you can find the dataset (under the "Data" tab),
submit predictions, and see your ranking,
The URL is right here:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![The house price prediction competition page.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Accessing and Reading the Dataset

Note that the competition data is separated
into training and test sets.
Each record includes the property value of the house
and attributes such as street type, year of construction,
roof type, basement condition, etc.
The features consist of various data types.
For example, the year of construction
is represented by an integer,
the roof type by discrete categorical assignments,
and other features by floating point numbers.
And here is where reality complicates things:
for some examples, some data is altogether missing
with the missing value marked simply as "na".
The price of each house is included
for the training set only
(it is a competition after all).
We will want to partition the training set
to create a validation set,
but we only get to evaluate our models on the official test set
after uploading predictions to Kaggle.
The "Data" tab on the competition tab
in :numref:`fig_house_pricing`
has links for downloading the data.

To get started, we will read in and process the data
using `pandas`, which we introduced in :numref:`sec_pandas`.
For convenience, we use the `d2l.download` helper to fetch and cache
the Kaggle housing dataset. It performs hash-checked caching: if a file
corresponding to this dataset already exists in the cache directory and its
SHA-1 matches `sha1_hash`, the cached copy is reused, avoiding redundant
downloads (its implementation details live in the d2l library).

```{.python .input #kaggle-house-price-accessing-and-reading-the-dataset-1  n=30}
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

The training dataset includes 1460 examples,
80 features, and one label, while the validation data
contains 1459 examples and 80 features.

```{.python .input #kaggle-house-price-accessing-and-reading-the-dataset-2  n=31}
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## Data Preprocessing

Let's take a look at the first four and final two features
as well as the label (SalePrice) from the first four examples.

```{.python .input #kaggle-house-price-data-preprocessing-1  n=10}
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

We can see that in each example, the first feature is the identifier.
This helps the model determine each training example.
While this is convenient, it does not carry
any information for prediction purposes.
Hence, we will remove it from the dataset
before feeding the data into the model.
Furthermore, given a wide variety of data types,
we will need to preprocess the data before we can start modeling.


Let's start with the numerical features.
First, we apply a heuristic,
replacing all missing values
by the corresponding feature's mean.
Then, to put all features on a common scale,
we *standardize* the data by
rescaling features to zero mean and unit variance:

$$x \leftarrow \frac{x - \mu}{\sigma},$$

where $\mu$ and $\sigma$ denote the feature's mean and standard deviation.
By the definition of mean and variance, the rescaled feature has
$E\!\left[\frac{x-\mu}{\sigma}\right] = 0$ and
$\mathrm{Var}\!\left[\frac{x-\mu}{\sigma}\right] = 1$,
so every column now lives on the same zero-mean, unit-variance scale.
Crucially, we compute $\mu$ and $\sigma$ from the *training* set only and
apply the very same transformation to the test set. Using statistics that
include the test data would let information about the test distribution
seep into our preprocessing, optimistically biasing every evaluation we
make afterwards. This pitfall, *test-set leakage*, is one of the most
common ways a model looks better offline than it ever does in deployment.

Intuitively, we standardize the data for three reasons.
First, it proves convenient for optimization, putting all coordinates on a
comparable scale. Second, because we do not know *a priori*
which features will be relevant, we do not want to penalize coefficients
assigned to one feature more than any other (a single scale lets weight
decay treat them even-handedly). Third, it makes our mean-imputation step
coherent: after standardization, filling a missing value with the column
mean is exactly filling it with $0$, the same neutral value regardless of
the feature's original units.

Next we deal with discrete values.
These include features such as "MSZoning".
We replace them by a one-hot encoding
in the same way that we earlier transformed
multiclass labels into vectors (see :numref:`subsec_classification-problem`).
For instance, "MSZoning" assumes the values "RL" and "RM".
Dropping the "MSZoning" feature,
two new indicator features
"MSZoning_RL" and "MSZoning_RM" are created with values being either 0 or 1.
According to one-hot encoding,
if the original value of "MSZoning" is "RL",
then "MSZoning_RL" is 1 and "MSZoning_RM" is 0.
The `pandas` package does this automatically for us.

```{.python .input #kaggle-house-price-data-preprocessing-2  n=32}
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns using training-set statistics only
    # (to avoid leaking test-set information into the normalization).
    numeric_features = features.select_dtypes(include='number').columns
    n_train = self.raw_train.shape[0]
    train_mean = features[numeric_features].iloc[:n_train].mean()
    train_std = features[numeric_features].iloc[:n_train].std()
    features[numeric_features] = (
        features[numeric_features] - train_mean) / train_std
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:n_train].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[n_train:].copy()
```

You can see that this conversion increases
the number of features from 79 to 331 (excluding ID and label columns).

```{.python .input #kaggle-house-price-data-preprocessing-3  n=33}
data.preprocess()
data.train.shape
```

## Error Measure

Before choosing a model, we need to decide what "good" means: the loss we train against and the metric we are scored on. Whatever model we fit, getting it to beat random guessing is a first sanity check that there is meaningful signal in the data and no data-processing bug, and the first working model becomes a baseline that tells us how much room fancier models have to improve. So the choice of error measure comes first.

With house prices, as with stock prices,
we care about relative quantities
more than absolute quantities.
Thus we tend to care more about
the relative error $\frac{y - \hat{y}}{y}$
than about the absolute error $y - \hat{y}$.
For instance, if our prediction is off by \$100,000
when estimating the price of a house in rural Ohio,
where the value of a typical house is \$125,000,
then we are probably doing a horrible job.
On the other hand, if we err by this amount
in Los Altos Hills, California,
this might represent a stunningly accurate prediction
(there, the median house price exceeds \$4 million).

One way to address this problem is to
measure the discrepancy in the logarithm of the price estimates.
In fact, this is also the official error measure
used by the competition to evaluate the quality of submissions.
After all, a small value $\delta$ for $|\log y - \log \hat{y}| \leq \delta$
translates into $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
This leads to the following root-mean-squared-error between the logarithm of the predicted price and the logarithm of the label price:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input #kaggle-house-price-error-measure  n=60}
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float),
                                      dtype=d2l.float32)
    # Logarithm of prices 
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## $K$-Fold Cross-Validation

You might recall that we introduced cross-validation
in :numref:`subsec_generalization-model-selection`, where we discussed how to deal
with model selection.
We will put this to good use to select the model design
and to adjust the hyperparameters.
The idea is shown in :numref:`fig_kfold`: we partition the data into $K$
equal folds and run $K$ training rounds. In round $i$, fold $i$ is held out
for validation and the model is trained on the remaining $K-1$ folds; our
generalization estimate is the average of the $K$ validation scores. With
only about $1500$ training examples here, this reuse of the data gives a far
steadier estimate than any single train/validation split would.

![In $K$-fold cross-validation with $K=5$, the data is partitioned into five equal folds. In each round $i$, fold $i$ is held out as the validation set (orange) and the model is trained on the remaining four folds (blue). The generalization estimate is the average of the five validation scores.](../img/mdl-mlp-kfold.svg)
:label:`fig_kfold`

We first need a function that returns
the $i^\textrm{th}$ fold of the data
in a $K$-fold cross-validation procedure.
It proceeds by slicing out the $i^\textrm{th}$ segment
as validation data and returning the rest as training data.
Note that this is not the most efficient way of handling data
and we would definitely do something much smarter
if our dataset was considerably larger.
But this added complexity might obfuscate our code unnecessarily
so we can safely omit it here owing to the simplicity of our problem.

```{.python .input #kaggle-house-price-k-fold-cross-validation-1}
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = list(range(j * fold_size, (j+1) * fold_size))
        rets.append(KaggleHouse(data.batch_size,
                                data.train.drop(index=idx),
                                data.train.iloc[idx]))
    return rets
```

The average validation error is returned
when we train $K$ times in the $K$-fold cross-validation. We pass in a
`model_fn` that builds a fresh model for each fold, so the *same*
cross-validation loop can score a linear baseline or an MLP without change.

```{.python .input #kaggle-house-price-k-fold-cross-validation-2}
%%tab pytorch
def k_fold(trainer, data, k, model_fn):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = model_fn()
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

```{.python .input #kaggle-house-price-k-fold-cross-validation-2}
%%tab mxnet, tensorflow
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

```{.python .input #kaggle-house-price-k-fold-cross-validation-2}
%%tab jax
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        # In JAX/Flax, params live in trainer.state, not the (frozen) model.
        # Capture each fold's trained params so the ensemble can use them.
        models.append((model, trainer.state.params))
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## Model Selection

We now have everything we need to compare model designs by their
$K$-fold cross-validation score. Finding a good choice can take time,
depending on how many variables one optimizes over.
With a large enough dataset,
and the normal sorts of hyperparameters,
$K$-fold cross-validation tends to be
reasonably resilient against multiple testing.
However, if we try an unreasonably large number of options
we might find that our validation
performance is no longer representative of the true error.

One honest caveat before we run the numbers. On *structured tabular data*
like this dataset, gradient-boosted tree ensembles (XGBoost and LightGBM)
typically outperform deep networks, including MLPs
:cite:`Grinsztajn.Oyallon.Varoquaux.2022,Shwartz-Ziv.Armon.2022`, and the
public leaderboard for this competition reflects that: the top submissions
are almost always tree-based, with neural networks some distance behind. This
does not diminish what we learn here, since the preprocessing pipeline,
log-RMSE loss, and $K$-fold cross-validation apply to *any* model class.
But it does set honest expectations about where neural networks shine
(images, text, audio, and sequences) and where they currently do not
(small-to-medium tabular data).

We start with a linear model. It is a fast, honest baseline that
sanity-checks the pipeline. One subtlety is easy to get wrong and worth
stating plainly: a baseline is only meaningful if it is *trained
competently*. Plain minibatch SGD on these standardized features needs more
than a handful of passes to converge, so we give the linear model a healthy
$100$ epochs at learning rate $0.03$ (a larger rate diverges on the
log-price target). Trained this way the linear model already reaches a
cross-validated log error well under $0.05$; stopping at the customary ten
epochs would instead leave it badly underfit (closer to $0.18$), which would
flatter every model we compared against it.

```{.python .input #kaggle-house-price-model-selection-linear}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=100)
linear_models = k_fold(trainer, data, k=5,
                       model_fn=lambda: d2l.LinearRegression(lr=0.03))
```

```{.python .input #kaggle-house-price-model-selection-linear}
%%tab mxnet, tensorflow, jax
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

Can a small neural network do better? Now that we have weight decay
(:numref:`sec_weight_decay`), dropout (:numref:`sec_dropout`), and sensible
initialization (:numref:`sec_numerical_stability`) in hand, we can try the
simplest possible upgrade: a single hidden layer with a ReLU
nonlinearity. The dataset is tiny (about $1460$ rows, $331$ features after
one-hot encoding), so capacity is the enemy. We therefore keep the network
*small* and lean on regularization: a modest $32$-unit hidden layer, a light
dropout of $0.1$, and a small amount of $L_2$ weight decay ($10^{-4}$) added
straight into SGD. A wider net or aggressive dropout (the $0.5$ that is
common on large datasets) simply overfits or fails to train on data this
small. We reuse the squared-error loss from `LinearRegression` and only
override the optimizer to attach weight decay.

```{.python .input #kaggle-house-price-mlp-model}
%%tab pytorch
class KaggleMLP(d2l.LinearRegression):
    def __init__(self, lr, num_hiddens=32, dropout=0.1, weight_decay=1e-4):
        super(d2l.LinearRegression, self).__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.Dropout(dropout), nn.LazyLinear(1))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
```

We train it with the *same* $K$-fold loop, learning rate, and epoch budget
as the linear baseline, so the only thing that changes is the model.

```{.python .input #kaggle-house-price-mlp-select}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=100)
models = k_fold(trainer, data, k=5, model_fn=lambda: KaggleMLP(lr=0.03))
```

On this run the small MLP edges out the (now competently trained) linear
baseline: both land near a cross-validated log error of $0.03$, with the MLP
a hair lower. The lesson is deliberately undramatic. A nonlinear model helps
a little here, but only once it is small enough and regularized enough to
survive a dataset of barely a thousand rows; the bulk of the gain over a
careless $0.18$ baseline came simply from training *either* model to
convergence. And as the caveat above promised, a gradient-boosted tree
ensemble would still be the stronger tabular choice. The exercises invite you
to try one and see.

Notice that sometimes the number of training errors
for a set of hyperparameters can be very low,
even as the number of errors on $K$-fold cross-validation
grows considerably higher.
This indicates that we are overfitting.
Throughout training you will want to monitor both numbers.
Less overfitting might indicate that our data can support a more powerful model.
Massive overfitting might suggest that we can gain
by incorporating regularization techniques.

##  Submitting Predictions on Kaggle

Now that we know what a good choice of hyperparameters should be,
we might 
calculate the average predictions 
on the test set
by all the $K$ models.
Saving the predictions in a csv file
will simplify uploading the results to Kaggle.
The following code will generate a file called `submission.csv`.

```{.python .input #kaggle-house-price-submitting-predictions-on-kaggle}
%%tab pytorch
preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
         for model in models]
# Each model predicts a log-price; exponentiate back to a price, then average
# across the K folds. (Averaging in log space, then exponentiating, makes this
# a geometric mean of the per-fold price predictions.)
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

```{.python .input #kaggle-house-price-submitting-predictions-on-kaggle}
%%tab tensorflow
preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
         for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

```{.python .input #kaggle-house-price-submitting-predictions-on-kaggle}
%%tab jax
preds = [model.apply({'params': params},
         d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
         for model, params in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

```{.python .input #kaggle-house-price-submitting-predictions-on-kaggle}
%%tab mxnet
preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
         for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

Next, as demonstrated in :numref:`fig_kaggle_submit2`,
we can submit our predictions on Kaggle
and see how they compare with the actual house prices (labels)
on the test set.
The steps are quite simple:

* Log in to the Kaggle website and visit the house price prediction competition page.
* Click the “Submit Predictions” or “Late Submission” button.
* Click the “Upload Submission File” button in the dashed box at the bottom of the page and select the prediction file you wish to upload.
* Click the “Make Submission” button at the bottom of the page to view your results.

![Submitting data to Kaggle.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Summary and Discussion

Real data is messy: a mix of numeric and categorical features, with missing
values and wildly different scales. The preprocessing pipeline in this
section (mean imputation, standardization with statistics fit on the training
set only to avoid test-set leakage, and one-hot encoding of categoricals) is
a sensible default that applies far beyond this competition. When the target
spans an order of magnitude, predicting the *logarithm* of the price and
scoring with root-mean-squared log error converts an asymmetric dollar-scale
problem into one where a $10\%$ error on a $\$100{,}000$ house and on a
$\$1{,}000{,}000$ house are penalized equally, which is what we actually care
about.

$K$-fold cross-validation is the right tool when training data is limited
(about $1500$ examples here): it spends $K$ training runs to buy a stable
estimate of generalization error, and that same loop doubles as the
infrastructure for hyperparameter search.

Two caveats belong in any honest account of this pipeline. First, none of it
is specific to neural networks. The same preprocessing, loss design, and
cross-validation loop work with any model class, including the
gradient-boosted tree ensembles (XGBoost, LightGBM) that routinely beat deep
networks on medium-sized tabular data
:cite:`Grinsztajn.Oyallon.Varoquaux.2022,Shwartz-Ziv.Armon.2022`. On images,
text, and audio the balance tips the other way, which is where the rest of
this book lives. Second, model capacity matters: adding hidden layers, tuning
the dropout rate, and searching over learning rate and weight decay can
improve substantially on the baseline shown here, which is exactly what the
exercises ask you to do.

Looking ahead, the moves we made here recur throughout supervised learning.
Feature scaling and imputation reappear in nearly every tabular pipeline, and
the competition recipe (download, preprocess, match the loss to the metric,
cross-validate, refit, submit) generalizes directly: later chapters apply the
same cross-validation discipline to image datasets, sequence tasks, and the
fine-tuning of pretrained models.



## Exercises

1. Submit your predictions for this section to Kaggle. How good are they?
1. Is it always a good idea to replace missing values by a mean? Hint: can you construct a situation where the values are not missing at random?
1. Improve the score by tuning the hyperparameters through $K$-fold cross-validation.
1. Improve the score by improving the model (e.g., layers, weight decay, and dropout).
1. What happens if we do not standardize the continuous numerical features as we have done in this section?
1. Swap the linear model for a gradient-boosted tree model (for example scikit-learn's `GradientBoostingRegressor`, or XGBoost or LightGBM if installed), trained on the same preprocessed features. How does its cross-validated log-RMSE compare? Why might tree ensembles have an edge on data like this?
1. Revisit the preprocessing choices. How does median imputation compare to mean imputation? What changes if you encode high-cardinality categorical features with target encoding or a learned embedding instead of one-hot vectors?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/237)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17988)
:end_tab:

<!-- slides -->

::: {.slide title="House prices: a full ML pipeline"}
**House Prices: Advanced Regression Techniques** —
predict sale prices of Ames, Iowa houses from 80 numeric
and categorical features. A small but realistic end-to-end
ML exercise.

What makes it interesting:

- **1460 train / 1459 test** — small data.
- **80 mixed features** — needs preprocessing.
- **Missing values** in dozens of columns.
- **Targets vary 10×** ($65k–$755k) — wrong loss
  overweights expensive houses.

The MLP is 5 lines. The lesson is the **pipeline** around
it — preprocessing, the right loss, CV, submission.
:::

::: {.slide title="Kaggle in 30 seconds"}
Kaggle hosts open ML competitions. Download train + test
CSVs, train locally, upload predictions, get scored on a
held-out portion of the test set.

![Kaggle competition page.](../img/kaggle.png){width=80%}
:::

::: {.slide title="The House Prices page"}
![The competition's data tab — download and inspect.](../img/house-pricing.png){width=82%}

Real-world ML practice: data isn't preprocessed for you,
the leaderboard tells you instantly if you're better than
baseline, and the public/private split keeps people honest
about overfitting.
:::

::: {.slide title="Setup and imports"}
One per-framework imports cell. We read the CSVs with
`pandas` and lean on `d2l.download`, a reusable hash-checked
download helper we keep using throughout the book:

@kaggle-house-price-predicting-house-prices-on-kaggle
:::

::: {.slide title="Reading the data"}
Wrap train and test CSVs in a `KaggleHouse(d2l.DataModule)`:

@kaggle-house-price-accessing-and-reading-the-dataset-1

. . .

@kaggle-house-price-accessing-and-reading-the-dataset-2
:::

::: {.slide title="What the rows look like"}
@kaggle-house-price-data-preprocessing-1

Mixed numeric and categorical columns; lots of missing
values; final column is the target `SalePrice`. Models eat
tensors, not DataFrames — preprocessing is mandatory.
:::

::: {.slide title="Three preprocessing transforms"}
**Fit statistics on train**, then apply the same transform
to train and test — so test data is mapped with the
training-time statistics (no test-set leakage):

1. **Numeric NaN → mean.** Simple imputation; median or
   zero are alternatives.
2. **Standardize** numeric columns to mean 0, std 1 —
   makes optimization well-conditioned across wildly
   different scales.
3. **One-hot encode** categorical columns. `NaN` becomes
   its own category — missing-as-a-signal.
:::

::: {.slide title="The transforms in code"}
@kaggle-house-price-data-preprocessing-2

. . .

@kaggle-house-price-data-preprocessing-3

After preprocessing: ~330 columns of well-scaled floats.
:::

::: {.slide title="Choosing the right loss"}
Plain squared error penalizes a $10k mistake on a $70k
house the same as on a $700k house. The *relative* error
is more honest — predict the **logarithm** of the price:

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \big(\log y_i - \log \hat y_i\big)^2}.$$

The official Kaggle metric for this competition. Mistakes
are measured **as percentages**, not dollars.
:::

::: {.slide title="Loss in code"}
@kaggle-house-price-error-measure
:::

::: {.slide title="K-fold cross-validation"}
With ~1500 training examples, a single 80/20 split is
noisy. **K-fold CV**: split into $K$ folds, train $K$
times holding each fold out, average the scores.

```
fold 1:  [ val ][ train ][ train ][ train ][ train ]
fold 2:  [train][  val  ][ train ][ train ][ train ]
fold 3:  [train][ train ][  val  ][ train ][ train ]
fold 4:  [train][ train ][ train ][  val  ][ train ]
fold 5:  [train][ train ][ train ][ train ][  val  ]
```

Costs $K\times$ more compute; gives a stable estimate of
generalization error.
:::

::: {.slide title="K-fold in code"}
@kaggle-house-price-k-fold-cross-validation-1

. . .

@kaggle-house-price-k-fold-cross-validation-2
:::

::: {.slide title="Model selection: a competent baseline"}
A baseline only means something if it is *trained to
convergence*. 100 epochs, lr 0.03 — not the customary 10:

@kaggle-house-price-model-selection-linear

Stopping early leaves the linear model badly underfit
(~0.18) and flatters everything compared against it.
:::

::: {.slide title="Model selection: a small MLP"}
With weight decay, dropout, and good init in hand, try the
simplest upgrade — *one small, regularized hidden layer*:

@kaggle-house-price-mlp-model

. . .

@kaggle-house-price-mlp-select

Same loop, lr, and epochs; only the model changes. It edges
out the linear baseline (~0.03). GBDTs still win tabular.
:::

::: {.slide title="Submitting predictions"}
Final step: re-fit on **all** training data (no
validation hold-out), predict the test set, write the
Kaggle-format CSV:

@kaggle-house-price-submitting-predictions-on-kaggle

![Upload the CSV; Kaggle scores it instantly against the held-out half of the test set.](../img/kaggle-submit2.png){width=84%}
:::

::: {.slide title="The general competition recipe"}
Same shape works for almost any tabular ML competition:

1. **Download** train + test data.
2. **Preprocess** — impute, scale, encode (combined stats).
3. **Choose the right loss** — match the scoring metric.
4. **K-fold CV** for generalization estimate + HP search.
5. **Refit on all training data** with the best HPs.
6. **Submit** predictions in the host's format.

GBDTs (XGBoost / LightGBM) usually win tabular; nets shine
on images, text, audio. The pipeline is the same.
:::

::: {.slide title="Recap"}
- Real-world ML is mostly **pipeline**, not model
  architecture.
- Heterogeneous tabular data → impute, standardize,
  one-hot encode.
- Match the loss to the metric — log-RMSE for prices, not
  squared error.
- K-fold CV for stable generalization estimates on small
  data.
- Refit on full data before final predictions.
- The MLP is a few lines; everything around it is the lesson.
:::
