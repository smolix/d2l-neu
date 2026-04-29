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
It is considerably larger than the famous [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) of Harrison and Rubinfeld (1978),
boasting both more examples and more features.


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

## Downloading Data

Throughout the book, we will train and test models
on various downloaded datasets.
Here, we implement two utility functions
for downloading and extracting zip or tar files.
Again, we skip implementation details of
such utility functions.

```{.python .input #kaggle-house-price-downloading-data  n=2}
def download(url, folder, sha1_hash=None):
    """Download a file to folder and return the local filepath."""

def extract(filename, folder):
    """Extract a zip/tar file into folder."""
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
For convenience, we can download and cache
the Kaggle housing dataset.
If a file corresponding to this dataset already exists in the cache directory and its SHA-1 matches `sha1_hash`, our code will use the cached file to avoid clogging up your Internet with redundant downloads.

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

where $\mu$ and $\sigma$ denote mean and standard deviation, respectively.
To verify that this indeed transforms
our feature (variable) such that it has zero mean and unit variance,
note that $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$
and that $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$.
Intuitively, we standardize the data
for two reasons.
First, it proves convenient for optimization.
Second, because we do not know *a priori*
which features will be relevant,
we do not want to penalize coefficients
assigned to one feature more than any other.

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

To get started we will train a linear model with squared loss. Not surprisingly, our linear model will not lead to a competition-winning submission but it does provide a sanity check to see whether there is meaningful information in the data. If we cannot do better than random guessing here, then there might be a good chance that we have a data processing bug. And if things work, the linear model will serve as a baseline giving us some intuition about how close the simple model gets to the best reported models, giving us a sense of how much gain we should expect from fancier models.

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
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

The average validation error is returned
when we train $K$ times in the $K$-fold cross-validation.

```{.python .input #kaggle-house-price-k-fold-cross-validation-2}
%%tab pytorch, mxnet, tensorflow
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

In this example, we pick an untuned set of hyperparameters
and leave it up to the reader to improve the model.
Finding a good choice can take time,
depending on how many variables one optimizes over.
With a large enough dataset,
and the normal sorts of hyperparameters,
$K$-fold cross-validation tends to be
reasonably resilient against multiple testing.
However, if we try an unreasonably large number of options
we might find that our validation
performance is no longer representative of the true error.

```{.python .input #kaggle-house-price-model-selection}
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

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
# Taking exponentiation of predictions in the logarithm scale
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

Real data often contains a mix of different data types and needs to be preprocessed.
Rescaling real-valued data to zero mean and unit variance is a good default. So is replacing missing values with their mean.
Furthermore, transforming categorical features into indicator features allows us to treat them like one-hot vectors.
When we tend to care more about
the relative error than about the absolute error,
we can 
measure the discrepancy in the logarithm of the prediction.
To select the model and adjust the hyperparameters,
we can use $K$-fold cross-validation .



## Exercises

1. Submit your predictions for this section to Kaggle. How good are they?
1. Is it always a good idea to replace missing values by a mean? Hint: can you construct a situation where the values are not missing at random?
1. Improve the score by tuning the hyperparameters through $K$-fold cross-validation.
1. Improve the score by improving the model (e.g., layers, weight decay, and dropout).
1. What happens if we do not standardize the continuous numerical features as we have done in this section?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17988)
:end_tab:

<!-- slides -->

::: {.slide}
A capstone for the chapter: take an MLP, plus everything we
just learned about regularization and stability, to a real
**Kaggle competition** — *House Prices: Advanced Regression
Techniques*.

What's different from Fashion-MNIST:

- **1460 training examples**, 1459 test — small data.
- **80 features** mixing numeric (`LotArea`, `YearBuilt`)
  and categorical (`Neighborhood`, `RoofStyle`) — needs
  *preprocessing*.
- **Missing values** in dozens of columns.
- **Targets vary by 10×** ($65k to $755k) — wrong loss
  function will weight expensive houses much more.

The MLP itself is 5 lines. The real work is the
**pipeline** around it — and that's the lesson of this
deck. Most production ML is plumbing.
:::

::: {.slide title="Kaggle in 30 seconds"}
Kaggle hosts open ML competitions. You download train and
test CSVs, train locally, upload predictions, get scored
on a held-out portion of the test set:

![Kaggle competition page.](../img/kaggle.png){width=68%}

The *House Prices* page:

![The competition's data tab — download and inspect.](../img/house-pricing.png){width=68%}

Real-world ML practice: the data isn't preprocessed for
you, the leaderboard tells you instantly if you're better
than baseline, and the public/private split keeps people
honest about overfitting.
:::

::: {.slide title="Setup and data download"}
A reusable hash-checked download helper we'll keep using
throughout the book:

@kaggle-house-price-predicting-house-prices-on-kaggle

. . .

@kaggle-house-price-downloading-data
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
Apply on **train + test together** so train statistics
match what we'll see at test time:

1. **Numeric NaN → mean**. Most simple imputation choice;
   median or zero are alternatives.
2. **Standardize** numeric columns to mean 0, std 1.
   Tabular features have wildly different scales
   (rooms: 1–10, area: 100–60000); standardization makes
   the optimization well-conditioned.
3. **One-hot encode** categorical columns. `NaN` becomes
   its own category — missing-as-a-signal.

@kaggle-house-price-data-preprocessing-2

. . .

@kaggle-house-price-data-preprocessing-3

After preprocessing: ~330 columns of well-scaled floats.
:::

::: {.slide title="Choosing the right loss"}
Plain squared error penalizes a $10k mistake on a $70k
house the same as on a $700k house — but the *relative*
error tells the more honest story. Predict the
**logarithm** of the price:

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \big(\log y_i - \log \hat y_i\big)^2}.$$

This is also the official Kaggle metric for this
competition. Two equivalent perspectives:

- Train on $\log y$ and use plain RMSE.
- Train on $y$ and use a log-loss.

Either way, mistakes are measured **as percentages**, not
dollars.

@kaggle-house-price-error-measure
:::

::: {.slide title="K-fold cross-validation"}
With ~1500 training examples, a single 80/20 split is
noisy — different splits give different validation
scores by ±0.02 RMSLE.

**K-fold CV**: split the training set into $K$ folds, train
$K$ times holding each fold out as validation, average the
scores. Costs $K\times$ more compute, but gives a stable
estimate of generalization error:

```
fold 1:  [ val ][ train ][ train ][ train ][ train ]
fold 2:  [train][  val  ][ train ][ train ][ train ]
fold 3:  [train][ train ][  val  ][ train ][ train ]
fold 4:  [train][ train ][ train ][  val  ][ train ]
fold 5:  [train][ train ][ train ][ train ][  val  ]
```

@kaggle-house-price-k-fold-cross-validation-1

. . .

@kaggle-house-price-k-fold-cross-validation-2
:::

::: {.slide title="Model selection"}
@kaggle-house-price-model-selection

. . .

@!kaggle-house-price-model-selection

In practice you'd grid- or random-search over learning
rate, hidden size, weight decay, dropout. Same loop,
different hyperparameters. Pick the config with the lowest
**average** val score.
:::

::: {.slide title="Submitting predictions"}
Final step: re-fit on **all** training data (no
validation hold-out), predict the test set, write the
Kaggle-format CSV:

@kaggle-house-price-submitting-predictions-on-kaggle

![Upload the CSV; Kaggle scores it instantly against the held-out half of the test set.](../img/kaggle-submit2.png){width=72%}
:::

::: {.slide title="The general competition recipe"}
Same shape works for almost any tabular ML competition:

1. **Download** train + test data.
2. **Preprocess** — impute, scale, encode. Use
   *combined* statistics so train and test match.
3. **Choose the right loss** — match the metric the
   competition is scored on.
4. **K-fold CV** to estimate generalization error and
   compare hyperparameters.
5. **Refit on all training data** with the best
   hyperparameters.
6. **Submit** predictions in the format the host
   requires.

Beyond MLPs: gradient-boosted trees (XGBoost / LightGBM)
typically win tabular competitions; neural networks shine
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
