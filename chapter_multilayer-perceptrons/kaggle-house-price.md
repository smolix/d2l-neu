```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Predicting House Prices on Kaggle
:label:`sec_kaggle_house`

Clean image benchmarks like Fashion-MNIST arrive with uniform pixel grids in $[0, 1]$ and zero missing values. Real-world tabular data rarely looks so tidy. In this capstone section for Part I, we tackle the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition compiled by :citet:`De-Cock.2011` from Ames, Iowa property sales (2006–2010): $1{,}460$ labeled training homes and $1{,}459$ unlabeled test homes, each described by $79$ heterogeneous numerical and categorical attributes with scales spanning five orders of magnitude and missing entries across $19$ columns.

Working through this end-to-end tabular pipeline brings together every tool from Chapters 1–4: `pandas` preprocessing without data leakage, logarithmic target transformation, $K$-fold cross-validation on shared folds, linear regression baselines, and He-initialized MLPs regularized with weight decay.

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #kaggle-house-price-predicting-house-prices-on-kaggle-1}
import pandas as pd
```


## The Dataset and the Log-RMSE Metric

We download and cache the training and test CSV files via `d2l.download` and load them into `pandas` DataFrames (:numref:`sec_pandas`):

```{.python .input #kaggle-house-price-accessing-and-reading-the-dataset-1  n=30}
def read_house_prices(root='../data'):
    """Download and read the training and test tables of the competition."""
    train = pd.read_csv(d2l.download(
        d2l.DATA_URL + 'kaggle_house_pred_train.csv', root,
        sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
    test = pd.read_csv(d2l.download(
        d2l.DATA_URL + 'kaggle_house_pred_test.csv', root,
        sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
    return train, test
```

Calling `read_house_prices()` loads the two tables into `raw_train` and `raw_test`, whose shapes confirm $1{,}460$ training homes across $81$ columns (including `SalePrice`) and $1{,}459$ test homes across $80$ columns:

```{.python .input #kaggle-house-price-accessing-and-reading-the-dataset-2  n=31}
raw_train, raw_test = read_house_prices()
print(raw_train.shape, raw_test.shape)
```

Slicing the first four rows across the first four and last three columns of `raw_train` via `iloc` displays a representative mix of integer codes (`Id`, `MSSubClass`), string categories (`MSZoning`, `SaleType`, `SaleCondition`), continuous measurements (`LotFrontage`), and the dollar target `SalePrice`:

```{.python .input #kaggle-house-price-data-preprocessing-1  n=10}
print(raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

Of the $79$ predictive attributes (after dropping the non-informative row index `Id`), $36$ are stored as numbers and $43$ as strings, with missing values often encoding the absence of a feature (such as `PoolQC = NaN` for homes without a swimming pool).

Because home prices range from $\$34{,}900$ to $\$755{,}000$, a $\$50{,}000$ error is catastrophic on a starter home yet modest on a mansion. Kaggle therefore scores predictions by *relative* accuracy via the root-mean-squared error of the *logarithms* of the prices:

$$\textrm{log-RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i - \log \hat{y}_i\right)^2}.$$

For small $\delta = |\log \hat{y} - \log y|$, $\hat{y}/y = e^{\pm \delta} \approx 1 \pm \delta$, so a log-RMSE of $0.135$ corresponds to roughly a $13.5\%$ typical percentage error. By training our models to predict $\log y$ under squared error loss, we optimize the square of the competition metric directly!

## Leakage-Free Preprocessing

Turning the raw DataFrames into float32 feature matrices requires four careful steps, implemented in `preprocess(train, *others)` below:

1. **Recast numeric category codes to strings:** `MSSubClass` uses numbers ($20, 60, 90, \ldots$) to name dwelling styles (1-story, 2-story, duplex); converting it to `str` prevents the model from treating code $90$ as a linear multiple of code $20$.
2. **Standardize numerical features using *training-set* statistics only:** Computing means $\mu$ and standard deviations $s$ exclusively on `train`—and applying $(x - \mu)/s$ to both training and validation/test tables—puts all numerical features on a unit scale for SGD and weight decay while preventing *data leakage* from validation or test rows.
3. **Impute missing numerical values with $0$ (the training mean):** Filling standardized `NaN` entries with `0` replaces missing numerical measurements by their training-set mean (Exercise :exref:`ex_kaggle-house-mean-imputation`).
4. **One-hot encode categoricals (`pd.get_dummies(..., dummy_na=True)`):** Expands each categorical column into binary indicators (including an explicit indicator for `NaN`) and aligns validation/test columns to the exact training column vocabulary via `reindex(..., fill_value=0)`:

```{.python .input #kaggle-house-price-data-preprocessing-2  n=32}
def preprocess(train, *others):
    """Fit the feature preprocessing on train; apply it to all tables."""
    frames = [df.drop(columns=['Id', 'SalePrice'], errors='ignore')
                .astype({'MSSubClass': str})  # A code, not a quantity
              for df in (train, *others)]
    numeric = frames[0].select_dtypes(include='number').columns
    mean, std = frames[0][numeric].mean(), frames[0][numeric].std()
    for df in frames:
        df[numeric] = ((df[numeric] - mean) / std).fillna(0)
    frames = [pd.get_dummies(df, dummy_na=True) for df in frames]
    return [df.reindex(columns=frames[0].columns, fill_value=0)
              .to_numpy(dtype=np.float32) for df in frames]
```

Running `preprocess(raw_train, raw_test)` on the full competition tables expands the $79$ raw attributes into aligned `float32` design matrices `X_train` and `X_test` with $345$ features per house:

```{.python .input #kaggle-house-price-data-preprocessing-3  n=33}
X_train, X_test = preprocess(raw_train, raw_test)
X_train.shape, X_test.shape
```

Finally, `KaggleHouse(d2l.DataModule)` pairs `preprocess` with logarithmic target centering, subtracting the training split's mean log price `self.offset` ($\approx 12.02$) from $\log(\textrm{SalePrice})$ so the network starts at the right baseline from step 1 (Exercise :exref:`ex_kaggle-house-scaling`):

```{.python .input #kaggle-house-price-error-measure  n=60}
class KaggleHouse(d2l.DataModule):
    """Features and centered log prices, fitted on the training rows."""
    def __init__(self, raw_train, raw_val, batch_size=64):
        super().__init__()
        self.save_hyperparameters(ignore=['raw_train', 'raw_val'])
        self.X_train, self.X_val = preprocess(raw_train, raw_val)
        self.num_inputs = self.X_train.shape[1]
        log_price = lambda df: np.log(
            df['SalePrice'].to_numpy(np.float32)).reshape(-1, 1)
        self.offset = log_price(raw_train).mean()  # Mean log price
        self.y_train = log_price(raw_train) - self.offset
        self.y_val = (log_price(raw_val) - self.offset
                      if 'SalePrice' in raw_val else None)

    def get_dataloader(self, train):
        X, y = ((self.X_train, self.y_train) if train
                else (self.X_val, self.y_val))
        if y is None:  # No prices to validate against
            return None
        return self.get_tensorloader((d2l.tensor(X), d2l.tensor(y)), train)
```

## Model Selection via $K$-Fold Cross-Validation

### Shared 5-Fold Splits and the Linear Baseline

With only $1{,}460$ labeled houses, a single $20\%$ validation split ($292$ houses) has a fold-to-fold standard error of several hundredths in log-RMSE. Our `k_fold_data(raw, k)` helper partitions `raw` into $K$ random folds and builds a separate `KaggleHouse` data module for each round so validation folds never leak into the training feature statistics:

```{.python .input #kaggle-house-price-k-fold-cross-validation-1}
def k_fold_data(raw, k):
    """Split the table raw into k random folds; one data module per round."""
    folds = np.array_split(np.random.default_rng().permutation(len(raw)), k)
    return [KaggleHouse(raw.drop(index=raw.index[f]), raw.iloc[f])
            for f in folds]
```

Given a list of pre-built `folds` and a model factory `make_model`, `k_fold` trains a fresh model on each fold for `max_epochs` epochs, averages the last five epochs of squared loss before taking the square root to damp single-epoch jitter, and prints the mean training and validation log-RMSEs across all folds:

```{.python .input #kaggle-house-price-k-fold-cross-validation-2}
def k_fold(folds, make_model, max_epochs=100):
    """Train a fresh model in each round; return the RMSEs of every fold."""
    rmse = []
    for fold in folds:
        trainer = d2l.Trainer(max_epochs=max_epochs)
        trainer.board.display = False
        trainer.fit(make_model(fold.num_inputs), fold)
        rmse.append([np.mean(trainer.history[f'{split}_loss'][-5:]) ** 0.5
                     for split in ('train', 'val')])  # Last five epochs
    train, val = np.array(rmse).T
    print(f'training RMSE {train.mean():.3f}, validation RMSE '
          f'{val.mean():.3f} (folds {val.min():.3f} to {val.max():.3f})')
    return train, val
```

We now generate our shared 5-fold partition `folds` once so every candidate model below is evaluated on the exact same five training and validation splits:

```{.python .input #kaggle-house-price-k-fold-cross-validation-3}
folds = k_fold_data(raw_train, 5)
```

First, let us define `make_linear` for our `d2l.LinearRegression` baseline ($100$ epochs, batch size $64$, learning rate $\eta = 0.02$):

```{.python .input #kaggle-house-price-model-selection-linear}
%%tab mxnet, pytorch, tensorflow
make_linear = lambda num_inputs: d2l.LinearRegression(lr=0.02)
```

```{.python .input #kaggle-house-price-model-selection-linear}
%%tab jax
make_linear = lambda num_inputs: d2l.LinearRegression(num_inputs, lr=0.02)
```

Running `k_fold(folds, make_linear)` trains the unregularized linear baseline across all five folds and records its training and validation log-RMSE vectors in `linear`:

```{.python .input #kaggle-house-price-a-linear-baseline-1}
linear = k_fold(folds, make_linear)
```

The linear baseline achieves a 5-fold validation log-RMSE of $\sim 0.15$ ($\sim 15\%$ typical price error) vs. $\sim 0.11$ training log-RMSE.

### An MLP with Weight Decay

Now let us build `HouseMLP`: a one-hidden-layer MLP with $64$ ReLU units ($\sim 22{,}000$ parameters, or $\sim 19$ parameters per training house in a fold), He-initialized weights, and $\ell_2$ weight decay `wd` applied exclusively to the weight matrices (excluding biases):

```{.python .input #kaggle-house-price-mlp-model}
%%tab mxnet
class HouseMLP(d2l.Module):
    """An MLP with one hidden layer, squared loss, and weight decay."""
    loss = d2l.LinearRegression.loss

    def __init__(self, num_hiddens, wd, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'), nn.Dense(1))
        self.net.initialize(init.Xavier(rnd_type='gaussian', factor_type='in',
                                        magnitude=2))

    def configure_optimizers(self):
        for p in self.collect_params('.*bias').values():
            p.wd_mult = 0
        return gluon.Trainer(self.collect_params(), 'sgd',
                             {'learning_rate': self.lr, 'wd': self.wd})

make_mlp = lambda num_inputs, wd: HouseMLP(num_hiddens=64, wd=wd, lr=0.02)
```

```{.python .input #kaggle-house-price-mlp-model}
%%tab pytorch
class HouseMLP(d2l.Module):
    """An MLP with one hidden layer, squared loss, and weight decay."""
    loss = d2l.LinearRegression.loss

    def __init__(self, num_hiddens, wd, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(1))

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)

    def configure_optimizers(self):
        weights = [p for p in self.parameters() if p.dim() > 1]
        biases = [p for p in self.parameters() if p.dim() == 1]
        return torch.optim.SGD([{'params': weights, 'weight_decay': self.wd},
                                {'params': biases}], lr=self.lr)

make_mlp = lambda num_inputs, wd: HouseMLP(num_hiddens=64, wd=wd, lr=0.02)
```

```{.python .input #kaggle-house-price-mlp-model}
%%tab tensorflow
class HouseMLP(d2l.Module):
    """An MLP with one hidden layer, squared loss, and weight decay."""
    loss = d2l.LinearRegression.loss

    def __init__(self, num_hiddens, wd, lr):
        super().__init__()
        self.save_hyperparameters()
        # One initializer per layer: an instance repeats its random values
        he = lambda: tf.keras.initializers.VarianceScaling(
            2.0, 'fan_in', 'untruncated_normal')
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation='relu',
                                  kernel_initializer=he()),
            tf.keras.layers.Dense(1, kernel_initializer=he())])

    def configure_optimizers(self):
        optimizer = tf.keras.optimizers.SGD(self.lr, weight_decay=self.wd)
        optimizer.exclude_from_weight_decay(var_names=['bias'])
        return optimizer

make_mlp = lambda num_inputs, wd: HouseMLP(num_hiddens=64, wd=wd, lr=0.02)
```

```{.python .input #kaggle-house-price-mlp-model}
%%tab jax
def is_matrix(params):
    """Mask that selects the parameters with more than one dimension."""
    return jax.tree.map(lambda p: p.ndim > 1, params)

class HouseMLP(d2l.Module):
    """An MLP with one hidden layer, squared loss, and weight decay."""
    loss = d2l.LinearRegression.loss

    def __init__(self, num_inputs, num_hiddens, wd, lr, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        self.net = nnx.Sequential(
            nnx.Linear(num_inputs, num_hiddens, kernel_init=he, rngs=rngs),
            nnx.relu,
            nnx.Linear(num_hiddens, 1, kernel_init=he, rngs=rngs))

    def configure_optimizers(self):
        return optax.chain(
            optax.masked(optax.add_decayed_weights(self.wd), is_matrix),
            optax.sgd(self.lr))

make_mlp = lambda num_inputs, wd: HouseMLP(num_inputs, num_hiddens=64,
                                           wd=wd, lr=0.02)
```

We now sweep `wd` across `wds = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]` on our shared 5-fold splits `folds` and plot the mean training and validation RMSEs against the linear baseline:

```{.python .input #kaggle-house-price-mlp-select}
wds = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
mlp = {}
for wd in wds:
    print(f'weight decay {wd}: ', end='')
    mlp[wd] = k_fold(folds, lambda num_inputs: make_mlp(num_inputs, wd))
d2l.plot(wds, [[mlp[wd][0].mean() for wd in wds],
               [mlp[wd][1].mean() for wd in wds], [linear[1].mean()] * 6],
         'weight decay', 'RMSE', xscale='log', fmts=('C0-o', 'C1-o', 'k:'),
         legend=['MLP, training', 'MLP, validation', 'linear, validation'],
         legend_loc='outside', figsize=(4.5, 3))
```

Selecting the weight decay `best` with the lowest 5-fold mean validation RMSE, we can also inspect the paired per-fold difference `mlp[best][1] - linear[1]` where fold-sampling noise cancels out:

```{.python .input #kaggle-house-price-mlp-paired}
best = min(wds, key=lambda wd: mlp[wd][1].mean())
print(f'weight decay {best}: MLP minus linear, per fold',
      np.round(mlp[best][1] - linear[1], 3))
```

The cross-validation curve traces a textbook U-shape: weak weight decay (`wd = 0.001`) overfits the training folds (training RMSE $\sim 0.09$, validation RMSE $\sim 0.16$, worse than linear regression!), intermediate weight decay (`wd = 0.03` or `0.1`) hits the sweet spot at validation RMSE $\sim 0.135$–$0.14$ (beating the linear baseline on nearly every fold), and excessive decay (`wd = 0.3`) underfits.

## Submitting Predictions
:label:`subsec_kaggle_submission`

With the winning weight decay `best` selected by 5-fold cross-validation, we retrain `HouseMLP` on all $1{,}460$ labeled houses (or ensemble the 5 fold models; see Exercise :exref:`ex_kaggle-house-fold-ensemble`), add back the training mean log-price `data.offset`, exponentiate to dollar predictions, and save `submission.csv`:

```{.python .input #kaggle-house-price-submitting-predictions-on-kaggle}
data = KaggleHouse(raw_train, raw_test)
model = make_mlp(data.num_inputs, wd=best)
trainer = d2l.Trainer(max_epochs=100)
trainer.board.display = False
trainer.fit(model, data)
log_price = d2l.numpy(model(d2l.tensor(data.X_val))).reshape(-1) + data.offset
submission = pd.DataFrame({'Id': raw_test['Id'],
                           'SalePrice': np.exp(log_price)})
submission.to_csv('submission.csv', index=False)
submission.head()
```

## Summary

- **Leakage-free tabular preprocessing:** Recast numeric category codes to strings, standardize numerical features using training-split means and standard deviations only, impute missing standardized values with $0$, one-hot encode categoricals with `dummy_na=True`, and center log targets by the training mean.
- **Cross-validated model selection:** Reusing identical $K = 5$ folds across models cancels fold-sampling noise in paired comparisons; tuning weight decay takes a 22,000-parameter MLP from overfitting ($\sim 0.16$ log-RMSE) past the linear baseline ($\sim 0.15$) down to $\sim 0.135$ log-RMSE (while gradient-boosted trees remain a strong tabular competitor; see Exercise :exref:`ex_kaggle-house-trees`).

## Exercises

1. [code] **Mean imputation.** `preprocess` replaces every missing
   numerical value by the training-set mean of its column. Its call to
   `pd.get_dummies` with `dummy_na=True` also adds a missing-value
   indicator column for every categorical feature.
    1. Identify a numerical feature in this dataset whose missingness is not
       random with respect to the sale price.
    1. Explain what bias mean imputation alone would introduce for that
       feature, and in which direction it would shift the predictions for
       the affected houses. Then determine whether the indicator columns
       that `preprocess` already creates let a linear model remove this
       bias.
    1. Replace mean imputation by median imputation, and separately add a
       binary missingness indicator column for each imputed feature. Compare
       the cross-validated errors of the three variants.

    *Adapted from Kaggle Learn,
    [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning),
    Exercise 2.*
   :exlabel:`ex_kaggle-house-mean-imputation`

1. [code] **Scaling the features and the target.** Remove the
   standardization from `preprocess`. Because `fillna(0)` imputes the
   column mean only for standardized columns, fill missing numerical values
   with the training-set column means explicitly instead.
    1. Rerun the cross-validation of the linear model at the section's
       learning rate of $0.02$, and describe the training loss.
    1. Find, to within a factor of ten, the largest learning rate at which
       the unstandardized linear model trains without diverging. Compare
       its cross-validated error after $100$ epochs with that of the
       standardized pipeline, and explain both observations.
    1. Restore the standardization, but train on the log prices without
       subtracting their mean. Compare the cross-validated error after
       $100$ epochs with that of the centered targets, and explain the
       difference.
    1. With the section's pipeline, record the validation RMSE of the linear
       model and the mean of its residuals on the validation fold after
       every epoch. In the epochs where the RMSE jumps, what has moved?
       Every house has 45 features equal to 1. Explain why a step of SGD
       changes the mean prediction more than it changes any single weight,
       and check whether halving the learning rate removes the jumps.
   :exlabel:`ex_kaggle-house-scaling`
1. [code] **Gradient-boosted trees.** On tabular data like this,
   ensembles of decision trees often predict better than neural networks
   :cite:`Grinsztajn.Oyallon.Varoquaux.2022,Shwartz-Ziv.Armon.2022`. Fit a
   gradient-boosted tree model (for example scikit-learn's
   `GradientBoostingRegressor`, or XGBoost or LightGBM if installed) to the
   centered log prices of the training rows of each fold in `folds`, and
   compute its validation RMSE on each fold. Compare the average with those
   of this section's models. Which properties of the data favor trees?
   :exlabel:`ex_kaggle-house-trees`
1. [code] **Target leakage.** A feature that is recorded only after, or
   because of, the event being predicted inflates validation scores without
   helping a deployed model.
    1. Identify at least one raw feature in this dataset that could
       plausibly be filled in only after the sale, and explain why
       cross-validation does not catch the problem.
    1. Verify the mechanism empirically: add a synthetic feature equal to
       the log sale price plus small Gaussian noise, rerun the
       cross-validation, and report the change in the validation RMSE.

    *Adapted from Kaggle Learn,
    [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning),
    Exercise 7.*
1. [code] **Fold ensembling versus retraining.**
   :numref:`subsec_kaggle_submission` retrains the selected MLP on all
   houses. Alternatively, the five models trained during cross-validation
   could each predict the test houses, with the preprocessing of their own
   round, and their log-price predictions could be averaged.
    1. Implement this fold ensemble and submit both its predictions and
       those of the retrained model. Compare the two leaderboard scores with
       each other and with the cross-validated RMSE.
    1. Each model of the ensemble was trained on four fifths of the houses.
       Why might the ensemble nevertheless predict better than the model
       retrained on all of them?
    1. Explain why the training error of the retrained model is not a
       substitute for the cross-validated estimate.
   :exlabel:`ex_kaggle-house-fold-ensemble`
1. [code] **Target encoding.** ● *Target encoding* replaces each category
   of a categorical column by the mean log sale price of the training rows
   in that category, so that the feature occupies a single column instead
   of the one column per category that `preprocess` creates.
    1. Target-encode every categorical column with more than ten categories
       instead of one-hot encoding it, computing the category means from
       the training rows of each fold. Give a category that does not occur
       in those rows the overall mean, and standardize the encoded columns
       like the numerical features. Compare the cross-validated error with
       that of one-hot encoding.
    1. The encoding above computes each training row's value from a mean
       that includes that row's own label. Encode the training rows out of
       fold instead: split the training rows of each fold into five parts,
       and encode each part with category means computed from the other four
       parts. Compare the cross-validated error with that of the in-sample
       encoding, and explain how the size of a category affects the
       difference between the two encodings.

1. **Geometric averaging.** ● An ensemble of $K$ models that predict the log
   prices $z_1, \ldots, z_K$ of a house can average them before
   exponentiating, which amounts to a geometric mean in price space.
    1. Compare $\exp\bigl(\frac{1}{K}\sum_{k=1}^K z_k\bigr)$ with the
       price-space average $\frac{1}{K}\sum_{k=1}^K e^{z_k}$. Show which of
       the two never exceeds the other, and state when they coincide.
    1. Let $\log y$ be the true log price of the house. Show that the
       log-space average satisfies

       $$\Bigl(\log y - \frac{1}{K}\sum_{k=1}^K z_k\Bigr)^2 \leq \frac{1}{K}\sum_{k=1}^K \left(\log y - z_k\right)^2.$$

       Determine whether the logarithm of the price-space average satisfies
       the same bound for all predictions $z_1, \ldots, z_K$.

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

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.7]{.kicker}

Predicting House Prices on Kaggle: End-to-End Tabular ML<br>**log-RMSE metric · leakage-free preprocessing · shared 5-fold cross-validation · weight-decayed MLP · submission**
:::
:::

::: {.slide title="From Clean Image Grids to Messy Tabular Data"}
[The Capstone Challenge]{.kicker}

In this capstone section for Chapters 1–4, we tackle the Kaggle Ames Housing competition (De Cock, 2011)—a realistic tabular dataset where data hygiene and validation protocol matter as much as model architecture:

| Dataset Property | Fashion-MNIST (§3.2–§4.6) | Ames House Prices (§4.7) |
|:--|:--|:--|
| **Labeled training examples** | $60{,}000$ images | $1{,}460$ houses ($+1{,}459$ unlabeled test houses) |
| **Raw input attributes** | $784$ uniform `uint8` pixels in $[0, 1]$ | $79$ mixed columns ($35$ numeric, $44$ categorical) |
| **Feature scales & missingness** | Identical scale; $0$ missing values | Scales span $10^0$ to $10^5$; `NaN` gaps in $19$ columns |
| **Target & evaluation metric** | $10$ balanced classes; accuracy | Sale price $\$34.9\textrm{k}$–$\$755\textrm{k}$ ($22\times$ range); **log-RMSE** |

::: {.d2l-note}
With only $1{,}460$ labeled homes and $345$ one-hot expanded features, an unregularized MLP has $\sim 19$ parameters per training house—making leakage-free preprocessing, $K$-fold cross-validation, and weight decay essential.
:::
:::

::: {.slide title="Inspecting the Raw Ames Housing Table"}
[Raw Data Anatomy]{.kicker}

Loading the CSV tables via `read_house_prices()` reveals $1{,}460 \times 81$ training rows (`Id`, $79$ features, `SalePrice`) and $1{,}459 \times 80$ test rows:

@!kaggle-house-price-data-preprocessing-1

- **Numeric vs. disguised categorical codes:** `LotFrontage` is a physical length in feet (with $259$ `NaN` rows), whereas `MSSubClass` uses integers ($20, 60, 90, \ldots$) to encode dwelling style (1-story, 2-story, duplex) and must be cast to `str`.
- **Informative missingness:** In many categorical attributes (such as `PoolQC` or `Alley`), `NaN` is not a recording error—it encodes the physical absence of a pool or alley!
- **Row identifier trap:** Column `Id` ($1, 2, \ldots, 1460$) carries zero predictive signal and must be dropped before training so the model cannot memorize row indices.
:::

::: {.slide title="Why Kaggle Scores Log-RMSE Instead of Dollar RMSE"}
[The Competition Metric]{.kicker}

A $\$50{,}000$ prediction error is catastrophic on a $\$60{,}000$ starter home ($83\%$ error) but minor on a $\$650{,}000$ estate ($8\%$ error). Kaggle therefore evaluates **relative** error via the root-mean-squared error of log prices:

$$\textrm{log-RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n \bigl(\log y_i - \log \hat{y}_i\bigr)^2}$$

- **Percentage error interpretation:** Since $\log \hat{y} - \log y = \log(\hat{y}/y) \approx (\hat{y} - y)/y$ for small relative errors $\delta$, a log-RMSE of $0.135$ means predictions typically lie within $e^{\pm 0.135} \approx \pm 14\%$ of the true sale price across all price brackets.
- **Direct loss alignment:** Training our network to predict $z_i = \log y_i$ under mean squared error $\frac{1}{n}\sum_i (z_i - \hat{z}_i)^2$ optimizes the exact square of the competition metric!
:::

::: {.slide title="Step 1 & 2: Standardization and Mean Imputation"}
[Leakage-Free Preprocessing]{.kicker}

Raw numerical columns span five orders of magnitude (lot area in $10^4\textrm{ sq ft}$ vs. bedroom count in $[0, 8]$), which would skew both gradient descent step sizes and $\ell_2$ weight decay penalties:

$$x_{ij} \leftarrow \frac{x_{ij} - \hat{\mu}_j}{\hat{s}_j}, \qquad \hat{\mu}_j, \hat{s}_j \textrm{ computed on the training split ONLY}$$

- **Unit-variance scaling:** Standardizing every numerical feature to zero mean and unit variance puts all weights $w_j$ on an equal footing under $\frac{\lambda}{2}\|\mathbf{W}\|_{\textrm{F}}^2$.
- **Zero imputation equals mean imputation:** After subtracting $\hat{\mu}_j$ and dividing by $\hat{s}_j$, replacing missing entries (`fillna(0)`) sets every missing numerical measurement to $0$—the exact training-set mean!
:::

::: {.slide title="Step 3 & 4: One-Hot Encoding With a Missingness Column"}
[Leakage-Free Preprocessing]{.kicker}

For the $44$ string/categorical attributes (including `MSSubClass`), `pd.get_dummies(df, dummy_na=True)` creates binary indicator columns for every observed category plus an explicit `_nan` column:

| House Row | Raw `MSZoning` | `MSZoning_RL` | `MSZoning_RM` | `MSZoning_FV` | `MSZoning_RH` | `MSZoning_C (all)` | `MSZoning_nan` |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| $1$ | `RL` | $1$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| $2$ | `RM` | $0$ | $1$ | $0$ | $0$ | $0$ | $0$ |
| $3$ | `NaN` | $0$ | $0$ | $0$ | $0$ | $0$ | $1$ |

- **Preserving informative `NaN`s:** Giving missing categories their own `_nan` indicator lets the model learn a distinct price offset for homes without a pool (`PoolQC_nan = 1`) or without a garage.
- **Vocabulary alignment (`reindex`):** Any test-set category never seen in `train` is dropped, and any `train` category absent from `test` is filled with $0$ (`reindex(columns=train_cols, fill_value=0)`).
:::

::: {.slide title="The Complete Leakage-Free Preprocessing Pipeline"}
[Preprocessing Code]{.kicker}

All four steps fit exclusively on `train` and transform `*others` (`val` or `test`) identically, expanding $79$ raw attributes into $345$ `float32` features:

@kaggle-house-price-data-preprocessing-2

@!kaggle-house-price-data-preprocessing-3
:::

::: {.slide title="Target Centering in KaggleHouse"}
[Target Normalization]{.kicker}

The log sale prices $\log y_i$ have mean $\approx 12.02$ ($\approx \$166{,}000$) and standard deviation $\approx 0.40$. In `KaggleHouse`, we subtract the training split's mean log price (`self.offset`) from the targets:

- **Why center $\log y$?** An initialized network starts with outputs near $0$. If targets sit around $12.02$, the initial squared error is $12^2 = 144$, and early SGD steps waste epochs shifting the output bias before learning feature weights.
- **Zero-mean targets ($y_i - \bar{y}_{\textrm{train}}$):** A network initialized near $0$ starts with RMSE $\approx 0.40$ from step $0$; at prediction time, we simply add `self.offset` back before exponentiating (`np.exp(pred + data.offset)`).
:::

::: {.slide title="Shared 5-Fold Cross-Validation Without Leakage"}
[Model Selection Protocol]{.kicker}

With only $1{,}460$ labeled houses, a single $20\%$ validation split ($292$ houses) has a sampling standard error of $\sim 0.015$ in log-RMSE—larger than the gap between competing models!

::: {.cols .vc}
::: {.col}
- **5-fold partition (`k_fold_data`):** Shuffles and splits `raw_train` once into $K = 5$ folds of $292$ houses each.
- **Fold-local preprocessing:** Each fold instantiates its own `KaggleHouse(train_4_folds, val_1_fold)`, refitting $\hat{\mu}, \hat{s}$, dummy columns, and `offset` on its $1{,}168$ training homes only.
- **Shared folds for paired comparison:** Evaluating every candidate model on the *exact same 5 folds* cancels fold-sampling noise when taking pairwise differences.
:::

::: {.col .fig .big}
![](../img/mdl-mlp-kfold.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Benchmarking the Linear Regression Baseline"}
[Baseline Model]{.kicker}

Before tuning an MLP, we benchmark `d2l.LinearRegression` across our $5$ folds ($100$ epochs, batch size $64$, learning rate $\eta = 0.02$, averaging the final $5$ epochs to smooth single-epoch SGD noise):

@!kaggle-house-price-a-linear-baseline-1

- **Baseline performance:** Achieves $\sim 0.11$ training log-RMSE and **$\sim 0.15$ validation log-RMSE** ($\sim 15\%$ typical price error), with individual folds ranging from $\sim 0.12$ to $\sim 0.19$ (due to a few outlier mansions in one fold).
:::

::: {.slide title="Building HouseMLP: 64 ReLU Units With Weight Decay"}
[Regularized MLP Architecture]{.kicker}

Next we construct `HouseMLP`: a one-hidden-layer ReLU network ($345 \to 64 \to 1$, $\sim 22{,}200$ parameters, or $19$ parameters per training house) with He initialization and $\ell_2$ weight decay applied strictly to the weight matrices $\mathbf{W}^{(1)}, \mathbf{W}^{(2)}$ (excluding biases):

@kaggle-house-price-mlp-model
:::

::: {.slide title="Cross-Validating Weight Decay: The U-Curve in Action"}
[Hyperparameter Sweep]{.kicker}

Sweeping weight decay `wd` across $\{0.001, 0.003, 0.01, 0.03, 0.1, 0.3\}$ on our $5$ shared folds traces a textbook generalization U-curve:

::: {.cols .vc}
::: {.col}
- **Under-regularized (`wd = 0.001`):** Overfits the $1{,}168$ training homes (train RMSE $\sim 0.09$), pushing validation RMSE to $\sim 0.16$—worse than linear regression!
- **Sweet spot (`wd = 0.03` or `0.1`):** Constrains weight norms so validation RMSE drops to **$\sim 0.135$–$0.140$**, beating the linear baseline.
- **Over-regularized (`wd = 0.3`):** Shrinks weights too aggressively, underfitting both training and validation splits ($\sim 0.17$).
:::

::: {.col .fig .big}
@!kaggle-house-price-mlp-select
:::
:::
:::

::: {.slide title="Paired Fold-by-Fold Comparison vs. Linear Baseline"}
[Statistical Verification]{.kicker}

Because both models were evaluated on the exact same $5$ folds, we can inspect the paired difference $\textrm{RMSE}_{\textrm{MLP}} - \textrm{RMSE}_{\textrm{linear}}$ on each fold:

@!kaggle-house-price-mlp-paired

- **Variance reduction via pairing:** Even though fold-to-fold validation RMSE varies by $\pm 0.04$ across splits (depending on which fold holds out unusual luxury sales), the paired difference is negative (or tied within $0.005$) on every fold!
- **Tabular reality check:** While our regularized 64-unit MLP beats linear regression by $\sim 0.01$–$0.015$ log-RMSE, on heterogeneous tabular data gradient-boosted decision trees (XGBoost / LightGBM) remain fierce competitors (Exercise 4.7.3).
:::

::: {.slide title="Retraining on All 1,460 Houses & Writing submission.csv"}
[Final Submission Pipeline]{.kicker}

Once 5-fold cross-validation selects the best weight decay `best`, we fit a fresh `KaggleHouse(raw_train, raw_test)` preprocessor and retrain `HouseMLP` on **all $1{,}460$ labeled houses** for $100$ epochs:

@!kaggle-house-price-submitting-predictions-on-kaggle

- **Un-centering and exponentiating:** Adding `data.offset` back to the predicted centered log prices and applying `np.exp(...)` converts log predictions back into dollars for `submission.csv`.
:::

::: {.slide title="Concept Check: Leakage, Log-RMSE, and Selection Bias"}
[Check]{.kicker}

Test your mastery of end-to-end tabular modeling:

1. **Relative nature of log-RMSE:** Home A sells for $\$100{,}000$ and is predicted at $\$125{,}000$; Home B sells for $\$400{,}000$ and is predicted at $\$500{,}000$. Compare their dollar squared errors and their log-price squared errors.
2. **Subtle data leakage bug:** Suppose someone runs `preprocess(raw_train)` once on all $1{,}460$ rows *before* splitting into $5$ cross-validation folds. What two statistics leak from the validation fold into training?
3. **Winner's curse in model selection:** When we pick the lowest validation RMSE among $M$ hyperparameter candidates on our 5 folds, is that winning score an unbiased estimate of test-set RMSE?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Equal log error:** Dollar squared errors differ $16\times$ ($\$25\textrm{k}^2$ vs. $\$100\textrm{k}^2$), but both have ratio $\hat{y}/y = 1.25$, giving the exact same log squared error $(\log 1.25)^2 \approx 0.223^2 \approx 0.050$.
2. **Leaked statistics:** The validation fold's feature means/standard deviations ($\hat{\mu}_j, \hat{s}_j$ used for scaling and mean imputation) and category vocabulary leak into every training fold.
3. **Downward selection bias:** No—$\mathbb{E}[\min_{m=1\dots M} \hat{R}_m] \le \min_m \mathbb{E}[\hat{R}_m]$. With $M = 6$ candidates on $5$ folds the optimistic bias is tiny, but across hundreds of hyperparameter trials a separate held-out test split is required.
:::
:::

::: {.slide title="Summary: End-to-End Tabular Pipeline on Kaggle"}
[Key Takeaways]{.kicker}

| Pipeline Stage | Implementation Choice | Mathematical / Statistical Rationale | Pitfall Prevented |
|:--|:--|:--|:--|
| **Metric & Target** | Predict $z_i = \log y_i - \bar{z}_{\textrm{train}}$ under MSE | $\sqrt{\textrm{MSE}(\log y, \log \hat{y})} = \textrm{log-RMSE}$ ($\approx \%$ error) | Dollar MSE over-weights mansions; uncentered $\log y \approx 12$ slows epoch 1 |
| **Type Hygiene** | Drop `Id`; cast `MSSubClass` to `str` | Row IDs are arbitrary; numeric codes are nominal categories | Prevents memorizing row order or treating duplex code $90 = 4.5 \times 20$ |
| **Numeric Scaling** | $x_{ij} \leftarrow (x_{ij} - \hat{\mu}_{\textrm{train}}) / \hat{s}_{\textrm{train}}$, then `fillna(0)` | Equalizes feature scales for SGD & $\ell_2$ penalty; $0 =$ train mean | Computing $\hat{\mu}, \hat{s}$ on full table leaks validation/test distribution |
| **Categoricals** | `get_dummies(dummy_na=True)` + `reindex` | Binary indicator per category + explicit `_nan` column ($345$ cols) | Preserves informative missingness (e.g., `PoolQC = NaN` means no pool) |
| **Model Selection** | 5-fold CV on shared folds (`k_fold`) | Averages $5$ validation splits ($292$ homes each); paired differences | Single $292$-home split has $\pm 0.015$ noise; unregularized MLP overfits ($0.16$) |
| **Submission** | Retrain with `wd = best` on all $1{,}460$ homes | $\hat{y} = \exp(\hat{z} + \bar{z}_{\textrm{train}})$ restores dollar scale | Training error of retrained model is not a valid generalization estimate |
:::

