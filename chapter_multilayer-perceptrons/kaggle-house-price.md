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

We download and cache the training and test CSV files via `d2l.download`, load them into `pandas` DataFrames (:numref:`sec_pandas`), and inspect the shapes and a slice of the first four rows:

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

```{.python .input #kaggle-house-price-accessing-and-reading-the-dataset-2  n=31}
raw_train, raw_test = read_house_prices()
print(raw_train.shape, raw_test.shape)
```

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
4. **One-hot encode categoricals (`pd.get_dummies(..., dummy_na=True)`):** Expands each categorical column into binary indicators (including an explicit indicator for `NaN`) and aligns validation/test columns to the exact training column vocabulary via `reindex(..., fill_value=0)`, yielding $345$ features on the full training table.

Finally, `KaggleHouse(d2l.DataModule)` centers the target $\log(\textrm{SalePrice})$ by subtracting the training split's mean log price `self.offset` ($\approx 12.02$) so the network starts at the right baseline from step 1 (Exercise :exref:`ex_kaggle-house-scaling`):

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

```{.python .input #kaggle-house-price-data-preprocessing-3  n=33}
X_train, X_test = preprocess(raw_train, raw_test)
X_train.shape, X_test.shape
```

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

With only $1{,}460$ labeled houses, a single $20\%$ validation split ($292$ houses) has a fold-to-fold standard error of several hundredths in log-RMSE. We instead partition `raw_train` once into $K = 5$ random folds via `k_fold_data`—fitting a fresh `KaggleHouse` preprocessor inside each round so validation folds never leak into feature statistics—and evaluate every candidate model across the *exact same five folds* (averaging the last 5 epochs of loss to damp single-step jitter):

```{.python .input #kaggle-house-price-k-fold-cross-validation-1}
def k_fold_data(raw, k):
    """Split the table raw into k random folds; one data module per round."""
    folds = np.array_split(np.random.default_rng().permutation(len(raw)), k)
    return [KaggleHouse(raw.drop(index=raw.index[f]), raw.iloc[f])
            for f in folds]
```

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

```{.python .input #kaggle-house-price-k-fold-cross-validation-3}
folds = k_fold_data(raw_train, 5)
```

First, let us benchmark our `d2l.LinearRegression` baseline across the five folds ($100$ epochs, batch size $64$, learning rate $\eta = 0.02$):

```{.python .input #kaggle-house-price-model-selection-linear}
%%tab mxnet, pytorch, tensorflow
make_linear = lambda num_inputs: d2l.LinearRegression(lr=0.02)
```

```{.python .input #kaggle-house-price-model-selection-linear}
%%tab jax
make_linear = lambda num_inputs: d2l.LinearRegression(num_inputs, lr=0.02)
```

```{.python .input #kaggle-house-price-a-linear-baseline-1}
linear = k_fold(folds, make_linear)
```

The linear baseline achieves a 5-fold validation log-RMSE of $\sim 0.15$ ($\sim 15\%$ typical price error) vs. $\sim 0.11$ training log-RMSE.

### An MLP with Weight Decay

Now let us build `HouseMLP`: a one-hidden-layer MLP with $64$ ReLU units ($\sim 22{,}000$ parameters, or $\sim 19$ parameters per training house in a fold), He-initialized weights, and $\ell_2$ weight decay applied to weight matrices (`wd`). We sweep `wd` across $\{0.001, 0.003, 0.01, 0.03, 0.1, 0.3\}$ on our shared 5-fold splits:

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

Predicting House Prices on Kaggle<br>**tabular data · preprocessing · log RMSE · cross-validated model selection**
:::
:::

::: {.slide title="1,460 Houses, 79 Mixed Attributes, Prices from $34,900 to $755,000"}
[The problem]{.kicker}

| | Fashion-MNIST | Ames house prices |
|---|---|---|
| examples with labels | 60,000 | 1,460 |
| inputs | 784 pixels in $[0, 1]$ | 35 numbers, 44 categories |
| missing values | none | in 19 attributes |
| label | one of 10 classes | a price, range 22× |

- A model cannot read the table as it is.
- With 1,460 houses, a careless error estimate is off by more than the
  differences between models.
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- turn a table of numbers and categories into features, without leaking
  validation data into the preprocessing;
- choose a loss that matches the competition's metric, the RMSE of log
  prices;
- estimate the error of a model by 5-fold cross-validation, on folds shared
  by all models;
- select the weight decay of an MLP by cross-validation;
- retrain the selected model and write a submission.

**Running example:** the validation RMSE of log prices; linear regression is
the baseline every MLP is compared with.
:::

::: {.slide title="The Chapter Ends with a Real Dataset"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Multilayer Perceptrons | hidden layers, universal approximation, activation functions |
| Implementation | an MLP trained on Fashion-MNIST |
| Backpropagation | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | vanishing gradients, Xavier and He initialization |
| Generalization in Deep Learning | memorization, double descent, implicit bias, early stopping |
| Dropout | random masks, slower memorization of noisy labels |
| **Predicting House Prices on Kaggle** | **the workflow on real data** |
:::

::: {.col .narrow}
- **Uses:** squared loss and SGD, standardization, $K$-fold
  cross-validation, weight decay, He initialization.
- **Skips:** feature engineering, tree ensembles (an exercise), stacking.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The Competition]{.dtitle}

[hidden labels, the data, and the score]{.dsub}
:::
:::

::: {.slide title="Kaggle Scores Predictions against Hidden Labels"}
[Competition · Kaggle]{.kicker}

- Training table: 1,460 houses with prices. Test table: 1,459 houses
  without.
- Upload one predicted price per test house; Kaggle scores it against
  prices it keeps hidden and ranks it on a leaderboard.
- No one can tune a model on the test labels.

::: {.d2l-note}
Data: sales in Ames, Iowa, 2006 to 2010, collected by De Cock (2011) as a
larger replacement for teaching datasets such as Boston Housing.
:::
:::

::: {.slide title="Numbers, Categories, and Gaps in One Table"}
[Competition · the data]{.kicker}

@!kaggle-house-price-data-preprocessing-1

- `MSZoning`, `SaleType`: categories. `LotFrontage`: feet, missing for 259
  houses.
- `MSSubClass`: a dwelling-type code stored as a number (20 to 190);
  we encode it as a category.
- Most gaps mean "absent": `PoolQC` is missing for the 1,453 houses
  without a pool.
- `Id` numbers the rows and is not a feature.
:::

::: {.slide title="Score Relative Errors: the RMSE of Log Prices"}
[Competition · the metric]{.kicker}

$$\textrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i - \log \hat{y}_i\right)^2}$$

- $|\log\hat{y} - \log y| \leq \delta$ means
  $e^{-\delta} \leq \hat{y}/y \leq e^{\delta}$: within about $\delta$ of the
  price.
- RMSE 0.1 ≈ typical error of 10%, for a \$35,000 and a \$750,000 house
  alike.

::: {.d2l-note}
Train on $\log y$ with the squared loss: SGD minimizes the square of the
score.
:::
:::

::: {.slide title="Check: How Much Does One Bad Prediction Cost?"}
[Check]{.kicker}

A house sold for \$200,000 is predicted at \$250,000. What is its squared
error in log prices? And for a \$100,000 house predicted at \$125,000?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Both are $(\log 1.25)^2 \approx 0.223^2 \approx 0.05$. The
metric sees only the ratio $\hat{y}/y$, not the dollars.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Preprocessing]{.dtitle}

[from a table to a matrix, without leakage]{.dsub}
:::
:::

::: {.slide title="Standardize Numbers; a Missing Number Becomes the Mean, 0"}
[Preprocessing · numbers]{.kicker}

$$x \leftarrow \frac{x - \mu}{s}, \qquad \mu, s \textrm{ from the training rows}$$

- Lot areas in the ten thousands, fireplace counts in single digits: one
  learning rate and one weight decay cannot suit both scales.
- After standardization, mean imputation fills a gap with 0.

::: {.d2l-note .warn}
Mean imputation assumes a gap is typical. For "year the garage was built",
a gap means "no garage".
:::
:::

::: {.slide title="One-Hot Encode Categories, with a Column for Missing"}
[Preprocessing · categories]{.kicker}

`MSZoning` takes 5 values in the training rows:

| house | `MSZoning` | `_RL` | `_RM` | `_FV` | `_RH` | `_C (all)` | `_nan` |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | RL | 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | RM | 0 | 1 | 0 | 0 | 0 | 0 |

- A category absent from the training rows gets no column: all its
  indicators are 0.
:::

::: {.slide title="Fit Every Statistic on the Training Rows Only"}
[Preprocessing · leakage]{.kicker}

```{.python #kaggle-house-price-exercises}
def preprocess(train, *others):
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

::: {.d2l-note .warn}
Means, standard deviations, and the set of categories come from `train`;
statistics that include validation rows make the error estimate optimistic.
:::
:::

::: {.slide title="79 Attributes Become 345 Features"}
[Preprocessing · result]{.kicker}

@!kaggle-house-price-data-preprocessing-3

- Log prices: mean 12.0, standard deviation 0.4. We subtract the mean of
  the training rows, so that a model starting near 0 is right on average.
- `KaggleHouse` holds one split: preprocessing and mean log price fitted on
  its training rows.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Model Selection]{.dtitle}

[five folds, one baseline, one sweep]{.dsub}
:::
:::

::: {.slide title="Five Folds: Every House Is Validated Once"}
[Model selection · cross-validation]{.kicker}

::: {.cols .vc}
::: {.col}
- One validation set of a fifth: 292 houses, a noisy estimate.
- 5 folds: train on 4, validate on 1, five times; average.
- Each round refits the preprocessing on its own training folds.
- The same folds for every model.
:::

::: {.col .fig .big}
![Each round holds out one fold for validation (orange) and trains on the other four (blue).](../img/mdl-mlp-kfold.svg){width=100%}
:::
:::
:::

::: {.slide title="Linear Regression: Validation RMSE About 0.15"}
[Model selection · baseline]{.kicker}

Minibatches of 64, learning rate 0.02, 100 epochs, the same for every model:

@!kaggle-house-price-a-linear-baseline-1

- Training RMSE about 0.11; the best fold 0.11 to 0.13, the worst about 0.2.
- The five-fold mean moves by about a hundredth between random splits.
- Each RMSE averages the last five epochs: a single epoch can land far off.
:::

::: {.slide title="An MLP with 64 Units: About 22,000 Parameters for 1,168 Houses"}
[Model selection · the MLP]{.kicker}

- One hidden layer of 64 ReLUs, one output; He initialization, zero biases.
- Squared loss reused from `LinearRegression`.
- Weight decay `wd` on the weights, not on the biases.

::: {.d2l-note}
Nineteen parameters per training house: expect it to fit their noise
unless something restrains it.
:::
:::

::: {.slide title="A Moderate Weight Decay Gives the Lowest Validation RMSE"}
[Model selection · the sweep]{.kicker}

@!kaggle-house-price-mlp-select

::: {.d2l-note}
Training RMSE rises from a weight decay of 0.03 on; validation RMSE falls,
then rises.
:::
:::

::: {.slide title="The Sweep Traces the U-Curve of Generalization"}
[Model selection · reading]{.kicker}

- **0.001, hardly any decay:** fits training folds better than the linear
  model (0.09 vs 0.11), validates worse (about 0.16 vs about 0.15): it fits
  noise.
- **0.03:** about 0.135, a typical error of about 14% of the price.
- **0.3:** about 0.17; it fits even the training folds worse than the
  linear model: it underfits.

::: {.d2l-note}
On the same folds, the selected MLP is ahead of the linear model by about a
hundredth on average.
:::
:::

::: {.slide title="Fold by Fold, the MLP's Advantage Holds Up, and It Is Small"}
[Model selection · paired comparison]{.kicker}

The folds are shared, so the two models can be compared on each fold:
validation RMSE of the best MLP minus that of the linear model.

@!kaggle-house-price-mlp-paired

- Ahead on most folds, by up to about four hundredths; on the others the
  two models are within half a hundredth.
- Shared folds cancel much of the variation between splits: the difference
  is measured more precisely than either error.

::: {.d2l-note}
For all its 22,000 parameters, the MLP finds little in this table that a
weighted sum had missed.
:::
:::

::: {.slide title="Check: Is the Lowest RMSE an Unbiased Estimate of the Score?"}
[Check]{.kicker}

We picked the weight decay with the lowest of six cross-validated RMSEs,
computed on the same folds. Is the selected RMSE an unbiased estimate of the
score on the test houses?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No. The minimum of six noisy estimates is biased downward, and
the selection used the folds that measure the error. With six candidates
the bias is small; with hundreds it is not.
:::
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Submission]{.dtitle}

[retrain, predict, upload]{.dsub}
:::
:::

::: {.slide title="Retrain on All 1,460 Houses, Then Predict"}
[Submission · code]{.kicker}

The selected MLP is trained once more, with the same hyperparameters, on all
houses. The first lines of `submission.csv`:

@!kaggle-house-price-submitting-predictions-on-kaggle

- The test table takes the place of the validation rows, so preprocessing and
  mean log price are fitted on all 1,460 houses.
- The model predicts centered log prices: add the mean back, exponentiate.
:::

::: {.slide title="Upload the File; the Score Should Be Near the Cross-Validated RMSE"}
[Submission · Kaggle]{.kicker}

- Competition page → "Submit Prediction" → upload `submission.csv`.
- The cross-validated RMSE estimates the score; the retrained model and the
  selection make it an estimate, not a guarantee.
- Alternative: average the log-price predictions of the five fold models.

::: {.d2l-note}
On tables like this, gradient-boosted trees often beat neural networks;
the pipeline stays the same.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Preprocess:** standardize, impute 0, one-hot with a missing column;
  fit on the training rows only.
- **Loss:** squared error of centered log prices = square of the
  competition's RMSE.
- **Estimate:** 5-fold cross-validation on shared folds.
:::

::: {.col}
- **Linear:** validation RMSE about 0.15.
- **MLP, 64 units:** about 0.16 with weight decay 0.001, about 0.135 with
  0.03; ahead of the linear model on most folds, by little.
- **Submit:** retrain the selected model on all houses.
:::
:::

**Next:** Computation: modules, parameters, and devices, the software
structure behind the models so far.
:::
