```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Data Preprocessing
:label:`sec_pandas`

So far, we have been working with synthetic data that arrived in ready-made tensors of whatever shape and data type we asked for. To apply deep learning in the wild, however, we must extract messy data stored in arbitrary file formats and preprocess it into clean numerical tensors. Turning a raw table of records into model-ready tensors involves more statistical judgment than the humble word *preprocessing* suggests.

Consider our running example for this section: a small table of ten residential home sales, each recording the number of rooms (`NumRooms`), the roof material (`RoofType`), the floor area in square feet (`Area`), and the sale price in dollars (`Price`). Small as it is, this table exhibits the three headaches that make real-world tabular data awkward to feed directly into a neural network:

1. **Missing values:** three of the ten homes are missing a room count, and four are missing a roof type.
2. **Categorical text fields:** the roof type is recorded as a word (`Slate` or `Tile`), whereas neural networks compute exclusively with numbers.
3. **Incommensurate scales:** floor area is measured in thousands of square feet, making its numerical values roughly $500$ times larger than the number of rooms.

How we resolve each of these three issues directly shapes what a downstream model can learn. Simply discarding every incomplete row would throw away more than half of our dataset. Encoding `Slate`, `Tile`, and missing roof entries as the integers `0`, `1`, and `2` would fabricate a numerical ordering among unordered categories. And leaving floor area unscaled would allow square footage to dominate any distance calculation or gradient step, rendering the room count almost invisible.

Fortunately, the [pandas](https://pandas.pydata.org/) library :cite:`McKinney.2010` is the standard Python workhorse for loading, inspecting, cleaning, and transforming tabular data. While no substitute for a complete pandas tutorial, this section walks through the core preprocessing pipeline shown in :numref:`fig_pandas_pipeline`, taking a raw CSV file on disk all the way to `float32` feature and target tensors.

![From a file to tensors. `read_csv` loads the text file into a DataFrame with gaps and text columns; imputation, encoding, and scaling make every entry a number; `to_numpy` hands the result to the framework as the tensors `X` and `y`.](../img/pandas-pipeline.svg)
:label:`fig_pandas_pipeline`

We begin by importing `pandas` under its customary alias `pd`, Python's `os` module for file-path handling, and our deep learning framework (needed at the final step when we convert the cleaned table into tensors).

:begin_tab:`mxnet`
As in the previous section, MXNet's array module is imported as `mnp` so that `np` continues to denote standard NumPy.
:end_tab:

```{.python .input #pandas-data-preprocessing}
%%tab mxnet
import os
import pandas as pd
from mxnet import np as mnp, npx
npx.set_np()
```

```{.python .input #pandas-data-preprocessing}
%%tab pytorch
import os
import pandas as pd
import torch
```

```{.python .input #pandas-data-preprocessing}
%%tab tensorflow
import os
import pandas as pd
import tensorflow as tf
```

```{.python .input #pandas-data-preprocessing}
%%tab jax
import os
import pandas as pd
from jax import numpy as jnp
```

## Loading and Inspecting

### CSV Files

*Comma-separated values* (CSV) files are ubiquitous for storing tabular, spreadsheet-like data in plain text. Each line in a CSV file represents one record (row), with its first line typically listing the column headers and commas separating consecutive fields within each line---for example, `"Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics"`.

To demonstrate how to load and inspect CSV files with `pandas`, let us write our ten-home dataset to `../data/house_tiny.csv`. Notice that several fields are left blank between consecutive commas (such as `,Tile,2100,290000`), indicating unrecorded values:

```{.python .input #pandas-reading-the-dataset-1}
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Area,Price
3,Slate,1500,210000
,Tile,2100,290000
2,,850,127500
4,Slate,1940,258000
,,1200,168000
3,Tile,1650,225000
5,Slate,2600,375000
2,,900,142000
,Tile,1750,240000
4,,2050,295000''')
```

Calling `pd.read_csv` parses the text file into a pandas *DataFrame*---a two-dimensional table of named, typed columns along with an index labeling the rows ($0$ through $9$ along the left margin):

```{.python .input #pandas-reading-the-dataset-2}
data = pd.read_csv(data_file)
data
```

Notice that `pandas` automatically replaced every empty field (as well as standard tokens like `NA`) with `NaN` (*not a number*), its sentinel for a *missing value*.

### Data Types and Summaries

Before modifying a newly loaded dataset, it is good practice to inspect what `pandas` inferred: how many rows and columns are present, how many non-missing entries each column contains, and what data type was assigned to each column. Calling `data.info()` summarizes all three in a single glance:

```{.python .input #pandas-knowing-your-data-1}
data.info()
```

We see $10$ rows and $4$ columns. While `Area` and `Price` have all $10$ non-null values present, `NumRooms` has only $7$ (so $3$ are missing) and `RoofType` has only $6$ (so $4$ are missing). Look closely at the inferred column data types (`Dtype`): `Area` and `Price` are stored as 64-bit integers (`int64`), yet `NumRooms`---which also counts whole rooms!---was promoted to 64-bit floating point (`float64`). The culprit is the missing data: under the IEEE 754 standard, `NaN` is a special floating-point bit pattern, so standard integer arrays cannot hold it. Meanwhile, `RoofType` has type `str` (or `object` in older pandas releases), indicating a text column that will need to be encoded numerically before any model can ingest it.

For the numerical columns, calling `data.describe()` computes summary statistics across the observed (non-missing) entries: the count, mean, sample standard deviation (`std`), minimum, quartiles ($25\%$, $50\%$ median, $75\%$), and maximum:

```{.python .input #pandas-knowing-your-data-2}
data.describe()
```

These summary statistics immediately highlight the scale mismatch across columns: `NumRooms` ranges from $2$ to $5$ (with a mean of $3.29$), `Area` ranges from $850$ to $2{,}600$ square feet (with a mean of $1{,}654$, roughly $500$ times the mean room count), and `Price` ranges from $\$127{,}500$ to $\$375{,}000$. We will address this scale disparity shortly when we standardize the numerical features.

### Separating Inputs and Targets

In supervised learning, our goal is to predict a designated *target* (or *label*) column---here, `Price`---from the remaining *input* (or *feature*) columns (`NumRooms`, `RoofType`, and `Area`). We separate the input features from the target column right at the start so that our subsequent feature transformations never accidentally alter or incorporate the target price. We can slice rows and columns by integer position using `.iloc` (which follows the exact same $0$-based slicing rules as tensors): `data.iloc[:, :3]` selects all rows and the first three columns, while `data.iloc[:, 3]` selects the fourth column:

```{.python .input #pandas-separating-inputs-and-targets-1}
inputs, targets = data.iloc[:, :3], data.iloc[:, 3]
list(inputs.columns), targets.name
```

While positional slicing with `.iloc` is compact, it is fragile in production pipelines: if an upstream data engineer inserts a new column or reorders the CSV headers, `data.iloc[:, 3]` will silently grab the wrong column without raising an error. Selecting columns explicitly by name---such as `targets = data['Price']` and `inputs = data.drop(columns='Price')`---guards against schema shifts, as you will verify in Exercise 1.

## Cleaning and Encoding

### Missing Values

Missing values are the "bed bugs" of data science---a persistent nuisance that you will confront throughout your career because sensor readings drop out, survey respondents skip questions, or historical records were never digitized. We can count how many entries are missing in each input column by chaining `.isna()` (which returns a Boolean DataFrame marking missing entries as `True`) with `.sum()` (which sums each column, treating `True` as $1$):

```{.python .input #pandas-handling-missing-values-1}
inputs.isna().sum()
```

Practitioners handle missing values using three main strategies, each carrying a distinct statistical trade-off:

* **Deletion** simply discards every row (or column) containing a missing value. While effortless to implement, deletion throws away valuable data---and whenever values are not missing completely at random (for instance, if owners of studio apartments disproportionately omit the room count), deleting incomplete rows biases the surviving sample.
* **Imputation** replaces each `NaN` with an estimated value, such as the column mean or median. Imputation preserves every row in the dataset, though the substituted values are only approximations and artificially reduce the apparent variance of the column.
* **Missingness indicators** augment imputation with an extra binary ($0/1$) column recording *whether* each entry was originally missing. This allows the model to learn whether the very fact that a value went unrecorded carries predictive signal (for example, if missing room counts occur predominantly on luxury estates). The trade-off is adding an extra feature column for every imputed variable.

To see why row deletion is untenable on our ten-home table, let us check how many rows survive `inputs.dropna()`:

```{.python .input #pandas-handling-missing-values-2}
len(inputs.dropna()), len(inputs)
```

Dropping incomplete rows leaves only $4$ of our $10$ homes---discarding $60\%$ of the dataset! We therefore turn to imputation. For numerical features, the most common heuristic is to replace missing entries with the mean of the observed values in that column (or the median, which is more robust when extreme outliers skew the mean). Calling `inputs.fillna(inputs.mean(numeric_only=True))` computes the mean of each numerical column across its observed entries and substitutes that mean wherever `NaN` appears. Because `RoofType` is a text column, it has no numerical mean and its missing entries remain `NaN` until we encode the categorical features in the next step:

```{.python .input #pandas-handling-missing-values-3}
inputs = inputs.fillna(inputs.mean(numeric_only=True))
inputs
```

The three missing room counts in rows $1$, $4$, and $8$ have now been filled with $23/7 \approx 3.285714$, the exact average of the seven observed room counts. Of course, no physical house has $3.29$ rooms; an imputed mean is a statistical placeholder representing an average-sized house with respect to room count.

### Categorical Features

Next, let us turn to the categorical text column `RoofType`. Including missing values as a distinct state, each home falls into one of three unordered categories: `Slate`, `Tile`, or missing (`NaN`). Because neural networks require numerical inputs, a tempting shortcut is to assign arbitrary integer codes such as $\textrm{Slate} = 0$, $\textrm{Tile} = 1$, and $\textrm{missing} = 2$. For unordered categories, however, single-integer coding is a conceptual trap: it forces any linear combination of the features to treat `Tile` as halfway between `Slate` and an unrecorded roof, inventing an algebraic ordering and distance ratio that do not exist in reality.

The standard solution is *one-hot encoding* (also called *dummy variable encoding*), which replaces a categorical column having $k$ distinct categories with $k$ separate binary ($0/1$) indicator columns---one per category. For any given row, the indicator column corresponding to that home's category is set to $1.0$ and the remaining $k - 1$ indicator columns are set to $0.0$, placing all categories on an equal, symmetric footing.

In `pandas`, `pd.get_dummies` performs one-hot encoding automatically across all string/categorical columns while leaving numerical columns untouched. Passing `dummy_na=True` instructs `pandas` to create an explicit indicator column (`RoofType_nan`) for missing entries---applying the missingness-indicator strategy directly to the categorical feature---and passing `dtype=float` stores the indicators as floating-point numbers ($0.0$ and $1.0$) rather than Booleans:

```{.python .input #pandas-encoding-categorical-features-1}
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=float)
inputs
```

As shown above, `RoofType` has been expanded into three binary columns: `RoofType_Slate`, `RoofType_Tile`, and `RoofType_nan`. Notice that in every row, exactly one of those three columns equals $1.0$ and the other two equal $0.0$ (so the three columns sum identically to $1$, making one of them algebraically redundant if a model already includes an intercept bias; similarly, a strictly binary feature with no missing values requires only a single $0/1$ column). Every entry in `inputs` is now a valid floating-point or integer number.

Keep in mind that one-hot encoding is ideal when a categorical feature has a modest number of distinct values (say, a few dozen). When a categorical column has tens of thousands of distinct values---such as postal ZIP codes, user IDs, or words in a natural-language vocabulary---one-hot encoding would create thousands of sparse columns filled almost entirely with zeros. For high-cardinality categories, deep learning models instead map each category ID to a compact, dense vector of learnable parameters called an *embedding*, which we will introduce in :numref:`sec_rnn`.

## Scaling and Converting

### Standardization

Although every column in `inputs` is now numerical, our two continuous features---`NumRooms` and `Area`---still live on vastly different measurement scales. In raw units, a difference of $2$ rooms is numerically smaller than a difference of $3$ square feet! If we compute the Euclidean distance between two homes or apply weight decay equally across parameters, the floor area in square feet will completely overwhelm the room count. Even worse, as we will see when we train linear regression from scratch (:numref:`sec_linear_scratch`) and study curvature in optimization (:numref:`chap_optimization`), features whose scales differ by orders of magnitude create elongated, ill-conditioned loss surfaces that force gradient descent to crawl at a tiny learning rate.

To put all continuous features on an equal footing, we apply *standardization* (also called *z-score normalization*), transforming each entry $x$ of a numerical column according to

$$\frac{x - \mu}{\sigma},$$

where $\mu$ is the column's sample mean and $\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2}$ is its sample standard deviation (we explain Bessel's $n - 1$ denominator correction in :numref:`sec_prob`). After standardization, every transformed numerical column has a mean of $0$ and a standard deviation of $1$, and each entry simply expresses how many standard deviations above or below the column average that observation lies---making the representation completely invariant to whether floor area was originally recorded in square feet, square meters, or acres. Because our one-hot indicator columns already live on $[0, 1]$, we standardize only the continuous columns `['NumRooms', 'Area']`:

```{.python .input #pandas-numerical-features-1}
numerical = ['NumRooms', 'Area']
mu, sigma = inputs[numerical].mean(), inputs[numerical].std()
inputs[numerical] = (inputs[numerical] - mu) / sigma
inputs
```

Look at row $6$, the largest home in the table ($5$ rooms and $2{,}600$ square feet): its standardized values are $1.89$ for `NumRooms` and $1.70$ for `Area`, immediately revealing that the house lies roughly $1.7$ to $1.9$ standard deviations above average on both dimensions. Meanwhile, the three imputed room counts in rows $1$, $4$, and $8$ have become $0.000000$, because mean imputation placed them at the exact column mean $\mu$. Notice, however, a subtle side effect of mean imputation: inserting three artificial observations at $\mu$ contributed zero squared deviation to the numerator of $\sigma$ while increasing the sample count $n$ from $7$ to $10$, shrinking the standard deviation of `NumRooms` from $1.11$ (across the $7$ observed homes) to $0.91$ (across all $10$ rows) and thereby inflating the standardized $z$-scores of the non-missing rooms (from $1.54$ to $1.89$ for the 5-room house).

Before moving on, we must highlight one critical methodological rule. In our ten-row illustration above, we computed the imputation means and the standardization parameters ($\mu$ and $\sigma$) across the entire table at once. In any real machine learning workflow where the dataset is split into a *training set* and a *test set*, **all preprocessing statistics---imputation means, medians, category vocabularies, and standardization $\mu$ and $\sigma$---must be computed exclusively on the training set**, and then applied unchanged to transform the test set. Computing preprocessing statistics across the combined dataset allows information from the held-out test set to *leak* into the training pipeline (a pitfall known as *data leakage*), which can lead to overly optimistic test scores. We enforce this training-only rule rigorously when we tackle a full real-estate benchmark in :numref:`sec_kaggle_house`, and you will quantify the effect of leakage first-hand in Exercise 4.

### Conversion to Tensors

Now that every entry in `inputs` and `targets` is a clean, properly scaled floating-point number, the final step is converting our pandas DataFrame and Series into framework tensors (recall :numref:`sec_ndarray`). Calling `.to_numpy(dtype='float32')` extracts the underlying table values as a contiguous 32-bit floating-point NumPy array (converting from pandas' default 64-bit `float64`), which we then pass directly to our framework's tensor constructor:

```{.python .input #pandas-conversion-to-tensor-format}
%%tab mxnet
X = mnp.array(inputs.to_numpy(dtype='float32'))
y = mnp.array(targets.to_numpy(dtype='float32'))
X.shape, X.dtype, y.shape, y.dtype
```

```{.python .input #pandas-conversion-to-tensor-format}
%%tab pytorch
X = torch.tensor(inputs.to_numpy(dtype='float32'))
y = torch.tensor(targets.to_numpy(dtype='float32'))
X.shape, X.dtype, y.shape, y.dtype
```

```{.python .input #pandas-conversion-to-tensor-format}
%%tab tensorflow
X = tf.constant(inputs.to_numpy(dtype='float32'))
y = tf.constant(targets.to_numpy(dtype='float32'))
X.shape, X.dtype, y.shape, y.dtype
```

```{.python .input #pandas-conversion-to-tensor-format}
%%tab jax
X = jnp.array(inputs.to_numpy(dtype='float32'))
y = jnp.array(targets.to_numpy(dtype='float32'))
X.shape, X.dtype, y.shape, y.dtype
```

Our feature matrix `X` has shape `(10, 5)`---one row for each of the $10$ homes and one column for each of the $5$ processed features ($2$ standardized numerical columns plus $3$ one-hot roof-type indicators)---and our target vector `y` has shape `(10,)`, both stored in `float32`.

While our ten-home table kept the mechanics transparent, data preprocessing in production systems can get considerably more involved. Rather than residing in a single flat CSV file, enterprise data are often distributed across multiple tables in a relational database or data warehouse (for example, customer profiles in one table, product catalogs in a second, and timestamped transaction logs in a third) that must be joined and aggregated before modeling. Furthermore, real-world datasets are frequently plagued by sensor glitches, corrupted records, and non-standard sentinel values (such as recording an unknown latitude as `-9999`), making exploratory data analysis and visualization tools such as [Matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) invaluable before fitting any model. Now that our data are assembled into a clean design matrix $\mathbf{X}$ with one example per row, the next section introduces the linear-algebraic operations that models apply to such matrices.

## Summary

Real-world tabular datasets almost always arrive with missing values, categorical text fields, and numerical columns measured on wildly different scales, whereas neural networks require clean `float32` tensors. Using `pandas`, we load CSV files into a DataFrame with `read_csv`, inspect column types, missing-value counts, and value distributions with `info()` and `describe()`, and separate the target column from the input features right at the outset. Missing values can be handled via deletion, imputation (such as filling numerical gaps with the training-set mean or median), or missingness indicator columns. Unordered categorical columns are converted into binary ($0/1$) indicator columns via one-hot encoding (`get_dummies`) to avoid fabricating a numerical order, and continuous numerical columns are standardized to zero mean and unit variance ($x \mapsto (x - \mu)/\sigma$) using statistics computed solely on the training split. Finally, `.to_numpy(dtype='float32')` bridges the cleaned DataFrame directly to deep learning framework tensors.

## Exercises

1. [code] **Selecting columns by name versus position.** In this section, we split `data` into features and targets by integer position using `data.iloc[:, :3]` and `data.iloc[:, 3]`. Suppose an updated version of `house_tiny.csv` inserts a new column `YearBuilt` between `Area` and `Price`.
    1. What will the positional split `data.iloc[:, :3]` and `data.iloc[:, 3]` place into `inputs` and `targets` on the updated file? Will any subsequent line of code in this section raise an exception, or will the pipeline train a model to predict the wrong target silently?
    1. Rewrite the feature/target split using column names (for example, via `data['Price']` and `data.drop(...)`) so that `targets` always extracts `Price` and `inputs` retains all other columns, regardless of column order or newly added features. Consult the pandas user guide on [indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html) for details.

1. [code] **Comparing imputation strategies.** Impute the missing entries in `NumRooms` under three different schemes: (i) mean imputation, (ii) median imputation, and (iii) mean imputation augmented with a binary `NumRooms_isna` indicator column.
    1. For both mean imputation and median imputation, compute the sample mean and sample standard deviation of the filled `NumRooms` column, compare them against the mean and standard deviation of the $7$ originally observed values, and explain why the standard deviation changes in the direction it does.
    1. Suppose that in a larger real-estate dataset, `NumRooms` goes unreported primarily for luxury mansions. What room count does a model see for those mansions under mean imputation alone? Explain how a linear model (which predicts price as a weighted sum of the input columns plus a bias) can use the extra `NumRooms_isna` indicator column to compensate for that systematic understatement.
    1. State clearly what statistical assumption each of the three treatments (deletion, mean imputation alone, and imputation with an indicator) makes about the mechanism causing values to go missing.

    *Adapted from Kaggle Learn, [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning), Exercise 2.*

1. **High-cardinality categorical features.** One-hot encoding expands a categorical column with $k$ distinct categories into $k$ binary columns (or $k + 1$ with `dummy_na=True`).
    1. Suppose a table of $n = 10^6$ homes contains a `ZipCode` column with $k = 10^4$ distinct postal codes and no missing values. Calculate how many gigabytes the resulting one-hot encoded matrix occupies if stored densely in 64-bit floating point (`float64`), and state what fraction of its entries are nonzero.
    1. Suppose the table also includes a `ListingID` column containing a unique identifier string for every single row. Compare three possible ways of handling `ListingID`: (i) one-hot encoding, (ii) mapping each ID to a learnable embedding vector, and (iii) dropping the column entirely. Under each treatment, what can a model learn from `ListingID` that generalizes to a newly listed house whose `ListingID` never appeared in the training set? Which treatment should you choose?

    *Adapted from Kaggle Learn, [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning), Exercise 3.*

1. [code] **Quantifying data leakage.** In the main text, we imputed missing `NumRooms` entries and standardized `NumRooms` and `Area` using the sample mean and standard deviation computed across all $10$ rows. Now partition `data` into a training set consisting of the first $8$ rows (`data.iloc[:8]`) and a test set consisting of the final $2$ rows (`data.iloc[8:]`).
    1. Re-run the preprocessing pipeline so that the imputation mean and the standardization parameters ($\mu$ and $\sigma$) are computed *exclusively on the $8$ training rows*, and then use those training parameters unchanged to impute and standardize both the training split and the test split. Compare the resulting standardized test rows against the values obtained when $\mu$ and $\sigma$ were computed from all $10$ rows.
    1. Explain precisely what information about the held-out test homes leaked into the training features under the whole-table recipe.
    1. Suppose a categorical or numerical feature happens to take a constant value across all training rows (so its training standard deviation is $\sigma = 0$), but varies on the test rows. What does the training-only standardization formula $(x - \mu)/\sigma$ produce on the training and test splits, and how should a robust preprocessing pipeline handle zero-variance features?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/195)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17967)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §1.2]{.kicker}

From Raw Tabular Data to Clean Tensors<br>**schema inspection · missing-data mechanisms · one-hot encoding · z-score standardization · data leakage**
:::
:::

::: {.slide title="Why Raw Tabular Data Cannot Go Straight into a Neural Network"}
[The Preprocessing Pipeline]{.kicker}

@fig:pandas-pipeline

::: {.cols}
::: {.col}
- **Heterogeneous types & gaps:** Real-world CSV logs mix numerical columns (`int64`, `float64`), free-text categorical strings (`"Slate"`, `"Tile"`), and unrecorded `NaN` entries—whereas tensor kernels require dense, homogeneous floating-point buffers.
- **Incommensurate scales:** In our house-price dataset, `Area` ($\sim 10^3\text{ sq ft}$) is roughly $500\times$ larger than `NumRooms` ($\sim 10^0$), skewing distances and gradients if left unscaled.
:::

::: {.col}
- **Statistical judgment, not boilerplate:** Every preprocessing choice—dropping vs. imputing `NaN`s, integer vs. one-hot encoding, whole-table vs. training-split scaling—imposes an explicit statistical assumption on what the downstream model can learn.
:::
:::
:::

::: {.slide title="Ingesting Raw CSV Records into a DataFrame"}
[Loading Tabular Data]{.kicker}

::: {.cols .vc}
::: {.col}
- **Plain-text tabular storage:** Comma-separated values (CSV) files store one record per line with comma-delimited fields; empty fields between consecutive commas (e.g., `,Tile,2100,290000`) denote unrecorded measurements.
- **Automatic sentinel parsing:** `pd.read_csv` parses the text file into a 2D `DataFrame` with named, typed columns and replaces blank entries with `NaN` (*Not a Number*).
- **Running dataset (`house_tiny.csv`):** 10 residential home sales recording `NumRooms`, `RoofType`, floor `Area` (sq ft), and sale `Price` ($).
:::

::: {.col}
@pandas-reading-the-dataset-2
:::
:::
:::

::: {.slide title="Auditing Schema Types and Summary Statistics"}
[Exploratory Inspection]{.kicker}

::: {.cols}
::: {.col}
- **`data.info()` reveals an IEEE 754 effect:** Even though room counts are whole numbers, `NumRooms` is promoted to `float64` because standard integer arrays cannot represent the floating-point `NaN` bit pattern.

@-pandas-knowing-your-data-1

| Column | Non-Null | Dtype | Preprocessing Role |
| :--- | :---: | :---: | :--- |
| `NumRooms` | 7 / 10 | `float64` | Continuous feature (3 `NaN`s) |
| `RoofType` | 6 / 10 | `str` | Categorical feature (4 `NaN`s) |
| `Area` | 10 / 10 | `int64` | Continuous feature (complete) |
| `Price` | 10 / 10 | `int64` | Supervised target label $y$ |
:::

::: {.col}
- **`data.describe()` exposes scale disparities** across observed entries ($\bar{x}_{\text{Area}} = 1{,}654\text{ sq ft}$ vs. $\bar{x}_{\text{Rooms}} = 3.29$):

@pandas-knowing-your-data-2
:::
:::
:::

::: {.slide title="Separating Input Features (X) from Target Labels (y) First"}
[Pipeline Hygiene]{.kicker}

::: {.cols .vc}
::: {.col}
- **Isolate $y$ before transforming $X$:** Split features from target `Price` *before* imputation or scaling so target statistics never leak into $\mathbf{X}$.
- **Select by column name, not position:** `data.drop(columns='Price')` survives upstream schema changes that would break `.iloc[:, :3]`.

@pandas-separating-inputs-and-targets-1
:::

::: {.col .fig}
@fig:pandas-xy-split
:::
:::
:::

::: {.slide title="Diagnosing Missing Values: Why Row Deletion Is Dangerous"}
[Missing Data]{.kicker}

::: {.cols .vc}
::: {.col}
- **Counting gaps (`isna().sum()`):** Sums the Boolean missingness mask per column (`NumRooms`: 3 missing, `RoofType`: 4 missing).
- **Why `dropna()` fails in practice:**
  1. **Sample destruction:** Dropping incomplete rows (`inputs.dropna()`) keeps only **4 of 10 homes**, destroying $60\%$ of our dataset!
  2. **Selection bias:** Whenever values are *not* missing completely at random (e.g., studio apartments or luxury estates disproportionately omitting room counts), row deletion biases the surviving sample.

@pandas-handling-missing-values-1
:::

::: {.col .fig}
@fig:pandas-missing

@pandas-handling-missing-values-2
:::
:::
:::

::: {.slide title="Numerical Imputation and Missingness Indicator Features"}
[Imputation Strategies]{.kicker}

::: {.cols .vc}
::: {.col}
- **Mean / median imputation:** Fills numerical `NaN`s with $\mu_{\text{Rooms}} \approx 3.29$, but piling imputed rows at $\mu$ shrinks sample spread $\sigma$ ($1.11 \to 0.91$).
- **Missingness indicator (`is_na`):** Adding a binary $0/1$ flag lets linear weights shift predictions when unrecorded values carry signal (e.g., luxury estates).

@pandas-handling-missing-values-3
:::

::: {.col .fig}
@fig:pandas-imputation-shrinkage
:::
:::
:::

::: {.slide title="Encoding Categorical Features as Orthogonal One-Hot Vectors"}
[Categorical Representation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why integer IDs fail on unordered categories:** Assigning $\textrm{Slate}=0, \textrm{Tile}=1, \textrm{missing}=2$ forces any linear layer $\mathbf{x}^\top \mathbf{w}$ to treat `Tile` as halfway between `Slate` and an unrecorded roof—inventing a false ordering and metric distance.
- **One-hot basis vectors in $\mathbb{R}^K$:** `pd.get_dummies(..., dummy_na=True, dtype=float)` expands $K$ categories into $K$ orthogonal $0/1$ columns (including `RoofType_nan` as a dedicated missingness channel), placing every category at equal Euclidean distance $\sqrt{2}$.
:::

::: {.col .fig .big}
@fig:pandas-onehot
:::
:::
:::

::: {.slide title="The Expanded One-Hot Table and High-Cardinality Limits"}
[Categorical Representation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Row-sum invariant:** In every row, `Slate + Tile + nan = 1.0`, placing categories at orthogonal corners of $\mathbb{R}^K$.
- **High-cardinality limit:** For $K = 50{,}000$ ZIP codes or tokens, one-hot vectors are $99.99\%$ zeros—use learned **embeddings** $\mathbf{E} \in \mathbb{R}^{K \times d}$ ($d \ll K$) instead.

@pandas-encoding-categorical-features-1
:::

::: {.col .fig}
@fig:pandas-categorical-pitfalls
:::
:::
:::

::: {.slide title="Feature Scaling: Why Unscaled Units Distort Optimization"}
[Standardization]{.kicker}

::: {.cols .vc}
::: {.col}
- **Unit mismatch:** In raw units, $\Delta\text{Area} = 3\text{ sq ft}$ dwarfs $\Delta\text{NumRooms} = 2\text{ rooms}$, distorting Euclidean distances, $L_2$ weight decay, and gradients.
- **Ill-conditioned valleys:** A $500\times$ scale gap elongates the loss Hessian $\nabla^2 \mathcal{L}$, forcing gradient descent to zig-zag at a tiny learning rate.
- **Z-score standardization** centers each continuous column to **mean $0$** and **unit variance ($1$)**:
  $$z_i = \frac{x_i - \mu}{\sigma}, \qquad \sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2}$$
:::

::: {.col .fig .big}
@fig:pandas-scale
:::
:::
:::

::: {.slide title="Standardizing Continuous Features While Preserving Binary Indicators"}
[Standardization in Code]{.kicker}

::: {.cols .vc}
::: {.col}
- **Scale continuous features only:** We standardize `['NumRooms', 'Area']` while leaving the one-hot columns `RoofType_*` untouched on $\{0.0, 1.0\}$ so binary presence/absence semantics remain intact.
- **Interpreting $z$-scores:** Row 6 (5 rooms, $2{,}600\text{ sq ft}$) maps to $(+1.89, +1.70)$—immediately revealing that the home sits $\approx 1.7\text{--}1.9\sigma$ above average on both dimensions.
- **Imputed entries map to $0$:** Rows 1, 4, and 8 were filled with $\mu_{\text{Rooms}}$, so their standardized room count is identically $0.0$.
:::

::: {.col}
@pandas-numerical-features-1
:::
:::
:::

::: {.slide title="The Golden Rule of Preprocessing: Fit on Train, Apply to Test"}
[Methodological Invariant]{.kicker}

::: {.cols .vc}
::: {.col}
- **How leakage happens:** Computing imputation means, category schemas, or $(\mu, \sigma)$ on the full table before splitting leaks test distribution statistics into training features.
- **Fit on $\mathcal{D}_{\text{train}}$ only:** Freeze $(\mu_{\text{train}}, \sigma_{\text{train}})$ and apply unchanged to $\mathcal{D}_{\text{val}}$ and $\mathcal{D}_{\text{test}}$.
- **Zero-variance guard:** If $\sigma_{\text{train}} = 0$, clamp $\sigma \leftarrow \max(\sigma, \epsilon)$ to avoid dividing by zero.
:::

::: {.col .fig .big}
@fig:pandas-leakage-split
:::
:::
:::

::: {.slide title="Concept Check: Spotting Data Leakage and Encoding Bugs"}
[Check Your Understanding]{.kicker}

Identify the statistical or systems bug in each preprocessing snippet:

1. **Snippet A:** Standardize the full DataFrame `df[num] = (df[num] - df[num].mean()) / df[num].std()`, and then split `train, test = df.iloc[:800], df.iloc[800:]`.
2. **Snippet B:** Encode `Neighborhood` (`"Downtown"`, `"Suburb"`, `"Rural"`) as integers `0, 1, 2` before feeding into a linear regression model $\hat{y} = \mathbf{x}^\top \mathbf{w} + b$.
3. **Snippet C:** Call `pd.get_dummies(train_df)` and `pd.get_dummies(test_df)` independently when `"Slate"` appears in `train_df` but never in `test_df`.

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Data leakage:** Test rows $800\ldots N$ influenced $\mu$ and $\sigma$. Compute `mu, sigma` on `train` only, then transform both `train` and `test` with those training parameters.
2. **False metric ordering:** A single weight $w_{\text{neigh}}$ forces $\hat{y}(\text{Suburb}) - \hat{y}(\text{Downtown}) = \hat{y}(\text{Rural}) - \hat{y}(\text{Suburb})$. Use one-hot encoding instead.
3. **Feature-dimension mismatch:** `test_df` will lack the `RoofType_Slate` column, causing a shape mismatch (or misaligned weights!) at matrix multiplication time. Align `test_df` to `train_df`'s columns.
:::
:::

::: {.slide title="Bridging Clean DataFrames to Contiguous float32 Tensors"}
[Tensor Conversion]{.kicker}

::: {.cols .vc}
::: {.col}
- **Downcast `float64` $\to$ `float32`:** Pandas defaults to 64-bit floats (8 B) or mixed `bool`/`object` columns; `.to_numpy(dtype='float32')` halves memory (4 B) and matches GPU layer weights.

@pandas-conversion-to-tensor-format
:::

::: {.col .fig}
@fig:pandas-dtype-cast
:::
:::
:::

::: {.slide title="Summary: From Messy Tables to Model-Ready Design Matrices"}
[Key Takeaways]{.kicker}

| Preprocessing Stage | pandas / Framework Primitive | Statistical Purpose | Production & Leakage Invariant |
| :--- | :--- | :--- | :--- |
| **Schema & Scale Audit** | `df.info()`, `df.describe()` | Detect `NaN` float promotion, text columns, scale gaps | Inspect distributions & sentinel codes before modeling |
| **Target Isolation** | `df.drop(columns='Price')`, `df['Price']` | Separate design matrix $\mathbf{X}$ from target labels $\mathbf{y}$ | Select by column name, never fragile `.iloc` positions |
| **Missing Numericals** | `fillna(train_mean)` + `isna()` flag | Preserve rows without selection bias; capture missingness signal | Fit imputation mean/median on **$\mathcal{D}_{\text{train}}$ only** |
| **Unordered Categories** | `pd.get_dummies(..., dummy_na=True)` | Map $K$ categories to orthogonal basis vectors in $\mathbb{R}^K$ | Lock category schema on **$\mathcal{D}_{\text{train}}$**; use embeddings if $K \gg 10^2$ |
| **Continuous Scaling** | $(x - \mu_{\text{train}}) / \sigma_{\text{train}}$ | Equalize feature units ($\mu=0, \sigma=1$); condition loss curvature | Fit $(\mu, \sigma)$ on **$\mathcal{D}_{\text{train}}$ only**; leave $0/1$ one-hots unscaled |
| **Tensor Construction** | `tensor(df.to_numpy(dtype='float32'))` | Pack heterogeneous columns into contiguous `float32` buffer | Verify `(N, D)` feature and `(N,)` target shapes |
:::
