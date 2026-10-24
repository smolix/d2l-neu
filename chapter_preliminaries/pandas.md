```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Data Preprocessing
:label:`sec_pandas`

A model may receive synthetic data as ready-made tensors, but real tabular
records arrive as text files, often with gaps, and with words where a model
needs numbers.
Consider ten home sales, each recorded with the number of rooms, the type of
roof, the floor area, and the sale price. In our table, three of the ten
homes lack the number of rooms and four lack the roof type. The roof type is
a word, such as `Slate` or `Tile`, not a number. And the floor area averages
1,654 square feet, about 500 times the average number of rooms, 3.3. A model
computes with tensors of numbers, so before it sees this table we have to
decide what to put in the gaps, how to turn words into numbers, and whether
to bring the columns to a common scale.

These decisions change what a model can learn. Dropping every incomplete row
would leave 4 of the 10 homes. Numbering slate roofs, tile roofs, and
missing entries 0, 1, and 2 would invent an order among them that does not
exist. And left unscaled, the floor area
dominates any computation that compares homes, so that the number of rooms
barely counts. This section turns the table into a tensor of inputs and a
tensor of targets with the [pandas](https://pandas.pydata.org/) library
:cite:`McKinney.2010`, the standard Python tool for loading, inspecting, and
transforming tables (:numref:`fig_pandas_pipeline`).

![From a file to tensors. `read_csv` loads the text file into a DataFrame with gaps and text columns; imputation, encoding, and scaling make every entry a number; `to_numpy` hands the result to the framework as the tensors `X` and `y`.](../img/pandas-pipeline.svg)
:label:`fig_pandas_pipeline`

We import pandas under its usual alias `pd`, the module `os` for file paths,
and the deep learning framework, which we need only at the end, for the
conversion to tensors.

:begin_tab:`mxnet`
MXNet's array module is imported under the name `mnp`, so that `np` can keep
meaning NumPy.
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

A *comma-separated values* (CSV) file stores a table as plain text. Each line
holds one row, the first line usually holds the column names, and commas
separate the fields within a line. We write the ten home sales to the file
`../data/house_tiny.csv`. The columns are the number of rooms (`NumRooms`),
the roof type (`RoofType`), the floor area in square feet (`Area`), and the
sale price in dollars (`Price`). An empty field marks a missing entry.

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

The function `read_csv` parses the file into a *DataFrame*, the table object
of pandas. A DataFrame consists of named columns, each holding values of one
type, and it labels its rows; the numbers 0 to 9 at the left of the output
are these labels.

```{.python .input #pandas-reading-the-dataset-2}
data = pd.read_csv(data_file)
data
```

Every empty field has become `NaN` (not a number), the marker that pandas
uses for a *missing value*.

### Data Types and Summaries

Before transforming a table, we check what pandas made of it: how many rows
and columns it has, how many entries of each column are present, and which
data type pandas inferred for each column. The method `info` reports all
three.

```{.python .input #pandas-knowing-your-data-1}
data.info()
```

The table has 10 rows and 4 columns. `NumRooms` has 7 entries present and
`RoofType` 6, so 3 and 4 entries are missing. The data types show how pandas
read each column. `Area` and `Price` hold 64-bit integers. `NumRooms` holds
whole numbers too, but pandas has made it a `float64` column, because `NaN`
is a floating-point value and an integer column cannot hold it. `RoofType`
has the type `str`, which pandas uses for text (versions before pandas 3.0
used `object`). A text column usually holds categories, as here, and must be
turned into numbers before a model can use it.

For the numerical columns, the method `describe` summarizes the values: how
many are present, their mean and standard deviation, and their quantiles.

```{.python .input #pandas-knowing-your-data-2}
data.describe()
```

The columns have very different scales. The number of rooms ranges from 2 to
5, the area from 850 to 2,600 square feet, and the price from 127,500 to
375,000 dollars. We return to this difference when we rescale the columns.

### Inputs and Targets

We want to predict the price of a home from its other attributes. The price is
the *target*, also called the *label*, and the remaining columns are the
inputs, also called *features*. We separate the two first, so that the steps
below transform only the inputs and the price never ends up among them. The
indexer `iloc` selects by position, with the slicing rules of tensors:
`data.iloc[:, :3]` takes all rows and the first three columns, and
`data.iloc[:, 3]` takes the fourth column. The names of the selected
columns confirm the split:

```{.python .input #pandas-separating-inputs-and-targets-1}
inputs, targets = data.iloc[:, :3], data.iloc[:, 3]
list(inputs.columns), targets.name
```

Selection by position is short, but it breaks without an error when a new
version of the file inserts or reorders columns. Selection by name, as in
`data.drop(columns='Price')`, does not; an exercise at the end of the
section takes this up.

## Cleaning and Encoding

### Missing Values

Values go missing in real data for many reasons: a field was not recorded,
did not apply, or was lost. Before we choose a remedy, we count the gaps. The
method `isna` returns a table of the same shape that holds `True` where an
entry is missing and `False` elsewhere, and `sum` adds up each column,
counting `True` as 1.

```{.python .input #pandas-handling-missing-values-1}
inputs.isna().sum()
```

There are three common responses, and each has a cost.

* *Deletion* drops every row, or every column, that contains a missing
  value. It is simple, but it discards data. If values are missing more
  often for some kind of home, such as small ones, deletion also biases
  what remains.
* *Imputation* fills each gap with an estimate, such as the mean of the
  column. It keeps every row, but the filled values are guesses, and they
  make the column look less spread out than it is.
* An *indicator* is an extra column that records which values were missing,
  usually added alongside imputation. With it, a model can learn whether a
  missing value carries information of its own, for example when the number
  of rooms is missing mostly for large homes.

On a table of ten rows, deletion is expensive. Dropping every row that has a
missing entry keeps only the complete rows:

```{.python .input #pandas-handling-missing-values-2}
len(inputs.dropna()), len(inputs)
```

Only 4 of the 10 homes remain, so we impute instead. For a numerical column,
the simplest estimate is the column mean; the median is the safer choice
when a few very large values pull the mean away from the typical value. The
method `fillna` fills the gaps of each column with the value given for that
column, and `mean(numeric_only=True)` computes the mean of each numerical
column, skipping its missing entries. A text column has no mean, so the gaps
in `RoofType` remain; we deal with them when we encode the column.

```{.python .input #pandas-handling-missing-values-3}
inputs = inputs.fillna(inputs.mean(numeric_only=True))
inputs
```

The three missing room counts are now 3.29, the mean $23/7$ of the seven
observed counts.

### Categorical Features

The column `RoofType` holds *categories*: each home belongs to one of a few
groups, and the groups have no order. Counting a missing entry as a group of
its own, there are three groups: `Slate`, `Tile`, and missing. A model needs
numbers, but numbering the three groups 0, 1, and 2 would invent an order
among them, and it would place a missing roof twice as far from slate as
tile is. A column with only two values is different: a single 0/1 column
represents it fully, because a weight of either sign can express any
difference between the two.

*One-hot encoding* avoids this. It replaces the column by one new column per
category, which holds 1 for the homes in that category and 0 for all others,
so that any two categories are equally far apart. The function
`get_dummies` performs the encoding. With `dummy_na=True`, it treats a
missing value as one more category, which is the indicator strategy applied
to a categorical column. With `dtype=float`, the new columns hold the
numbers 0.0 and 1.0 rather than the values `False` and `True`.

```{.python .input #pandas-encoding-categorical-features-1}
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=float)
inputs
```

`RoofType` has become the three columns `RoofType_Slate`, `RoofType_Tile`,
and `RoofType_nan`, and each row has a single 1 among them. The numerical
columns pass through unchanged, so every column of `inputs` now holds
numbers.

One-hot encoding suits columns with a few categories. A column with
thousands of distinct values, such as a postal code, would become thousands
of columns that are almost entirely zero. Such columns are better
represented by *embeddings*, short vectors of numbers that a model learns
for each value; they appear later in the book.

## Scaling and Converting

### Standardization

Every entry of `inputs` is now a number, but the two numerical columns still
measure different things in different units. In raw units, two homes with
the same area that differ by two rooms are closer to each other than two
homes with the same number of rooms that differ by three square feet. Any
computation that adds up differences between features, such as the distance
between two homes, is then dominated by the area, and the number of rooms
barely counts. A model could compensate by giving the area a weight hundreds
of times smaller than that of the number of rooms. However, the training
methods of later chapters adjust all weights with a common step size, and a
step size that suits one feature is then far too large or far too small for
the other (:numref:`subsec_mdl-quadratic-model` analyzes this
effect).

The standard remedy is *standardization*. We replace every entry $x$ of a
column by

$$\frac{x - \mu}{\sigma},$$

where $\mu$ and $\sigma$ are the mean and the standard deviation of the column.
For a column with entries $x_1, \ldots, x_n$, pandas computes the standard
deviation, the typical distance of an entry from the mean, as $\sigma =
\sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2}$. After this step every column
has mean 0 and standard deviation 1. An entry then says how many standard
deviations it lies above or below the mean of its column, whatever the units of
the column. The one-hot columns already share a scale, 0 and 1, so we leave
them as they are.

```{.python .input #pandas-numerical-features-1}
numerical = ['NumRooms', 'Area']
mu, sigma = inputs[numerical].mean(), inputs[numerical].std()
inputs[numerical] = (inputs[numerical] - mu) / sigma
inputs
```

The largest home, in row 6, has 5 rooms and 2,600 square feet. After
standardization these entries are 1.89 and 1.70: in both columns the home
lies 1.7 to 1.9 standard deviations above the mean, and the two numbers can
now be compared directly. The imputed room counts have become 0, because
mean imputation placed them at the mean of the column.

All statistics in this section, the means used for imputation and the means
and standard deviations used for standardization, were computed from all ten
homes. In practice we hold back part of the data to test a trained model,
and the test has to imitate data that the model has never seen. The
statistics must therefore be computed from the training data alone and then
applied, unchanged, to the test data. Computing them from all the data lets
information about the test data *leak* into training, and the test then
overstates how well the model will do on new data.
:numref:`sec_kaggle_house` follows this rule on a real dataset of home
prices.

### Conversion to Tensors

Every column of `inputs` now holds numbers, so the table can become a
tensor. The method `to_numpy` converts a DataFrame to a NumPy array, and the
framework's tensor constructor converts the array to a tensor. A DataFrame
stores floating-point columns as 64-bit numbers, whereas deep learning
frameworks compute in 32-bit floating point by default, so we ask `to_numpy`
for `float32` directly.

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

`X` has one row per home and one column per feature: the two standardized
numerical features and the three indicators of the roof type. `y` holds the
ten prices.

We can now turn a table with missing entries and text columns into tensors,
and we know the decisions this involves: how to fill gaps, how to encode
categories, and how to bring features to a common scale. The pandas library
holds a whole table in main memory and processes it on the CPU, which suits
the tables in this book. Tables larger than memory need tools that read them in
pieces, and images, audio, and text need pipelines of their own, which later
chapters build. The tensor `X` is a matrix with one row per example, the
form in which models receive their data, and the next section studies the
operations that models apply to such matrices.

## Summary

Real tables have missing entries, text columns, and columns on very
different scales, whereas a model needs a tensor of numbers. The pandas
library loads such a table into a DataFrame, and `info` and `describe` show its size, its
gaps, its data types, and the ranges of its columns. Missing values can be
deleted, imputed, or flagged with an indicator column; on small data,
imputation keeps the rows that deletion would discard. Categorical columns
become one 0/1 column per category, which avoids inventing an order, and
numerical columns are standardized to mean 0 and standard deviation 1, with
statistics computed from the training data only. `to_numpy` then hands the
table to the framework as a `float32` tensor.

## Exercises

1. [code] **Indexing by name.** This section splits `data` by position
   with `data.iloc[:, :3]` and `data.iloc[:, 3]`. Suppose a new version of
   `house_tiny.csv` adds a column `YearBuilt` between `Area` and `Price`.
    1. State what the positional split then puts into `inputs` and
       `targets`, and whether any later line of the section's code raises
       an error.
    1. Rewrite the split with column names so that `targets` holds `Price`
       and `inputs` holds every other column, for the new file and for the
       original one. The pandas user guide on
       [indexing](https://pandas.pydata.org/docs/user_guide/indexing.html)
       lists the options.

1. [code] **Imputation strategies.** Fill the missing values in `NumRooms`
   in three ways: with the column mean, with the column median, and with
   the column mean plus a "was-missing" indicator column.
    1. For the mean fill and the median fill, compute the mean and the
       standard deviation of the filled column, compare them with those of
       the observed values, and explain the direction of each change.
    1. Suppose that in a larger dataset `NumRooms` is missing mostly for
       large homes. State what a model sees for those homes after mean
       imputation alone. Then explain how a model that predicts the price
       as a weighted sum of its features plus a constant can use the
       indicator column to correct its predictions for them.
    1. State what each of the three treatments assumes about why the values
       are missing.

    *Adapted from Kaggle Learn,
    [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning),
    Exercise 2.*

1. **High-cardinality categories.** One-hot encoding turns a categorical
   column with $k$ distinct values into $k$ columns, plus one more with
   `dummy_na=True`.
    1. A table of $n = 10^6$ homes has a `ZipCode` column with no missing
       entries and $k = 10^4$ distinct values. Compute how many bytes its
       one-hot columns occupy when stored densely as `float64`, and how many
       of their entries are nonzero.
    1. A column `ListingID` holds a value unique to each row. Consider three
       treatments: one-hot encoding, an embedding (a learned vector for each
       distinct value), and dropping the column. For each, state what a
       model could learn about a home whose `ListingID` does not occur in
       the training data, and choose one treatment.

    *Adapted from Kaggle Learn,
    [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning),
    Exercise 3.*

1. [code] **Where leakage hides.** This section fills the missing
   `NumRooms` entries with the mean of all observed values and
   standardizes `NumRooms` and `Area` with the mean and standard deviation
   of all ten rows. Treat the first eight rows of `data` as training data
   and the last two as test data.
    1. Redo the imputation and the standardization with statistics
       computed on the training rows only and applied unchanged to the test
       rows. Compare the standardized test rows with those of the
       whole-dataset recipe.
    1. Explain what the whole-dataset recipe lets the model learn about the
       test rows that the training-only recipe does not.
    1. Suppose a feature is constant on the training rows but not on the
       test rows. State what the training-only recipe produces for that
       feature on each split, and propose a fix.

1. [extended] **A real messy dataset, end to end.** Take a messy public
   dataset, for example the San Francisco restaurant-inspection data used
   in Berkeley's Data 100. Run the full pipeline of this section: detect a
   non-standard missing-value sentinel (a value such as `-9999` rather than
   a blank field), choose and justify a handling strategy, encode at least
   one categorical column with an explicit judgment about its number of
   distinct values, standardize the numerical columns, and convert the
   result to a tensor. Report the shape and data type of the final tensor.

    *Adapted from UC Berkeley Data 100,
    [homework 2A "Food Safety"](https://github.com/DS-100/sp25-student/blob/main/hw/hw02A/hw02A.ipynb).*

1. [code] **Scaling limits.** ● Estimate how many rows of a table with the
   columns of `house_tiny.csv` you can load with `pd.read_csv` on your
   machine. Write synthetic CSV files with the four columns of
   `house_tiny.csv` and $10^5$, $10^6$, and $10^7$ random rows. For each
   file, time `pd.read_csv` and record the memory of the resulting
   DataFrame. Extrapolate both measurements linearly. Find the number of
   rows at which the DataFrame would fill your machine's memory and the
   number at which loading would take ten minutes, and state which limit
   comes first.

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
[Dive into Deep Learning · §2.2]{.kicker}

Tensors from a table with gaps<br>**inspect · impute · one-hot encode · standardize · convert**
:::
:::

::: {.slide title="Ten Homes: Gaps, Words, and Scales 500× Apart"}
[The problem]{.kicker}

::: {.cols .vc}
::: {.col}
@!pandas-reading-the-dataset-2
:::

::: {.col .narrow}
- 3 room counts and 4 roof types are missing (`NaN`).
- `RoofType` is a word, not a number.
- Mean area 1,654 sq ft, mean rooms 3.3.

::: {.d2l-note}
A model needs a tensor of numbers.
:::
:::
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- inspect a table's size, gaps, and types with `info` and `describe`;
- choose among deleting, imputing, and flagging missing values;
- one-hot encode a categorical column;
- standardize numerical columns without leaking test data;
- convert the result to `float32` tensors.

**Running example:** `house_tiny.csv`, 10 homes, 4 columns, the target
(label) is `Price`.
:::

::: {.slide title="Preprocessing Turns a CSV File Into Tensors"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | tool |
|---|---|
| Data Manipulation | tensors |
| **Data Preprocessing** | **pandas** |
| Linear Algebra | matrix products, norms, eigenvalues |
| Calculus | derivatives, gradients, chain rule |
| Automatic Differentiation | autograd |
| Probability and Statistics | Bayes' theorem, standard error |
| Documentation | `help`, small tests |
:::

::: {.col .narrow}
- **Uses:** tensors, from the previous deck.
- **Skips:** joining several tables, tables larger than memory, images,
  and text.
:::
:::
:::

::: {.slide title="Three Transformations Stand Between read_csv and to_numpy"}
[Pipeline]{.kicker}

@fig:pandas-pipeline

- `read_csv` loads the text into a DataFrame, gaps and words included.
- Imputation, encoding, and scaling make every entry a number.
- `to_numpy` hands the result to the framework as `X` and `y`.
:::

::: {.slide title="info Shows Size, Gaps, and Types in One Call"}
[Inspect]{.kicker}

::: {.cols .vc}
::: {.col}
@-pandas-knowing-your-data-1

10 rows; per column, `info` reports:

| column | present | type |
|---|---|---|
| `NumRooms` | 7 of 10 | `float64` |
| `RoofType` | 6 of 10 | `str` |
| `Area` | 10 of 10 | `int64` |
| `Price` | 10 of 10 | `int64` |
:::

::: {.col .narrow}
- `NumRooms` holds whole numbers but is `float64`: `NaN` is a float.
- `RoofType` is `str`: a category to encode.
:::
:::
:::

::: {.slide title="describe Shows Scales 500× Apart"}
[Inspect]{.kicker}

@-pandas-knowing-your-data-2

| | `NumRooms` | `Area` (sq ft) | `Price` (dollars) |
|---|---|---|---|
| mean | 3.29 | 1,654 | 233,050 |
| std | 1.11 | 556 | 76,129 |
| min | 2 | 850 | 127,500 |
| max | 5 | 2,600 | 375,000 |

The mean area is about 500 times the mean number of rooms.
:::

::: {.slide title="Split Off the Target Before Any Transformation"}
[Inspect]{.kicker}

@pandas-separating-inputs-and-targets-1

- **Inputs** (features): `NumRooms`, `RoofType`, `Area`.
- **Target** (label): `Price`, which must never appear among the inputs.

::: {.d2l-note .warn}
`iloc` selects by position and breaks silently if columns move; selecting
by name does not.
:::
:::

::: {.slide title="Count the Gaps Before Choosing a Remedy"}
[Missing values]{.kicker}

::: {.cols .vc}
::: {.col}
@pandas-handling-missing-values-1

- **delete** the row: loses data;
- **impute** the mean: keeps rows, guesses values;
- add an **indicator**: the gap itself becomes a feature.
:::

::: {.col .fig .big}
@fig:pandas-missing
:::
:::
:::

::: {.slide title="Deletion Keeps Only 4 of 10 Homes"}
[Missing values]{.kicker}

Dropping every row with a gap keeps the complete rows only:

@pandas-handling-missing-values-2

Six of ten homes, and their prices, are lost. On small data we impute
instead.
:::

::: {.slide title="Mean Imputation Fills the Gaps With 23/7 ≈ 3.29"}
[Missing values]{.kicker}

@-pandas-handling-missing-values-3

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `NumRooms` before | 3 | NaN | 2 | 4 | NaN | 3 | 5 | 2 | NaN | 4 |
| after | 3 | **3.29** | 2 | 4 | **3.29** | 3 | 5 | 2 | **3.29** | 4 |

The seven observed counts sum to 23. `RoofType` has no mean; its gaps are
handled by the encoding.
:::

::: {.slide title="Check: Which Remedy Fits?"}
[Check]{.kicker}

1. A column is missing in 95% of the rows.
2. `NumRooms` is missing mostly for large homes.
3. `RoofType` is missing for 4 of 10 homes.

::: {.d2l-note .fragment .fade-up}
**Answers.**

1. Drop the column, or keep only an indicator: imputation would invent
   most of it.
2. Impute and add an indicator: the gap itself predicts a high price.
3. Make "missing" a category of its own (`dummy_na=True`).
:::
:::

::: {.slide title="One-Hot Encoding Adds a 0/1 Column per Category"}
[Categories]{.kicker}

::: {.cols .vc}
::: {.col}
@-pandas-encoding-categorical-features-1

- Codes 0, 1, 2 for `Slate`, `Tile`, missing would invent an order.
- One-hot: all categories are equally far apart.
- `dummy_na=True`: "missing" gets its own column.
- Two values only: one 0/1 column suffices.
:::

::: {.col .fig .big}
@fig:pandas-onehot
:::
:::
:::

::: {.slide title="In Raw Units, Two Rooms Count Less Than Three Square Feet"}
[Scaling]{.kicker}

Standardize each numerical column: $x \mapsto (x - \mu)/\sigma$, with
mean $\mu$ and standard deviation $\sigma$.

@-pandas-numerical-features-1

::: {.cols .vc}
::: {.col}
Largest home: 5 rooms $\to$ 1.89, 2,600 sq ft $\to$ 1.70. Both now
count in standard deviations from the mean.
:::

::: {.col .fig .big}
@fig:pandas-scale
:::
:::
:::

::: {.slide title="Statistics Come From the Training Rows Only"}
[Scaling]{.kicker}

- Training rows 0 to 7: compute $\mu$, $\sigma$, and the imputation means.
- Test rows 8 and 9: apply the same numbers, unchanged.
- This deck's code used all 10 rows, to keep it short. With rows 0 to 7
  only, the largest home becomes 1.86 and 1.66 instead of 1.89 and 1.70.

::: {.d2l-note .warn}
Statistics from all rows leak the test data into training, and the test
score then overstates performance on new data.
:::
:::

::: {.slide title="A (10, 5) float32 Tensor and Ten Targets"}
[Conversion]{.kicker}

@pandas-conversion-to-tensor-format

::: {.d2l-note}
Rows: homes. Columns: 2 standardized features and 3 roof-type indicators.
:::
:::

::: {.slide title="Check: How Many Columns Does X Get?"}
[Check]{.kicker}

A table has the numerical columns `NumRooms`, `Area`, and `YearBuilt`,
some with gaps, a column `RoofType` with the values `Slate`, `Tile`, and
`Metal` and some gaps, and the target `Price`. How many columns does `X`
have after `get_dummies(..., dummy_na=True)`?

::: {.d2l-note .fragment .fade-up}
**Answer.** 7: three numerical columns, three roof types, and one column
for a missing roof type. `Price` is the target, not a feature.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| step | running example |
|---|---|
| inspect with `info`, `describe` | 10 rows; 3 + 4 gaps; scales 500× apart |
| delete, impute, or flag gaps | deletion keeps 4 of 10; the mean fills 3.29 |
| one-hot encode categories | `RoofType` becomes 3 columns |
| standardize (here with all 10 rows; in practice training rows only) | 5 rooms $\to$ 1.89, 2,600 sq ft $\to$ 1.70 |
| convert with `to_numpy(dtype='float32')` | `X`: (10, 5), `y`: (10,) |

**Next:** the operations that models apply to such matrices.
:::
