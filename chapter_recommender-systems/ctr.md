# Feature-Rich Recommender Systems

In click-through-rate (CTR) prediction, one example is an impression: an item was displayed to a user in a particular context. Its input $\mathbf{x}$ contains user, item, and context features, and its label $y\in\{0,1\}$ records whether the impression produced a click. A model outputs a logit $s(\mathbf{x})$ or probability $\hat p(\mathbf{x})=\sigma(s(\mathbf{x}))$ and is commonly trained with binary log loss. Unlike interaction-only collaborative filtering, this formulation can use item attributes, device type, time, and other features that may support predictions when a user--item history is sparse :cite:`McMahan.Holt.Sculley.ea.2013`.


For a collection of impressions, the empirical click-through rate is

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

This aggregate rate is not itself a measure of model quality: it also changes with the population, candidate-selection policy, and display position. The prediction task is to estimate a conditional probability for each impression. Evaluation should therefore use a proper scoring rule such as held-out log loss, and any train/test split should respect the time or policy shift the model is expected to face.

```{.python .input #ctr-feature-rich-recommender-systems}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

```{.python .input #ctr-feature-rich-recommender-systems}
#@tab pytorch
from collections import defaultdict
from d2l import torch as d2l
import torch
import os
```

## An Online Advertising Dataset

The anonymous advertising dataset contains a binary click label followed by 34 categorical fields. The undisclosed fields may encode quantities such as an advertisement, site, application, device, time bucket, or user group; because their semantics are hidden, the experiment can test feature-interaction models but cannot support a substantive interpretation of individual coefficients. Each categorical value is mapped to an integer index within its field, with an additional index reserved for rare or unseen values.

The following code downloads the dataset from our server and saves it into the local data folder.

```{.python .input #ctr-an-online-advertising-dataset  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

```{.python .input #ctr-an-online-advertising-dataset  n=15}
#@tab pytorch
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

There are a training set and a test set, consisting of 15000 and 3000 samples/lines, respectively.

## Dataset Wrapper

For the convenience of data loading, we implement a `CTRDataset` which loads the advertising dataset from the CSV file and can be used by a standard data loader.

```{.python .input #ctr-dataset-wrapper-1  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                instance['y'] = [float(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(sorted(feat_values))}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        # Wrap label in np.array so DataLoader batching yields an ndarray
        # (not a list-of-lists), matching the pytorch tab's torch.tensor(...).
        return feat + self.offsets, np.array(self.data[idx]['y'])
```

```{.python .input #ctr-dataset-wrapper-1  n=13}
#@tab pytorch
#@save
class CTRDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = torch.zeros(self.NUM_FEATS, dtype=torch.long)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                instance['y'] = [float(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(sorted(feat_values))}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = torch.tensor(
            (0, *torch.cumsum(self.field_dims, dim=0).numpy()[:-1]))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        feat = torch.tensor([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                             for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, torch.tensor(self.data[idx]['y'])
```

The following example loads the training data and print out the first record.

```{.python .input #ctr-dataset-wrapper-2  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

```{.python .input #ctr-dataset-wrapper-2  n=16}
#@tab pytorch
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

The encoded example contains one integer index per categorical field and a binary label; it does not materialize the corresponding high-dimensional one-hot vector. The wrapper can be adapted to other field-based datasets, but datasets with continuous variables require a separate numerical preprocessing path rather than categorical vocabulary lookup.

## Summary 
* CTR prediction is binary probabilistic prediction conditional on an observed impression and its user, item, and context features.
* Field-wise categorical indices permit sparse inputs to share compact embedding tables; aggregate CTR should not be confused with a model-evaluation metric.

## Exercises

* Extend `CTRDataset` with an explicit path for continuous fields. How will you fit normalization or bin boundaries without leaking information from the test set?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/405)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/405)
:end_tab:

<!-- slides -->

::: {.slide title="CTR Prediction"}
Pure (user, item) collaborative filtering breaks for cold
start — new users and new items have no history. Real
recommenders integrate **side features**: item attributes,
user profiles, time, device, query context, …

This deck sets up the **online advertising CTR
prediction** problem: predict click probability from a
sparse vector of categorical features. Feature-rich
recommendation in its purest form. The next two decks
(FM and DeepFM) train models on this loader.

@ctr-feature-rich-recommender-systems
:::

::: {.slide title="Each impression carries a binary click label"}
Tab-separated; each row has many one-hot categorical
fields plus a binary click label. Sparsity is extreme —
think "1 of 10000 in each field":

@ctr-an-online-advertising-dataset
:::

::: {.slide title="Per-field vocabularies avoid dense one-hot vectors"}
Build per-field vocabularies, encode each row as a sparse
feature index vector, yield (features, label) pairs:

@ctr-dataset-wrapper-1

. . .

@ctr-dataset-wrapper-2
:::

::: {.slide title="CTR conditions on an observed impression"}
- CTR prediction = binary classification on sparse
  categorical features.
- Side features handle cold start; pure collaborative
  filtering can't.
- Output of this deck: indexed-categorical mini-batches
  the FM / DeepFM decks consume.
- Real-world systems extend this with continuous
  features, multi-task heads, and embedding tables on
  the order of billions of entries.
:::
