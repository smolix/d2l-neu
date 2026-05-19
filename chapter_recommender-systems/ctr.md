# Feature-Rich Recommender Systems

Interaction data is the most basic indication of users' preferences and interests. It plays a critical role in former introduced models. Yet, interaction data is usually extremely sparse and can be noisy at times. To address this issue, we can integrate side information such as features of items, profiles of users, and even in which context that the interaction occurred into the recommendation model. Utilizing these features are helpful in making recommendations in that these features can be an effective predictor of users interests especially when interaction data is lacking. As such, it is essential for recommendation models also have the capability to deal with those features and give the model some content/context awareness. To demonstrate this type of recommendation models, we introduce another task on click-through rate (CTR) for online advertisement recommendations :cite:`McMahan.Holt.Sculley.ea.2013` and present an anonymous advertising dataset. Targeted advertisement services have attracted widespread attention and are often framed as recommendation engines. Recommending advertisements that match users' personal taste and interest is important for click-through rate improvement.


Digital marketers use online advertising to display advertisements to customers. Click-through rate is a metric that measures the number of clicks advertisers receive on their ads per number of impressions and it is expressed as a percentage calculated with the formula: 

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

Click-through rate is an important signal that indicates the effectiveness of prediction algorithms. Click-through rate prediction is a task of predicting the likelihood that something on a website will be clicked. Models on CTR prediction can not only be employed in targeted advertising systems but also in general item (e.g., movies, news, products) recommender systems, email campaigns, and even search engines. It is also closely related to user satisfaction, conversion rate, and can be helpful in setting campaign goals as it can help advertisers to set realistic expectations.

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

With the considerable advancements of Internet and mobile technology, online advertising has become an important income resource and generates vast majority of revenue in the Internet industry. It is important to display relevant advertisements or advertisements that pique users' interests so that casual visitors can be converted into paying customers. The dataset we introduced is an online advertising dataset. It consists of 34 fields, with the first column representing the target variable that indicates if an ad was clicked (1) or not (0). All the other columns are categorical features. The columns might represent the advertisement id, site or application id, device id, time, user profiles and so on. The real semantics of the features are undisclosed due to anonymization and privacy concern.

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

For the convenience of data loading, we implement a `CTRDataset` which loads the advertising dataset from the CSV file and can be used by `DataLoader`.

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

As can be seen, all the 34 fields are categorical features. Each value represents the one-hot index of the corresponding entry. The label $0$ means that it is not clicked. This `CTRDataset` can also be used to load other datasets such as the Criteo display advertising challenge [dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and the Avazu click-through rate prediction [dataset](https://www.kaggle.com/c/avazu-ctr-prediction).  

## Summary 
* Click-through rate is an important metric that is used to measure the effectiveness of advertising systems and recommender systems.
* Click-through rate prediction is usually converted to a binary classification problem. The target is to predict whether an ad/item will be clicked or not based on given features.

## Exercises

* Can you load the Criteo and Avazu dataset with the provided `CTRDataset`. It is worth noting that the Criteo dataset consisting of real-valued features so you may have to revise the code a bit.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/405)
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

::: {.slide title="The advertising dataset"}
Tab-separated; each row has many one-hot categorical
fields plus a binary click label. Sparsity is extreme —
think "1 of 10000 in each field":

@ctr-an-online-advertising-dataset
:::

::: {.slide title="Dataset wrapper"}
Build per-field vocabularies, encode each row as a sparse
feature index vector, yield (features, label) pairs:

@ctr-dataset-wrapper-1

. . .

@ctr-dataset-wrapper-2
:::

::: {.slide title="Recap"}
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
