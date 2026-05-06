# Personalized Ranking for Recommender Systems

In the former sections, only explicit feedback was considered and models were trained and tested on observed ratings.  There are two demerits of such methods: First, most feedback is not explicit but implicit in real-world scenarios, and explicit feedback can be more expensive to collect.  Second, non-observed user-item pairs which may be predictive for users' interests are totally ignored, making these methods unsuitable for cases where ratings are not missing at random but because of users' preferences.  Non-observed user-item pairs are a  mixture of real negative feedback (users are not interested in the items) and missing values (the user might interact with the items in the future). We simply ignore the non-observed pairs in matrix factorization and AutoRec. Clearly, these models are incapable of distinguishing between observed and non-observed pairs and are usually not suitable for personalized ranking tasks.

To this end, a class of recommendation models targeting at generating ranked recommendation lists from implicit feedback have gained popularity. In general, personalized ranking models can be optimized with pointwise, pairwise or listwise approaches. Pointwise approaches consider a single interaction at a time and train a classifier or a regressor to predict individual preferences. Matrix factorization and AutoRec are optimized with pointwise objectives. Pairwise approaches consider a pair of items for each user and aim to approximate the optimal ordering for that pair. Usually, pairwise approaches are more suitable for the ranking task because predicting relative order is reminiscent to the nature of ranking. Listwise approaches approximate the ordering of the entire list of items, for example, direct optimizing the ranking measures such as Normalized Discounted Cumulative Gain ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)). However, listwise approaches are more complex and compute-intensive than pointwise or pairwise approaches. In this section, we will introduce two pairwise objectives/losses, Bayesian Personalized Ranking loss and Hinge loss, and their respective implementations.

## Bayesian Personalized Ranking Loss and its Implementation

Bayesian personalized ranking (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009` is a pairwise personalized ranking loss that is derived from the maximum posterior estimator. It has been widely used in many existing recommendation models. The training data of BPR consists of both positive and negative pairs (missing values). It assumes that the user prefers the positive item over all other non-observed items.

In formal, the training data is constructed by tuples in the form of $(u, i, j)$, which represents that the user $u$ prefers the item $i$ over the item $j$. The Bayesian formulation of BPR which aims to maximize the posterior probability is given below:

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

Where $\Theta$ represents the parameters of an arbitrary recommendation model, $>_u$ represents the desired personalized total ranking of all items for user $u$. We can formulate the maximum posterior estimator to derive the generic optimization criterion for the personalized ranking task.

$$
\begin{aligned}
\textrm{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j) \in D} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j) \in D} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j) \in D} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$


where $D \stackrel{\textrm{def}}{=} \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ is the training set, with $I^+_u$ denoting the items the user $u$ liked, $I$ denoting all items, and $I \backslash I^+_u$ indicating all other items excluding items the user liked. $\hat{y}_{ui}$ and $\hat{y}_{uj}$ are the predicted scores of the user $u$ to item $i$ and $j$, respectively. The prior $p(\Theta)$ is a normal distribution with zero mean and variance-covariance matrix $\Sigma_\Theta$. Here, we let $\Sigma_\Theta = \lambda_\Theta I$.

![Illustration of Bayesian Personalized Ranking](../img/rec-ranking.svg)

:begin_tab:`mxnet`
We will implement the base class `mxnet.gluon.loss.Loss` and override the `forward` method to construct the Bayesian personalized ranking loss. We begin by importing the Loss class and the np module.
:end_tab:

:begin_tab:`pytorch`
We will subclass `nn.Module` and implement the BPR loss in its `forward` method.
:end_tab:

```{.python .input #ranking-bayesian-personalized-ranking-loss-and-its-implementation-1  n=5}
#@tab mxnet
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input #ranking-bayesian-personalized-ranking-loss-and-its-implementation-1  n=5}
#@tab pytorch
import torch
from torch import nn
```

The implementation of BPR loss is as follows.

```{.python .input #ranking-bayesian-personalized-ranking-loss-and-its-implementation-2  n=2}
#@tab mxnet
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

```{.python .input #ranking-bayesian-personalized-ranking-loss-and-its-implementation-2  n=2}
#@tab pytorch
#@save
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, positive, negative):
        distances = positive - negative
        loss = -torch.sum(torch.log(torch.sigmoid(distances)), dim=0,
                          keepdim=True)
        return loss
```

## Hinge Loss and its Implementation

The Hinge loss for ranking has a different form from the standard hinge loss that is often used in classifiers such as SVMs.  The loss used for ranking in recommender systems has the following form.

$$
 \sum_{(u, i, j) \in D} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

where $m$ is the safety margin size. It aims to push negative items away from positive items. Similar to BPR, it aims to optimize for relevant distance between positive and negative samples instead of absolute outputs, making it well suited to recommender systems.

```{.python .input #ranking-hinge-loss-and-its-implementation  n=3}
#@tab mxnet
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

```{.python .input #ranking-hinge-loss-and-its-implementation  n=3}
#@tab pytorch
#@save
class HingeLossbRec(nn.Module):
    def __init__(self):
        super(HingeLossbRec, self).__init__()

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = torch.sum(torch.clamp(-distances + margin, min=0))
        return loss
```

These two losses are interchangeable for personalized ranking in recommendation.

## Summary

- There are three types of ranking losses available for the personalized ranking task in recommender systems, namely, pointwise, pairwise and listwise methods.
- The two pairwise losses, Bayesian personalized ranking loss and hinge loss, can be used interchangeably.

## Exercises

- Are there any variants of BPR and hinge loss available?
- Can you find any recommendation models that use BPR or hinge loss?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:

<!-- slides -->

::: {.slide}
Most real-world recommender data is **implicit** —
clicks, watches, purchases. There are no explicit ratings,
and the unobserved (user, item) pairs are a *mix* of
"didn't like it" and "haven't seen it yet". MSE on a 0/1
target is wrong.

Better framing: **personalized ranking** — given an
observed positive (user, $i$), the model should rank $i$
*above* sampled unobserved items. Treating every unobserved
pair as a literal negative target is usually misaligned with
ranking because exposure is missing-not-at-random.

Two pairwise losses for this:

- **BPR** (Bayesian Personalized Ranking, Rendle et al.
  2009) — log-sigmoid of score margin:
  $-\log \sigma(\hat r_{ui} - \hat r_{uj})$ for sampled
  negatives $j$.
- **Hinge** — max-margin variant:
  $\max(0, m - (\hat r_{ui} - \hat r_{uj}))$.

Both turn implicit feedback into pairwise comparisons; the
model learns to put positives above negatives.
:::

::: {.slide title="Training triples"}
For each user $u$, let $I_u^+$ be observed positives
(clicked, watched, bought) and sample negatives
$j \notin I_u^+$ from the item catalog. Training examples are
triples:

$$D = \{(u,i,j): i\in I_u^+,\, j\notin I_u^+\}.$$

The model never needs an absolute rating target. It only needs
the score gap

$$\Delta_{uij} = \hat r_{ui} - \hat r_{uj}.$$

Large positive gaps mean the positive item outranks the sampled
negative. The sampled negative is a training contrast, not proof
that the user would dislike the item.
:::

::: {.slide title="BPR loss"}
Sampled negatives $j$ per positive $(u, i)$; loss is
log-sigmoid of the score margin:

@ranking-bayesian-personalized-ranking-loss-and-its-implementation-1

. . .

@ranking-bayesian-personalized-ranking-loss-and-its-implementation-2
:::

::: {.slide title="Hinge loss"}
Hard-margin alternative — equivalent to a max-margin
classifier over score differences:

@ranking-hinge-loss-and-its-implementation
:::

::: {.slide title="BPR vs hinge"}
Both losses reward positive margins, but their gradients behave
differently:

$$\ell_\textrm{BPR}(\Delta) = -\log \sigma(\Delta), \qquad
  \ell_\textrm{hinge}(\Delta) = \max(0, m-\Delta).$$

- BPR keeps a smooth, nonzero gradient for every sampled pair.
- Hinge stops updating once the margin is satisfied.
- The most important implementation choice is often the negative
  sampler, not the algebraic form of the loss.
:::

::: {.slide title="Recap"}
- Personalized ranking turns implicit feedback into a
  pairwise comparison task.
- BPR: log-sigmoid of the (positive - negative) score
  margin. Soft, differentiable, the most-used choice.
- Hinge: hard margin; sometimes better with very
  imbalanced data.
- Negative sampling is the implementation hammer that
  makes either loss tractable on large item catalogs.
:::
