# Recommender Systems
:label:`chap_recsys`


**Shuai Zhang** (*Amazon*), **Aston Zhang** (*Amazon*), and **Yi Tay** (*Google*)

A catalog may contain millions of items, while any one user will inspect only a
small fraction of them. A recommender system uses the interactions that have
been observed---ratings, clicks, purchases, or viewing histories---to score or
rank the remaining candidates for that user. Unlike search, which begins with
an explicit query, recommendation must infer a useful ranking from partial and
selectively observed behavior.

This setting creates two recurring difficulties. First, an unobserved
user--item pair is not an observed dislike: the user may never have encountered
the item. Second, the evaluation protocol determines the question being asked.
A random interaction split measures a warm-start interpolation problem, whereas
a chronological split asks whether past behavior predicts a later choice.

The chapter develops three families of tasks. Rating models predict explicit
scores; ranking models learn from implicit interactions; and feature-rich
models predict events such as clicks from user, item, and contextual fields.
The progression from matrix factorization to AutoRec, NeuMF, Caser,
factorization machines, and DeepFM shows how the input data and objective---not
the presence of a neural network alone---determine what a model can learn.

The models can be compared by the evidence supplied to them and the question
used for evaluation:

| Model | Input | Training objective | Assumption about unobserved pairs | Evaluation used here |
|:--|:--|:--|:--|:--|
| MF | Explicit ratings | Masked squared error | Omitted from the loss | RMSE on a random warm-start holdout |
| AutoRec | Partially observed rating vectors | Masked reconstruction error | Omitted from the loss | RMSE on the same holdout |
| NeuMF | User, observed item, sampled item | BPR pairwise loss | Sampled unobserved items act as comparisons | Hit rate and AUC on a chronological holdout |
| Caser | Ordered recent items and user ID | BPR pairwise loss | Same sampling assumption as NeuMF | Next-item ranking after chronological holdout |
| FM | Categorical impression fields | Binary log loss | Every row is an observed impression with a click label | Held-out log loss |
| DeepFM | The same impression fields | Binary log loss | The same labeled-impression assumption as FM | Held-out log loss |

```toc
:maxdepth: 2

recsys-intro
movielens
mf
autorec
ranking
neumf
seqrec
ctr
fm
deepfm
```
