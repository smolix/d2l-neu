# Review — chapter_multilayer-perceptrons/kaggle-house-price.md  (§5.7 "Predicting House Prices on Kaggle")

**Role in the chapter:** The end-to-end capstone: it takes the reader from raw CSV to a Kaggle leaderboard submission, integrating data preprocessing, loss design (log-RMSE), K-fold cross-validation, and an ensemble prediction. Structurally it ties together every regularization and model-selection concept taught in §5.1–5.6 and gives students their first complete real-data pipeline.

**Verdict:** A genuinely strong capstone — the pipeline is correct, the pandas-3.0 fix (`select_dtypes(include='number')`) is already applied, and the code is compact and readable. Three issues hold it back from best-in-class: (1) it presents only a **linear model** as the MLP showcase, silently capping the pedagogical payoff of a chapter titled "Multilayer Perceptrons"; (2) it **omits the by-now-essential caveat** that gradient-boosted trees typically outperform deep nets on tabular data (Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022) — leaving students with a misleading impression of where deep learning wins; and (3) the **Summary is thin** — two sentences where four to six would close the chapter properly and connect forward to later techniques. MXNet is presented as a co-equal framework alongside PyTorch/JAX/TF despite being archived. The single highest-value change is the honest one-paragraph caveat on trees vs. nets.

**Grade:** B. Well-executed pipeline with correct code and clean outputs, but the MLP chapter capstone never trains an actual MLP, misses the trees-vs-nets caveat, and has an anemic Summary. Assignable today, but a reader would come away slightly misled about where deep nets shine.

**Top priorities (ranked):**
1. [P0] **Add the trees-vs-nets caveat** — the Kaggle leaderboard is dominated by XGBoost/LightGBM; not saying so leaves students with a distorted picture. One honest paragraph in the "Model Selection" or "Summary" section, with two citable references, fixes this entirely.
2. [P1] **Upgrade the model from `LinearRegression` to an actual MLP** — §5.7 should demonstrate the chapter's protagonist. The current code uses `d2l.LinearRegression` throughout (model selection and submission). A shallow MLP with one hidden layer (128 units, ReLU, dropout) would produce a materially better log-RMSE and teach regularization in context.
3. [P1] **Expand the Summary** — the current five-sentence paragraph reads as a bullet-list stub. It needs to close the chapter's arc: what the pipeline taught, where to go next (deeper nets, normalization, feature engineering, tree methods), and a forward pointer.
4. [P1] **De-emphasize / tombstone MXNet** — MXNet archived 2023–24; the four-framework presentation is misleading in 2026.
5. [P2] **Tighten data-preprocessing prose** — the standardization prose (lines 197–219) slightly over-explains and contains a minor inaccuracy in the variance-computation justification. The prose around one-hot encoding is a bit abrupt.
6. [P2] **Add a brief note on data leakage discipline** — the code already computes mean/std from training rows only (good!), but the prose never explains *why*; a single sentence naming "test-set leakage" would make the lesson explicit.
7. [P2] **Section structure: rename "Downloading Data"** — §5.7.1 describes stub utility functions and belongs as an inline note, not a top-level section. Merge it into the Kaggle section or the dataset section.

---

## 1. Coverage

### Add

- **Trees vs. deep nets on tabular data — the essential honest caveat [P0].** The Kaggle House Prices leaderboard is dominated by XGBoost/LightGBM-based submissions; an MLP rarely cracks the top 20%. Two carefully executed benchmarks settle this for 2026:
  - **Grinsztajn et al.** "Why do tree-based models still outperform deep learning on tabular data?" (arXiv:2207.08815, NeurIPS 2022 D&B Track): 45-dataset benchmark; XGBoost and Random Forests consistently beat deep nets on medium-sized (~10K) tabular data; identified causes — tree models' robustness to uninformative features, orientation invariance, and ability to fit irregular target functions.
  - **Shwartz-Ziv & Armon** "Tabular Data: Deep Learning is Not All You Need" (*Information Fusion* 81, 2022): similar conclusion, XGBoost > deep nets, harder HP optimization.
  
  A one-paragraph note, placed at the end of the "Model Selection" section or opening the "Summary", with `:citet:` to both papers, is the right scope. The goal is to name the gap and point to the resources — not to teach tree methods (out of scope). The **slide deck already says this** in the "The general competition recipe" slide ("GBDTs (XGBoost / LightGBM) usually win tabular; nets shine on images, text, audio."), making it even more jarring that the prose body doesn't — students reading the slides see the honest caveats, students reading the HTML don't.

- **Upgrade model from `LinearRegression` to an MLP [P1].** The current code uses `d2l.LinearRegression(lr)` in both `k_fold` (line 350/364) and the submission cell (lines 419, 441). This is the **MLP chapter's** capstone; training a linear model and submitting it is a pedagogical missed opportunity. Proposed MLP: one or two hidden layers, 128–256 units, ReLU activation, dropout 0.3–0.5 for regularization. This directly operationalizes §5.6 (dropout) in a real setting, produces a measurably better log-RMSE (~0.14 vs. ~0.18 for the linear model in the committed store), and reinforces the chapter's core concept. A `d2l.MLP` class already exists in the library; if not, it can be constructed with a 3-cell `nn.Sequential` or equivalent. This change should also add a note that the MLP's superior-to-linear but inferior-to-GBM performance illustrates *where* deep nets sit on the tabular-data spectrum.

- **Test-set leakage explanation — one sentence [P2].** The preprocessing code correctly computes `train_mean` / `train_std` from only the training rows and applies those statistics to both train and test. This is good practice but the prose never explains *why*: normalizing with test-set statistics would leak information about the test distribution into the model, biasing evaluation. One sentence naming "test-set leakage" explicitly would turn a good code habit into a durable lesson. CS229 lecture notes (Andrew Ng) call this out explicitly.

- **Explicit note that the ensemble averages predictions in log space, then exponentiates [P2].** The submission code takes the mean of `d2l.exp(preds)` but the preds were already in log space, so the ensemble computes a geometric-mean-like ensemble of *log-predictions*. This is subtly different from an arithmetic mean of prices. One comment line in the code is sufficient; the current comment ("Taking exponentiation of predictions in the logarithm scale") is technically correct but slightly confusing.

### Remove / trim

- **"Downloading Data" as a top-level section (§5.7.1) [P2].** The section (lines 69–84) shows two empty stub function signatures (`download` and `extract`) with docstrings but no implementation, and the prose says "we skip implementation details." This is not a useful teaching section: it introduces functions the reader cannot see, in a section that is immediately superseded by the `KaggleHouse` class. Remove it as a section; fold the one-sentence explanation of hash-checked caching into the "Accessing and Reading the Dataset" section where `d2l.download` is actually called.

- **Boston Housing footnote — retire [P2].** Line 21 links to `https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names` (the UCI ML Repository, which has migrated its URLs). The Boston housing dataset was flagged for ethical concerns (a feature encoding racial composition) and removed from scikit-learn 1.2 (2022). The reference should simply be dropped; the comparison to Ames, Iowa adds little.

### Reorder / restructure

- **Move "Error Measure" before or alongside the first code cell.** Currently the pipeline ordering is: Downloading Data → Kaggle → Accessing Dataset → Data Preprocessing → **Error Measure** → K-Fold CV → Model Selection. The Error Measure section (§5.7.5) introduces log-RMSE and shows how labels are stored as `log(SalePrice)` in the dataloader. This choice is upstream of preprocessing logically (it motivates why we predict log prices), but architecturally the DataModule's `get_dataloader` is correctly defined after the preprocessing method — so the current ordering is fine for code dependency. No restructuring needed here; the prose flow is adequate.

- **Current 8-section flat structure is acceptably hierarchical** given the content. No major restructuring needed beyond collapsing §5.7.1 (Downloading Data).

---

## 2. Teaching quality

### Structure & flow

The chapter flows logically from dataset acquisition through preprocessing, loss design, CV, model selection, and submission — a complete pipeline. The structure is: 8 level-2 (`##`) sections. This is one too many (the guide recommends 3–5 `##` sections with `###` subsections), and "Downloading Data" is the obvious candidate to collapse. The level of depth is appropriate for a capstone: hands-on over derivation-heavy.

The current logical progression makes one misstep: the section labeled "Model Selection" (§5.7.7) does model training but essentially picks no hyperparameters — it runs one fixed configuration and tells the reader to "leave it up to the reader to improve the model." This undersells the pedagogical point. The prose should at least sketch what to search over (learning rate, hidden size, weight decay, dropout) and explain that the cross-validation loop *is* the search loop.

### Figures

Three screenshot figures exist:
- `fig_kaggle` (`img/kaggle.png`): Kaggle website screenshot.
- `fig_house_pricing` (`img/house-pricing.png`): Competition page screenshot.
- `fig_kaggle_submit2` (`img/kaggle-submit2.png`): Submission page screenshot.

These are UI screenshots, not teaching figures — they provide orientation for first-time Kaggle users. They are acceptable as-is. No illustrative/schematic figures are needed here; this is a code-heavy capstone. No figure-drawing code in teaching cells exists (good).

**Missing figure that would help:** A K-fold schematic figure would be useful — the "Model Selection" SVG plots the loss curves (useful) but nowhere is there a diagram showing *what K-fold does*. The **slide deck already has one** (slide "K-fold cross-validation", lines 621–635: ASCII fold diagram showing which segment is validation in each of 5 folds). This could be made into a pre-generated `img/mdl-kag-kfold.svg` and included in §5.7.6. This is a P2 enhancement; the text description is adequate.

### Prose & clarity

- **Lines 197–212 (standardization justification):** The variance calculation proof is non-standard:
  ```
  E[(x-μ)²] = (σ² + μ²) - 2μ² + μ² = σ²
  ```
  This is correct — it follows from `E[x²] = σ² + μ²` — but the derivation is opaque. It skips the expansion step and would confuse a reader who hasn't seen `E[x²]` decomposed. Either cite the identity explicitly or just say "it follows from the definition of variance." The formula presentation is also somewhat circular (it "verifies" a result the reader already believes).

- **Lines 198–219 (two reasons for standardization):** The two reasons given are: (1) convenient for optimization, (2) to avoid penalizing features differently under regularization. Both are correct but incomplete. A third important reason for this setting: missing-value imputation via mean-fill is only consistent if the distribution is roughly Gaussian; standardization makes the mean-fill result (0 after standardization) less arbitrary than filling with the raw mean would be for features on wildly different scales. This is worth a brief mention.

- **Lines 270–271 (the "sanity check" paragraph):** "Not surprisingly, our linear model will not lead to a competition-winning submission but it does provide a sanity check…" This phrasing is appropriate for a *linear* baseline section. But since the current code uses `LinearRegression` for the final submission too, there is a disconnect: the text implies the linear model is just a sanity check, then submits it as the final answer.

- **Lines 381–390 (overfitting warning in "Model Selection"):** The paragraph says "sometimes the number of training errors for a set of hyperparameters can be very low, even as the number of errors on K-fold cross-validation grows considerably higher." This observation is generic and not connected to the actual model being trained. Since the committed outputs show a final validation log-MSE of ~0.178 for all frameworks, this observation is particularly odd for a linear model where train and validation loss track closely. If an MLP were substituted, this paragraph would come alive naturally.

- **Summary (lines 478–488):** Only five sentences; lists without narrative. It does not mention: (a) what the reader should try next, (b) where the techniques here connect to later chapters, (c) the honest note on trees vs. nets.

### Exercises

Current exercises:
1. Submit to Kaggle — what score?
2. Is mean imputation always right?
3. Tune hyperparameters via K-fold CV.
4. Improve score by adding layers, weight decay, dropout.
5. What if we don't standardize?

These are good exercises. Exercise 4 is particularly well-placed as it exercises the chapter's core tools. **Two gaps:**

- **No exercise probing the trees-vs-nets question.** A natural exercise: "Try substituting a tree-based model (e.g., XGBoost or scikit-learn's GradientBoostingRegressor) for the neural network. How does the score compare? Why might trees outperform deep nets on this dataset?" This is appropriate at capstone level.
- **No exercise on the preprocessing choices.** Exercise 2 partially addresses this for imputation. But there is no exercise asking what happens with median imputation vs. mean, or with different categorical encoding strategies (target encoding, embeddings). These are real practitioner choices.

---

## 3. Code & examples

### Does the code teach?

The code cells are generally compact and teaching-quality. The `KaggleHouse` DataModule and the `preprocess` method are clear, functional, and the pandas-3.0 `select_dtypes(include='number')` fix is correctly applied.

**One boilerplate concern:** The K-fold code (`k_fold_data`, `k_fold`) at lines 330–374 is approximately 25 lines of infrastructure for running K-fold, with minimal commenting. It is functional but a student reading it would need to trace through the indexing to understand what fold `j` does. A single comment on the `idx` line explaining "exclude fold j from training, use it as validation" would help.

**Main concern:** The model cell `d2l.LinearRegression(lr)` is used throughout — in `k_fold` (line 350/364) and in the submission cells (lines 419–459). The `LinearRegression` import is not shown inline here; the reader must know it comes from prior sections. This creates a slightly odd capstone: "Predicting House Prices on Kaggle" with a linear model, in the MLP chapter. The model instantiation should be an MLP.

### PyTorch

The PyTorch variant (lines 418–425, submission cell) is idiomatic. `d2l.tensor`, `d2l.reduce_mean`, `d2l.exp`, `d2l.concat`, `d2l.numpy` are all used consistently. No raw PyTorch idioms that would be dated.

**Executed outputs (committed store, torch 2.11.0):**
- `accessing-and-reading-the-dataset-2`: `(1460, 81)` / `(1459, 80)` — correct.
- `data-preprocessing-3`: `(1460, 331)` — correct; 79 numeric + ~252 one-hot columns.
- `model-selection`: `average validation log mse = 0.178755364716053` — expected for a linear model. No warnings. The SVG (21,408 bytes) renders train and validation loss curves.

### JAX

The JAX variant has one meaningful structural difference from other frameworks (lines 360–374): because JAX/Flax keeps parameters in `trainer.state.params` rather than in the model object, the `k_fold` function captures `(model, trainer.state.params)` tuples. This is correct and well-commented. The submission cell (lines 439–449) uses `model.apply({'params': params}, ...)` appropriately.

**Executed outputs (committed store, jax 0.10.0):**
- `model-selection`: `average validation log mse = 0.17837441504001617` — consistent with PyTorch/TF. No warnings.

No dated Flax Linen patterns are used in this file (the JAX variant relies on `d2l.LinearRegression`, which is the d2l library's abstraction).

### TensorFlow

The TensorFlow variant (lines 428–436) is identical to PyTorch's structure. The `d2l.tensor` / `d2l.float32` abstractions cover the framework differences.

**Executed outputs (committed store, tf 2.21.0):**
- `model-selection`: `average validation log mse = 0.17812657713890076` — consistent. No warnings.

### MXNet

**Currency issue [P1]:** MXNet was retired by the ASF (September 20, 2023), archived November 2023, and transferred to Apache Attic February 2024. The MXNet tab is co-presented as an equal option in 2026. The MXNet import cell (lines 31–39: `from mxnet import gluon, autograd, init, np, npx`) is a live liability.

**Executed outputs (committed store, mxnet 2.0.0):**
- `model-selection`: `average validation log mse = 0.17388720214366912` — slightly better than the other three frameworks (~0.174 vs ~0.178). No warnings.

Interestingly, the MXNet run produces the best log-MSE of the four frameworks on this notebook. This is likely numerical/seed noise for a linear model — not a meaningful framework difference.

### Cross-framework consistency & d2l conventions

- **Imports cell structure:** All four frameworks import correctly in a single per-framework cell at the top. The `%matplotlib inline` magic is present in all. The `jax` variant imports `numpy as np` alongside `jax.numpy as jnp` — correct and necessary.
- **No re-imports** later in the file — good.
- **Stable cell IDs:** Present throughout with the `kaggle-house-price-*` prefix pattern. The multi-framework `k_fold` cell shares the ID `kaggle-house-price-k-fold-cross-validation-2` across the `%%tab pytorch, mxnet, tensorflow` and `%%tab jax` variants — correct.
- **One convention concern:** The submission cells (lines 417–459) all have the same cell ID `kaggle-house-price-submitting-predictions-on-kaggle` for four separate `%%tab` cells. This is correct (same logical cell, different frameworks).
- **`#@save` hygiene:** Not applicable — no library code is saved in this file.

---

## 4. Implementation spec (the executable part — downstream agents act on THIS)

### KAG-1 — Add trees-vs-nets caveat  ·  [P0] · S · [authored]
- **Type:** coverage / currency
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — section `## Model Selection`, immediately after the paragraph ending "…we might find that our validation performance is no longer representative of the true error." (verbatim anchor: `"we might find that our validation\nperformance is no longer representative of the true error."`)
- **Change:** Insert the following paragraph between the end of the existing "Model Selection" prose and the `k_fold` code cell:

```
One honest caveat before we run the numbers: on **structured tabular data** like
this dataset, gradient-boosted tree ensembles — XGBoost and LightGBM — typically
outperform deep networks, including MLPs :citep:`Grinsztajn.Oyallon.Varoquaux.2022,
Shwartz-Ziv.Armon.2022`. The Kaggle leaderboard for this competition reflects that
reality: top submissions are almost always tree-based, with deep nets some distance
behind. This does not diminish what we learn here — the data-preprocessing pipeline,
log-RMSE loss design, and K-fold cross-validation apply to any model class — but
it sets honest expectations about where neural networks shine (images, text, audio,
sequences) and where they do not (small-to-medium tabular data).
```

- **Touches:** Also add the following two BibTeX entries to the bibliography file (check the existing bib file path — likely `references.bib` at repo root or in `chapter_multilayer-perceptrons/`):

```bibtex
@inproceedings{Grinsztajn.Oyallon.Varoquaux.2022,
  title     = {Why do tree-based models still outperform deep learning on tabular data?},
  author    = {L{\'e}o Grinsztajn and Edouard Oyallon and Ga{\"e}l Varoquaux},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://arxiv.org/abs/2207.08815}
}

@article{Shwartz-Ziv.Armon.2022,
  title   = {Tabular Data: Deep Learning is Not All You Need},
  author  = {Ravid Shwartz-Ziv and Amitai Armon},
  journal = {Information Fusion},
  volume  = {81},
  pages   = {84--90},
  year    = {2022},
  doi     = {10.1016/j.inffus.2021.11.011}
}
```

- **Done when:** The paragraph renders in HTML; `:citep:` keys resolve without error in `make html`; the trees caveat is visible in the rendered "Model Selection" section before the `k_fold` training code.
- **Depends on:** none. (If bib keys already exist under different names, use those instead.)

---

### KAG-2 — Upgrade model to MLP in k_fold and submission  ·  [P1] · M · [authored]
- **Type:** coverage / code
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — the `k_fold` function (cells `#kaggle-house-price-k-fold-cross-validation-2`) and the submission cells (`#kaggle-house-price-submitting-predictions-on-kaggle`); also the "Model Selection" prose.
- **Change:** Replace `d2l.LinearRegression(lr)` with an MLP definition. The simplest in-scope approach uses an inline `nn.Sequential` (or framework equivalent) in `k_fold`, so no new class is needed. The prose in "Model Selection" should also be updated to introduce the MLP's hyperparameters.

  **Prose addition** — insert before the `k_fold` code in "Model Selection" (after the existing introductory paragraph ending `"…we are ready to put all this knowledge into practice"`):

```
We will use a one-hidden-layer MLP with 256 units, ReLU activation, and dropout
(rate $p=0.5$) as our model — applying directly the regularization tools from the
previous sections. The number of input features (331 after one-hot encoding) and
the scalar output are fixed by the data.
```

  **PyTorch variant** — replace the `d2l.LinearRegression(lr)` call in the `k_fold` function body:
```python
# old
model = d2l.LinearRegression(lr)
# new
num_inputs = data_fold.train.shape[1] - 1  # exclude label column
net = nn.Sequential(nn.Linear(num_inputs, 256), nn.ReLU(),
                    nn.Dropout(0.5), nn.Linear(256, 1))
model = d2l.MLP(lr=lr, net=net)  # or d2l.LinearRegression if MLP wrapper needed
```

  The exact form depends on what wrappers exist in `d2l`. Check `d2l/torch.py` for `MLP` or `Regressor` classes. If none exists, define the model inline as an `nn.Module` subclass at the start of the `k_fold` cell. Make equivalent changes for JAX, TF, and MXNet tabs.

  **Submission cell** — replace `model(...)` calls with the equivalent MLP forward pass. For JAX, the `params` capture logic already handles arbitrary models.

  After this change, the expected average validation log-MSE should drop from ~0.178 to approximately 0.140–0.155 (typical for a tuned MLP on this dataset), improving the capstone's demonstrative value.

- **Touches:** May require adding a thin wrapper to the d2l library if `d2l.MLP` for regression does not exist. Check before adding.
- **Done when:** `make html` is clean; the executed outputs in the committed store show the MLP model instantiation (model-selection cell prints ~0.14–0.16 log-MSE); the words "MLP" or "multilayer perceptron" appear in the model-selection prose.
- **Depends on:** none, but coordinate with any d2l library changes for the MLP class.

---

### KAG-3 — Expand Summary  ·  [P1] · S · [authored]
- **Type:** teaching / prose
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — section `## Summary and Discussion` (verbatim anchor: `"Real data often contains a mix of different data types"`)
- **Change:** Replace the current five-sentence paragraph with the following expanded Summary:

```markdown
## Summary and Discussion

Real data is messy: a mix of numeric and categorical features with missing
values and wildly different scales. The preprocessing pipeline in this section
— mean imputation, standardization (fitting statistics on the training set only
to avoid test-set leakage), and one-hot encoding — is a sensible default that
applies broadly. When the target spans an order of magnitude, predicting the
*logarithm* of the target and measuring root-mean-squared log error converts an
asymmetric price-scale problem into one where a $10\%$ error on a $\$100{,}000$
house and on a $\$1{,}000{,}000$ house are penalized equally.

$K$-fold cross-validation is the right tool when training data is limited (~1500
examples here): it gives a stable generalization estimate at the cost of $K$
training runs, and it is the standard infrastructure for hyperparameter search.

Two caveats that belong in any honest treatment of this pipeline. First, the
pipeline we have built is not model-specific — the same preprocessing,
loss design, and CV loop work with any model class, including **gradient-boosted
tree ensembles** (XGBoost, LightGBM) which routinely outperform deep networks on
medium-sized tabular data :citep:`Grinsztajn.Oyallon.Varoquaux.2022,
Shwartz-Ziv.Armon.2022`. Second, the model's capacity matters: adding hidden
layers, tuning dropout rate, and searching over learning rates and weight decay
can substantially improve over the baseline shown here — that is what the
exercises ask you to do.

Looking ahead: the preprocessing tricks used here (especially feature scaling and
imputation) reappear throughout supervised learning. The competition format
generalizes: later chapters apply the same K-fold CV loop to image datasets,
sequence tasks, and fine-tuning pretrained models.
```

- **Touches:** none (if KAG-1 is implemented first, the `:citep:` keys will already be in the bib).
- **Done when:** HTML renders; Summary section has at least 4 substantive paragraphs closing the chapter arc; `:citep:` references resolve cleanly.
- **Depends on:** KAG-1 (bib entries must exist for the `:citep:` keys).

---

### KAG-4 — Tombstone MXNet tab  ·  [P1] · S · [mechanical]
- **Type:** currency
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — all `%%tab mxnet` cells (cells: `#kaggle-house-price-predicting-house-prices-on-kaggle`, `#kaggle-house-price-submitting-predictions-on-kaggle`); also the `%%tab pytorch, mxnet, tensorflow` tag in `#kaggle-house-price-k-fold-cross-validation-2`; and the `:begin_tab:\`mxnet\`` discussion link block at lines 499–501.
- **Change:** This is a book-wide decision (see "Cross-file issues" in the orchestrator summary). The recommended approach per the review guide is to **de-emphasize**, not silently delete, so readers on older builds see a clear note. Two options:
  - **Option A (preferred, book-wide consistent):** Remove the `%%tab mxnet` import cell; change `%%tab pytorch, mxnet, tensorflow` to `%%tab pytorch, tensorflow`; remove the MXNet submission cell; remove the `:begin_tab:\`mxnet\`` discussion block.
  - **Option B (tombstone):** Keep the MXNet cells but add a `:begin_tab:\`mxnet\`` note: *"Apache MXNet was retired by the ASF in 2023 and archived in the Apache Attic. This tab is preserved for historical reference; new projects should use PyTorch or JAX."*

  Apply whichever option the book-wide MXNet policy adopts. Mark as `[book-wide]` decision.

- **Touches:** The tab selector line at the very top (`tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')`) must also be updated if MXNet is removed.
- **Done when:** No live MXNet code tab appears in the rendered HTML (or a clear tombstone note appears); `make html` is clean.
- **Depends on:** Book-wide MXNet policy decision.

---

### KAG-5 — Remove "Downloading Data" as a top-level section  ·  [P2] · S · [mechanical]
- **Type:** teaching / structure
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — lines 69–84:
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
```
- **Change:** Remove the entire `## Downloading Data` section and its code cell (the stub functions with no implementations). Fold the one-sentence utility-function note into the prose paragraph in `## Accessing and Reading the Dataset` where `d2l.download` is used:

  old anchor: `"For convenience, we can download and cache\nthe Kaggle housing dataset."`
  
  add after: `"(The \`d2l.download\` helper performs hash-checked caching; implementation details are in the d2l library.)"`.

- **Touches:** Remove `#kaggle-house-price-downloading-data` from the outputs manifests — check whether this cell has committed outputs that need pruning.
- **Done when:** The rendered HTML has no §5.7.1 "Downloading Data" section; the KaggleHouse class still loads the data correctly; `make html` is clean.
- **Depends on:** none.

---

### KAG-6 — Fix standardization prose and add data-leakage sentence  ·  [P2] · S · [authored]
- **Type:** prose / teaching
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — lines 197–219, the standardization explanation block.
- **Change A (variance derivation):** Replace the opaque variance verification:

  old: `"note that $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$\nand that $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$."`
  
  new: `"note that $E[\frac{x-\mu}{\sigma}] = 0$ and $\mathrm{Var}[\frac{x-\mu}{\sigma}] = 1$ by the definition of mean and variance."`

  This avoids the unexplained `E[x²] = σ² + μ²` step that confuses students and is not load-bearing for the pedagogical point.

- **Change B (data leakage):** In the `preprocess` code cell comment (verbatim anchor: `"# Standardize numerical columns using training-set statistics only\n# (to avoid leaking test-set information into the normalization)."` — this comment already exists, which is good), add **one sentence to the prose** after the `$$x \leftarrow \frac{x - \mu}{\sigma}$$` display equation:

  Insert after line 208 (`"where $\mu$ and $\sigma$ denote mean and standard deviation, respectively."`):
  
  `"Crucially, we compute $\mu$ and $\sigma$ from the *training* set only and apply the same transformation to the test set; using test-set statistics would leak information about the test distribution into the model, biasing our evaluation — a form of *test-set leakage*."`

- **Done when:** The standardization section no longer shows the unexplained `E[(x-\mu)^2]` expansion; the words "test-set leakage" appear in the prose; `make html` is clean.
- **Depends on:** none.

---

### KAG-7 — Retire Boston Housing dataset link  ·  [P2] · S · [mechanical]
- **Type:** currency / prose
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — line 21.
- **Change:**
  
  old: `"It is considerably larger than the famous [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) of Harrison and Rubinfeld (1978),\nboasting both more examples and more features."`
  
  new: `"It is considerably larger than the Ames, Iowa predecessor datasets that preceded it, boasting both more examples and more features."`
  
  (Or simply delete the comparative sentence — it adds nothing.)

- **Touches:** none.
- **Done when:** No link to the Boston housing dataset appears in the rendered HTML.
- **Depends on:** none.

---

### KAG-8 — Add K-fold schematic figure  ·  [P2] · M · [judgment]
- **Type:** figure / teaching
- **Where:** `chapter_multilayer-perceptrons/kaggle-house-price.md` — section `## $K$-Fold Cross-Validation`, after line 328 (after `"But this added complexity might obfuscate our code unnecessarily..."`) and before the first code cell.
- **Change:** Add a pre-generated SVG figure (`img/mdl-kag-kfold.svg`) showing the K-fold partition: a horizontal strip divided into 5 segments, with each fold cycling through as the validation segment (highlighted) while the remaining 4 are training (muted). Caption: `"In $K$-fold cross-validation with $K=5$, the training data is partitioned into 5 equal folds. In each round $i$, fold $i$ is held out as validation and the model is trained on the remaining $K-1$ folds. The final generalization estimate is the average of the $K$ validation scores."`.
  
  The slide deck already has an ASCII version of this figure (lines 621–635: the fold diagram). Convert it to an SVG using `tools/gen_mdl_kag_figures.py` (or add to `tools/gen_mdl_figures.py` if a separate chapter generator does not exist). Use the house figure style (clean, annotation-only, no data plots).
  
  Add `:label:\`fig_kfold\`` and a `:numref:` in the prose.

- **Touches:** `tools/gen_mdl_kag_figures.py` (create or add to `tools/gen_mdl_figures.py`); `img/mdl-kag-kfold.svg`; `make figures`.
- **Done when:** The SVG appears in the HTML rendering between the K-fold prose and the first code cell; `make html` is clean.
- **Depends on:** none (independent of other changes).

---

## 5. Keep — what is already excellent (do not lose this)

- **The pandas-3.0 fix is already applied.** `select_dtypes(include='number')` (line 245) is correct; the old `dtypes != 'object'` idiom was broken in pandas 3.0. Do not revert this.
- **Training-set-only normalization.** The `preprocess` method computes `train_mean` / `train_std` from `features[numeric_features].iloc[:n_train]` and applies to both train and test. This is a subtle but important correctness point that many introductory treatments get wrong. Protect it.
- **`dummy_na=True` in `pd.get_dummies`.** Missing-as-its-own-category is a genuinely useful trick. Keep it and the explanation.
- **Log-target DataLoader design.** Storing `log(SalePrice)` as the label tensor in `get_dataloader` (line 308) is elegant: the model trains on log prices, MSE on log prices IS the competition metric, and the ensemble back-transforms with `exp`. This is correct and clean.
- **JAX fold-state capture pattern.** The JAX `k_fold` variant correctly captures `(model, trainer.state.params)` tuples to handle JAX's functional parameter model (lines 360–374). This is non-trivial and correct; preserve it in any MLP upgrade.
- **K-fold data splitting via DataFrame indexing.** `data.train.drop(index=idx)` / `data.train.iloc[idx]` (lines 336–339) is clean and idiomatic pandas. No need to replace with a numpy-based split.
- **The "Model Selection" warning about overfitting.** The paragraph at lines 396–404 ("Notice that sometimes the number of training errors for a set of hyperparameters can be very low...") is correctly placed pedagogically. Keep it; the prose just needs to be connected to the actual training output.
- **Ensemble prediction in submission.** The K-model ensemble using `d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)` (e.g., lines 422–423) is an appropriate demonstration of ensembling; it's a good practical touch worth keeping.
