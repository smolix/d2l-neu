# The Dataset for Pretraining BERT
:label:`sec_bert-dataset`

To pretrain the BERT model as implemented in :numref:`sec_bert`,
we need to generate the dataset in the ideal format to facilitate
the two pretraining tasks:
masked language modeling and next sentence prediction.
On the one hand,
the original BERT model is pretrained on the concatenation of
two huge corpora BookCorpus and English Wikipedia (see :numref:`subsec_bert_pretraining_tasks`),
making it hard to run for most readers of this book.
On the other hand,
the off-the-shelf pretrained BERT model
may not fit for applications from specific domains like medicine.
Thus, it is getting popular to pretrain BERT on a customized dataset.
To facilitate the demonstration of BERT pretraining,
we use a smaller corpus WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`.

Comparing with the PTB dataset used for pretraining word2vec in :numref:`sec_word2vec_data`,
WikiText-2 (i) retains the original punctuation, making it suitable for next sentence prediction; (ii) retains the original case and numbers; (iii) is over twice larger.

```{.python .input #bert-dataset-the-dataset-for-pretraining-bert-1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input #bert-dataset-the-dataset-for-pretraining-bert-1}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

```{.python .input #bert-dataset-the-dataset-for-pretraining-bert-1}
#@tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import os
import random
```

```{.python .input #bert-dataset-the-dataset-for-pretraining-bert-1}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import os
import random
```

In the WikiText-2 dataset,
each line represents a paragraph where
space is inserted between any punctuation and its preceding token.
Paragraphs with at least two sentences are retained.
To split sentences, we only use the period as the delimiter for simplicity.
We leave discussions of more complex sentence splitting techniques in the exercises
at the end of this section.

```{.python .input #bert-dataset-the-dataset-for-pretraining-bert-2}
#@save
WIKITEXT_2_URL = ('https://huggingface.co/datasets/Salesforce/wikitext/'
                  'resolve/main/wikitext-2-v1/train-00000-of-00001.parquet')
WIKITEXT_2_SHA1 = '98ee727e59fcc34fddaadae93e15b1f8ed5561a4'

#@save
def _read_wiki(data_dir=None):
    import contextlib
    import io
    import pandas as pd
    with contextlib.redirect_stdout(io.StringIO()):
        fname = d2l.download(WIKITEXT_2_URL, folder='../data',
                              sha1_hash=WIKITEXT_2_SHA1)
    lines = pd.read_parquet(fname)['text'].tolist()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Defining Helper Functions for Pretraining Tasks

In the following,
we begin by implementing helper functions for the two BERT pretraining tasks:
next sentence prediction and masked language modeling.
These helper functions will be invoked later
when transforming the raw text corpus
into the dataset of the ideal format to pretrain BERT.

### Generating the Next Sentence Prediction Task

According to descriptions of :numref:`subsec_nsp`,
the `_get_next_sentence` function generates a training example
for the binary classification task.

```{.python .input #bert-dataset-generating-the-next-sentence-prediction-task-1}
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

The following function generates training examples for next sentence prediction
from the input `paragraph` by invoking the `_get_next_sentence` function.
Here `paragraph` is a list of sentences, where each sentence is a list of tokens.
The argument `max_len` specifies the maximum length of a BERT input sequence during pretraining.

```{.python .input #bert-dataset-generating-the-next-sentence-prediction-task-2}
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### Generating the Masked Language Modeling Task
:label:`subsec_prepare_mlm_data`

In order to generate training examples
for the masked language modeling task
from a BERT input sequence,
we define the following `_replace_mlm_tokens` function.
In its inputs, `tokens` is a list of tokens representing a BERT input sequence,
`candidate_pred_positions` is a list of token indices of the BERT input sequence
excluding those of special tokens (special tokens are not predicted in the masked language modeling task),
and `num_mlm_preds` indicates the number of predictions (recall 15% random tokens to predict).
Following the definition of the masked language modeling task in :numref:`subsec_mlm`,
at each prediction position, the input may be replaced by
a special “&lt;mask&gt;” token or a random token, or remain unchanged.
In the end, the function returns the input tokens after possible replacement,
the token indices where predictions take place and labels for these predictions.

```{.python .input #bert-dataset-generating-the-masked-language-modeling-task-1}
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

By invoking the aforementioned `_replace_mlm_tokens` function,
the following function takes a BERT input sequence (`tokens`)
as an input and returns indices of the input tokens
(after possible token replacement as described in :numref:`subsec_mlm`),
the token indices where predictions take place,
and label indices for these predictions.

```{.python .input #bert-dataset-generating-the-masked-language-modeling-task-2}
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Transforming Text into the Pretraining Dataset

Now we are almost ready to customize a `Dataset` class for pretraining BERT.
Before that, 
we still need to define a helper function `_pad_bert_inputs`
to append the special “&lt;pad&gt;” tokens to the inputs.
Its argument `examples` contain the outputs from the helper functions `_get_nsp_data_from_paragraph` and `_get_mlm_data_from_tokens` for the two pretraining tasks.

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-1}
#@tab mxnet
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-1}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-1}
#@tab jax
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(jnp.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=jnp.int32))
        all_segments.append(jnp.array(segments + [0] * (
            max_len - len(segments)), dtype=jnp.int32))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(jnp.array(len(token_ids), dtype=jnp.float32))
        all_pred_positions.append(jnp.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=jnp.int32))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            jnp.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=jnp.float32))
        all_mlm_labels.append(jnp.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=jnp.int32))
        nsp_labels.append(jnp.array(is_next, dtype=jnp.int32))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-1}
#@tab tensorflow
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)))
        all_segments.append(segments + [0] * (max_len - len(segments)))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(float(len(token_ids)))
        all_pred_positions.append(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            [1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)))
        all_mlm_labels.append(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)))
        nsp_labels.append(int(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

Putting the helper functions for
generating training examples of the two pretraining tasks,
and the helper function for padding inputs together,
we customize the following `_WikiTextDataset` class as the WikiText-2 dataset for pretraining BERT.
By implementing the `__getitem__ `function,
we can arbitrarily access the pretraining (masked language modeling and next sentence prediction) examples 
generated from a pair of sentences from the WikiText-2 corpus.

The original BERT model uses WordPiece embeddings whose vocabulary size is 30000 :cite:`Wu.Schuster.Chen.ea.2016`.
The tokenization method of WordPiece is a slight modification of
the original byte pair encoding algorithm in :numref:`subsec_Byte_Pair_Encoding`.
For simplicity, we use the `d2l.tokenize` function for tokenization.
Infrequent tokens that appear less than five times are filtered out.

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-2}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-2}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-2}
#@tab jax
#@save
class _WikiTextDataset:
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-2}
#@tab tensorflow
#@save
class _WikiTextDataset:
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __len__(self):
        return len(self.all_token_ids)
```

By using the `_read_wiki` function and the `_WikiTextDataset` class,
we define the following `load_data_wiki` to download the WikiText-2 dataset
and generate pretraining examples from it.

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-3}
#@tab mxnet
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    paragraphs = _read_wiki()
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-3}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    paragraphs = _read_wiki()
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-3}
#@tab jax
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    paragraphs = _read_wiki()
    train_set = _WikiTextDataset(paragraphs, max_len)
    # Create an index array and shuffle it
    indices = list(range(len(train_set)))
    random.shuffle(indices)

    # Return a callable so each call yields a fresh iterator (one-shot
    # generators can't be re-entered for multi-epoch training).
    def data_iter():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size:
                continue
            batch = [train_set[idx] for idx in batch_indices]
            yield (jnp.stack([b[0] for b in batch]),
                   jnp.stack([b[1] for b in batch]),
                   jnp.stack([b[2] for b in batch]),
                   jnp.stack([b[3] for b in batch]),
                   jnp.stack([b[4] for b in batch]),
                   jnp.stack([b[5] for b in batch]),
                   jnp.stack([b[6] for b in batch]))
    return data_iter, train_set.vocab
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-3}
#@tab tensorflow
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    paragraphs = _read_wiki()
    train_set = _WikiTextDataset(paragraphs, max_len)
    AUTOTUNE = tf.data.AUTOTUNE
    train_iter = tf.data.Dataset.from_tensor_slices((
        tf.constant(train_set.all_token_ids, dtype=tf.int32),
        tf.constant(train_set.all_segments, dtype=tf.int32),
        tf.constant(train_set.valid_lens, dtype=tf.float32),
        tf.constant(train_set.all_pred_positions, dtype=tf.int32),
        tf.constant(train_set.all_mlm_weights, dtype=tf.float32),
        tf.constant(train_set.all_mlm_labels, dtype=tf.int32),
        tf.constant(train_set.nsp_labels, dtype=tf.int32),
    )).shuffle(buffer_size=10000).batch(batch_size).prefetch(AUTOTUNE)
    return train_iter, train_set.vocab
```

Setting the batch size to 512 and the maximum length of a BERT input sequence to be 64,
we print out the shapes of a minibatch of BERT pretraining examples.
Note that in each BERT input sequence,
$10$ ($64 \times 0.15$) positions are predicted for the masked language modeling task.

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-4}
#@tab mxnet, pytorch, tensorflow
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-4}
#@tab jax
batch_size, max_len = 512, 64
# train_iter is a callable returning a fresh iterator on each call.
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter():
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

In the end, let's take a look at the vocabulary size.
Even after filtering out infrequent tokens,
it is still over twice larger than that of the PTB dataset.

```{.python .input #bert-dataset-transforming-text-into-the-pretraining-dataset-5}
len(vocab)
```

## Summary

* Comparing with the PTB dataset, the WikiText-2 dataset retains the original punctuation, case and numbers, and is over twice larger.
* We can arbitrarily access the pretraining (masked language modeling and next sentence prediction) examples generated from a pair of sentences from the WikiText-2 corpus.


## Exercises

1. For simplicity, the period is used as the only delimiter for splitting sentences. Try other sentence splitting techniques, such as the spaCy and NLTK. Take NLTK as an example. You need to install NLTK first: `pip install nltk`. In the code, first `import nltk`. Then, download the Punkt sentence tokenizer: `nltk.download('punkt')`. To split sentences such as `sentences = 'This is great ! Why not ?'`, invoking `nltk.tokenize.sent_tokenize(sentences)` will return a list of two sentence strings: `['This is great !', 'Why not ?']`.
1. What is the vocabulary size if we do not filter out any infrequent token?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1496)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1496)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1496)
:end_tab:

<!-- slides -->

::: {.slide title="BERT Pretraining Data"}
The previous deck specified BERT's *model*. This one
specifies the *data*: how to turn raw text into the
(masked tokens, NSP label, segment IDs, valid lengths)
tuples that the pretraining loop expects.

We use **WikiText-2** — a small, readable Wikipedia
subset. Real BERT was pretrained on BookCorpus + English
Wikipedia (~3.3B tokens); the recipe is identical, just
scaled up.
:::

::: {.slide title="Read WikiText-2"}
WikiText-2 keeps punctuation, case, and numbers. The loader
returns paragraphs as sentence lists so NSP can sample adjacent
or random sentence pairs:

@bert-dataset-the-dataset-for-pretraining-bert-1

. . .

@bert-dataset-the-dataset-for-pretraining-bert-2
:::

::: {.slide title="Generating NSP examples"}
For each sentence, with probability 0.5 pair it with the
*next* sentence (`is_next=1`); otherwise pair with a
*random* sentence (`is_next=0`):

@bert-dataset-generating-the-next-sentence-prediction-task-1

. . .

@bert-dataset-generating-the-next-sentence-prediction-task-2
:::

::: {.slide title="Generating Masked LM labels"}
Pick 15% of token positions. For those:

- 80%: replace with `<mask>`.
- 10%: replace with a random token.
- 10%: leave the original (so the model can't tell which
  position was selected for prediction).

@bert-dataset-generating-the-masked-language-modeling-task-1

. . .

@bert-dataset-generating-the-masked-language-modeling-task-2
:::

::: {.slide title="Padding"}
Pad to the batch max length; track `valid_lens` for
attention masking; pad MLM labels with zero so the loss
ignores them:

@bert-dataset-transforming-text-into-the-pretraining-dataset-1
:::

::: {.slide title="Custom Dataset class"}
Wraps the per-example generators into a `__getitem__`
interface — the standard PyTorch / framework idiom:

@bert-dataset-transforming-text-into-the-pretraining-dataset-2
:::

::: {.slide title="Loader factory"}
Download corpus → tokenize → generate NSP + MLM pairs →
DataLoader:

@bert-dataset-transforming-text-into-the-pretraining-dataset-3
:::

::: {.slide title="Inspect a minibatch"}
Verify shapes: `tokens`, `segments`, `valid_lens`,
`pred_positions`, `mlm_weights`, `mlm_labels`,
`nsp_labels`. `mlm_weights` marks which padded prediction
slots should contribute to the MLM loss:

@bert-dataset-transforming-text-into-the-pretraining-dataset-4

. . .

@bert-dataset-transforming-text-into-the-pretraining-dataset-5
:::

::: {.slide title="Recap"}
- A BERT minibatch carries seven tensors: tokens,
  segments, valid_lens, masked positions, MLM weights,
  MLM labels, NSP label.
- 15% MLM masking with the 80/10/10 split is the original
  recipe; modern variants (RoBERTa) drop NSP, increase
  masking, and dynamic-mask each epoch.
- Pretraining BERT for real takes 16+ TPU/GPU-days; the
  WikiText-2 demo is small enough to play with.
:::
