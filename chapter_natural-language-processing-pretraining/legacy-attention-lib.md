# Legacy Attention Library (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml` and is never rendered or executed
as a notebook — `make lib` (`tools/build_lib.py`) only scans it for `#@save`
blocks. It carries the frozen 2017-era Transformer-encoder classes that
`chapter_natural-language-processing-pretraining/bert.md` still builds on, in
all four frameworks, plus the tensorflow/mxnet variants of the attention
primitives whose PyTorch/JAX versions now live in `chapter_attention/`.
Semantics are frozen exactly as they were in
`chapter_attention-mechanisms-and-transformers/` (post-LN, ReLU, original
signatures) so that BERT's committed outputs stay reproducible from source
without a re-capture. This file is deleted, and `TransformerEncoderBlock`
modernized, when the Language-Models part is next modernized (atomically with
the BERT re-capture).
-->

**`masked_softmax` (tensorflow, mxnet only — PyTorch/JAX version is
`chapter_attention/attention-scoring.md`'s `#@save`).**

```{.python .input}
%%tab mxnet
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
%%tab tensorflow
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = tf.shape(X)[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)
        return tf.where(mask, X, value)

    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = tf.shape(X)
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
        else:
            valid_lens = tf.reshape(valid_lens, shape=(-1,))
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(tf.reshape(X, (-1, shape[-1])), valid_lens,
                           value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape), axis=-1)
```

**`DotProductAttention` (tensorflow, mxnet only).**

```{.python .input}
%%tab mxnet
class DotProductAttention(nn.Block):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of keys
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class DotProductAttention(tf.keras.layers.Layer):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def call(self, queries, keys, values, valid_lens=None, training=False,
             **kwargs):
        d = tf.cast(tf.shape(queries)[-1], dtype=tf.float32)
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(
            self.attention_weights, training=training), values)
```

**`MultiHeadAttention` + `transpose_qkv`/`transpose_output` (tensorflow,
mxnet only — PyTorch/JAX version is `chapter_attention/multihead-attention.md`'s
`#@save`).**

```{.python .input}
%%tab mxnet
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab tensorflow
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens, training=False, **kwargs):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
            
        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens,
                                training=training)
        
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab mxnet
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input X: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). Shape of output X: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # Shape of output X: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = X.transpose(0, 2, 1, 3)
    # Shape of output: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """Reverse the operation of transpose_qkv."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input X: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). Shape of output X: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = tf.reshape(X, (tf.shape(X)[0], tf.shape(X)[1], self.num_heads, -1))
    # Shape of output X: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    # Shape of output: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return tf.reshape(X, (-1, tf.shape(X)[2], tf.shape(X)[3]))

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """Reverse the operation of transpose_qkv."""
    X = tf.reshape(X, (-1, self.num_heads, tf.shape(X)[1], tf.shape(X)[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, (tf.shape(X)[0], tf.shape(X)[1], -1))
```

**`PositionWiseFFN` (all four frameworks — no name collision with the new
chapters).**

```{.python .input}
%%tab mxnet
class PositionWiseFFN(nn.Block):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))
```

```{.python .input}
%%tab pytorch
class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab tensorflow
class PositionWiseFFN(tf.keras.layers.Layer):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab jax
class PositionWiseFFN(nnx.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs,
                 ffn_num_inputs=None, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        ffn_num_inputs = (ffn_num_hiddens if ffn_num_inputs is None
                          else ffn_num_inputs)
        self.dense1 = nnx.Linear(ffn_num_inputs, ffn_num_hiddens, rngs=rngs)
        self.dense2 = nnx.Linear(ffn_num_hiddens, ffn_num_outputs, rngs=rngs)

    def __call__(self, X):
        return self.dense2(nnx.relu(self.dense1(X)))
```

**`AddNorm` (all four frameworks — no name collision with the new chapters).**

```{.python .input}
%%tab mxnet
class AddNorm(nn.Block):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab pytorch
class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab tensorflow
class AddNorm(tf.keras.layers.Layer):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # `norm_shape` mirrors PyTorch's `nn.LayerNorm` convention: it gives
        # the shape of the trailing dims to normalize over. Convert that to
        # Keras's `axis` argument (negative axis indices counting from the end).
        self.ln = tf.keras.layers.LayerNormalization(
            axis=list(range(-len(norm_shape), 0)))

    def call(self, X, Y, training=False, **kwargs):
        return self.ln(self.dropout(Y, training=training) + X)
```

```{.python .input}
%%tab jax
class AddNorm(nnx.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, num_hiddens, dropout, rngs=None):
        rngs = nnx.Rngs(params=0, dropout=1) if rngs is None else rngs
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.ln = nnx.LayerNorm(num_hiddens, rngs=rngs)

    def __call__(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

**`TransformerEncoderBlock` (all four frameworks — no name collision with the
new chapters; frozen post-LN, ReLU, original signature. The PyTorch/JAX
tabs' `d2l.MultiHeadAttention` call now resolves to the new
`chapter_attention/multihead-attention.md` class — constructor and call
shape verified compatible, no adaptation needed.)**

```{.python .input}
%%tab mxnet
class TransformerEncoderBlock(nn.Block):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab pytorch
class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab tensorflow
class TransformerEncoderBlock(tf.keras.layers.Layer):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, bias)
        self.addnorm1 = AddNorm([num_hiddens], dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm([num_hiddens], dropout)

    def call(self, X, valid_lens, training=False, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens,
                          training=training), training=training)
        return self.addnorm2(Y, self.ffn(Y), training=training)
```

```{.python .input}
%%tab jax
class TransformerEncoderBlock(nnx.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, rngs=None):
        rngs = nnx.Rngs(params=0, dropout=1) if rngs is None else rngs
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias, rngs=rngs)
        self.addnorm1 = AddNorm(num_hiddens, dropout, rngs=rngs)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens,
                                   num_hiddens, rngs=rngs)
        self.addnorm2 = AddNorm(num_hiddens, dropout, rngs=rngs)

    def __call__(self, X, valid_lens):
        output, attention_weights = self.attention(X, X, X, valid_lens)
        Y = self.addnorm1(X, output)
        return self.addnorm2(Y, self.ffn(Y)), attention_weights
```

## Legacy Recurrent Classes (tensorflow, mxnet only)

**Why these are here (2026-07-19).** `chapter_recurrent-modern/lstm.md` now
carries PyTorch/JAX tabs only (the Advanced-part policy), but `seq2seq.md` —
relocated to this part as its historical baseline — still builds its
translator on `d2l.GRU` in all four frameworks. These are the frozen
tensorflow/mxnet `LSTMScratch`/`LSTM`/`GRU` classes, byte-identical to the
last four-framework revision of `lstm.md`, kept so seq2seq's committed
outputs stay reproducible from source. Deleted together with the rest of
this file when the Language-Models part is modernized.

```{.python .input #lstm-implementation-from-scratch-1}
%%tab mxnet
class LSTMScratch(d2l.Module):  #@save
    """The long short-term memory (LSTM) cell implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: d2l.randn(*shape) * sigma
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          d2l.zeros(num_hiddens))
        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node
```

```{.python .input #lstm-implementation-from-scratch-1}
%%tab tensorflow
class LSTMScratch(d2l.Module):  #@save
    """The long short-term memory (LSTM) cell implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          tf.Variable(d2l.zeros(num_hiddens)))
        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node
```

```{.python .input #lstm-implementation-from-scratch-2}
%%tab mxnet
@d2l.add_to_class(LSTMScratch)  #@save
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                      ctx=inputs.ctx)
        C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                      ctx=inputs.ctx)
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) +
                        d2l.matmul(H, self.W_hi) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

```{.python .input #lstm-implementation-from-scratch-2}
%%tab tensorflow
@d2l.add_to_class(LSTMScratch)  #@save
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = tf.zeros((tf.shape(inputs)[1], self.num_hiddens))
        C = tf.zeros((tf.shape(inputs)[1], self.num_hiddens))
    else:
        H, C = H_C
    outputs = []
    for X in tf.unstack(inputs):
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) +
                        d2l.matmul(H, self.W_hi) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

```{.python .input #lstm-concise-implementation-1}
%%tab mxnet
class LSTM(d2l.RNN):  #@save
    """The multilayer LSTM model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)

    def forward(self, inputs, H_C=None):
        if H_C is None:
            H_C = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        return self.rnn(inputs, H_C)
```

```{.python .input #lstm-concise-implementation-1}
%%tab tensorflow
class LSTM(d2l.RNN):  #@save
    """The multilayer LSTM model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.lstms = [tf.keras.layers.LSTM(
            num_hiddens, return_sequences=True, return_state=True,
            dropout=dropout) for _ in range(num_layers)]

    def forward(self, inputs, H_C=None):
        X = tf.transpose(inputs, perm=[1, 0, 2])  # To batch-major layout
        if H_C is None:
            H_C = [None] * self.num_layers
        new_H_C = []
        for lstm, state in zip(self.lstms, H_C):
            X, *state = lstm(X, initial_state=state)
            new_H_C.append(state)
        return tf.transpose(X, perm=[1, 0, 2]), new_H_C
```

```{.python .input #lstm-implementation-and-comparison-1}
%%tab mxnet
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
```

```{.python .input #lstm-implementation-and-comparison-1}
%%tab tensorflow
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        gru_cells = [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
                     for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True,
                                       return_state=True)

    def forward(self, X, state=None):
        outputs, *state = self.rnn(tf.transpose(X, perm=[1, 0, 2]), state)
        state = [s[0] if isinstance(s, list) else s for s in state]
        return tf.transpose(outputs, perm=[1, 0, 2]), state
```
