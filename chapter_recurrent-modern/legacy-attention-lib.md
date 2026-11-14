# Legacy Recurrent Library (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml` and is never rendered or executed
as a notebook: `make lib` (`tools/build_lib.py`) only scans it for `#@save`
blocks. It holds the PyTorch/JAX `d2l.GRU` that `seq2seq.md` builds its
translator on; `lstm.md` teaches the GRU but does not save it to the library.

History. It came from `chapter_natural-language-processing-pretraining/` when
that chapter was retired (2026-11), carrying the frozen 2017-era
Transformer-encoder classes of the retired BERT section, the tensorflow/mxnet
attention primitives, and the tensorflow/mxnet `LSTMScratch`/`LSTM`/`GRU`.
When `seq2seq.md` dropped its TensorFlow and MXNet tabs (2026-11-14, the
Advanced part carries PyTorch and JAX only) none of those had a user left in
any framework, and they were removed. The file keeps its name so that
`LIB_ONLY_FILES` in `tools/build_lib.py` does not change.
-->

## GRU (pytorch, jax)
:label:`sec_gru`

`seq2seq.md` builds its translator on these. They follow the label of
their original home in `lstm.md`, so that their docstrings say "Defined
in" `sec_gru`.

```{.python .input #lstm-implementation-and-comparison-1}
%%tab pytorch
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

```{.python .input #lstm-implementation-and-comparison-1}
%%tab jax
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0,
                 rngs=None):
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.num_layers = num_layers
        self.rnns = nnx.List([
            nnx.RNN(nnx.GRUCell(num_inputs if i == 0 else num_hiddens,
                                num_hiddens, rngs=rngs),
                    time_major=True, return_carry=True, rngs=rngs)
            for i in range(num_layers)])
        self.dropouts = nnx.List([
            nnx.Dropout(dropout, rngs=rngs) for _ in range(num_layers - 1)])

    def __call__(self, X, state=None):
        outputs = X
        state = [None] * self.num_layers if state is None else state
        new_state = []
        for i, rnn in enumerate(self.rnns):
            carry, outputs = rnn(outputs, initial_carry=state[i])
            new_state.append(carry)
            if i < self.num_layers - 1:
                outputs = self.dropouts[i](outputs)
        return outputs, new_state
```
