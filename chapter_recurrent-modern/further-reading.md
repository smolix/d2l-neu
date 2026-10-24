# Further Reading
:label:`sec_ssm-further-reading`

The sections of this chapter teach one representative of each idea. This
section points to topics they had no room for: the theory behind the
initialization of state space models, other ways to build and write to a
fixed-size state, results on what such models can compute, other ways to
combine recurrence and attention, and applications beyond language. Each
topic states the problem it addresses and what it changes relative to the
chapter. Implementations, tutorials, and courses for the material that the
chapter does teach are listed in its introduction,
:numref:`chap_modern_rnn`. Nothing else in the book depends on this section.

## State Space Models

### Memory by Polynomial Projection

:numref:`sec_ssm` initialized the rates of S4D as $\lambda_n = -(n + 1)$ and
gave one sentence of justification. The full argument asks what a fixed-size
state should remember about an input $x(s)$, $s \le t$, that keeps growing.
HiPPO :cite:`Gu.Dao.Ermon.ea.2020` answers: the coefficients of the best
approximation of the history by a polynomial of degree $N - 1$, measured
with a weight that spreads uniformly over the whole past. These coefficients
obey a linear differential equation, $\mathbf{h}'(t) =
\frac{1}{t}\big(\mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t)\big)$, whose matrix
$\mathbf{A}$, the HiPPO matrix, is lower triangular and fixed by the choice
of polynomials and weight. S4 :cite:`Gu.Goel.Re.2022` dropped the factor
$1/t$, initialized an SSM with this matrix, and made it trainable by writing
it as a normal matrix plus a low-rank correction, which allows its
convolution kernel to be computed quickly. S4D :cite:`Gu.Gupta.Goel.Re.2022`
uses a diagonal matrix instead: either the normal part without the
correction, whose eigenvalues are complex, or the real rates $-(n + 1)$ on
the diagonal of the HiPPO matrix itself, the choice of :numref:`sec_ssm`.
Both work nearly as well as the full matrix.
:citet:`Gu.Johnson.Timalsina.ea.2023` explain what the model without the
factor $1/t$ remembers, a projection of the history onto polynomials under a
weight that decays exponentially into the past, and interpret the
convolution kernels of these models as projections onto orthogonal bases.
Gu's thesis :cite:`Gu.2023` develops the whole line, including the
control-theoretic view of SSMs as linear systems with inputs, states, and
outputs.

### Recurrences Without a Differential Equation

The continuous-time view is one way to arrive at stable, well-initialized
linear recurrences, not the only one. The linear recurrent unit
:cite:`Orvieto.Smith.Gu.ea.2023` starts from a plain diagonal recurrence
with complex decays, parameterizes the decays so that their magnitude stays
below one, initializes the magnitudes close to one, and normalizes the input
so that the state does not grow. It matches S4 on long
sequence benchmarks without discretization. S5
:cite:`Smith.Warrington.Linderman.2023` replaces the $d$ independent SSMs of
an S4D layer by one SSM with vector input and output, computed with a
parallel scan rather than a convolution, which is the route the selective
models of :numref:`sec_mamba` take.

### Long Convolutions

The kernel of S4D in :numref:`sec_ssm` is a sum of $N$ decaying
exponentials, one per state dimension, which is what gives the layer a
recurrent form with a small state. H3 :cite:`Fu.Dao.Saab.ea.2023` kept such
kernels and chained two SSMs, with elementwise products of input projections
between them in the roles of queries, keys, and values. The first SSM only
shifts its input, which pairs each token with its predecessor, and the
second, diagonal SSM accumulates these pairs. A layer can then compare a
token with earlier ones and recall what followed them, and H3 closed much of
the gap to attention on language. Hyena
:cite:`Poli.Massaroli.Nguyen.ea.2023` drops the constraint on the kernel: a
small network maps each position to the kernel value there, the convolution
over the whole sequence is computed with the FFT in $\mathcal{O}(T \log T)$
time, and elementwise gates sit between the convolutions. Such kernels have
no recurrent form with a small state, so generation either recomputes the
convolution or approximates the kernel by a recurrence afterwards.

### Kernels for Selective Scans

The selective scan of :numref:`sec_mamba` produces a state of $H \times N$
numbers at every position, too much to write to GPU memory for long
sequences. The Mamba kernel :cite:`Gu.Dao.2023` loads the inputs into fast
on-chip memory, discretizes and scans there, writes only the outputs, and
recomputes the states during the backward pass instead of storing them, the
trade that :numref:`sec_memory_precision` discusses as activation
checkpointing. Mamba-2 :cite:`Dao.Gu.2024` restricts the decay to one
scalar per head so that most of the work becomes the matrix products of
:numref:`sec_matrix-state`, which GPUs perform much faster than elementwise
scans. Mamba-3 :cite:`Lahoti.Li.Chen.ea.2026` makes three changes to the
selective model: a trapezoidal discretization, which combines the current
and the previous input in each update instead of the current one alone; a
complex-valued state, which lets the state rotate and track quantities such
as parity; and a multi-input, multi-output update, which raises the
arithmetic work per byte of state read during generation.

## Writing to a Matrix Memory

### Exponential Gates

The LSTM's gates are sigmoids, so a forget gate can keep at most all of the
state and an input gate can write at most the full candidate. The mLSTM of
xLSTM :cite:`Beck.Poppel.Spanring.ea.2024` uses a matrix memory with an
exponential input gate, which lets a single important token outweigh
everything stored before it, and keeps a second state, the sum of the
gated keys, by which it divides each read. Exponentials overflow quickly,
so the layer carries the logarithm of the largest gate seen so far and
rescales both states by it, the same device that the online softmax of
fused attention kernels uses.

### Generalized Delta Rules

The delta rule of :numref:`sec_deltanet` removes from the memory the
component along the key it is about to write, by the same amount
$\beta_t$ for every coordinate. RWKV-7 :cite:`Peng.Zhang.Goldstein.ea.2025`
decouples these choices: it decays each coordinate of the memory at its own
rate, removes along one vector, and writes along another, so that its
transition is no longer symmetric. DeltaProduct
:cite:`Siems.Carstensen.Zela.ea.2025` applies several delta-rule steps per
token, so that each transition is a product of several reflections or
projections and can rotate the memory, which helps on tasks that require
tracking a state through permutations.

### Memories That Are Networks

In :numref:`sec_deltanet` the memory was a matrix fitted to the pairs of the
current sequence by one gradient step per token. TTT layers
:cite:`Sun.Li.Dalal.ea.2024` replace the matrix by a small multilayer
perceptron and fit it by gradient steps as the sequence is read. Titans
:cite:`Behrouz.Zhong.Mirrokni.2025` add momentum to these steps, so that a
surprising token keeps influencing the next few updates, and a forgetting
term like the decay of Gated DeltaNet. A network can represent nonlinear
associations between keys and values that a matrix cannot, but whether
this recalls more per byte of state than a matrix of the same size is a
current research question.

### Expressivity of Fixed-State Models

Some tasks are hard for the linear recurrences of this chapter,
independently of memory size. :citet:`Merrill.Petty.Sabharwal.2024` show
that linear recurrences with diagonal transitions, like transformers, fall
in a class of parallel computations that cannot track the state of a general
finite automaton over arbitrary lengths, for instance the result of a long
sequence of permutations. Nonlinear recurrent networks and recurrences with
non-diagonal transitions can. :citet:`Grazzi.Siems.Franke.ea.2025` show that
allowing negative eigenvalues, as the delta rule with $\beta_t > 1$ does, is
enough for parity and helps on related tasks, and DeltaProduct extends the
range further.

## Recurrence and Attention Together

### Hybrid Models in Practice

:numref:`sec_hybrids` showed that one attention layer among several
recurrent ones restores recall. Released models make different choices
for the ratio and the kind of attention. Griffin
:cite:`De.Smith.Fernando.ea.2024` alternates a gated diagonal recurrence
with attention restricted to a local window. Jamba
:cite:`Lieber.Lenz.Bata.ea.2024,Jamba.Team.2024` uses one attention layer
for every seven Mamba layers, together with mixture-of-experts MLPs.
:citet:`Waleffe.Byeon.Riach.ea.2024` trained matched transformers, Mamba
models, and hybrids on the same data and found that hybrids with a small
fraction of attention layers matched or exceeded the transformer on most
tasks, including recall. Samba :cite:`Ren.Liu.Lu.ea.2024` combines Mamba
layers with sliding-window attention, and Zamba
:cite:`Glorioso.Anthony.Tokpanov.ea.2024` shares the weights of one
attention block among several positions in the stack. More recent models
build their recurrent layers on the delta rule: Qwen3-Next
:cite:`Qwen.Team.2025` interleaves Gated DeltaNet layers with full attention
layers, three to one, and Kimi Linear :cite:`Kimi.Team.2025b` does the same
with a Gated DeltaNet variant whose decay acts on each key coordinate.

### Between a Fixed State and a Full Cache

A recurrent layer stores a fixed amount, and attention stores every token.
Two designs sit in between. Based :cite:`Arora.Eyuboglu.Zhang.ea.2024`
combines linear attention, whose feature map approximates the exponential of
the softmax by its Taylor expansion, with exact attention over a short
window of recent tokens. The size of the feature map and of the window then
set how much the layer stores, and the authors trace how recall improves as
the state grows. Log-linear attention :cite:`Guo.Yang.Goel.ea.2025` keeps
one matrix state per block of the past, with blocks that double in length
the further back they lie, so the memory grows with the logarithm of the
length rather than staying fixed or growing linearly. The construction
applies to the gated recurrences of this chapter, Mamba-2 and Gated DeltaNet
among them.

### Distillation from Transformers

Training a large recurrent model from scratch is as expensive as training a
transformer. An alternative is to convert one: initialize the recurrent
layers from the attention weights of a trained transformer, using the
correspondence between linear attention and matrix-state recurrences of
:numref:`sec_matrix-state`, and train the result to match the transformer's
outputs. :citet:`Wang.Paliotta.May.ea.2024` do this to turn a transformer
into a hybrid that keeps some of its attention layers, and
:citet:`Bick.Li.Xing.ea.2024` distill in stages, first matching the mixing
matrices of attention, then the hidden states, then the outputs.

## Beyond Language

### Images

An image has no natural order. If it is flattened into a sequence of patches
and scanned causally, as a language model scans text, each patch sees only
the patches above it and to its left. Vision Mamba
:cite:`Zhu.Liao.Zhang.ea.2024` scans the sequence of patches forward and
backward and adds the two outputs. VMamba :cite:`Liu.Tian.Zhao.ea.2024`
scans the grid of patches along rows and along columns, each in both
directions, and merges the four results, so that every patch receives
information from the whole image.

### Audio

Raw audio has tens of thousands of samples per second, far too many for
attention over a recording of any length. SaShiMi
:cite:`Goel.Gu.Donahue.ea.2022` stacks S4 layers in a multiscale network
that pools the sequence to coarser rates and upsamples it again, and
generates waveforms one sample at a time with the recurrent form of each
layer.

### Genomics

A stretch of DNA can regulate a gene tens of thousands of bases away, and a
strand and its reverse complement encode the same information. HyenaDNA
:cite:`Nguyen.Poli.Faizi.ea.2023` trains Hyena on contexts of up to a
million bases with one token per nucleotide. Caduceus
:cite:`Schiff.Kao.Gokaslan.ea.2024` builds on Mamba, scans in both
directions, and makes each layer equivariant to taking the reverse
complement, so that a sequence and its reverse complement receive consistent
predictions.

[Discussions](https://d2l.discourse.group/)
