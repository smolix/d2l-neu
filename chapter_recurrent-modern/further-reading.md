# Further Reading
:label:`sec_ssm-further-reading`

By now you have built the recurrent layers that recent long-context models
put next to attention, and the LSTM they descend from, and you have
measured where a state of fixed size runs out. Most of the literature
behind these layers is less than five years old, and not all of it will
last. This section points to the topics the chapter had no room for and
says, where it can, what has held up. Each topic states the problem it
addresses and what it changes relative to the chapter.

If you have one evening, spend it on the papers behind two entries:
"Hybrid Models in Practice", where the question of :numref:`sec_hybrids`
is being decided at scale, and "Expressivity of Fixed-State Models", on
what a model with a fixed state can compute at all. With a second evening,
read "Memory by Polynomial Projection", the derivation of the rates $-(n +
1)$ of S4D that the chapter cut short. The applications at the end are
pointers for readers who work with images, audio, or DNA. Implementations,
tutorials, and courses for what the chapter does teach are listed in
:numref:`chap_modern_rnn`. Nothing else in the book depends on this
section.

## Recurrence and Attention Together

This is the group to read first. The designs of released models are being
decided here, and none of its questions is settled.

### Hybrid Models in Practice

:numref:`sec_hybrids` ended on three questions: how few attention layers are
enough, where in the stack they belong, and whether attention over a window of
recent tokens can replace some of them. The matched comparison of
:citet:`Waleffe.Byeon.Riach.ea.2024`, which that section cited, gives the most
careful answer to the first two. Their hybrid of 8B parameters mixes tokens
with 24 Mamba-2 layers and 4 attention layers, one attention layer in seven.
They chose it with small hybrids trained at different shares, whose
validation loss was lowest when about 8% of all layers, the MLP layers
included, were attention. They put a Mamba-2 layer first, so that the model
needs no position embeddings, and spread the attention layers evenly through
the rest of the stack. At this share the hybrid exceeded a transformer trained
on the same data on all twelve standard tasks of the study, while the pure
Mamba models fell behind on five-shot MMLU and on looking up a number in a
phone book.

Released models do not agree on the share. Qwen3-Next :cite:`Qwen.Team.2025`
and Kimi Linear :cite:`Kimi.Team.2025b` use one attention layer in four, Jamba
:cite:`Lieber.Lenz.Bata.ea.2024,Jamba.Team.2024` one in eight, and nobody has
yet shown which share is best for which task. Nor do they agree on the
recurrent layer: the first two build on the delta rule of
:numref:`sec_deltanet`, while Nemotron-H :cite:`NVIDIA.2025` keeps the Mamba-2
layers of Waleffe et al. The third question has its boldest answer in Griffin
:cite:`De.Smith.Fernando.ea.2024`, which has no global attention at all: it
interleaves two layers of a gated diagonal recurrence with one layer of the
sliding-window attention of :numref:`sec_windowed-linear-attention`. Its
authors report what the experiment of :numref:`sec_hybrids` predicts: lookup
in a phone book degrades once the book is longer than the window.

### Distillation from Transformers

Training a large recurrent model from scratch costs as much as training a
transformer. It is cheaper to convert a trained transformer: initialize the
recurrent layers from its attention weights, using the correspondence
between linear attention and a matrix state of :numref:`sec_matrix-state`,
and train them to match the transformer's outputs.
:citet:`Wang.Paliotta.May.ea.2024` keep some of the attention layers and
obtain a hybrid. :citet:`Bick.Li.Xing.ea.2024` match three things in turn:
first the attention matrices, the weights each token gives the others,
then the hidden states, then the outputs. Their distilled models need 3B
to 5B tokens of training, which they put at less than 1% of what training
from scratch uses.

### Between a Fixed State and a Full Cache

A recurrent layer stores a fixed amount, and attention stores every token. Two
designs sit in between. Based :cite:`Arora.Eyuboglu.Zhang.ea.2024` addresses
the weakness on recall that :numref:`sec_windowed-linear-attention` noted in
linear attention. It combines linear attention, whose feature map approximates
$\exp(\mathbf{q}^\top\mathbf{k})$ by its Taylor expansion to second order (an
exercise of that section works out this truncation), with exact attention over
a short window of recent tokens. The size of the feature map and of the window
then set how much the layer stores. The authors find that recall grows with
the size of the state for every model of this kind, and that Based, by
changing these two sizes, traces the best trade-off between state size and
recall among the models they compare. Log-linear attention
:cite:`Guo.Yang.Goel.ea.2025` keeps one matrix state per block of the past,
with blocks that double in length the further back they lie, so the memory
grows with the logarithm of the length. It applies to Mamba-2 and to Gated
DeltaNet. Both designs turn the choice between a fixed state and a full cache
into a dial.

## Writing to a Matrix Memory

The first entry says what no fixed state of this chapter can compute, and
why the delta rule comes closer than the others. The other three change how
the memory is written, or what the memory is.

### Expressivity of Fixed-State Models

Some tasks are hard for the linear recurrences of this chapter whatever the
size of their state. Take the shell game: five cups are swapped many times,
and the model must say where the ball ended up. Following the ball means
composing the swaps, that is, composing permutations of five objects.
:citet:`Merrill.Petty.Sabharwal.2024` show that state space models whose
transitions are diagonal or fixed, S4 and Mamba among them, compute only
functions in the circuit class $\mathsf{TC}^0$, as transformers do.
Roughly, $\mathsf{TC}^0$ holds what a network of threshold units can
compute with a number of layers fixed in advance, however long the input.
Composing permutations of five objects is among the hardest problems of a
larger class, $\mathsf{NC}^1$, which allows a number of layers that grows
with the logarithm of the length. It therefore lies outside
$\mathsf{TC}^0$ unless $\mathsf{TC}^0 = \mathsf{NC}^1$, which complexity
theorists conjecture is false. Surprisingly, the number of cups matters:
with three or four cups the permutations form what group theory calls a
solvable group, and the shell game stays inside $\mathsf{TC}^0$; five is
the smallest number of cups for which it does not. Under the conjecture no
such model of fixed depth can follow the ball over arbitrary lengths. A
nonlinear recurrent network can, and so can a linear recurrence whose
transition is non-diagonal and depends on the input.

The delta rule's transition $\mathbf{I} - \beta_t
\mathbf{k}_t\mathbf{k}_t^\top$ is of that kind, so this result does not
rule it out. A second result does, for the layer of
:numref:`sec_deltanet`. The shell game contains parity: let a swap of the
first two cups stand for a one and a swap of the last two for a zero, and
the ball, which starts under the first cup, is back there exactly when the
number of ones is even. :citet:`Grazzi.Siems.Franke.ea.2025` show that in
finite precision a linear recurrence whose transitions have eigenvalues
only between zero and one cannot compute parity at arbitrary lengths. The
`GatedDeltaNet` we built, whose $\beta_t$ comes from a sigmoid, is such a
recurrence, so it cannot play the shell game over arbitrary lengths either.
Once $\beta_t$ may reach 2, neither result applies, and the delta rule can
follow the ball: with $\beta_t = 2$ and the key $(\mathbf{e}_i -
\mathbf{e}_j)/\sqrt{2}$, its transition is the swap of cups $i$ and $j$, so
one delta step per token can carry out one swap (Exercise
:exref:`ex_dn-parity` doubles the range of $\beta_t$ for parity). These
two results are the most likely to last in this section: they are proofs,
not benchmarks, though the first rests on a conjecture.

### Generalized Delta Rules

The delta rule ties three choices together: it removes along the key, it
writes along the same key, and one strength $\beta_t$ serves every
coordinate. RWKV-7 :cite:`Peng.Zhang.Goldstein.ea.2025`, released as a
family of language models, unties them. It decays each coordinate of the
memory at its own rate, removes along the key by a different amount in
each coordinate, and writes along a second, adjusted key. Its transition is
then no longer symmetric, and its authors show that the layer can track
states and recognize every regular language, that is, every language a
finite automaton recognizes, the shell game among them. Under the
conjecture of the previous entry, transformers cannot. DeltaProduct
:cite:`Siems.Carstensen.Zela.ea.2025` keeps the delta rule and takes
several of its steps per token, so that each transition is a product of
several factors $\mathbf{I} - \beta\mathbf{k}\mathbf{k}^\top$. At $\beta =
2$ each factor is a reflection, which reverses the component along its key
(:numref:`sec_deltanet`). A swap of two cups is one reflection and a cycle
of three cups is two, so several steps per token let one token apply any
rearrangement of the cups, not only a swap. Each step adds the work of one
delta-rule update.

### Exponential Gates

A sigmoid gate writes a token with weight at most one, while the memory
holds the sum of everything before it, so a single important token cannot
overrule what is stored. The mLSTM of xLSTM
:cite:`Beck.Poppel.Spanring.ea.2024` keeps a matrix memory with an
exponential input gate, whose weight has no such ceiling. With exponential
gates the scale of the memory is arbitrary, so the layer divides each read
by a second state, the sum of the gated keys, which cancels that scale as
the denominator of the softmax does. To keep the exponentials from
overflowing, it carries a running maximum of the logarithms of the gates,
decayed by the forget gates, and rescales both states by it, the device of
the online softmax in :numref:`sec_attention-at-scale`. The same group has
since trained an xLSTM language model of 7B parameters
:cite:`Beck.Poppel.Lippe.ea.2025`, so the gate has been tried at scale, but
it has not spread to the other released models of this section.

### Memories That Are Networks

Nothing in the delta rule requires the memory to be a matrix. In
:numref:`sec_deltanet` it was one, fitted to the pairs of the current
sequence by one gradient step per token. TTT layers
:cite:`Sun.Li.Dalal.ea.2024` keep the gradient steps and make the memory
and its update rule design choices: the memory is a linear model or a
small multilayer perceptron. Titans :cite:`Behrouz.Zhong.Mirrokni.2025`
add two things to the steps: momentum, so that a surprising token keeps
influencing the next few updates, and a forgetting term like the decay of
Gated DeltaNet. A network can represent nonlinear associations between
keys and values that a matrix cannot, but whether this recalls more per
byte of state than a matrix of the same size is a current research
question. The chapter sets a bar and says how to read it. By the capacity
law of :numref:`sec_matrix-state`, a memory that adds its pairs reads with
squared error $(n - 1)/d_k$ after $n$ pairs with random keys, so the pairs
it holds at a given error grow in proportion to the key dimension. And
:numref:`sec_deltanet` found that a smaller error need not mean more pairs
recalled. A network memory has to recall more pairs than a matrix with the
same number of parameters.

## State Space Models

This group is history and engineering: where the S4D of :numref:`sec_ssm`
came from, what became of Mamba, and two other routes to a long memory.

### Memory by Polynomial Projection

:numref:`sec_ssm` initialized the rates of S4D as $\lambda_n = -(n + 1)$ and
gave one sentence of justification. The full argument asks what a
fixed-size state should remember about the history $x(s)$, $s \le t$, of an
input, a history that keeps growing. HiPPO :cite:`Gu.Dao.Ermon.ea.2020`
answers: the coefficients of the best approximation of the history by a
polynomial of degree $N - 1$, measured with a weight that spreads uniformly
over the whole past. These coefficients obey a linear differential
equation, $\mathbf{h}'(t) = \frac{1}{t}\big(\mathbf{A}\mathbf{h}(t) +
\mathbf{B}x(t)\big)$, whose matrix $\mathbf{A}$, the HiPPO matrix, is lower
triangular and fixed by the choice of polynomials and weight.

S4 :cite:`Gu.Goel.Re.2022` dropped the factor $1/t$. Without it the system is
the same at every step, so it has the convolutional form of
:numref:`subsec_ssm-conv`. :citet:`Gu.Johnson.Timalsina.ea.2023` later worked
out what it then remembers: the projection of the history onto Legendre
polynomials in the warped time $e^{-(t-s)}$, which resolve the recent past
finely and the distant past coarsely. The uniform window over all the past
became an exponentially fading one. S4 then had to train an SSM started at
this matrix. Diagonalizing it outright fails, because its eigenvectors have
entries that grow exponentially with $N$, so S4 wrote it as a normal matrix,
which has orthogonal eigenvectors, plus a low-rank correction, a form whose
convolution kernel can be computed quickly.

That correction was the hardest part of S4 and the first to go. Within a
year, S4D :cite:`Gu.Gupta.Goel.Re.2022` kept only a diagonal: either the
eigenvalues of the normal part, which are complex, or the real rates $-(n +
1)$ on the diagonal of the HiPPO matrix itself, the choice of
:numref:`sec_ssm`. In that paper's ablations the complex variants come
close to the full matrix, and the real rates, simpler to implement and to
explain, lose some accuracy. S5 :cite:`Smith.Warrington.Linderman.2023`
replaces the independent SSMs of an S4D layer by one SSM with vector input
and output, still discretized from continuous time and started from the
normal part of the HiPPO matrix, and computes it with a parallel scan
(:numref:`subsec_parallel-scans`) rather than a convolution. The selective
models of :numref:`sec_mamba` also use the scan, but keep one SSM per
channel. Gu's thesis :cite:`Gu.2023` develops the whole line.

### Successors of Mamba

Two later versions changed the layer of :numref:`sec_mamba`. Mamba-2
:cite:`Dao.Gu.2024` gives each head one decay so that most of the work becomes
the matrix products of :numref:`sec_matrix-state`; it is the recurrent layer
of the hybrids of Waleffe et al. and of Nemotron-H. Mamba-3
:cite:`Lahoti.Li.Chen.ea.2026` changes three things. Where the zero-order hold
of :numref:`sec_ssm` keeps the input constant over a step, its trapezoidal
discretization combines the current and the previous input in each update. Its
state is complex, so it can rotate and track parity, which a positive real
decay cannot. And its update takes several inputs and outputs at once, which
does more arithmetic per byte of state read during generation, where reading
the state, not the arithmetic, limits the speed. It is too recent to say which
of the three will last.

### Recurrences Without a Differential Equation

The continuous-time view is one way to arrive at stable, well-initialized
linear recurrences, not the only one. The linear recurrent unit
:cite:`Orvieto.Smith.Gu.ea.2023` takes a diagonal recurrence with complex
decays, keeps their magnitudes below one and starts them close to one, and
scales the input by $\sqrt{1 - |\lambda|^2}$, so that the state keeps its
size as the magnitude $|\lambda|$ of a decay approaches one. With no
discretization at all it matches deep state space models on the Long Range
Arena, a set of classification tasks on sequences of one to sixteen
thousand elements. It agrees with the recall experiment of
:numref:`sec_ssm`, where a minGRU started at the decays of S4D recalled a
symbol after 1,000 steps as S4D did: the decays and their initialization
matter more than the differential equation they came from. With gates
computed from the input and real decays in place of complex ones, the
linear recurrent unit became the recurrence of Griffin.

### Long Convolutions

The kernel of S4D is a sum of $N$ decaying exponentials, which is what
gives the layer a recurrent form with a small state. Long convolutions drop
that constraint, and the step toward them was H3
:cite:`Fu.Dao.Saab.ea.2023`. It chained two SSMs of the recurrent kind,
with elementwise products between them. The first only shifts its input,
pairing each token with its predecessor, the job of the short convolution
in front of every layer of :numref:`sec_matrix-state`; the second
accumulates the pairs, so a layer can find an earlier occurrence of a token
and recall what followed it. H3 came within 0.4 points of a transformer's
perplexity on OpenWebText, and a hybrid that kept two attention layers beat
the transformer by 1.0, an early sign of the result of
:numref:`sec_hybrids`. Hyena :cite:`Poli.Massaroli.Nguyen.ea.2023` keeps
H3's chain of convolutions and elementwise products and drops the
constraint on the kernel: a small network maps each position to the kernel
value there, and the convolution over the whole sequence is computed with
the FFT in $O(T \log T)$ time. Such a kernel has no recurrent form with a
small state, so generation recomputes the convolution or approximates it by
a recurrence afterwards. A long convolution sees the whole past, but it has
to look at it again for every new token. Released language models have
mostly taken the recurrent route, and the best-known uses of long
convolutions are in genomics.

## Beyond Language

The same layers have been applied wherever sequences are long, and each
field added one idea of its own. An image has no natural order, and a
causal scan over its patches would let each patch see only those above it
and to its left. Vision Mamba :cite:`Zhu.Liao.Zhang.ea.2024` therefore
scans the patches forward and backward, and VMamba
:cite:`Liu.Tian.Zhao.ea.2024` scans along rows and columns in both
directions. Raw audio has tens of thousands of samples per second. SaShiMi
:cite:`Goel.Gu.Donahue.ea.2022` stacks S4 layers in a network that pools to
coarser rates and upsamples again, and generates one sample at a time with
the recurrent form. In DNA, a stretch can regulate a gene tens of thousands
of bases away, and a strand and its reverse complement carry the same
information. HyenaDNA :cite:`Nguyen.Poli.Faizi.ea.2023` trains Hyena on
contexts of up to a million single nucleotides, and Caduceus
:cite:`Schiff.Kao.Gokaslan.ea.2024` makes each layer of a bidirectional
Mamba equivariant to the reverse complement, so that both strands receive
consistent predictions.

[Discussions](https://d2l.discourse.group/t/36786)
