# Advanced
:label:`part_advanced`

The Basics ended with a small recurrent network trained on *The Time
Machine*. Asked to continue the phrase "The Time Traveller", it writes real
words and punctuation, but its grammar holds over a few words at most. The
language models that answer questions and write programs today are built
from the same parts: layers, a loss, its gradient and an optimizer. What
does it take to get from one to the other? This part treats four problems
that the small network never met.

The first is training. A learning rate tuned for one model may make a model
ten times wider diverge. The second is memory. A recurrent network squeezes
everything it has read into a state of fixed size, written before it knows
what will be asked. In 2014 attention let a translation model consult every
source word. In 2017 Ashish Vaswani and colleagues at Google built a
translation model from attention alone, without recurrence. That model, the
transformer, underlies most large language models built since.

The third is hardware. A GPU computes much faster than it moves data, so a
model's cost depends on how its operations use memory, not only on how many
it performs. The fourth is the objective. An agent that plays a game learns
from rewards rather than labels. A model that generates images has no label
to match, so its loss must judge whether a new sample could have come from
the data.

:numref:`chap_optimization` asks how gradient methods behave on curved,
noisy losses, and how to choose a learning rate, schedule and batch size.
:numref:`chap_attention` asks how a position can retrieve what it needs from
any other. :numref:`chap_transformers` asks how attention and feed-forward
layers stack into a deep model that trains reliably.
:numref:`chap_modern_rnn` asks how a state of fixed size can be trained on
all positions at once, and what it can store. :numref:`chap_performance`
asks what limits the speed of a training step on one GPU and on several.
:numref:`chap_reinforcement_learning` asks how an agent learns when its
actions change the data it sees, and :numref:`chap_deep_rl` asks how to keep
that learning stable with neural networks. :numref:`chap_gans` asks how to
train a generator that offers samples but no density, and
:numref:`chap_diffusion` asks how to generate data by learning to remove
noise.

The Advanced chapters are implemented in PyTorch and JAX. The Language
Models and Image Models parts apply them to text and images.
