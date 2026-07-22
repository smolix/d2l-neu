# Advanced

The Advanced part picks up where the Basics leave off and turns to the methods
that make modern, large-scale models work. It begins with *optimization* — how
gradient descent, momentum, Adam, learning-rate schedules, and newer geometry-
aware optimizers like Muon actually behave, and how to reason about optimizer
claims rather than take them on faith. It then develops *attention* and the
*transformer*, the architecture behind current language and vision models,
followed by *state space models* and linear recurrences that revisit sequence
modelling through a scalable, linear-time lens.

A chapter on *computational performance* grounds all of this in hardware: the
roofline model, compilation, mixed precision, and multi-GPU training. The part
closes with three families of generative and interactive models —
*reinforcement learning*, *generative adversarial networks*, and *diffusion
models*. Following the book's coverage policy, the Advanced chapters are
implemented in PyTorch and JAX.
