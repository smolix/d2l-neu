# Tools for Deep Learning

A notebook that runs on a laptop may fail as soon as the model, batch, or
context grows. The next decision is not automatically to buy the accelerator
with the highest advertised throughput. First determine whether the workload
fits in memory, whether computation or data movement is limiting it, and what a
completed experiment will cost. Training and serving then impose different
constraints: distributed training incurs accelerator-communication costs, whereas
serving must also meet latency and throughput targets while storing persistent
model and cache state.

This part develops these decisions from a single notebook to a deployed system.
Use the following table to find the section relevant to your problem.

| Task | Section | Principal question |
| :-- | :-- | :-- |
| Run and inspect the book locally | :numref:`sec_interactive_development` | How should code, state, and outputs be organized for reproducible iteration? |
| Start without a local installation | :numref:`sec_hosted_notebooks` | Which hosted environment provides the required framework and accelerator? |
| Rent compute | :numref:`sec_cloud_instances` | What is the total cost per completed run, including setup and data transfer? |
| Choose hardware | :numref:`sec_hardware_buyers` | Does the workload fit, and will compute, memory bandwidth, or interconnect limit it? |
| Find models, datasets, and evidence | :numref:`sec_software_ecosystem` | Which artifact, implementation, or benchmark can be reproduced and trusted? |
| Scale training | :numref:`sec_training_systems` | How should parameters, activations, data, and communication be partitioned? |
| Serve a model | :numref:`sec_model_serving` | How do batching, latency, throughput, and cache memory interact? |
| Contribute to the book | :numref:`sec_developers_guide` | How are changes tested and kept consistent across four frameworks? |
| Look up book utilities | :numref:`sec_utils` and :numref:`sec_d2l` | Where is a class or function defined and documented? |

Readers building a system can follow the table from top to bottom. Readers with
a specific operational question can consult the corresponding section
directly. Hardware names, prices, cloud interfaces, and library versions change
quickly, so the linked sections date such facts and emphasize measurements that
remain useful when the products change. The utility and `d2l` API pages are
searchable references rather than part of the teaching sequence.
