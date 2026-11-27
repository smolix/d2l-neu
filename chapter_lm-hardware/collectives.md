# Collective Communication at Scale
:label:`sec_lm_collectives`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.7 Collective Communication at Scale · T4 (T3 core; T0 simulator; T2 pair)").
Blueprint: docs/llm/blueprints/ch05_hardware.md section 5.5.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A collective's time is an algorithm's step count times a protocol's latency plus the bytes over the slowest link it uses, and the library picks both per message size and topology.
**Opening:** The reader fitted alpha and beta to all-reduces on the build box in `sec_multi_gpu_concise` and to each link class in 5.5, and read in `sec_performance-further-reading` that NCCL combines two binary trees for small messages. One concrete example: across two nodes a 1 MB all-reduce takes longer than the alpha-beta line of 5.6 predicts, and a 1 GB all-reduce runs faster than the single-rail ring allows (to be measured). The open question: what the library does between the ring and the tree, which protocol carries each message, and whether its choice is the right one.
**Teaches:**
- Algorithms: the ring (restated), the pipelined double binary tree (every rank is a leaf in one tree and interior in the other, so each sends 2S bytes in 2d latency steps), the hierarchical all-reduce across nodes (reduce-scatter in the node, all-reduce of each shard across the ranks on its rail, all-gather in the node), in-network reduction (NVLS inside an NVSwitch fabric, SHARP in InfiniBand switches) with its bandwidth term S / beta, and the cross-node all-to-all with the PXN detour over NVLink to the GPU on the destination's rail.
- Protocols: Simple (bulk data, then a flag), LL (eight-byte lines of data and flag, lowest latency, half the bandwidth), LL128 (128-byte lines, needs NVLink's ordering), and why small messages take LL and large ones Simple; the latency floor per step is a protocol property, so alpha is fitted per (algorithm, protocol) pair.
- Channels, SMs and buffers: a collective is a kernel that runs on SMs, one channel per parallel ring or tree with its own buffer, so channels trade SM time against bandwidth and compete with compute kernels on the same GPU; what `NCCL_DEBUG=INFO` prints about channels, trees and transports.
- Topology detection and the tuner: NCCL reads the PCIe, NVLink and NIC graph, searches ring and tree orders, estimates the time of each (algorithm, protocol) pair per size from a table of latencies and bandwidths and takes the minimum; `NCCL_ALGO` and `NCCL_PROTO` force a pair, which is how the notebook tests the tuner.
**Derivation:** Ring against pipelined double binary tree and their crossover (blueprint ch05 5.5 D1). With G ranks, per-rank bandwidth beta and per-step latency alpha, T_ring = 2(G-1) alpha + 2((G-1)/G) S / beta and T_tree = 2 d alpha + 2 S / beta with d the hops from a leaf to the root, so the tree wins iff S < S* = G alpha beta (G - 1 - d). Smallest numeric case: G = 16 ranks over two nodes, alpha = 10 microseconds, beta = 25 GB/s per rank and d = 4 give S* = 16 x 1e-5 x 2.5e10 x 11 = 44 MB; the hierarchical variant replaces the single-rail ring's (G-1) steps at NIC bandwidth by (g-1) steps at NVLink bandwidth plus (n-1) steps at NIC bandwidth on a shard of S / g, and its speedup over a flat single-rail ring tends to g when NVLink is much faster than the NIC (D2 of the same blueprint section).
**Notebook:**
1. Question: does NCCL's choice of algorithm and protocol per message size follow the alpha-beta crossovers derived above, on one node and across two, and does the hand-written hierarchical all-reduce match the library's default?
2. T0: `d2l.sim_collective` simulates ring and tree event by event on a graph of full-duplex links and reproduces the closed forms within 1% far from the crossover and 5% near it, for G in {4, 8, 16, 32}.
3. T3 and T4 (committed sweeps from the one allocation shared with 5.6): nccl-tests all-reduce from 8 bytes to 4 GiB with `NCCL_ALGO` in {Ring, Tree, NVLS where available} crossed with `NCCL_PROTO` in {LL, LL128, Simple} and the default, on 8 and on 16 ranks; all-gather, reduce-scatter and all-to-all at the default; the `NCCL_DEBUG=INFO` header of one run.
4. Fit alpha and beta per (algorithm, protocol) with `d2l.fit_alpha_beta`; overlay the forced pairs and the default; locate the measured crossovers.
5. The hierarchical all-reduce written in PyTorch from reduce-scatter, all-reduce and all-gather (`d2l.hier_allreduce`) against the default across two nodes, with all NICs and with one NIC (`NCCL_IB_HCA` restricted), which isolates the rail effect.
6. T2 pair: the LL against Simple crossover at g = 2 over the NVLink bridge (ring and tree coincide at two ranks, so only the protocol crossover is visible; the text says so).
7. Reduced T1 configuration: the simulator, the committed logs, and the gloo loopback group, which shows only the latency floor and is labelled as not a datacenter network.
8. Claims that survive noise: the default's time is never above the best forced pair by more than noise at any size (the tuner is right to within noise); LL wins at small sizes and Simple at large ones, with the measured crossover within a factor of two of the fitted prediction; the hand-written hierarchical all-reduce matches the default on all rails and beats the flat single-rail ring by a factor that approaches g at large sizes (direction, number printed).
**Figures:** `fig_lm_hw_ring_tree` (ring against double binary tree: data halves, each rank a leaf in one tree and interior in the other); `fig_lm_hw_protocols` (LL line of data and flag against Simple's bulk-then-flag; bytes on the wire); `fig_lm_hw_algo_crossover` (plot: time against size for the forced pairs and the default, crossovers marked).
**Restate / forward:** `subsec_mg-ring` and `eq_ring_traffic` (the ring moves 2(k-1)/k of the message); `sec_performance-further-reading` "Collective Algorithms beyond the Ring" and "Bus Bandwidth" (double trees in about log k steps; the busbw convention), in one sentence each; `tab_collectives`; 5.5's alpha-beta per link. Forward: 6.7 (which collective each axis pays).
**Exercises:** 7 to 9. Derive that each rank sends 2S bytes over the two trees (derive); derive S* for 2, 4 and 16 nodes with the fitted alpha and beta (derive); derive the in-network bound and the busbw it implies (derive); measure busbw against `NCCL_MIN_NCHANNELS` (measure); measure a matmul running concurrently with an all-reduce and the slowdown of each (measure); check the `NCCL_DEBUG=INFO` header for channels and transports and compare with the fitted bandwidth (check); open: write a two-dimensional all-reduce in `torch.distributed` that beats the default on a one-NIC node.
**Writer notes:** Demystifying NCCL (arXiv 2507.04786) is the primary reference for protocols, channels and the tuner, and the NCCL environment page is the source for every variable the notebook sets; the NCCL 2.4 post is the source for the double binary tree and the 2.12 post for PXN. Pitfalls (blueprint 5.5): `NCCL_TESTS_SPLIT` prints per-group bandwidth; a single-NIC restriction is also a common accidental misconfiguration (wrong `NCCL_IB_HCA`, a NIC down), so read the header before trusting a slow result; bus bandwidth across nodes can exceed the NIC rate for all-to-all (intra-node share) and for in-network reduction, so it is not a link measurement; alpha from NCCL differs from `ib_write_lat`. Hardware fallbacks: no NVLS or SHARP (omit the in-network series; state the bound and check it in the simulator only); the T2 pair's TCP-loopback trick may fail across NCCL versions (drop it and let the gloo run carry the T1 role); a shared RoCE fabric (repeat three times at different hours, widen the claim threshold). Not re-taught: the ring derivation, the alpha-beta definition and fit (5.5), the busbw convention beyond one sentence, DDP bucketing (`sec_multi_gpu_concise`). Channels consume SMs, which is why 6.2's overlap is not free; say so in one sentence and let 6.8 show it in a trace. Exports: `d2l.ring_allreduce_time`, `d2l.tree_allreduce_time`, `d2l.hier_allreduce_time`, `d2l.a2a_inter_node_time`, `d2l.sharp_allreduce_time`, `d2l.sim_collective`, `d2l.hier_allreduce`; artifact `lm-hw-cluster-nccl` gains the per-(algorithm, protocol) sweeps, and its `alpha_beta.json` records carry `algo` and `proto` fields (schema change against `01_interfaces.md`; 6.x consumers read the default-pair rows unless they ask for a pair). Detail: blueprint ch05 5.5 (N2, N3, D1 to D3, risks). NVLS is forced on the T3 node if it is NVSwitch (one series); the hand-written hierarchical version uses two process groups, as the device mesh is 6.1's.
**Sources:**
- Papers: [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786) (protocols, algorithms, channels, tuner); Patarasuk and Yuan, [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://doi.org/10.1016/j.jpdc.2008.09.002) (the ring's optimality); Thakur, Rabenseifner and Gropp, "Optimization of Collective Communication Operations in MPICH" (IJHPCA 2005; cited by title, the book's existing bibliography key) (recursive halving and doubling); [TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches](https://arxiv.org/abs/2111.04867) (synthesized topology-specific algorithms, one sentence); [Rail-only](https://arxiv.org/abs/2307.12169) (what rails mean for the hierarchical all-reduce).
- Code and documentation: [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MIN_NCHANNELS`, `NCCL_IB_HCA`); [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html); NVIDIA, [Massively scale deep learning training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) (double binary trees) and [Doubling all2all performance with NCCL 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/) (PXN); [nccl-tests](https://github.com/NVIDIA/nccl-tests); [SHARP documentation](https://docs.nvidia.com/networking/display/sharpv300).
- Courses, tutorials and blogs: [Scaling Book: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) (collectives on GPU networks); Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (NCCL tuning and benchmark scripts); [CMU 11-968 LLM Systems syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf) (communication before parallelism).
-->

<!-- Placeholder (prose): opening, from the reader. The reader fitted an
alpha and a beta to all-reduces in sec_multi_gpu_concise and to every link
class of this box in sec_lm_server_paths (one line t = alpha + S / beta per
collective), and that line fits NCCL's default well. Two symptoms on the
box (the cell below prints both from the committed box-coll run, which the
tuner section computes further down): at one mid-size message, forcing
another (algorithm, protocol) pair beats NCCL's own choice (quote the size,
both pairs and both times as printed; if the line says NCCL is within
noise everywhere, say so and drop this symptom); and NCCL's own model of
its large-message pair assumes a bus bandwidth about ten times what the
box measures (quote both numbers: the tuner's model is a table of
latencies and bandwidths, not a measurement). The
question: what the library does between the ring and the tree, which
protocol carries each message, and whether its choice is right. Idea (plan):
a collective's time is an algorithm's step count times a protocol's latency
plus the bytes over the slowest link it uses, and the library picks both per
message size and topology. Hand-off: no new alpha-beta theory (22.5), no
ring derivation (subsec_mg-ring, eq_ring_traffic), busbw in one sentence. -->

```{.python .input #collectives-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #collectives-extras}
%%tab pytorch
import collections
import heapq
import itertools
import pandas as pd
from torch import distributed as dist
```

```{.python .input #collectives-opening}
%%tab pytorch
n = torch.cuda.device_count()  # sizes the live checks below
ab = d2l.box_alpha_beta()  # sec_lm_server_paths' all-reduce on all cards
d2l.coll_symptoms(d2l.hw_box_log('box-coll'), ab)
```

## Algorithms
:label:`subsec_lm_hw_coll_algos`

<!-- Placeholder (prose): the algorithms as step counts and bytes per
rank. The ring, restated in two sentences (2 (G - 1) steps of S / G each;
eq_ring_traffic). The pipelined double binary tree (NCCL 2.4 post): two
binary trees over the ranks, each carrying half the message in small
chunks, reduced up and broadcast down; every rank is a leaf in one tree
and interior in the other, so each rank sends 2 S bytes in all and pays 2 d
latency steps (d hops from a leaf to the root), the depth growing as log G
instead of G. Across nodes the tree runs inside a node as a chain of the g
GPUs, so d = (g - 1) hops at the node's latency plus log2 n hops across
nodes. The hierarchical all-reduce: reduce-scatter in the node,
all-reduce of each GPU's shard of S / g across the nodes on its own NIC (a
rail, Rail-only paper), all-gather in the node; the slow network carries
S / g per GPU instead of S. In-network reduction (SHARP in InfiniBand
switches, NVLS inside an NVSwitch fabric): each GPU sends S once and
receives the sum once, bandwidth term S / beta, a factor 2 (G - 1) / G
below the ring's, so its busbw can approach twice the link rate. The
cross-node all-to-all: a share g / G of each GPU's bytes stays in the node,
the rest crosses its NIC; PXN (NCCL 2.12 post) first moves a byte over the
node's links to the GPU on the destination's rail (its time function,
d2l.a2a_inter_node_time, is in the library: this section computes nothing
with it). One sentence that these are the time functions chapter 23
composes. -->

<!-- Figure: fig_lm_hw_ring_tree (generator ring-tree): a ring of 8 ranks
against NCCL's double binary tree on 8 ranks (data halves; each rank a leaf
in one tree and interior in the other, colours per tree). -->

```{.python .input #collectives-closed-forms-def}
%%tab pytorch
def ring_allreduce_time(S, G, alpha, beta):  #@save
    """Seconds of a ring all-reduce of S bytes on G ranks: 2 (G - 1)
    steps, each sending S / G bytes over a link of beta bytes/s and paying
    alpha seconds of latency."""
    return 2 * (G - 1) * (alpha + S / (G * beta))

def tree_allreduce_time(S, n, g, a_in, a_out, beta, pipelined=True):  #@save
    """Seconds of a double-binary-tree all-reduce of S bytes across n
    nodes whose g GPUs form a chain: a leaf is g - 1 hops (latency a_in)
    plus ceil(log2 n) hops (a_out) from its root, paid up and down.
    Pipelined in small chunks, every rank sends 2 S bytes in all;
    unpipelined (one tree), every hop carries the whole message."""
    up = math.ceil(math.log2(n))
    latency = (g - 1) * a_in + up * a_out
    if pipelined:
        return 2 * latency + 2 * S / beta
    return 2 * (latency + (g - 1 + up) * S / beta)
```

### Ring against Tree
:label:`subsec_lm_hw_ring_tree`

<!-- Placeholder (prose): derivation D1 (blueprint ch05 5.5): with G ranks,
per-rank bandwidth beta and per-step latency alpha, T_ring = 2 (G - 1) alpha
+ 2 ((G - 1) / G) S / beta and T_tree = 2 d alpha + 2 S / beta; their
difference 2 alpha (G - 1 - d) - 2 S / (G beta) changes sign at S* = G alpha
beta (G - 1 - d): below it the tree's log-depth latency wins, above it the
ring's slightly smaller bandwidth term. The smallest numeric case (plan):
G = 16 ranks, alpha = 10 us, beta = 25 GB/s (a 200 Gb/s NIC). With one rank
per node the tree is binary, d = log2 16 = 4, and S* = 44 MB (first line).
On two nodes of eight (the plan's "two nodes") NCCL's tree runs as a chain
of the eight GPUs inside each node plus one hop across, d = 7 + 1 = 8, and
S* drops to 28 MB (second line; the same alpha inside and across the
nodes, an idealization). -->

```{.python .input #collectives-closed-forms}
%%tab pytorch
G, alpha, beta = 16, 10e-6, 25e9  # ranks, s, bytes/s (a 200 Gb/s NIC)
for n_nodes, g in ((16, 1), (2, 8)):  # one rank per node; two nodes of 8
    d = (g - 1) + math.ceil(math.log2(n_nodes))  # hops, leaf to root
    S_star = G * alpha * beta * (G - 1 - d)  # D1
    t = ring_allreduce_time(S_star, G, alpha, beta)
    assert math.isclose(
        t, tree_allreduce_time(S_star, n_nodes, g, alpha, alpha, beta))
    print(f'{n_nodes} x {g}: d = {d}, S* = {S_star / 1e6:.0f} MB (ring = '
          f'tree = {t * 1e3:.2f} ms there)')
```

<!-- Placeholder (prose): D2, the hierarchical all-reduce and in-network
reduction as time functions. The hierarchical one replaces the flat ring's
(G - 1) steps at NIC bandwidth by (g - 1) steps at NVLink bandwidth plus
(n - 1) steps at NIC bandwidth on a shard of S / g. At 1 GB on 2 nodes of
8, against a flat ring that leaves each node through one NIC (every step
as slow as the NIC), with NVLink 3's per-direction rate inside the node
(links table, quoted): quote both hierarchical times. Most of the gain is
the rails: with one NIC per GPU the eight shards leave the node in
parallel (first speedup); squeezed through one NIC they do not (second).
Without latency and with NVLink much faster than the NIC the speedup is
(G - 1) / (n - 1): 15 on two nodes, falling toward g as the node count
grows (third line; derive it in one line). In-network reduction: each GPU
sends S once and receives the sum once, so the bandwidth term is S / beta,
a factor 2 (G - 1) / G below the ring's, and the bus bandwidth (the
nccl-tests convention) exceeds the link rate (fourth line). -->

```{.python .input #collectives-hier-forms-def}
%%tab pytorch
def hier_allreduce_time(S, n, g, a_in, b_in, a_out, b_out, rails=1):  #@save
    """Seconds of a hierarchical all-reduce on n nodes of g GPUs:
    reduce-scatter in the node, a ring all-reduce of each GPU's shard of
    S / g across the nodes, all-gather in the node. The g shards leave a
    node on its `rails` NICs (rails = g: one per GPU), each at b_out
    min(rails, g) / g."""
    intra = 2 * (g - 1) * (a_in + S / (g * b_in))  # both intra stages
    b_shard = b_out * min(rails, g) / g
    return intra + ring_allreduce_time(S / g, n, a_out, b_shard)

def sharp_allreduce_time(S, alpha, beta):  #@save
    """Seconds of an in-network all-reduce (SHARP, NVLS): each GPU sends S
    once to the switches and receives the reduced S once."""
    return alpha + S / beta
```

```{.python .input #collectives-hier-forms}
%%tab pytorch
n_nodes, g, S = 2, 8, 1e9  # two nodes of eight GPUs, one 1 GB all-reduce
b_in = d2l.links_table(show=False).loc['NVLink 3 (A100 SXM)',
                                       'gbs_per_dir'] * 1e9
flat = ring_allreduce_time(S, n_nodes * g, alpha, beta)  # through one NIC
hier = {rails: hier_allreduce_time(S, n_nodes, g, alpha, b_in, alpha, beta,
                                   rails) for rails in (g, 1)}
print(f'1 GB on {n_nodes} x {g}: flat ring through one NIC per node '
      f'{flat * 1e3:.0f} ms')
print(f'hierarchical: {hier[g] * 1e3:.0f} ms on {g} NICs per node '
      f'({flat / hier[g]:.1f}x), {hier[1] * 1e3:.0f} ms on one '
      f'({flat / hier[1]:.1f}x)')
limit = {n: ring_allreduce_time(S, n * g, 0, beta) / hier_allreduce_time(
    S, n, g, 0, math.inf, 0, beta, rails=g) for n in (2, 8, 64)}
print(f"alpha = 0, b_in = inf, on {', '.join(map(str, limit))} nodes: "
      + ', '.join(f'{x:.1f}x' for x in limit.values()) + f' (g = {g})')
t_net = sharp_allreduce_time(S, alpha, beta)
busbw = d2l.nccl_busbw_factor('all_reduce', n_nodes * g) * S / t_net
print(f'in-network: {t_net * 1e3:.0f} ms (flat ring {flat * 1e3:.0f} ms), '
      f'busbw {busbw / 1e9:.0f} GB/s on {beta / 1e9:.0f} GB/s links')
```

<!-- Placeholder (prose): NCCL's two trees. In the first tree the odd ranks
are leaves and each even rank r has the children r - b / 2 and r + b / 2
(b its lowest set bit); the second tree mirrors the ranks (G - 1 - r) for
even G and shifts them by one for odd G, so a leaf of one tree is interior
in the other. One sentence: double_binary_tree builds both as parent
maps, as NCCL's trees.cc does. -->

```{.python .input #collectives-trees-def}
%%tab pytorch
def double_binary_tree(G):  #@save
    """NCCL's two trees over G ranks as {rank: parent} dicts (None at the
    root). In the first, the odd ranks are leaves; the second mirrors it
    (even G) or shifts it by one (odd G), so that every leaf of one tree
    is interior in the other."""
    def tree(rank_at):  # rank_at(position) -> rank
        parent = {}
        for p in range(G):
            bit = p & -p  # the lowest set bit: p's level in the tree
            up = None if p == 0 else (p ^ bit) | (bit << 1)
            if up is not None and up >= G:
                up = p ^ bit
            parent[rank_at(p)] = None if up is None else rank_at(up)
        return parent
    second = (lambda p: G - 1 - p) if G % 2 == 0 else (lambda p: (p + 1) % G)
    return tree(lambda p: p), tree(second)
```

<!-- Placeholder (prose): the simulator, the T0 check of the closed forms
(not of the hardware). Every rank has one sending and one receiving port of
beta bytes/s (full duplex, one switch); a message passes through both
ports at once (cut-through) and arrives alpha later; a port serves its
waiting messages in chunk order, as a channel processes its chunks in
order. The ring moves one segment per step, as the closed form counts it;
the tree cuts each half of S into 1,000 chunks. The sizes are in units of
the tree's own balance point S_b = d alpha beta, where its latency term
2 d alpha equals its bandwidth term 2 S / beta (S*, the ring/tree
crossover, grows as G squared and is no reference for the tree). Then claim
C1 as the table below reads it (ruling in the spec log): the ring is
exact; the tree is within 1% of 2 d alpha + 2 S / beta far from S_b on
both sides (0.01 and 100 S_b), and at S_b the simulated tree is faster
than the closed form by up to about a quarter, never slower: while one
tree's first chunks climb toward its root, each rank's ports already carry
the other tree's chunks, so part of the latency hides behind the transfer.
The closed form adds the two terms and is an upper bound, provided the
chunks are fine enough that the pipeline's fill (about 2 d S / (2 C beta)
for C chunks per half) is negligible: with 10 chunks instead of 1,000 the
simulator runs up to a third above it. Quote the largest deficit as
printed. -->

<!-- Placeholder (prose): fix pass F2 (C1, coordinator decision): the line
under the table states C1 as measured: ring exact; the tree exceeds the
closed form by at most 0.3% (at 100 S_b, G = 16 and 32), is within 0.5% of
it far from S_b, and up to 24.9% below it at S_b (G = 4), not monotone in G
(-24.9, -8.2, -12.3, -4.7%). Never "never above". Drop the cause sentence
above ("while one tree's first chunks climb ..."): F2's single-tree check
(one tree carrying all of S, same rules) sits near its own closed form at
S_b only for G = 4 (+0.1%) and 7-12% above it for G = 8-32, so the overlap
of the two trees is not shown to be the cause; state the pattern only. -->

```{.python .input #collectives-simulator-def}
%%tab pytorch
def sim_collective(topo, algo, S, chunks):  #@save
    """Seconds of an all-reduce of S bytes, simulated message by message.
    Every rank has a sending and a receiving port of topo['beta'] bytes/s;
    a port serves its waiting messages lowest chunk first, a message
    passes both ports at once and arrives topo['alpha'] s later. 'ring':
    2 (G - 1) steps of one S / G segment; 'tree': each half of S, cut into
    chunks, is reduced up one of the two trees and broadcast down it."""
    import heapq, itertools  # not in the d2l library preamble
    G, alpha, beta = topo['G'], topo['alpha'], topo['beta']
    Msg = collections.namedtuple('Msg', 'src dst nbytes chunk tag')
    tick, events, free, waiting = itertools.count(), [], {}, {}
    def at(t, *event):
        heapq.heappush(events, (t, next(tick), event))
    def serve(t, port):  # start the next waiting message if the port is idle
        if waiting.get(port) and free.get(port, 0.0) <= t:
            msg = heapq.heappop(waiting[port])[2]
            free[port] = t + msg.nbytes / beta
            at(free[port], 'port free', port, msg)
            if port[0] == 'out':  # cut-through: the receiver starts too
                offer(t, ('in', msg.dst), msg)
    def offer(t, port, msg):  # queue msg at port, ordered by its chunk
        heapq.heappush(waiting.setdefault(port, []),
                       (msg.chunk, next(tick), msg))
        serve(t, port)
    def send(t, *fields):  # fields: src, dst, nbytes, chunk, tag
        msg = Msg(*fields)
        offer(t, ('out', msg.src), msg)
    trees = double_binary_tree(G)
    kids = [{r: [c for c in tr if tr[c] == r] for r in tr} for tr in trees]
    def reduced(t, k, r, j):  # chunk j of tree k is summed at rank r
        for dst in kids[k][r] if trees[k][r] is None else [trees[k][r]]:
            send(t, r, dst, S / 2 / chunks, j,
                 (k, 'down' if trees[k][r] is None else 'up'))
    if algo == 'ring':
        for r in range(G):
            send(0.0, r, (r + 1) % G, S / G, 0, 0)  # tag: the step
    else:
        for j, k in itertools.product(range(chunks), range(2)):
            for r in trees[k]:
                if not kids[k][r]:  # a leaf starts chunk j up its tree
                    reduced(0.0, k, r, j)
    got, end = collections.Counter(), 0.0
    while events:
        t, _, (kind, port, msg) = heapq.heappop(events)
        if kind == 'port free':
            serve(t, port)
            if port[0] == 'in':
                end = max(end, t + alpha)
                at(t + alpha, 'arrived', None, msg)
            continue
        if algo == 'ring':  # tag: the step
            if msg.tag + 1 < 2 * (G - 1):
                send(t, msg.dst, (msg.dst + 1) % G, msg.nbytes, msg.chunk,
                     msg.tag + 1)
            continue
        k, way = msg.tag  # tree k, 'up' or 'down'
        if way == 'down':
            for c in kids[k][msg.dst]:
                send(t, msg.dst, c, msg.nbytes, msg.chunk, msg.tag)
        else:
            got[k, msg.dst, msg.chunk] += 1
            if got[k, msg.dst, msg.chunk] == len(kids[k][msg.dst]):
                reduced(t, k, msg.dst, msg.chunk)
    return end
```

```{.python .input #collectives-simulator}
%%tab pytorch
factors = (0.01, 1, 100)  # S / S_b: far below, at and far above balance
closed_form = {
    'ring': lambda S, G: ring_allreduce_time(S, G, alpha, beta),
    'tree': lambda S, G: tree_allreduce_time(S, G, 1, alpha, alpha, beta)}
cols = [f'{f:g} S_b (%)' for f in factors]
rows = []
for G in (4, 8, 16, 32):
    d = math.ceil(math.log2(G))  # depth of NCCL's binary tree
    S_b, topo = d * alpha * beta, d2l.sim_topology(G, alpha, beta)
    for algo, model in closed_form.items():
        rows.append([G, algo] + [100 * (sim_collective(
            topo, algo, f * S_b, chunks=1000) / model(f * S_b, G) - 1)
            for f in factors])
err = pd.DataFrame(rows, columns=['G', 'algo'] + cols)
err[cols] = err[cols].round(1) + 0.0  # + 0.0: no '-0.0' in the table
d2l.show_table(err, digits=dict.fromkeys(cols, 1), width=79)
d2l.sim_summary(err)  # claim C1, read from the table
assert (err[err.algo == 'ring'][cols].abs() < 1e-6).all().all()  # exact
assert (err[[cols[0], cols[-1]]].abs() < 1).all().all()  # C1: far from S_b
assert (err[cols] < 1).all().all()  # C1: less than 1% above the closed form
```

## Protocols
:label:`subsec_lm_hw_protocols`

<!-- Placeholder (prose): a protocol is how a receiver learns that the
bytes it waits for have arrived (Demystifying NCCL). Simple: bulk data,
then a flag after a memory fence; full bandwidth, the highest latency per
step. LL (low latency): every 8-byte line carries 4 bytes of data and a
4-byte flag written together, so the receiver polls the line itself; no
fence, lowest latency, half the bandwidth. LL128: 128-byte lines with 120
bytes of data and an 8-byte flag; near full bandwidth at low latency, but
it relies on 128-byte stores arriving in order, which NVLink guarantees and
PCIe does not, so NCCL disables it on PCIe-only nodes (the box's INFO
digest below shows it disabled: quote the line). Forced with NCCL_PROTO,
LL128 still runs here and passes a one-iteration result check, but that is
no evidence that it is safe: the NCCL environment page warns that forcing
a protocol NCCL has not enabled can corrupt data. The tuner section
therefore shows it (dashed, "off") and never counts it as a better
choice. Small messages take
LL, large ones Simple, and alpha belongs to an (algorithm, protocol) pair,
which is why the tuner section fits one alpha and beta per pair. -->

<!-- Figure: fig_lm_hw_protocols (generator protocols): bytes on the wire
for Simple (data block, fence, flag) against LL (8-byte lines of 4 data +
4 flag bytes) and LL128 (120 + 8). -->

## Channels, Buffers and SMs
:label:`subsec_lm_hw_channels`

<!-- Placeholder (prose): a collective is a kernel. NCCL launches one CUDA
block per channel; each channel runs its own ring (or tree) over its own
slice of the message with its own buffers in each peer's memory (or, on
this box, in shared host memory). More channels move more bytes in
parallel and take more SMs, so a collective competes with compute kernels
for SMs and memory bandwidth: one sentence that this is why overlapping
communication with computation in Chapter 23 is not free. Then the box's
own account, read from NCCL_DEBUG=INFO (the cell runs one all-reduce
sweep and digests the header): the version, the transport between every
pair of ranks (SHM/direct/direct: shared host memory, since peer access is
refused, sec_lm_server_paths), the number of channels and their ring
order, the tree of the first channels (inside one node NCCL's tree is a
chain of the GPUs, so ring and tree differ only in protocol choices here),
the tuner's model of each (algorithm, protocol) pair it may use (latency
and bandwidth; NCCL's table holds algorithm bandwidths, the digest converts
them to the bus bandwidth nccl-tests reports, the convention of every fit
in this chapter, so they compare directly with the measured 2.4 GB/s) and
the pairs it disabled, and the pair it chose at each size. Quote these
lines; they are the box's facts, deterministic for the pinned NCCL. A
reader on another machine sees their own. -->

```{.python .input #collectives-debug-header}
%%tab pytorch
info = d2l.nccl_info(gpus=tuple(range(min(n, 4))))  # NCCL_DEBUG=INFO, digested
```

## The Tuner
:label:`subsec_lm_hw_tuner`

<!-- Placeholder (prose): how NCCL chooses. At communicator creation it
detects the topology (PCIe tree, NVLink, NICs; the graph of
sec_lm_server_paths), searches ring and tree orders over it, and fills a
table of latency and bandwidth per (algorithm, protocol) (the model lines
above); for each call it predicts latency + size / bandwidth for every
enabled pair and takes the smallest. NCCL_ALGO and NCCL_PROTO force a pair
(NCCL environment page), which is how to test the tuner: run every pair
over all sizes and see whether the default is ever slower than the best
forced pair. The plot: each forced pair's time relative to the default's
(thin; the default is the bold black line at 1) against message size; the
dotted line marks where Ring/Simple overtakes Ring/LL; the LL128 pairs are
dashed and labelled "off" (NCCL's tuner disables them on this node,
Protocols). Forcing changes more than the pair: even the forced copy of
the pair NCCL itself chose sits off 1 at some sizes (the tuner verdict
below measures this "cost of forcing"), so a curve below 1 is not yet a
better choice. A pair NCCL refuses, that hangs (each sweep has a timeout),
returns wrong results, that NCCL quietly replaces (its INFO lines report
another pair) or whose run reports no choice at all is dropped from the
plot and the fits; the printed lines name it and why: quote them. Claim
C2, first half: LL wins at the smallest size and Simple at the largest;
the two lines under the plot give each gap in noise units, and a gap
within 3 units is called within noise (quote the verdicts; the assert
fails only if a size is clearly the other way round). Inside one node the
tree is a chain, so ring and tree differ little here; the ring/tree
crossover of D1 is a multi-node effect (pending C-T4). -->

```{.python .input #collectives-forced-pairs}
%%tab pytorch
pairs = d2l.nccl_algo_proto_sweeps('box')  # every NCCL_ALGO x NCCL_PROTO
d2l.plot_algo_crossover(pairs)
units = d2l.protocol_order(pairs)  # Ring/LL at 8 B, Ring/Simple at 256 MiB
assert (units > -3).all()  # C2: neither size clearly the other way round
```

<!-- Placeholder (prose): one alpha and one beta per pair, the two-regime
fit of sec_lm_server_paths (alpha from the messages up to 64 KiB, beta, in
the bus-bandwidth convention, from those of 4 MiB and more), with 95%
intervals over the three regeneration processes. The fits predict where
Ring/Simple overtakes Ring/LL, alpha_Simple - alpha_LL over phi (1 /
beta_LL - 1 / beta_Simple); claim C2, second half (spec fallback: quote
both sizes, not asserted). On this box the prediction is ill-conditioned:
every pair runs through the same host-staged shared-memory path, which
caps all of them near the same beta (quote the betas: LL is not at half
of Simple's here, because the protocol is not the bottleneck, the path
is), so the formula divides a small alpha difference by a small
difference of inverse betas (if a regeneration fits LL's beta at or above
Simple's, the line says that no crossover is predicted and why: say so).
The measured crossover comes from what the lines do not model (NCCL uses
fewer channels for small messages, visible in the INFO choice lines). Say
this plainly: the alpha-beta line of one pair is good, the difference of
two nearly equal lines is not. The "(off)" rows are LL128, fitted but not
a choice NCCL would make here. -->

<!-- Placeholder (prose): fix pass F2 (C2): the last line under the table
gives each fit against its measured times from 16 KiB to 1 MiB (box-coll
v2: Ring/LL +2% to +78%, Ring/Simple -2% to +47%) and the two fitted lines
against each other (+11% at 16 KiB, 0% at 1 MiB): the fits miss by more
than the lines differ, so they cannot place the crossover. C2 fails the 2x
bar (87 KiB measured, 600 KiB predicted; v1 gave 280: two nearly parallel
lines put their intersection anywhere, which is the point); say so, do not
widen the bar. -->

```{.python .input #collectives-fit-pairs}
%%tab pytorch
fits = d2l.fit_pairs(pairs)
d2l.show_pair_fits(fits)
```

<!-- Placeholder (prose): is the default right? Forcing a pair changes
more than the pair (the first line: the default against a forced copy of
its own choice, the "cost of forcing"; quote it), so the test compares
forced with forced: at every size the forced copy of the pair NCCL chose
against the best other forced pair NCCL's tuner enables (LL128 left out),
in noise units (the largest of the median absolute deviation over the
regeneration's processes, the cost of forcing at that size, and 2% of the
time). Claim C3: no enabled pair beats NCCL's choice by more than 3 noise
units. If the second line shows more, name the size and the pair as
printed: the tuner's table is a model of the hardware (its last line:
where the table switches from LL to Simple against where the forced pairs
cross; the INFO digest above showed how far its bandwidths are from the
measured ones), and on a PCIe node that stages every byte through host
memory its model can be wrong at some sizes; that misprediction is the
teaching point, stated, not hidden. -->

```{.python .input #collectives-tuner-verdict}
%%tab pytorch
regret = d2l.tuner_regret(pairs)
```

<!-- Placeholder (prose): fix pass F11 (claim C3, its fallback, as
measured): NCCL's choice is within 3 noise units of the best enabled forced
pair at 24 of 26 sizes, not at every size. The largest miss: at 128 KiB
forced Ring/Simple is 15% faster than NCCL's Ring/LL (6.9 noise units;
box-coll v2). The
tuner's model is the reason, stated as the teaching point: it credits
Ring/Simple with 24 GB/s busbw on this node against the measured 2.4, and
its table switches from LL to Simple at 470 KiB while the forced pairs
cross at 87 KiB. Quote size, pair and units from the printed lines. -->

<!-- Placeholder (prose): the hierarchical all-reduce by hand, from three
torch.distributed calls on two process groups (the device mesh is Chapter
23's): reduce-scatter in the intra group, all-reduce of the shard in the
inter group, all-gather in the intra group. The groups may run different
backends: here the intra group is NCCL and the inter group gloo, which
works on CPU tensors only, so the shard crosses through a pinned host
copy. -->

```{.python .input #collectives-hier-code}
%%tab pytorch
def hier_allreduce(tensor, intra_group, inter_group):  #@save
    """All-reduce tensor in place over every rank in three stages:
    reduce-scatter among the ranks of intra_group, all-reduce of each
    rank's shard across inter_group, all-gather in intra_group. The groups
    may use different backends: a CUDA tensor crosses a gloo group through
    a pinned host copy. tensor.numel() must be a multiple of the size of
    intra_group."""
    from torch import distributed as dist  # not in the d2l preamble
    flat = tensor.view(-1)
    shard = flat.new_empty(flat.numel() // dist.get_world_size(intra_group))
    dist.reduce_scatter_tensor(shard, flat, group=intra_group)
    if shard.is_cuda and dist.get_backend(inter_group) == 'gloo':
        host = torch.empty(shard.shape, dtype=shard.dtype, pin_memory=True)
        host.copy_(shard)
        dist.all_reduce(host, group=inter_group)
        shard.copy_(host)
    else:
        dist.all_reduce(shard, group=inter_group)
    dist.all_gather_into_tensor(flat, shard, group=intra_group)
    return tensor
```

<!-- Placeholder (prose): the box as two "nodes": the pairs 0-1 and 2-3
(each behind one host bridge) are the nodes, NCCL's shared-memory path is
their fast link, and gloo over the loopback interface between 0-2 and 1-3
is the slow network (an emulated slow network, labelled so; not a
datacenter network). The rows: the flat all-reduce on a gloo group of all
four (the all-slow baseline), on NCCL (the default), the hierarchical one,
and its two stages alone; measured time against hier_allreduce_time and
ring_allreduce_time with the committed fits of sec_lm_server_paths (the
NCCL pair, all four cards) and sec_lm_networks (gloo on 2 and 4 ranks).
Claim C4 takes the branch the last line before the live lines prints (the
gate fixed in the spec log, "The 22.6 gate rule", computed from gloo's
4-rank fit against NCCL's shared-memory path on the four cards; the
"2-rank stages" line is not the gate's quantity). If it says "C4 stands",
the hierarchical all-reduce beats the flat gloo baseline by a factor that
approaches the intra/inter bandwidth ratio at large sizes (quote the gain
and the ratio as printed); if it says "C4 fallback", the box cannot
emulate a slow network by bandwidth (gloo is the slow group by latency
only) and C4 is only the next sentence. Either way the closed form
predicts the measured gain within 20% (asserted when the fits are there),
and the bandwidth and latency ratios of the gate line are stated. The
model leaves out the host copies of the gloo stage, which is why the
inter stage runs above it. Correctness: hier_allreduce equals a
CPU sum of every rank's input (the line below; on a machine without four
GPUs it runs on four CPU processes with gloo). -->

```{.python .input #collectives-hier-run}
%%tab pytorch
hier = d2l.hier_vs_flat('box')  # 2 x 2: NCCL pairs, gloo across them
at = hier[hier['size'] == 2**28].set_index('variant')  # 256 MiB
gain = at.us['flat gloo'] / at.us['hierarchical']
predicted = at.model_us['flat gloo'] / at.model_us['hierarchical']
if not np.isnan(predicted):  # NaN without the two earlier sections' fits
    assert abs(gain / predicted - 1) < 0.2  # C4: the closed form predicts it
```

<!-- Placeholder (prose): fix pass F6: the gloo ranks are pinned to physical cores from box-net v2 and box-coll v2 on (sec_lm_networks), which raises gloo's beta and halves its alpha; C4's branch, the gain and the gate's bandwidth and latency ratios are quoted from the v2 pair only, and stated as properties of this box's gloo setup, not of gloo itself. -->

<!-- Placeholder (prose): fix pass F12 (claim C4, from the v2 pair): the gate
fails clearly now (gloo's pinned 4-rank beta 1.46 GB/s, 95% 1.45-1.47,
against NCCL's shared memory 2.36: 1.6x the bandwidth, 1/37 the latency),
so C4 is its fallback: the closed form predicts the measured
hierarchical/flat gain within 20% (at 256 MiB 1.4x measured, 1.6x by the
model; the v1 "at its edge" wording is gone). Separately, stated and not
part of the 20% test: the intra/inter bandwidth ratio is 2.0x measured and
1.2x from the fits, and the model's inter stage (66 ms at 256 MiB) is about
half the measured 122 ms (candidate cause, not measured: the loopback fit
leaves out the gloo stage's copies between GPU and host), so the measured
gain falls short of the model's. By the gate's rule this box's gloo setup cannot emulate a slow
network by bandwidth (it is slower by latency). -->

<!-- Placeholder (prose): fix pass F12 review (C4): the latency ratio in
the gate line ("1/37") comes from box-net v2's 4-rank gloo alpha, which a
start-up transient inflates by about 10% (sec_lm_networks, N2); about 1/34
from v2's steady points. Quote it from box-net v3. C4's branch, its gain
and every bandwidth figure are unchanged (256 MiB is bandwidth-bound). -->

<!-- Placeholder (prose): fix pass runs (C4 gate from box-net v3): gloo on
4 ranks 1.42 GB/s busbw (95% 1.25-1.58) against NCCL's shared memory 2.36:
1.7x the bandwidth, 1/34 the latency; the interval is not below half, so
C4 stays at its fallback. At 256 MiB hierarchical 1.4x faster than flat
gloo, 1.6x by the model (flat gloo model 285 ms, from v3's beta). -->

<!-- Placeholder (prose): what the box cannot show. The ring/tree crossover
of D1 needs a tree deeper than a chain (several nodes), the rail effect of
D2 needs one NIC per GPU, and SHARP and NVLS need the switches. Campaign
C-T4 (two 8 x A100 nodes with RDMA, shared with sec_lm_networks) runs the
same forced pairs, NVLS or CollNet where the fabric has them, and the
hierarchical all-reduce against the default with all NICs and with one
(NCCL_IB_HCA). Claim C5 (pending): the measured ring/tree crossover within
2x of D1, and the hierarchical all-reduce close to the default on all rails
and faster than it on one NIC. Until the log is committed the cell prints
one status line and the prose says "pending: part of the A100 campaign". -->
<!-- Placeholder (prose and code, fix pass F14 review): C5's second half is
the log's entry `hier` (tools/hw_campaign/hier_t4.py: the page's
hier_allreduce with NCCL inside and across the two nodes against NCCL's
default, column nics 'all' or 'one'); when the log is registered the cell
below also prints the default and the hierarchical time at the largest
size for each, and the sentence quotes them. -->
<!-- campaign: lm-hw-cluster-nccl-v1 -->

```{.python .input #collectives-cluster-twin}
%%tab pytorch
d2l.show_pair_fits(d2l.fit_pairs(d2l.hw_campaign_log('cluster-nccl')))
```

## Summary

<!-- Placeholder (prose): 3-6 sentences: a collective's time is steps times
a protocol's latency plus bytes over the slowest link; the tree trades the
ring's G - 1 latency steps for log G, the hierarchical all-reduce keeps
the slow network's bytes at S / g, in-network reduction halves the
bandwidth term; NCCL picks the pair from a table and on this box (quote
the verdict of C3); forward to Chapter 23 (which collective each parallel
axis pays) by :numref: if its label resolves. -->

## Exercises

<!-- Placeholder (prose): 7-9 exercises per S/spec/22.7.md part 8: each
rank sends 2 S over the two trees (derive); S* for 2, 4 and 16 nodes with
the fitted alpha and beta (derive); the in-network bound and its busbw
(derive); busbw against NCCL_MIN_NCHANNELS on this box (measure, 3 min, 4
GPUs); a matmul running concurrently with an all-reduce and the slowdown of
each (measure, 3 min, 2 GPUs); the INFO header against the fitted bandwidth
(check); open: a two-dimensional all-reduce that beats the default on a
one-NIC node. -->
