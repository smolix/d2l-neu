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
collective). Two symptoms on the box (the cell below prints both from the
committed box-coll run, which the tuner section computes further down):
a 4 KiB all-reduce on the four cards against the 22.5 line; the bus
bandwidth of NCCL's default at 256 MiB against the best single forced
(algorithm, protocol) pair. Quote both numbers as printed and say which way
each one points (the line is a fit of NCCL's default, so a small message
off the line means the default changes how it sends small messages). The
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
import pandas as pd
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
node's links to the GPU on the destination's rail. One sentence that these
are the time functions chapter 23 composes. -->

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

def hier_allreduce_time(S, n, g, a_in, b_in, a_out, b_out, rails=1):  #@save
    """Seconds of a hierarchical all-reduce on n nodes of g GPUs:
    reduce-scatter in the node, a ring all-reduce of each GPU's shard of
    S / g across the nodes, all-gather in the node. The g shards leave a
    node on its `rails` NICs (rails = g: one per GPU), each at b_out
    min(rails, g) / g."""
    intra = 2 * (g - 1) * (a_in + S / (g * b_in))  # both intra stages
    b_shard = b_out * min(rails, g) / g
    return intra + ring_allreduce_time(S / g, n, a_out, b_shard)

def a2a_inter_node_time(S, n, g, a_out, b_out, a_in, b_in):  #@save
    """Seconds of an all-to-all of S bytes per GPU across n nodes of g
    GPUs, one NIC per GPU (PXN routes each byte to the GPU on its
    destination's rail first): the share g / G stays in the node, the rest
    crosses the GPU's own NIC, and the two run at the same time."""
    G = n * g
    inside = (g - 1) * a_in + S * (g - 1) / (G * b_in)
    across = (n - 1) * a_out + S * (G - g) / (G * b_out)
    return max(inside, across)

def sharp_allreduce_time(S, alpha, beta):  #@save
    """Seconds of an in-network all-reduce (SHARP, NVLS): each GPU sends S
    once to the switches and receives the reduced S once."""
    return alpha + S / beta
```

### Ring against Tree
:label:`subsec_lm_hw_ring_tree`

<!-- Placeholder (prose): derivation D1 (blueprint ch05 5.5): with G ranks,
per-rank bandwidth beta and per-step latency alpha, T_ring = 2 (G - 1) alpha
+ 2 ((G - 1) / G) S / beta and T_tree = 2 d alpha + 2 S / beta; their
difference 2 alpha (G - 1 - d) - 2 S / (G beta) changes sign at S* = G alpha
beta (G - 1 - d): below it the tree's log-depth latency wins, above it the
ring's slightly smaller bandwidth term. The smallest numeric case (plan):
G = 16 ranks, alpha = 10 us, beta = 25 GB/s (a 200 Gb/s NIC), d = 4 gives
S* = 44 MB, the first line below. D2: the hierarchical all-reduce at 1 GB
on 2 nodes of 8 against a flat ring that leaves each node through one NIC
(every step as slow as the NIC), with NVLink 3's per-direction rate inside
the node (links table, quoted) and one NIC per GPU: quote the speedup
printed; as the intra/inter ratio and the node count grow it tends to g
(third line). -->

```{.python .input #collectives-closed-forms}
%%tab pytorch
G, alpha, beta, d = 16, 10e-6, 25e9, 4  # ranks, s, bytes/s, d = log2 G
S_star = G * alpha * beta * (G - 1 - d)
t = ring_allreduce_time(S_star, G, alpha, beta)
assert math.isclose(t, tree_allreduce_time(S_star, G, 1, alpha, alpha, beta))
print(f'S* = {S_star / 1e6:.0f} MB (ring = tree = {t * 1e3:.2f} ms there)')
n_nodes, g, S = 2, 8, 1e9  # two nodes of eight GPUs, one 1 GB all-reduce
b_in = d2l.links_table(show=False).loc['NVLink 3 (A100 SXM)',
                                       'gbs_per_dir'] * 1e9
flat = ring_allreduce_time(S, n_nodes * g, alpha, beta)  # one NIC per node
hier = hier_allreduce_time(S, n_nodes, g, alpha, b_in, alpha, beta, rails=g)
print(f'1 GB on {n_nodes} x {g}: flat ring {flat * 1e3:.0f} ms, '
      f'hierarchical {hier * 1e3:.0f} ms, speedup {flat / hier:.1f}x')
big = 10**4  # b_in / b_out and the number of nodes, both large
limit = (ring_allreduce_time(S, big * g, alpha, beta) /
         hier_allreduce_time(S, big, g, alpha, big * beta, alpha, beta,
                             rails=g))
print(f'speedup with b_in / b_out = {big:,} on {big:,} nodes: '
      f'{limit:.2f} (g = {g})')
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
the tree cuts each half of S into 1,000 chunks. Then claim C1 as the table
below reads it: the ring is exact; the tree matches 2 d alpha + 2 S / beta
in the bandwidth limit (100 S*, within 1%), and where latency and
bandwidth are of similar size (G = 4 at S*, G = 16 at S* / 100) the
simulated tree is up to about a sixth faster than the closed form, never
slower: while one tree's first chunks climb toward its root, each rank's
ports already carry the other tree's chunks, so part of the latency hides
behind the transfer. The closed form adds the two terms and is an upper
bound; quote the largest deficit as printed. -->

```{.python .input #collectives-simulator-def}
%%tab pytorch
def sim_collective(topo, algo, S, chunks):  #@save
    """Seconds of an all-reduce of S bytes, simulated message by message.
    Every rank has a sending and a receiving port of topo['beta'] bytes/s;
    a port serves its waiting messages lowest chunk first, a message
    passes both ports at once and arrives topo['alpha'] s later. 'ring':
    2 (G - 1) steps of one S / G segment; 'tree': each half of S, cut into
    chunks, is reduced up one of the two trees and broadcast down it."""
    import collections, heapq, itertools
    G, alpha, beta = topo['G'], topo['alpha'], topo['beta']
    tick, events, free, waiting = itertools.count(), [], {}, {}
    def at(t, *event):
        heapq.heappush(events, (t, next(tick), event))
    def serve(t, port):  # start the next waiting message if the port is idle
        if waiting.get(port) and free.get(port, 0.0) <= t:
            msg = heapq.heappop(waiting[port])[2]
            free[port] = t + msg[2] / beta
            at(free[port], 'port free', port, msg)
            if port[0] == 'out':  # cut-through: the receiver starts too
                offer(t, ('in', msg[1]), msg)
    def offer(t, port, msg):  # msg: (src, dst, nbytes, chunk, tag)
        heapq.heappush(waiting.setdefault(port, []), (msg[3], next(tick), msg))
        serve(t, port)
    def send(t, src, dst, nbytes, chunk, tag):
        offer(t, ('out', src), (src, dst, nbytes, chunk, tag))
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
        src, dst, nbytes, j, tag = msg
        if algo == 'ring':
            if tag + 1 < 2 * (G - 1):
                send(t, dst, (dst + 1) % G, nbytes, j, tag + 1)
        elif tag[1] == 'down':
            for c in kids[tag[0]][dst]:
                send(t, dst, c, nbytes, j, tag)
        else:
            got[tag[0], dst, j] += 1
            if got[tag[0], dst, j] == len(kids[tag[0]][dst]):
                reduced(t, tag[0], dst, j)
    return end
```

```{.python .input #collectives-simulator}
%%tab pytorch
factors = (0.01, 1, 100)  # S / S*: far below, at and far above the crossover
rows = []
for G in (4, 8, 16, 32):
    d = math.ceil(math.log2(G))  # depth of NCCL's binary tree
    S_star, topo = G * alpha * beta * (G - 1 - d), d2l.sim_topology(
        G, alpha, beta)
    for algo in ('ring', 'tree'):
        model = (lambda S: ring_allreduce_time(S, G, alpha, beta)) if (
            algo == 'ring') else (lambda S: tree_allreduce_time(
                S, G, 1, alpha, alpha, beta))
        rows.append(dict(G=G, algo=algo, **{
            f'{f:g} S* (%)': 100 * (sim_collective(topo, algo, f * S_star,
                                                   chunks=1000)
                                    / model(f * S_star) - 1)
            for f in factors}))
err = pd.DataFrame(rows)
cols = [c for c in err.columns if c.endswith('(%)')]
err[cols] = err[cols].round(1) + 0.0  # + 0.0: no '-0.0' in the table
d2l.show_table(err, digits=dict.fromkeys(cols, 1), width=79)
assert (err[err.algo == 'ring'][cols].abs() < 1e-6).all().all()  # exact
assert (err[err.algo == 'tree']['100 S* (%)'].abs() < 1).all()  # C1
assert (err[cols] < 1).all().all()  # the closed form is an upper bound
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
digest below shows it disabled or not: quote the line). Small messages take
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
and bandwidth) and the pairs it disabled, and the pair it chose at each
size. Quote these lines; they are the box's facts, deterministic for the
pinned NCCL. A reader on another machine sees their own. -->

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
(thin; the default is the bold black line at 1) against message size; a
pair below 1 beats the tuner at that size; the dotted line marks where
Ring/Simple overtakes Ring/LL. A pair NCCL refuses, that hangs (each sweep
has a timeout), returns wrong results, or that NCCL quietly replaces (its
INFO lines report another pair) is dropped from the plot and the fits; the
printed lines name it and why: quote them. Claim C2, first half: LL wins at
the smallest sizes and Simple at the largest (asserted). Inside one node
the tree is a chain, so ring and tree differ little here; the ring/tree
crossover of D1 is a multi-node effect (pending C-T4). -->

```{.python .input #collectives-forced-pairs}
%%tab pytorch
pairs = d2l.nccl_algo_proto_sweeps('box')  # every NCCL_ALGO x NCCL_PROTO
d2l.plot_algo_crossover(pairs)
smallest, largest = 8, 2**28  # bytes
for S, faster, slower in ((smallest, 'Ring/LL', 'Ring/Simple'),
                          (largest, 'Ring/Simple', 'Ring/LL')):
    assert (d2l.pair_time(pairs, faster, S)
            < d2l.pair_time(pairs, slower, S))  # C2: LL small, Simple large
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
difference of inverse betas. The measured crossover comes from what the
lines do not model (NCCL uses fewer channels for small messages, visible
in the INFO choice lines). Say this plainly: the alpha-beta line of one
pair is good, the difference of two nearly equal lines is not. -->

```{.python .input #collectives-fit-pairs}
%%tab pytorch
fits = d2l.fit_pairs(pairs)
d2l.show_pair_fits(fits)
```

<!-- Placeholder (prose): is the default right? At every size the default's
time against the best forced pair's, in units of the run-to-run noise (the
median absolute deviation over the three regeneration processes, at least
2% of the time). Claim C3: the default is never slower than the best forced
pair by more than 3 noise units. If the first line shows more, name the
size and the pair as printed: the tuner's table is a model of the hardware,
and on a PCIe node that stages every byte through host memory its model can
be wrong at some sizes; that misprediction is the teaching point, stated,
not hidden. -->

```{.python .input #collectives-tuner-verdict}
%%tab pytorch
regret = d2l.tuner_regret(pairs)
```

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
    from torch import distributed as dist
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
Claim C4 takes the branch the 22.6 gate chose (S/spec/spec-log.md, "The
22.6 gate rule"): if gloo's loopback beta is below half the shared-memory
beta, the hierarchical all-reduce beats the flat gloo baseline by a factor
that approaches the intra/inter bandwidth ratio at large sizes (quote the
gain and the ratio as printed); otherwise the box cannot emulate a slow
network by bandwidth and C4 is only the next sentence. Either way the
closed form predicts the measured gain within 20% (asserted), and the
intra/inter bandwidth ratio is stated. Correctness: hier_allreduce equals a
CPU sum of every rank's input (the line below; on a machine without four
GPUs it runs on four CPU processes with gloo). -->

```{.python .input #collectives-hier-run}
%%tab pytorch
hier = d2l.hier_vs_flat('box')  # 2 x 2: NCCL pairs, gloo across them
at = hier[hier['size'] == 2**28].set_index('variant')  # 256 MiB
gain, predicted = (at[c]['flat gloo'] / at[c]['hierarchical']
                   for c in ('us', 'model_us'))
assert abs(gain / predicted - 1) < 0.2  # C4: the closed form predicts it
```

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
