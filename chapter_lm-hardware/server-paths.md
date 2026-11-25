# Data Paths inside a Server
:label:`sec_lm_server_paths`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.5 Data Paths inside a Server · T3 and T2 pair (T1 reduced)").
Blueprint: docs/llm/blueprints/ch05_hardware.md section 5.4.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A server is a graph whose edges are links with their own alpha and beta, and the topology decides which pairs of devices can talk fast and what an all-to-all costs.
**Opening:** The reader read the build box's PCIe tree in `subsec_hw-interconnects` and saw NCCL fall to an effective 2 GB/s on its host-staged path. One concrete example: `nvidia-smi topo -m` on an 8-GPU node prints a matrix of `NV12`, `PIX`, `NODE` and `SYS` entries, and the same all-reduce runs five times faster on one pair of GPUs than on another (to be measured). The open question: how to read the matrix as a graph, measure each edge, and predict which collective suffers on which topology.
**Teaches:**
- The server as a graph: GPUs, NVSwitches or NVLink bridges, PCIe switches, CPU sockets, NICs and NVMe drives; SXM against PCIe parts; NVLink generations and NVSwitch, PCIe generations and switch trees, CPU-GPU links (PCIe against NVLink-C2C), AMD's Infinity Fabric; a link table with per-direction rates computed from lanes and encoding for PCIe and transcribed from vendor pages otherwise.
- GPUDirect peer-to-peer and RDMA, one NIC per GPU on the same PCIe switch, NUMA and CPU-NIC affinity; copy engines against SM-driven copies; the latency floor of a small transfer.
- The alpha-beta model as a model: t(m) = alpha + m / beta per link, fitted by weighted least squares, with the half-bandwidth message size alpha beta; collectives inherit the form with the algorithm's byte factor as slope.
- Bisection bandwidth: on a full crossbar all-reduce and all-to-all cost about the same per byte, but on a two-level tree whose uplink is slower the all-to-all is bisection-limited and the ring all-reduce is not; why expert parallelism needs the switch and data parallelism tolerates the tree; what the bus-bandwidth normalization hides.
- How the server's topology decides where each parallel axis goes, in one paragraph (Chapter 6 develops it).
**Derivation:** Bisection against the ring. On a crossbar of k GPUs with per-direction bandwidth beta, an all-to-all of S bytes per GPU takes ((k-1)/k) S / beta and a ring all-reduce 2((k-1)/k) S / beta; on a two-level tree with two groups of k/2 joined by one uplink of bandwidth beta_u, the ring all-reduce is unchanged when beta_u >= beta because the ring crosses the uplink once per direction per step, while the all-to-all pushes k S / 4 bytes per direction across the uplink, so its time is at least k S / (4 beta_u) and the ratio to the crossbar all-to-all is (k^2 / (4 (k-1))) (beta / beta_u) (blueprint ch05 5.4 D2). Smallest numeric case: k = 8 gives 64/28, about 2.3 times beta / beta_u, and it grows linearly in k.
**Notebook:**
1. Question: do the fitted alpha-beta parameters of each link class on the node, and the predicted all-to-all against all-reduce ratio, match what nccl-tests measures on the switched node and on a host-staged path?
2. Read `nvidia-smi topo -m` into `d2l.parse_topo_matrix` and draw the graph with `d2l.topo_graph`; the committed `topo.txt` decides whether the node is NVSwitch or NVLink-bridged PCIe pairs, and the text says which was measured.
3. The GPU-to-GPU bandwidth and latency matrix with `p2pBandwidthLatencyTest` (or `nvbandwidth`'s device-to-device matrices); host-to-device per NUMA node from 5.4's logs.
4. Intra-node nccl-tests sweeps for all-reduce, all-gather, reduce-scatter and all-to-all at g = 2, 4, 8, default algorithm and `NCCL_ALGO=Ring`, with and without `NCCL_P2P_DISABLE=1` (the node's emulation of a host-staged path); parse with `d2l.parse_nccl_tests`; fit per link class and operation with `d2l.fit_alpha_beta`; write `alpha_beta.json`.
5. The T2 pair: the same at g = 2 over the NVLink bridge and with peer-to-peer disabled, which is the bridge-against-host contrast every reader of the logs can see.
6. The prediction: the busbw ratio of all-to-all to all-reduce near 1 on the switched node and below 1 under `NCCL_P2P_DISABLE=1`; `d2l.tree_link_loads` routes both patterns through the tree and returns the busiest link exactly.
7. Reduced T1 configuration: the build box runs no GPU-to-GPU traffic, so T1 captures its topology, runs the host-device series on one GPU, and tests the parsers and the fit on committed samples.
8. Claims that survive noise: fitted beta per link class within 20% of the datasheet per-direction rate where one exists; the all-to-all ratio falls below 1 on the host-staged path and stays near 1 on the switch (direction, with the measured number printed).
**Figures:** `fig_lm_hw_server_graph` (the 8-GPU node as a graph, generated from `topo.txt`, edges labelled with measured beta); `fig_lm_hw_p2p_matrix` (plot: the bandwidth matrix as a heat map); `fig_lm_hw_bisection` (crossbar against two-level tree, the uplink load of all-reduce against all-to-all).
**Restate / forward:** `subsec_hw-interconnects` and `fig_pcie_topology` (the build box as the worked host-staged example; regenerated from text, not redrawn); `eq_ring_traffic` and `subsec_mg-accounting` (the ring moves 2(k-1)/k of the message; the busbw convention); `tab_collectives` (all-to-all as the expert collective); the `NCCL_SHM_USE_CUDA_MEMCPY` finding of `sec_multi_gpu_concise`. Forward: 5.7 (which algorithm NCCL chose).
**Exercises:** 6 to 8. Derive the alpha-beta estimates in closed form and the reason for the 1/t^2 weights (derive); derive the uplink load of a ring whose order crosses the uplink four times instead of twice (derive); measure the bandwidth matrix with and without NUMA pinning (measure); measure one process driving 8 GPUs against eight processes with one GPU each (measure); check that ACS or IOMMU settings explain a slow pair before blaming the fabric (check); open: write a topology-aware ring order for a bridged-pair node and compare with NCCL's.
**Writer notes:** Pitfalls (blueprint 5.4): NVLink figures on vendor pages are bidirectional aggregates per GPU, a per-direction measurement is half; `busbw` is not a link measurement when the algorithm does not use each link once (tree, in-switch reduction); the first NCCL call includes communicator setup, discard warm-ups (`-w`); `-c 0` for timing and a separate `-c 1` run for correctness; peer-to-peer disabled by the driver, ACS or the IOMMU looks like a slow fabric, so read the capability matrix first; MIG, power caps and shared hosts change the numbers and are recorded. Risks: the node's interconnect is unknown until `topo.txt` is read, and the section runs unchanged in all three cases (NVSwitch, bridged pairs, plain PCIe) with the links table carrying both A100 rows; `p2pBandwidthLatencyTest` may move inside `cuda-samples`, so the parser is tested on committed text; NCCL version changes may alter output columns, so the version is pinned in the log. Exports: `d2l.links_table`, `d2l.parse_topo_matrix`, `d2l.topo_graph`, `d2l.parse_p2p`, `d2l.parse_nccl_tests`, `d2l.nccl_busbw_factor`, `d2l.coll_time` (alpha-beta time of ring, binary tree and direct all-to-all; 5.7 extends it), `d2l.tree_link_loads`; artifacts `lm-hw-node-p2p` (with `alpha_beta.json`: fields `scope, link, op, g, algo, alpha_s, beta_Bps, alpha_ci95, beta_ci95, n_half_bytes, resid_rel, n_points`, consumed by 6.x, 13.7 and 13.8), `lm-hw-t2pair-p2p`, `lm-hw-buildbox-p2p`. Do not re-teach the ring derivation (`subsec_mg-ring`), DDP bucketing, pinned memory, or the alpha-beta exercises of `sec_multi_gpu` beyond one recap sentence. Detail: blueprint ch05 5.4 (commands, parsers, D1 to D3, risks). `NCCL_P2P_DISABLE=1` on the node is labelled an emulation of a PCIe-only node; the T2-pair contrast is the clean case.
**Sources:**
- Papers: [Evaluating Modern GPU Interconnect: PCIe, NVLink, NV-SLI, NVSwitch and GPUDirect](https://arxiv.org/abs/1903.04611) (the measurement method per link class); [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786) (what the intra-node collectives do on each topology).
- Code and documentation: [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) and its [environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) (`NCCL_P2P_DISABLE`, `NCCL_ALGO`); [nccl-tests](https://github.com/NVIDIA/nccl-tests) (sweeps and the busbw convention); [cuda-samples](https://github.com/NVIDIA/cuda-samples) (`p2pBandwidthLatencyTest`); [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) (device matrices).
- Courses, tutorials and blogs: [Scaling Book: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) (node networking levels); Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (intra-node network chapter); [Modal GPU Glossary](https://modal.com/gpu-glossary) (NVLink, NVSwitch vocabulary).
-->

<!-- Placeholder (prose): opening. The reader met this box's PCIe tree in
subsec_hw-interconnects (fig_pcie_topology) and NCCL's host-staged path in
sec_multi_gpu_concise. The cell below reads the description of the machine
that runs it (on the build box, the box itself; a reader elsewhere sees
their own machine, while the measurements further down show the committed
box run): nvidia-smi topo -m (how each pair of devices is connected:
PHB, both GPUs
behind one PCIe host bridge; NODE, across two bridges of one CPU) and
topo -p2p r (whether the driver lets one GPU read another's memory: CNS,
"chipset not supported", everywhere on this box). Two published samples of
other machines (user posts on GitHub issues, labelled "published sample,
not measured here"): an 8-GPU A100 SXM node behind NVSwitch (every pair
NV12; nccl-tests issue 149) and a 5-GPU workstation with two NVLink-bridged
pairs of A100 PCIe cards (jax issue 11478). The workstation is not the
8-GPU bridged node that fig_lm_hw_server_graph draws; the text never calls
them the same machine. The question: how to read the matrix as a graph, measure each
edge, and predict which collective suffers on which topology. Idea: a
server is a graph whose edges are links with their own alpha and beta. -->

```{.python .input #server-paths-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #server-paths-topology}
%%tab pytorch
topo = d2l.parse_topo_matrix(d2l.hw_capture('topo'))  # nvidia-smi topo -m
p2p = d2l.parse_topo_matrix(d2l.hw_capture('topo-p2p'))  # topo -p2p r
d2l.show_topo(topo, p2p, samples=('nvswitch-node', 'bridged-workstation'))
```

## The Server as a Graph
:label:`subsec_lm_hw_server_graph`

<!-- Placeholder (prose): the devices of a server (GPUs, NVSwitches or
NVLink bridges, PCIe switches, the CPU's root complexes and host bridges,
CPU sockets, NICs, NVMe drives) and the links between them. SXM parts sit on
a baseboard whose NVLinks run to NVSwitch chips; PCIe parts plug into slots
and at most pair up over a bridge. The link table: PCIe per direction
computed from lanes, rate and encoding (16 x 16 GT/s x 128/130 / 8 = 31.5
GB/s for Gen 4, the A100 datasheet's 64 GB/s counts both directions);
NVLink, NVLink-C2C and Infinity Fabric transcribed from the vendors' pages,
which quote both directions together (the second column), so a
per-direction measurement is half the headline. One sentence: links_table
loads the sourced table and computes the PCIe rows. -->

```{.python .input #server-paths-links-table}
%%tab pytorch
links = d2l.links_table()
```

<!-- Placeholder (prose): the matrix becomes a graph. Devices joined by PIX
or PXB share a PCIe switch, by PHB a host bridge, by NODE a CPU socket, and
SYS crosses sockets; a pair marked NV# has # NVLinks, and when every pair
does, the GPUs hang off NVSwitch. One sentence: topo_graph infers the
switch-class nodes from the codes and labels each edge with the table's
rate; the links inside the CPU (host bridge to socket) have no datasheet
rate and are left for the measurement. The figure draws an 8-GPU A100 node
both ways, SXM with NVSwitch and bridged PCIe pairs, side by side (drawn
from the datasheets, not from the 5-GPU workstation sample); this box is
fig_pcie_topology, cited, not redrawn. If the prose quotes the MI300X row
of the links table, it cites AMD's datasheet (7 Infinity Fabric links,
128 GB/s each both directions together, 896 GB/s aggregate). -->

```{.python .input #server-paths-graph}
%%tab pytorch
g = d2l.topo_graph(topo, links)
d2l.show_edges(g)
```

<!-- Figure: fig_lm_hw_server_graph (generator server-graph). -->

### Peer-to-Peer, Copy Engines and NUMA
:label:`subsec_lm_hw_p2p`

<!-- Placeholder (prose): GPUDirect peer-to-peer (one GPU reads or writes
another's memory across PCIe or NVLink without touching host memory) and
GPUDirect RDMA (a NIC does the same); why the driver refuses peer access on
this box (CNS: the GeForce driver does not enable it on this chipset) and
what it does instead: stage every byte through a pinned buffer in host
memory. Copy engines (DMA units that move bytes without SMs) against
SM-driven copies (NCCL's kernels load and store across the link). One NIC
per GPU on the same PCIe switch on training nodes; NUMA and CPU-NIC
affinity (this box: one NUMA node, CPUs 0-63 for every GPU, a 10 GbE NIC,
no RDMA). The latency floor of a small transfer. -->

## Measuring Links
:label:`subsec_lm_hw_measure_links`

<!-- Placeholder (prose): the alpha-beta model as a model, t(m) = alpha +
m / beta per link, fitted by weighted least squares with 22.4's
fit_alpha_beta; the half-bandwidth size alpha beta, the message at which
half the time is the fixed cost. The GPU-to-GPU matrix: every ordered pair
copies 256 MiB (bandwidth) and 4 bytes (latency) with torch's copy_
between devices, which the driver stages through host memory. Then
concurrent pairs, to locate the shared resource: two pairs each behind its
own bridge (0 to 1 with 2 to 3), two pairs that both cross the bridges (0
to 2 with 1 to 3), one card sending to two (0 to 1 with 0 to 2). One
sentence: hw_gpu_matrix shows the committed run of the whole box and
re-measures two pairs (of the same relation class as on the box). The heat
map shows GB/s only, each cell labelled with its rate and relation: the
latency of a 4-byte copy is flat (18-19 us on every pair) and is read from
the printed line. The printed medians per relation class carry the split
the text reads: a pair behind one host bridge (PHB) about 19.5 GB/s, a
pair across the bridges (NODE) about 22. Claim S1: P2P refused, staged
copies below one host link. -->

```{.python .input #server-paths-gpu-matrix}
%%tab pytorch
mat = d2l.hw_gpu_matrix(nbytes=2**28)
d2l.plot_gpu_matrix(mat)  # GB/s per ordered pair, labelled by relation
host = links.loc['PCIe Gen 4 x16', 'gbs_per_dir']  # one host link, GB/s
assert not mat['peer'].any()  # S1: the driver refuses peer access
assert np.nanmax(mat['gbs']) < host  # every copy crosses two host links
```

### Collectives on the Box
:label:`subsec_lm_hw_box_collectives`

<!-- Placeholder (prose): bus bandwidth in one sentence (eq_ring_traffic,
subsec_mg-accounting): nccl-tests divides the message by the time
(algorithm bandwidth) and multiplies by the bytes each GPU's link carries
per message byte, 2(k-1)/k for a ring all-reduce and (k-1)/k for the
others, so that a link-bound collective reads as the link's rate. The
alpha-beta time of a ring, a binary tree and a direct all-to-all, with
alpha per step. -->

```{.python .input #server-paths-busbw-def}
%%tab pytorch
def nccl_busbw_factor(op, k):  #@save
    """Bus bandwidth over algorithm bandwidth (nccl-tests' convention): the
    bytes each GPU's link carries per byte of the message."""
    op = op.replace('_', '')
    if op == 'allreduce':
        return 2 * (k - 1) / k  # reduce-scatter, then all-gather
    if op in ('allgather', 'reducescatter', 'alltoall'):
        return (k - 1) / k
    if op in ('broadcast', 'reduce'):
        return 1.0
    raise ValueError(f'unknown collective {op!r}')

def coll_time(op, S, k, alpha, beta, algo='ring'):  #@save
    """Seconds for a collective of S bytes per GPU on k GPUs, alpha (s)
    per step, beta (bytes/s) per link and direction: a ring (direct for
    all-to-all on a crossbar), or algo='tree' for the binary-tree
    all-reduce (reduce up, broadcast down, the whole message per hop)."""
    op = op.replace('_', '')
    if op not in ('allreduce', 'allgather', 'reducescatter', 'alltoall'):
        raise ValueError(f'coll_time does not model {op!r}')
    if algo == 'tree' and op == 'allreduce':
        return 2 * math.ceil(math.log2(k)) * (alpha + S / beta)
    if algo != 'ring':
        raise ValueError(f'no {algo} {op} in this model')
    steps = 2 * (k - 1) if op == 'allreduce' else k - 1
    return steps * (alpha + S / (k * beta))  # S / k bytes per step
```

<!-- Placeholder (prose): the sweeps, 8 B to 256 MiB, on a pair behind one
host bridge (PHB), a pair across the bridges (NODE) and all cards;
all-reduce and all-to-all on each, all-gather and reduce-scatter on all
cards. NCCL's own account (NCCL_DEBUG=INFO, one line read): the transport
between the cards is SHM, shared memory, since peer access is refused.
Every curve tops out at 2 to 4 GB/s, an order of magnitude below the staged
copies of the matrix and the pinned host link (dashed, 22.4's committed
plateau): the link is not what limits NCCL here. One sentence:
nccl_sweeps shows the committed run and re-runs the all-reduce on one pair
and on all cards. -->

```{.python .input #server-paths-nccl-sweeps}
%%tab pytorch
nccl = d2l.nccl_sweeps('box')
d2l.plot_nccl(nccl, host_link=d2l.hw_box_log('box-storage'))
```

<!-- Placeholder (prose): one alpha and one beta per collective and
placement, beta in the bus-bandwidth convention (the rate each GPU's link
carries), n_1/2 = alpha beta. The curves are not alpha-beta shaped: bus
bandwidth peaks near 1 MiB and settles lower, so one line through every
size overstates beta by up to 20% (for the four-card all-reduce it would
predict a 32 MiB message 20% faster than measured). Hence two regimes, as
22.4 did for the host link (fit_h2d): alpha from the fit of the messages
up to 64 KiB, beta the median rate of the messages of 4 MiB and more.
The 95% intervals come from the three regeneration processes (each
process's own two-regime fit, Student t over the three): run-to-run
noise, not model misfit. Claim S2: a few GB/s for a large all-reduce on
two cards on either pair. The pair behind one host bridge (PHB) is slower
than the pair across the bridges (smoke run: about 2.4 against 3.6 GB/s,
at every size from 1 MiB); the prose reports the gap as measured, and
what the copy-engine cell below says about its cause. Claim S5: alpha is
tens of microseconds on the shared-memory path. The fits are the box's
alpha_beta.json, which 22.3, 22.6 and 22.7 read through box_alpha_beta. -->

```{.python .input #server-paths-fits}
%%tab pytorch
ab = d2l.fit_collectives(nccl, alpha_below=2**16, plateau_from=2**22)
d2l.show_fits(ab)
pairs = ab.loc[('all_reduce', 2)].beta / 1e9  # GB/s: PHB and NODE pairs
assert ((pairs > 1) & (pairs < 5)).all()  # S2: a few GB/s on either pair
assert ((ab.alpha > 1e-6) & (ab.alpha < 1e-3)).all()  # S5: microseconds
```

<!-- Placeholder (prose): NCCL_SHM_USE_CUDA_MEMCPY=1 makes the
shared-memory transport move data with the copy engines instead of SM
loads and stores across PCIe; in a bare all-reduce microbenchmark it
multiplies the bandwidth (claim S3, quoted with the pair it was measured
on). It runs on one pair per relation class (PHB and NODE), which also
tests the cause of the PHB/NODE gap of S2: if the knob closed the gap, the
SM-driven shared-memory copy would set it. Smoke run: it does not; the gap
widens (PHB 2.4 to 11.5 GB/s, 4.7x; NODE 3.7 to 19.8, 5.4x), as in the
staged copy_ matrix (PHB 19, NODE 22): with the copy engines doing the
work, a pair behind one host bridge, whose staged traffic goes up and
down through that one bridge, stays slower. The prose says this is
consistent with the shared bridge and is not a direct test of it. It is a
workaround to measure, never a setting for training: at the torch 2.11
pin it deadlocks DDP (sec_multi_gpu_concise). -->

```{.python .input #server-paths-copy-engine}
%%tab pytorch
ce = d2l.copy_engine_effect(nbytes=2**28)  # one pair per relation class
assert (ce.knob / ce.default > 2).all()  # S3: copy engines, not the link
```

## Bisection and the All-to-All
:label:`subsec_lm_hw_bisection`

<!-- Placeholder (prose): derivation D2. On a crossbar of k GPUs with
per-direction rate beta, an all-to-all of S bytes per GPU takes ((k-1)/k)
S / beta and a ring all-reduce 2((k-1)/k) S / beta. Two groups of k/2 under
one uplink each, rate beta_u: the ring crosses the uplink once per
direction, so it is unchanged while beta_u >= beta; the all-to-all sends
(k/2)(k/2) S / k = k S / 4 across it, so it takes at least k S / (4
beta_u), (k^2 / (4 (k-1))) (beta / beta_u) times the crossbar's: 64/28
for k = 8. tree_link_loads routes each pattern through the tree and
returns every link's load exactly; pattern_time divides the busiest link's
bytes by its rate. -->

```{.python .input #server-paths-tree-loads-def}
%%tab pytorch
def tree_link_loads(k, groups, pattern, S=1.0, staged=False):  #@save
    """Bytes on each link of a two-level tree, its busier direction, when
    k GPUs (groups: the GPUs below each uplink) run pattern on S bytes per
    GPU: 'alltoall' (S / k to every other GPU) or 'allreduce' (a ring in
    the groups' order, 2 (k - 1) / k S over each ring edge). Keys 'GPU i'
    and 'uplink g'. staged: every byte passes through host memory above
    the uplinks, even between two GPUs of one group."""
    group = {i: gi for gi, grp in enumerate(groups) for i in grp}
    assert len(group) == k, 'groups must hold the k GPUs once each'
    if pattern == 'alltoall':
        flows = [(i, j, S / k) for i in group for j in group if i != j]
    elif pattern == 'allreduce':
        ring = [i for grp in groups for i in grp]
        flows = [(i, j, 2 * (k - 1) / k * S)
                 for i, j in zip(ring, ring[1:] + ring[:1])]
    else:
        raise ValueError(f'unknown pattern {pattern!r}')
    load = {}  # (link, direction): bytes
    for i, j, b in flows:
        hops = [(f'GPU {i}', 'up'), (f'GPU {j}', 'down')]
        if staged or group[i] != group[j]:
            hops += [(f'uplink {group[i]}', 'up'),
                     (f'uplink {group[j]}', 'down')]
        for hop in hops:
            load[hop] = load.get(hop, 0) + b
    out = {}
    for (link, _), b in load.items():
        out[link] = max(out.get(link, 0), b)
    return out

def pattern_time(k, groups, p, beta, beta_u, S=1.0, staged=False):  #@save
    """Seconds of pattern p on the two-level tree: the busiest link's
    bytes over that link's rate (bytes/s: beta for a GPU's link, beta_u
    for an uplink)."""
    loads = tree_link_loads(k, groups, p, S, staged)
    return max(b / (beta_u if link.startswith('uplink') else beta)
               for link, b in loads.items())

def a2a_busbw_ratio(k, groups, beta, beta_u, staged=False):  #@save
    """Predicted bus bandwidth of the all-to-all over the ring
    all-reduce's on a two-level tree (busbw = factor x S / time)."""
    busbw = {p: nccl_busbw_factor(p, k)
             / pattern_time(k, groups, p, beta, beta_u, staged=staged)
             for p in ('alltoall', 'allreduce')}
    return busbw['alltoall'] / busbw['allreduce']
```

<!-- Placeholder (prose): D2 in numbers, uplink as fast as a GPU's link:
at k = 8 the all-to-all puts 2 S on the uplink and the ring 1.75 S, and
the all-to-all's bus bandwidth is 28/64 of the ring's; at k = 4, 12/16. -->

```{.python .input #server-paths-d2}
%%tab pytorch
for k in (4, 8):  # two groups of k / 2, one uplink each
    halves = [tuple(range(k // 2)), tuple(range(k // 2, k))]
    a2a = tree_link_loads(k, halves, 'alltoall')['uplink 0']
    ring = tree_link_loads(k, halves, 'allreduce')['uplink 0']
    print(f'k = {k}: uplink load {a2a:.2f} S (all-to-all), {ring:.2f} S '
          f'(ring); busbw ratio {a2a_busbw_ratio(k, halves, 1, 1):.3f}')
```

<!-- Placeholder (prose): the box as that tree. Its "uplinks" are the host
bridges (beta_u: the two pairs that cross the bridges together, from the
matrix), its GPU links the staged copies (beta). Every byte is staged
through host memory, so even two GPUs behind one bridge talk over both
uplinks: host memory is the hub of a star (staged=True). NCCL cannot test
the model here: it runs at about 2.3 GB/s busbw, ten times below these
links, so its all-to-all / all-reduce ratio (about 0.95) says only that
the shared-memory transport treats both patterns alike; the tree (0.96),
the star (1.00) and a crossbar (1.00) all predict the same. The copies
can: hw_staged_patterns runs each pattern as concurrent staged copies (the
all-to-all as 12 flows of S/4, the ring as 4 flows of 1.5 S), at link
rates, where the tree and the star predict times 1.5 times apart. Claim
S4 (smoke run): both patterns take longer than even the star predicts
(all-to-all about 24 ms against 10 as a tree and 14 as a star; ring 40
against 18 and 29): no single bridge binds, a resource every staged copy
shares does (the copies run at 33-43 GB/s in all, whatever the pattern:
host memory and the driver's staging path). On such a hub both patterns
cost total bytes over one shared rate, so the all-to-all is not
bisection-limited on this box; it is everything-limited. Why expert
parallelism needs the switch and data parallelism tolerates the tree. -->

```{.python .input #server-paths-bisection}
%%tab pytorch
groups = mat['groups']  # the GPUs behind each host bridge
k = sum(len(grp) for grp in groups)
beta = np.nanmedian(mat['gbs']) * 1e9  # one staged copy, bytes/s
conc = mat['concurrent'].set_index('pattern').gbs
beta_u = conc['two pairs across the bridges'] * 1e9  # a bridge, one way
S = 2**28  # bytes per GPU
flows = d2l.hw_staged_patterns(S=S)  # each pattern as concurrent copies
for p, name in (('alltoall', 'all-to-all'), ('allreduce', 'ring')):
    tree = pattern_time(k, groups, p, beta, beta_u, S)
    star = pattern_time(k, groups, p, beta, beta_u, S, staged=True)
    print(f'{name:>10}: tree {tree * 1e3:.0f} ms, star {star * 1e3:.0f} '
          f'ms, measured {flows.seconds[p] * 1e3:.0f} ms '
          f'({flows.gbs[p]:.0f} GB/s in all)')
    assert flows.seconds[p] > star  # S4: a shared resource, not a bridge
ratio = (d2l.busbw_at(nccl, 'alltoall', 'all', S)
         / d2l.busbw_at(nccl, 'all_reduce', 'all', S))
print(f'NCCL all-to-all / all-reduce busbw at {S / 2**20:.0f} MiB: '
      f'{ratio:.2f} (tree {a2a_busbw_ratio(k, groups, beta, beta_u):.2f})')
```

<!-- Figure: fig_lm_hw_bisection (generator bisection). -->

<!-- Placeholder (prose): what the box's link costs one tensor-parallel
all-reduce: 4,096 tokens of Qwen3-8B activations in bf16, with the fitted
alpha per step and beta of each placement, against the bandwidth term
alone at NVLink 3's per-direction rate (datasheet, quoted; alpha left
out). -->

```{.python .input #server-paths-tp-cost}
%%tab pytorch
cfg_8b = d2l.hw_config('Qwen/Qwen3-8B')
S = 4096 * cfg_8b.hidden_size * 2  # 4,096 tokens of activations in bf16
nv = links.loc['NVLink 3 (A100 SXM)', 'gbs_per_dir'] * 1e9
print(f'one all-reduce of {S / 2**20:.0f} MiB per GPU:')
for (op, k, path), fit in ab.loc[['all_reduce']].iterrows():
    box = coll_time(op, S, k, fit.alpha_step, fit.beta)
    print(f'  g = {k}, {path}: {box * 1e3:.1f} ms on the box, '
          f'{coll_time(op, S, k, 0, nv) * 1e3:.2f} ms at NVLink 3 rates')
```

### Where the Parallel Axes Go
:label:`subsec_lm_hw_axes_in_node`

<!-- Placeholder (prose): one paragraph (Chapter 23 develops it): the axis
with the most frequent and the largest collectives (tensor parallelism,
an all-reduce per layer) goes on the fastest links, inside an NVLink
domain; the expert axis needs the switch's bisection for its all-to-all;
data parallelism, one all-reduce per step that overlaps the backward pass,
tolerates the slowest path, across servers. On this box no axis has a fast
link. -->

<!-- Placeholder (prose): the node twins. The campaign logs C-T3 (one 8 x
A100 node, NVSwitch or bridged pairs, decided by its committed topo.txt;
NCCL_P2P_DISABLE=1 labelled an emulation of a PCIe-only node) and C-T2pair
(two A100s with an NVLink bridge, with and without peer-to-peer) give the
same fits; claim S6 (pending): NVLink beta within 20% of the per-direction
datasheet rate, the all-to-all ratio near 1 on NVSwitch and below 1 with
peer-to-peer disabled. Until they are committed the cell prints one status
line each and the sentence says so. -->
<!-- campaign: lm-hw-node-p2p-v1 -->
<!-- campaign: lm-hw-t2pair-p2p-v1 -->

```{.python .input #server-paths-node-twin}
%%tab pytorch
for log in ('node-p2p', 't2pair-p2p'):
    d2l.show_fits(d2l.fit_collectives(d2l.hw_campaign_log(log)))
```

## Summary

<!-- Placeholder (prose): 3-6 sentences; forward to sec_lm_networks (the
network between servers) and sec_lm_collectives (which algorithm NCCL
chose on this box). -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per S/spec/22.5.md part 9:
closed-form alpha-beta estimates and the 1/t^2 weights (derive); a ring
order crossing the uplink four times (derive); the GPU matrix with and
without numactl CPU pinning (measure, 2 min; on this one-node box the
expected answer is no difference, and the exercise says why); one process
driving all GPUs against one process per GPU (measure, 2 min); check that
ACS, the IOMMU or the driver (CNS) explains a slow pair before blaming the
fabric (check); open: a topology-aware ring order for a bridged-pair
node. -->
