# Scale-Up Domains and Scale-Out Networks
:label:`sec_lm_networks`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.6 Scale-Up Domains and Scale-Out Networks · T4 (T0 model; T2 pair)").
Blueprint: docs/llm/blueprints/ch05_hardware.md section 5.5; docs/llm/blueprints/ch05_hardware.md section 5.6.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A cluster has two networks, one with memory-class bandwidth inside a domain and one with NIC-class bandwidth between domains, and the domain boundary decides which parallel axes fit where.
**Opening:** The reader has the server graph of 5.5 and one number for the fastest link in it. One concrete example: a 70B model in bf16 needs eight 40 GB nodes just for its weights, so some collective must cross a NIC, and the NIC is twenty times slower than NVLink. The open question: how the two networks are built, what each costs per byte and per hop, and which parallel axes can live on the slow one.
**Teaches:**
- Scale-up domains: the 8-GPU node, NVL72 (72 GPUs on one NVSwitch fabric), Helios (72 MI455X over UALink), Trn3 UltraServers (up to 144 chips), TPU pods with torus links and optical circuit switches, UALink as a standard; the domain's memory, bandwidth per chip and chip count as the three numbers that matter; a systems table as a data file.
- Scale-out networks: NICs and link speeds, InfiniBand against RDMA over Converged Ethernet, Ultra Ethernet; the latency budget of a cross-node message (NIC, switch hops, cables, software); rail-optimized fat trees, oversubscription, rail-only designs that lean on the domain; congestion control and adaptive routing in one paragraph; scale-across in one paragraph (synchronizing pseudo-gradients between datacentres every H steps hides behind the inner steps iff the WAN carries the bytes in time).
- Which axes fit: the communication-to-compute ratio of each parallel axis on the link it uses, the chattiest axis on the fastest link; the layout calculator applied to a node, NVL72, Helios, a Trn3 UltraServer and a TPU pod slice.
**Derivation:** Communication-to-compute ratios per axis on a link of ridge R_l = P_eff / beta_l (blueprint ch05 5.6 D1): tensor parallel rho_tp = (4/3)(t-1)(d b / N_layer) R_l; expert parallel rho_ep = (2 b (1-1/e) / (9 d_ff)) R_l; pipeline rho_pp = (d b / (3 N_stage)) R_l; data parallel rho_dp = (b_g / (3 T)) R_l (k-1)/k. Smallest numeric case: Qwen3-8B, N_layer about 193M, d = 4,096, b = 2, t = 8 on an NVLink link gives a ratio below one tenth, while the same t = 8 across a NIC twenty times slower multiplies the ratio by twenty; the cell prints both and marks the fit boundary at epsilon = 0.15. A fat tree's sizing (r^2/2 endpoints for two tiers, per-endpoint cross-leaf bandwidth beta / o under oversubscription o) is stated and checked by construction in a cell.
**Notebook:**
1. Question: does a two-node measurement of the scale-out network give the alpha and beta per rail that the layout calculator needs, and does the calculator then say which axes of Qwen3-8B and Qwen3-30B-A3B fit in a node, a rack and a pod?
2. T4 (two nodes, once per release): `ib_write_bw` and `ib_write_lat` between the nodes for message sizes from 8 bytes to 4 GiB, on one rail and on all rails concurrently; parse with `d2l.parse_perftest`; fit alpha and beta per rail with `d2l.fit_alpha_beta`; record `ibstat` and the leaf identifiers so the text can say whether the nodes share a leaf.
3. The latency budget: decompose the measured one-way latency into NIC, switch hops and software from the perftest numbers and the NCCL alpha fitted in 5.7; the software floor of a TCP loopback on the T2 pair as the contrast.
4. T0: `d2l.fat_tree` builds leaf-spine graphs and checks the endpoint and switch counts and the max-flow across an oversubscribed leaf exactly.
5. T0: `d2l.systems_table()` and `d2l.layout_fit(cfg, system, layout, tokens_per_step)` for the five systems and the two models: per axis the link, the bytes per step, the ratio, fit or no fit, and memory per chip.
6. Reduced T1 configuration: the gloo loopback group with 2 and 4 ranks shows only the software latency floor and is labelled "not a datacenter network"; the T2 pair with NCCL forced onto TCP loopback gives the NCCL-level alpha for the same contrast.
7. Claims that survive noise: per-rail beta within 20% of the NIC line rate; the latency floor of RDMA is an order of magnitude below the TCP loopback; the calculator's verdicts (tensor and expert parallelism inside the domain, data and pipeline parallelism across) follow from the ratios, and a change of epsilon moves the boundary, which the plot shows.
**Figures:** `fig_lm_hw_two_networks` (a rack of domains joined by rails to a leaf-spine fabric); `fig_lm_hw_fat_tree` (two-tier tree with oversubscription and a rail-only variant); `fig_lm_hw_layout_fit` (plot: ratio per axis against link, for the five systems).
**Restate / forward:** `subsec_hw-interconnects` (NVL72's scale-up bandwidth as given); `eq_ring_traffic` and `eq_dp_cost` (the data-parallel term); `eq_six_nd` (the compute side of each ratio); `sec_cloud_instances` (the "NVLink or PCIe" exercise); 5.5's alpha-beta fits. Forward: 6.7 (the composed cost model refines the first-order ratios).
**Exercises:** 6 to 8. Derive the layouts that fit in one NVL72 for Qwen3-235B-A22B and the memory per chip of each (derive); derive the WAN bandwidth that hides a synchronization every 100 steps for an 8B model (derive); measure all-rail against single-rail bandwidth and attribute the difference (measure); check the systems table's bidirectional-against-per-direction columns against the vendor pages (check); derive the oversubscription at which a job spanning two leaves loses half its data-parallel bandwidth (derive); open: add a UALink system row when one ships and re-run the calculator.
**Writer notes:** Pitfalls (blueprint 5.5 and 5.6): vendor scale-up figures are bidirectional aggregates per chip, use per-direction numbers in the ratios; RoCE needs the right GID index and a lossless or congestion-controlled fabric, so `ib_write_bw` failures are often configuration; latency from `ib_write_lat` is not the latency of an NCCL step (5.7 fits alpha from NCCL itself); oversubscription is invisible at small scale and bites only when a job spans leaves, so ask the scheduler where the nodes are; results depend on neighbours' traffic on a shared fabric (repeat at different times, widen the claim threshold to 5x MAD); a larger domain raises the bound on tensor and expert parallelism but the all-to-all latency and switch radix grow with it. The ratios ignore overlap and attention FLOPs, stated and bounded in the check cell; 6.2 to 6.7 refine them and each notebook asserts agreement with `d2l.axis_ratio` in its limit. Hardware fallbacks: no multi-node allocation (the notebook runs from committed logs; the T0 and T1 cells keep every cell executable; the release notes state the log date); fewer than 8 NICs per node (rails equal the actual count from `ibstat`); a TPU slice whose per-link rate is not on the chosen generation's page (mark "not stated", omit from `layout_fit`). Exports: `d2l.parse_perftest`, `d2l.fat_tree`, `d2l.systems_table`, `d2l.axis_ratio`, `d2l.layout_fit`; `d2l.diloco_wan_bandwidth` for the scale-across paragraph; artifacts `lm-hw-cluster-nccl` (the perftest files and `ibstat` output; 5.7 adds the collective sweeps to the same tar and the two sections share one job script and one allocation), `lm-hw-systems`, `lm-hw-t2pair-net`, `lm-hw-buildbox-net`. The cost functions belong to 5.8. Do not re-teach the ring, the alpha-beta definition, or the build box's topology. Detail: blueprint ch05 5.5 (N1 to N4 commands, D4) and 5.6 (D1, systems table columns, `layout_fit`). Authoring check: confirm whether Helios's UALink per-direction rate is published at writing time (AMD's brochure gives aggregate figures). The scale-across paragraph cites DiLoCo and Streaming DiLoCo, one sentence each.
**Sources:**
- Papers: [Rail-only: A Low-Cost High-Performance Network for Training LLMs with Trillion Parameters](https://arxiv.org/abs/2307.12169) (rail-only design); "RDMA over Ethernet for Distributed AI Training at Meta Scale" and "Alibaba HPN: A Data Center Network for Large Language Model Training" (SIGCOMM 2024; cited by title); [TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings](https://arxiv.org/abs/2304.01433) (torus links and optical circuit switches); [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/abs/2505.09343) (scale-up and scale-out from a model builder's side); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (cluster design, RoCE at scale); [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627) (network design and topology-aware placement); [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105) and [Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch](https://arxiv.org/abs/2501.18512) (scale-across, one sentence each).
- Code and documentation: NVIDIA, [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) and [NVL72 for training and real-time inference](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/) (rack row); AMD, [Helios](https://www.amd.com/en/products/rackscale-solutions/helios.html) and its [launch post](https://www.amd.com/en/blogs/2026/amd-launches-helios-the-highest-performing-rackscale-ai-infrastructure-solution.html) (second rack row); AWS, [Trn3 UltraServers](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/); [UALink Consortium](https://ualinkconsortium.org/) and [Ultra Ethernet Consortium](https://ultraethernet.org/) (standards rows); [perftest](https://github.com/linux-rdma/perftest) (`ib_write_bw`, `ib_write_lat`); [SHARP documentation](https://docs.nvidia.com/networking/display/sharpv300) (in-network reduction exists; 5.7 uses it).
- Courses, tutorials and blogs: [Scaling Book: sharding](https://jax-ml.github.io/scaling-book/sharding/) (placement and communication models); [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (fat trees, oversubscription); Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (inter-node network chapter).
-->

<!-- Placeholder (prose): opening. The reader has the server graph of
sec_lm_server_paths and one number for the fastest link in it. A 70B
model's mixed-precision training state (16 bytes per parameter: bf16
weights and gradients, fp32 master weights and two fp32 moments) does not
fit in one 8 x A100 40 GB node, so some collective must cross a NIC; quote
the three numbers of the cell below (state in TB, nodes needed, the
NVLink/NIC ratio per direction, computed from the systems table, never a
rounded "twenty"). The open question: how the two networks are built, what
each costs per byte and per hop, and which parallel axes can live on the
slow one. Idea: a cluster has two networks, memory-class bandwidth inside a
domain and NIC-class bandwidth between domains, and the domain boundary
decides which parallel axes fit where. -->

```{.python .input #networks-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #networks-extras}
%%tab pytorch
import pandas as pd
```

```{.python .input #networks-opening}
%%tab pytorch
systems = d2l.systems_table()
a100 = d2l.chip_specs().loc['A100 SXM 80 GB']  # bf16 312 TFLOP/s, as 40 GB
cfg_70b = d2l.HW_SHAPES['dense-70b']
n_params = d2l.ServingModel(cfg_70b, a100).n_params()[0]
state = d2l.TrainStepModel.state['mixed16'] * n_params  # bytes
node = systems.loc['8 x A100 SXM 40 GB']
node_bytes = node.chips * node.hbm_gb_per_chip * 1e9
print(f'training state of the 70B model: {state / 1e12:.2f} TB')
print(f'8 x A100 40 GB nodes to hold it: {math.ceil(state / node_bytes)} '
      f'({node_bytes / 1e9:.0f} GB each)')
print(f'NVLink / NIC per direction: {node.up_gbs_dir:.0f} / '
      f'{node.scaleout_gbs_per_chip:.0f} GB/s = '
      f'{node.up_gbs_dir / node.scaleout_gbs_per_chip:.0f}x')
```

## Scale-Up Domains
:label:`subsec_lm_hw_scaleup`

<!-- Placeholder (prose): a scale-up domain is a set of chips with
memory-class bandwidth between every pair: the 8-GPU node (NVSwitch on the
baseboard; subsec_hw-interconnects for NVL72's numbers, taken as given),
GB200 NVL72 and Vera Rubin NVL72 (72 GPUs on one NVLink switch fabric),
Helios (72 MI455X over UALink), Trn3 UltraServers (up to 144 chips over
NeuronSwitch-v1), TPU pods (a 3-D torus of 4x4x4 cubes joined by optical
circuit switches, up to 9,216 chips; a torus gives each chip six links,
collectives run per axis and bisection falls with size: cite the Scaling
Book, do not derive), UALink 1.0 as an open standard with no shipping
system. Three numbers per domain: chips, bandwidth per chip and direction,
memory. Pitfall: vendors quote both directions added; the table halves
them (up GB/s/dir), and on a torus a collective along one axis uses that
axis's two links. One sentence: systems_table loads the sourced table
(page and date per row; rack power of the NVL72 racks "as reported",
MLSS slide 40); the box is its last row, measured in sec_lm_server_paths
(NCCL's shared-memory path, host-staged). Helios: AMD states only the
rack's aggregate (260 TB/s), halved like the NVIDIA figures (authoring
check of the plan: per-direction rate not published at writing time).
Trn3: no interconnect or bf16 figure published. -->

```{.python .input #networks-systems}
%%tab pytorch
d2l.show_table(systems[list(d2l.SYSTEMS_SHORT)], headers=d2l.SYSTEMS_SHORT)
```

## Scale-Out Networks
:label:`subsec_lm_hw_scaleout`

<!-- Placeholder (prose): NICs and line rates (one NIC per GPU on training
nodes: 200 Gb/s HDR on a DGX A100, 400 Gb/s NDR on H100/B200/GB200, 800
Gb/s next), InfiniBand against RDMA over Converged Ethernet (credit-based
against priority flow control; RoCE needs the right GID index and a
lossless or congestion-controlled fabric), Ultra Ethernet. The latency
budget of a cross-node message: NIC and software, switch hops, cable (5
ns/m in fibre), serialization m / beta. Rail-optimized fat trees (NIC i of
every node on rail i's leaves), oversubscription, rail-only designs that
send cross-rail traffic over NVLink first (Rail-only paper). Congestion
control and adaptive routing in one paragraph (Meta's RoCE and Alibaba HPN
papers, SIGCOMM 2024, by title; Llama 3, MegaScale). -->

<!-- Figure: fig_lm_hw_two_networks (generator two-networks): a rack of
scale-up domains joined by rails to a leaf-spine fabric. -->

### Building a Fat Tree
:label:`subsec_lm_hw_fat_tree`

<!-- Placeholder (prose): D4. Radix-r switches; a leaf with d down and u
up ports has oversubscription o = d / u. Two tiers: r leaves (one port of
every spine each), u spines, r d endpoints: r^2/2 and 3r/2 switches when
non-blocking; three tiers r^3/4 endpoints and 5r^2/4 switches. Uniform
cross-leaf traffic from d endpoints shares u uplinks: beta u / d = beta / o
per endpoint; traffic inside a leaf is not affected. With rails, host h's
NIC i hangs off rail i's leaf in h's group: the counts do not change, only
which leaf a message enters. One sentence: cross_leaf_flow checks the
cross-leaf rate by a max-flow over the built graph. Figure
fig_lm_hw_fat_tree. -->

```{.python .input #networks-fat-tree-def}
%%tab pytorch
def fat_tree(radix, tiers, oversub=1.0, rails=1):  #@save
    """A two-tier (leaf-spine) or three-tier fat tree of radix-port
    switches whose edge switches have `down` endpoint ports and
    up = down / oversub uplinks; rails > 1 wires NIC i of each host to
    rail i's edge switch of its group. Counts and the graph (networkx)."""
    import networkx as nx
    r, G = radix, nx.Graph()
    up = int(r // (1 + oversub))
    down = int(round(oversub * up))
    if tiers == 2:  # every leaf has one link to each spine
        edge = [f'leaf{i}' for i in range(r)]
        G.add_edges_from((l, f'spine{j}') for l in edge for j in range(up))
    else:  # r pods: r/2 edge and `up` aggregation switches, r/2 cores each
        edge = [f'edge{p}.{i}' for p in range(r) for i in range(r // 2)]
        G.add_edges_from((e, f'agg{e[4:].split(".")[0]}.{j}')
                         for e in edge for j in range(up))
        G.add_edges_from((f'agg{p}.{j}', f'core{j}.{c}') for p in range(r)
                         for j in range(up) for c in range(r // 2))
    attach = {}  # endpoint -> its edge switch
    for g in range(len(edge) // rails):
        for h in range(down):
            for nic in range(rails):
                attach[f'host{g}.{h}/{nic}'] = edge[g * rails + nic]
    G.add_edges_from(attach.items())
    return dict(endpoints=len(attach), hosts=len(attach) // rails,
                switches=G.number_of_nodes() - len(attach),
                links=G.number_of_edges(), down=down, up=up, graph=G,
                edge_switches=edge, attach=attach)
```

```{.python .input #networks-fat-tree}
%%tab pytorch
r, beta = 64, 400e9 / 8  # switch radix; one 400 Gb/s port, bytes/s per dir
for o in (1.0, 2.0):
    ft = fat_tree(radix=r, tiers=2, oversub=o, rails=8)
    flow = d2l.cross_leaf_flow(ft, beta) / (ft['down'] * beta)
    print(f"o = {o:g}: {ft['endpoints']:,} endpoints on {ft['hosts']} hosts, "
          f"{ft['switches']} switches, {ft['links']:,} links")
    print(f"  cross-leaf max-flow: {flow:.3f} beta per endpoint "
          f"(1 / o = {1 / o:.3f}, u/d = {ft['up']}/{ft['down']})")
    assert math.isclose(flow, ft['up'] / ft['down'])  # N1, exact
    if o == 1:
        assert (ft['endpoints'], ft['switches']) == (r**2 // 2, 3 * r // 2)
```

### The Software Floor
:label:`subsec_lm_hw_software_floor`

<!-- Placeholder (prose): what the box can measure of a network: not the
wire, only the software above it. A gloo group of 2 and 4 CPU processes
over TCP on the loopback interface, and NCCL forced onto its socket
transport over loopback between two cards (the four variables in the call
disable peer-to-peer, shared memory and InfiniBand and pin the socket to
lo); all labelled "not a datacenter network". First the kernel's own path:
a TCP ping-pong between two processes (TCP_NODELAY), whose round trip is
what the operating system's network stack costs on loopback, with no
collective library on top. The alpha-beta fit of sec_lm_server_paths (not
re-taught): alpha from the messages up to 64 KiB, beta from those of 4 MiB
and more, intervals over the regeneration's three processes. Quote alpha
and beta at 2 significant digits from the committed lines, never the live
check. Claim N2, two layers of software: the TCP stack costs tens of
microseconds per round trip; gloo adds hundreds of microseconds per ring
step on top (its all-reduce on k ranks takes 2(k-1) steps, so the 4-rank
alpha is about three times the 2-rank one): that is library overhead, not
the network stack. NCCL over sockets costs about a hundred microseconds
per all-reduce. -->

```{.python .input #networks-gloo-floor}
%%tab pytorch
tcp = d2l.loopback_sweep(backend='tcp', ranks=(2,), max_bytes=2**24)
floor = d2l.loopback_sweep(backend='gloo', ranks=(2, 4), max_bytes=2**24)
```

<!-- Placeholder (prose): fix pass F6 (N2 as measured, committed values kept): the gloo alpha is gloo's own per-message cost, not the sweep's settings: a gloo send/recv round trip takes 200-300 us against 41 us for the Python TCP round trip, and a 2-rank all-reduce is two such steps; OMP threads change nothing, pinning each rank to a core lowers it by at most 1.9x and only in some runs, the 4-rank alpha not at all. -->

```{.python .input #networks-nccl-socket}
%%tab pytorch
sock = d2l.loopback_sweep(
    backend='nccl', ranks=(2,), max_bytes=2**28,
    env='NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 '
        'NCCL_SOCKET_IFNAME=lo')
```

<!-- Placeholder (prose): the ladder of transports: the measured software
floor (the TCP round trip, gloo, NCCL sockets), the box's shared-memory
path of sec_lm_server_paths, and two quoted rows (labelled quoted): an RDMA
write round trip of about 5 us and one NVLink hop of about 1.5 us (MLSS
latency ladder, slide 33), with one NDR port's 50 GB/s and NVLink 3's 300
GB/s per direction. Note the column "alpha of": a measured collective's
alpha is a whole all-reduce (2(k-1) ring steps), the TCP and RDMA figures
one round trip; "per msg" puts them on one scale (an all-reduce's alpha
over its ring steps, half a round trip), which separates the kernel's TCP
path (tens of us) from the collective library (hundreds of us per step).
Pitfall: ib_write_lat's latency is not the latency of an NCCL step;
sec_lm_collectives fits alpha from NCCL itself. -->

```{.python .input #networks-floor-compare}
%%tab pytorch
quoted = d2l.quoted_latencies()
ladder = d2l.transport_ladder(tcp, floor, sock, d2l.box_alpha_beta(),
                              quoted=quoted)
rdma = quoted.set_index('transport').alpha_us['RDMA, one NDR port']
assert (ladder[ladder.source == 'measured'].alpha_us > rdma).all()  # N2
```

## Which Axes Fit Where
:label:`subsec_lm_hw_layout`

<!-- Placeholder (prose): D1. On a link of ridge R_l = P_eff / beta_l
(FLOP per byte, beta per direction), the communication-to-compute ratio of
each parallel axis, ignoring overlap and attention FLOPs: tensor parallel
(degree t, two all-reduces forward and two backward per layer, ring
factor 2(t-1)/t) rho_tp = (4/3)(t-1)(d b / N_layer) R_l; expert parallel
(width e, four all-to-alls per MoE layer, fraction 1 - 1/e off the rank,
18 K d d_ff FLOPs per token) rho_ep = (2 b (1 - 1/e) / (9 d_ff)) R_l;
pipeline (boundary activations forward and backward) rho_pp = (d b / (3
N_stage)) R_l; data parallel (gradient all-reduce, eq_dp_cost) rho_dp =
(b_g / (3 T)) R_l (k - 1)/k. Derive tp in five steps (T cancels; the ratio
falls as 1/d for N_layer ~ c d^2), the others in one line each; compute
from eq_six_nd. Names: b bytes per activation element, b_g per gradient
element, N_layer the matmul parameters of one layer a token uses
(layer_params), N_stage those of one pipeline stage, T the tokens a
data-parallel replica processes per step. -->

```{.python .input #networks-axis-ratio-def}
%%tab pytorch
def layer_params(cfg):  #@save
    """Matmul parameters of one transformer layer that a token uses: the
    attention projections plus the MLP (K experts of width d_ff for a
    mixture of experts)."""
    d, d_h = cfg.hidden_size, cfg.head_dim
    attn = 2 * d * d_h * (cfg.num_attention_heads + cfg.num_key_value_heads)
    K = getattr(cfg, 'num_experts_per_tok', 0) or 1
    d_ff = (cfg.moe_intermediate_size if K > 1 else cfg.intermediate_size)
    return attn + K * 3 * d * d_ff

def axis_ratio(axis, cfg, link_ridge, b, t, e, n_stage, b_g, T, k):  #@save
    """D1: communication over compute time of one parallel axis ('tp',
    'ep', 'pp', 'dp') on a link of ridge R = P_eff / beta (FLOP per byte),
    with no overlap and attention FLOPs ignored; NaN for 'ep' of a dense
    model."""
    d, R, N_layer = cfg.hidden_size, link_ridge, layer_params(cfg)
    d_ff = getattr(cfg, 'moe_intermediate_size', None) or math.nan
    N_stage = cfg.num_hidden_layers * N_layer / n_stage
    rho = {'tp': lambda: 4 / 3 * (t - 1) * d * b / N_layer * R,
           'ep': lambda: 2 * b * (1 - 1 / e) / (9 * d_ff) * R,
           'pp': lambda: d * b / (3 * N_stage) * R,
           'dp': lambda: b_g / (3 * T) * R * (k - 1) / k}
    return rho[axis]()
```

<!-- Placeholder (prose): the smallest numeric case. Qwen3-8B (N_layer
193M by layer_params, d = 4,096, b = 2), a global batch of 1,024 sequences
of 4,096 tokens over k = 8 data-parallel replicas, 4 pipeline stages; the
expert row is Qwen3-30B-A3B (e = 8, experts of width 768). P_eff is the
achieved rate, not the datasheet peak: one A100's quoted bf16 312 TFLOP/s
at an MFU of 0.4 (a stated assumption, about what the Llama 3 report gives
for its 16K-GPU runs; at the peak every ratio would be 1/MFU = 2.5x
higher), behind three links: NVLink, the node's HDR NIC (the ratio grows
by the NVLink/NIC ratio of the opening) and the box's host-staged path
(beta from sec_lm_server_paths; the A100's rate behind the box's link, so
the column isolates the link). Rows name the degree (tp8 = tensor parallel
over 8). Quote the table's numbers as computed: tensor parallelism at t =
8 on NVLink is not below one tenth (the plan's sentence was wrong for this
model); at t = 2 it is, and across the NIC it fits at no degree. That
contrast is the section's idea for tp. The expert row: fine-grained
experts (d_ff = 768) do little work per byte moved, so ep is chatty;
dispatching in fp8 (b = 1) halves it. The ratios only shrink with the
ignored terms (last line). -->

```{.python .input #networks-axis-ratios}
%%tab pytorch
cfg_8b = d2l.hw_config('Qwen/Qwen3-8B')
cfg_moe = d2l.hw_config('Qwen/Qwen3-30B-A3B')
s, tokens_per_step, k = 4096, 1024 * 4096, 8  # sequence, tokens, replicas
par = dict(b=2, b_g=2, t=8, e=8, n_stage=4, T=tokens_per_step // k, k=k)
mfu = 0.4  # achieved share of the peak (an assumption)
P_eff = mfu * a100.p16_tflops * 1e12  # FLOP/s
ridges = {link: P_eff / beta for link, beta in
          d2l.link_betas(systems, d2l.box_alpha_beta()).items()}
cases = {'tp8 (8B)': ('tp', cfg_8b, {}), 'tp2 (8B)': ('tp', cfg_8b, {'t': 2}),
         'pp4 (8B)': ('pp', cfg_8b, {}), 'dp8 (8B)': ('dp', cfg_8b, {}),
         'ep8 (30B-A3B)': ('ep', cfg_moe, {})}
rho = pd.DataFrame({link: {row: axis_ratio(axis, cfg, R, **(par | deg))
                           for row, (axis, cfg, deg) in cases.items()}
                    for link, R in ridges.items()})
d2l.show_ratios(rho)
attn = d2l.flops_per_token(cfg_8b, T=s, causal_half=True) / \
    d2l.flops_per_token(cfg_8b) - 1
print(f'every column: one A100 at MFU {mfu:g}, {P_eff / 1e12:.0f} TFLOP/s')
print(f'ignored, both lower the ratios: attention adds {attn:.0%} to the '
      f'FLOPs at s = {s:,};')
print('overlap f multiplies a ratio by 1 - f')
assert (rho.loc['pp4 (8B)'] < rho.loc['tp8 (8B)']).all()  # D1's ordering
```

<!-- Placeholder (prose): the layout calculator. A layout gives each axis a
degree (t, e, p and k data-parallel replicas; expert groups are carved out
of the replicas, tensor parallelism innermost); an axis lands on the
scale-up link when the ranks it spans fit in one domain, else on the
scale-out network; it fits when its ratio after overlap is at most epsilon
= 0.15. Overlap is a parameter, not a fact (blueprint: 0 for tp, 0.5 for
ep, 0.9 for pp and dp; AXIS_OVERLAP). P_eff is mfu times the chip's dense
bf16 peak (the same 0.4 as above). Two corrections to D1 inside the
calculator, one sentence each: the dp ratio counts the gradients a chip
holds, while the FLOPs count the layers' matmul parameters a token uses
(D1 cancels N only when the two are equal), so dp's ratio grows by
held x t p / (L N_layer): 1.2 for Qwen3-8B (its embeddings), about 1.9
for Qwen3-30B-A3B at e = 8 (every chip holds its share of all experts),
and about 11 without expert parallelism; and the bytes per step of ep
come from the expert FLOPs its ratio is measured against. Memory per chip: the
16-byte mixed-precision training state, sharded by t and p (experts also
by e); activations not counted. -->

```{.python .input #networks-layout-def}
%%tab pytorch
AXIS_OVERLAP = {'tp': 0.0, 'ep': 0.5, 'pp': 0.9, 'dp': 0.9}  #@save

def layout_fit(cfg, system, layout, tokens_per_step, p_eff=None, eps=0.15,
               overlap=AXIS_OVERLAP, b=2, b_g=2, mfu=1.0):  #@save
    """Per axis of a layout (degrees t, e, p, k) on a systems-table row:
    the link it lands on (scale-up when the ranks it spans fit in one
    domain), that link's ridge, GB per chip and step, the ratio after
    overlap and whether it fits; attrs: the training state per chip.
    P_eff is p_eff, else mfu times the chip's dense bf16 peak."""
    P = p_eff or mfu * system.p16_tflops * 1e12
    t, e, p, k = (layout[a] for a in 'tepk')
    degree = {'tp': t, 'ep': e, 'pp': p, 'dp': k}
    span = {'tp': t, 'ep': t * e, 'pp': t * e * p, 'dp': t * p * k}
    par = dict(b=b, b_g=b_g, t=t, e=e, n_stage=p, T=tokens_per_step / k, k=k)
    mem = d2l.state_per_chip(cfg, system, layout)
    t_c = (6 * cfg.num_hidden_layers * layer_params(cfg) * tokens_per_step
           / (t * p * k * P))  # compute seconds per chip and step
    t_axis = {'ep': t_c * d2l.expert_flop_share(cfg)}  # ep: expert FLOPs
    held = mem['n_held'] * t * p / (cfg.num_hidden_layers * layer_params(cfg))
    rows = {}
    for axis in (a for a in degree if degree[a] > 1):
        up = span[axis] <= system.domain_chips
        beta = 1e9 * (system.up_gbs_dir if up
                      else system.scaleout_gbs_per_chip)
        rho = axis_ratio(axis, cfg, P / beta, **par) * (
            held if axis == 'dp' else 1)
        rows[axis] = dict(degree=degree[axis], ridge=P / beta,
                          link='scale-up' if up else 'scale-out',
                          gb_per_step=rho * t_axis.get(axis, t_c) * beta / 1e9,
                          ratio=rho * (1 - overlap[axis]))
    fit = pd.DataFrame.from_dict(rows, orient='index')
    fit['fit'] = fit.ratio <= eps
    fit.attrs.update(mem, eps=eps)
    return fit
```

<!-- Placeholder (prose): the usual placement as a rule (the chattiest
axis on the fastest link): tensor parallelism inside a domain for the
dense model, expert parallelism inside a domain for the mixture of
experts (no tensor parallelism: its d = 2,048 is too narrow), each at the
largest power-of-two degree, up to 8, whose ratio on the scale-up link is
at most epsilon (a faster chip behind the same link needs a smaller
degree); 4 pipeline stages; data parallelism over the rest of a 1,024-chip
job; a system without a scale-out network (the box) runs the job on its
own chips. -->

```{.python .input #networks-layout-for}
%%tab pytorch
def layout_for(cfg, system, degrees, job_chips, tokens_per_step, **kw):  #@save
    """The usual placement of a job of job_chips chips (all the system's
    chips when it has no scale-out network): the largest power-of-two t,
    then e, up to degrees and one domain, whose ratio fits (layout_fit
    with the keywords kw); p pipeline stages; data parallelism over the
    rest."""
    n = job_chips if system.has_scaleout else int(system.chips)
    D = int(min(system.domain_chips, n))

    def place(t, e):
        p = min(degrees['p'], n // (t * e))
        return dict(t=t, e=e, p=p, k=n // (t * p))

    def fits(axis, t, e):
        return layout_fit(cfg, system, place(t, e), tokens_per_step,
                          **kw).fit[axis]

    powers = lambda cap: [2**i for i in range(int(math.log2(cap)), 0, -1)]
    t = next((t for t in powers(min(degrees['t'], D)) if fits('tp', t, 1)), 1)
    e = next((e for e in powers(min(degrees['e'], D // t))
              if fits('ep', t, e)), 1)
    return place(t, e)
```

<!-- Placeholder (prose): the calculator on five systems and the two
models: the 8 x A100 node, HGX B200, GB200 NVL72, the TPU v7 pod (its
per-link ICI rate is on Google's page, so it is included) and the box.
Helios and Trn3 are left out, as the plan says for a system whose rates
are not stated (first line: which rate is missing). Each cell of the table
names the axes with their degrees (tp2 = tensor parallel over 2; in
parentheses: over epsilon). Claim N3, stated as computed: read the table.
Tensor and expert parallelism land inside the domain at the degree that
fits, and tp fits across a NIC at no degree (the tp2 row above); pp and dp
go across. On NVL72 pp moves inside. On the box, Qwen3-8B's 131 GB
training state fits no layout of 4 x 24 GB (the memory line). The plot
shows every axis as a point at its link's ridge, one marker per system: at
a fixed degree a ratio grows in proportion to the ridge, so for each axis
the boundary at epsilon is a ridge, and a faster chip behind the same link
moves every point right. -->

```{.python .input #networks-layout-fit}
%%tab pytorch
job_chips = 1024
degrees = {'8B': (cfg_8b, dict(t=8, e=1, p=4)),
           '30B-A3B': (cfg_moe, dict(t=1, e=8, p=4))}
chosen = d2l.stated_rates(systems.loc[[
    '8 x A100 SXM 40 GB', 'HGX B200', 'GB200 NVL72', 'Helios (72 x MI455X)',
    'Trn3 UltraServer', 'TPU v7 pod', 'this box (4 x RTX 4090)']])
fits = []
for _, system in chosen.iterrows():
    for cfg, deg in degrees.values():
        layout = layout_for(cfg, system, deg, job_chips, tokens_per_step,
                            mfu=mfu)
        fits.append(layout_fit(cfg, system, layout, tokens_per_step, mfu=mfu))
d2l.plot_layout_fit(fits)
d2l.show_layout_fit(fits)
```

<!-- Placeholder (prose): epsilon is a choice: the lines below re-read the
same ratios (same layouts) at a stricter and a looser threshold and name
every verdict that changes (claim N4); the plot's dashed line moves with
it. -->

```{.python .input #networks-epsilon}
%%tab pytorch
d2l.show_epsilon(fits, eps_values=(0.05, 0.3))
flips = [x for f in fits for x in f.ratio.dropna()
         if (x <= 0.05) != (x <= 0.3)]
assert flips  # N4: the boundary moves with epsilon
```

<!-- Placeholder (prose): scale-across in one paragraph. Training across
datacentres synchronizes pseudo-gradients every H inner steps (DiLoCo, one
sentence; Streaming DiLoCo overlaps the outer step with the next inner
steps, one sentence): hidden iff the WAN carries N b bytes within H
t_step seconds. The 2 s per optimizer step of an 8B run is a stated
assumption. -->

```{.python .input #networks-diloco}
%%tab pytorch
def diloco_wan_bandwidth(n_params, bytes_per_param, h_steps, t_step_s):  #@save
    """Bytes/s a WAN must carry to hide one synchronization of the
    parameters every h_steps steps of t_step_s seconds."""
    return n_params * bytes_per_param / (h_steps * t_step_s)

t_step_s = 2.0  # seconds per optimizer step of an 8B run (an assumption)
n_8b = d2l.ServingModel(cfg_8b, a100).n_params()[0]
wan = diloco_wan_bandwidth(n_8b, bytes_per_param=2, h_steps=100,
                           t_step_s=t_step_s)
print(f'WAN rate that hides a bf16 sync every 100 steps: '
      f'{wan * 8 / 1e9:.2f} Gb/s ({wan / 1e6:.0f} MB/s)')
```

<!-- Placeholder (prose): the two-node campaign C-T4 (shared with
sec_lm_collectives: one job script, one allocation): ib_write_bw and
ib_write_lat between two 8 x A100 nodes, 8 B to 4 GiB, one rail and all
rails, ibstat and the leaf identifiers; parse_perftest and fit_perftest
give alpha and beta per rail. Claim N5 (pending): per-rail beta within 20%
of the NIC line rate, and the measured RDMA latency floor. Until the log is
committed the cell prints one status line and the sentence says "pending:
part of the A100 campaign". -->
<!-- campaign: lm-hw-cluster-nccl-v1 -->

```{.python .input #networks-cluster-twin}
%%tab pytorch
d2l.show_fits(d2l.fit_perftest(d2l.hw_campaign_log('cluster-nccl')))
```

## Summary

<!-- Placeholder (prose): 3-6 sentences; forward to sec_lm_collectives
(how NCCL moves bytes over both networks) and to Chapter 23's composed
cost model (by :numref: if its label resolves), which refines these
first-order ratios. -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per S/spec/22.6.md part 10: the
layouts of Qwen3-235B-A22B (config only, d2l.hw_config at its pinned
revision) that fit one NVL72 and their memory per chip (derive); the WAN
rate for a sync every 100 steps of an 8B model at other H (derive);
all-rail against single-rail bandwidth (measure, needs two nodes; readers
without them read the committed C-T4 log when it exists); the systems
table's both-directions against per-direction columns against the vendor
pages (check); the oversubscription at which a job spanning two leaves
loses half its data-parallel bandwidth (derive); open: a UALink system row
when one ships. -->
