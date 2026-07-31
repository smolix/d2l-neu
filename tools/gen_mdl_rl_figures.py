"""Illustrative figures for the Reinforcement Learning chapters (14 and 15).

Generates img/mdl-rl-*.svg in the shared mdl house style. Assembled from six
reviewed batches; every figure passed a render-and-inspect loop and the whole
set is byte-idempotent (fixed svg.hashsalt via gen_mdl_figures.save, seeded
RNGs only, no timestamps).

Run:
    .venv-pytorch/bin/python tools/gen_mdl_rl_figures.py
or via `make figures` (picked up by the gen_mdl_*_figures.py glob).
"""
import os
import sys

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Patch, Polygon, Rectangle)
from matplotlib.textpath import TextPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402  (applies the shared rcParams)

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
PURPLE, RED = '#9467bd', '#d62728'

# --- module-level constants (from batch fragments) ---
MAP4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
DELTA = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}
LAKE_EXPERT = {0: DOWN, 4: DOWN, 8: RIGHT, 9: DOWN, 13: RIGHT, 14: RIGHT}
BANDIT_ARMS = np.array([0.50, 0.42, 0.90, 0.25, 0.55, 0.38, 0.60, 0.32, 0.50,
                        0.45])
BANDIT_KAPPA = 0.5
TOY_MU, TOY_FLOOR, TOY_PEAK, TOY_WIDTH = 0.0, 0.4, 2.0, 1.0
TOY_RESCALE = 5.0                                  # only for drawing R(a)
TRIAD_R = 2.35
TRIAD_C = np.array([[-1.275, -0.736], [1.275, -0.736], [0.000, 1.472]])
OFFLINE_GAMMA, OFFLINE_ALPHA = 0.95, 0.2
OFFLINE_EPISODES, OFFLINE_SWEEPS = 500, 200
COLLAPSE = ["discounting", "bootstrapping", "the TD error",
            "the Bellman equations", "value iteration", "Q-learning",
            "replay", "the deadly triad", "per-token credit assignment"]
SURVIVE = ["the score function", "baselines and advantages",
           "variance grows with length", "on-policy staleness",
           "importance ratios and clipping", "trust regions in policy space",
           "entropy collapse", "over-optimization, and its two cures"]

# --- shared helpers ---

def _frozen_lake_mdp(slippery=False, desc=MAP4):
    """Return (P, R, terminal, n_states, n_actions) for the 4x4 lake.

    ``P[s, a, s2]`` is the transition probability and ``R[s, a]`` the expected
    immediate reward.  With ``slippery=True`` the intended direction and the
    two perpendicular ones each get probability 1/3, matching Gymnasium
    (whose slip set is ``[(a-1) % 4, a, (a+1) % 4]`` under the action order
    LEFT, DOWN, RIGHT, UP).
    """
    nrow, ncol = len(desc), len(desc[0])
    n_states, n_actions = nrow * ncol, 4
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    terminal = np.array([desc[s // ncol][s % ncol] in "HG" for s in range(n_states)])

    def move(s, a):
        r, c = divmod(s, ncol)
        dr, dc = DELTA[a]
        r2, c2 = min(max(r + dr, 0), nrow - 1), min(max(c + dc, 0), ncol - 1)
        return r2 * ncol + c2, 1.0 if desc[r2][c2] == "G" else 0.0

    for s in range(n_states):
        for a in range(n_actions):
            if terminal[s]:                      # absorbing, no further reward
                P[s, a, s] = 1.0
                continue
            slips = [(a - 1) % 4, a, (a + 1) % 4] if slippery else [a]
            p = 1.0 / len(slips)
            for b in slips:
                s2, rew = move(s, b)
                P[s, a, s2] += p
                R[s, a] += p * rew
    return P, R, terminal, n_states, n_actions

def _box(ax, cx, cy, w, h, text, color, fontsize=13, ls="-", lw=1.7,
         fc=None, tc="black"):
    """Rounded box, faint fill, coloured edge, centred text (as in perf/ssm)."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        fc="white" if fc is None else fc, ec=color,
        lw=lw, linestyle=ls, zorder=3))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                color=tc, zorder=4)

def _arc_arrow(ax, p, q, rad, color, lw=2.0, mut=17, ls="-"):
    """A curved arrow between two points (the loop edges of F1/F11/F16)."""
    ax.add_patch(FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
        mutation_scale=mut, color=color, lw=lw, linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=4))

def _grid(ax, desc, values=None, cmap="Blues", vmax=1.0, annotate=True,
          outline=None, path=None, faint=False):
    """Draw the 4x4 lake: cells, H/G/S glyphs, optional value shading, an
    optional orange outline set (the wavefront), an optional arrow path.

    Cell ``(r, c)`` is the unit square with corner ``(c, r)``; the caller sets
    an inverted ``ylim`` so row 0 is on top.  ``faint=True`` draws the map as
    a light backdrop (LIGHT edges, no fills) for figures that overlay their
    own colour on it.
    """
    nrow, ncol = len(desc), len(desc[0])
    fills = {"H": (GRAY, 0.55), "G": (GREEN, 0.35), "S": (BLUE, 0.18)}
    for r in range(nrow):
        for c in range(ncol):
            ch, s = desc[r][c], r * ncol + c
            fc, alpha = "white", 1.0
            if values is not None and ch not in "HG":
                fc, alpha = plt.get_cmap(cmap)(float(values[s]) / vmax), 1.0
            elif ch in fills and not faint:
                fc, alpha = fills[ch]
            ax.add_patch(Rectangle(
                (c, r), 1, 1, fc=fc, alpha=alpha,
                ec=LIGHT if faint else "black", lw=1.0 if faint else 1.2,
                zorder=1))
            if ch in "HGS":
                # Bottom-right of the cell: the index owns the top-left corner
                # and a path arrow owns the centre and the four edge midpoints.
                ax.text(c + 0.77, r + 0.75, ch, ha="center", va="center",
                        fontsize=16, fontweight="bold", zorder=2,
                        color=GRAY if faint else ("white" if ch == "H" else "black"),
                        alpha=0.55 if faint else 1.0)
            if annotate:
                ax.text(c + 0.09, r + 0.09, f"{s}", ha="left", va="top",
                        fontsize=12, color=GRAY, zorder=2)
    for s in outline or []:
        r, c = divmod(s, ncol)
        ax.add_patch(Rectangle((c, r), 1, 1, fc="none", ec=ORANGE, lw=2.4,
                               zorder=5))
    for s, s2 in zip((path or [])[:-1], (path or [])[1:]):
        (r, c), (r2, c2) = divmod(s, ncol), divmod(s2, ncol)
        p0, p1 = np.array([c + 0.5, r + 0.5]), np.array([c2 + 0.5, r2 + 0.5])
        u = (p1 - p0) / np.linalg.norm(p1 - p0)      # trimmed ends: successive
        fl.arrow(ax, p0 + 0.09 * u, p1 - 0.09 * u,   # arrows stay separate and
                 color=BLUE, lw=2.2, mut=13)         # clear of the S/G glyphs
    ax.set_aspect("equal")
    ax.axis("off")

def _value_iteration(P, R, gamma, iters, V0=None):
    """Exact synchronous value iteration; returns the ``(iters+1, S)`` stack of
    iterates so callers can draw a sweep or a convergence curve."""
    V = np.zeros(P.shape[0]) if V0 is None else np.asarray(V0, float).copy()
    hist = [V.copy()]
    for _ in range(iters):
        V = (R + gamma * P @ V).max(axis=1)      # one Bellman optimality sweep
        hist.append(V.copy())
    return np.array(hist)

def _state(ax, xy, color="black", r=0.13, fill="white", lw=1.6, label=None):
    """Backup-diagram state node: open circle.  ``label`` goes to its left."""
    ax.add_patch(Circle(xy, r, fc=fill, ec=color, lw=lw, zorder=5))
    if label:
        ax.text(xy[0] - r - 0.07, xy[1], label, ha="right", va="center",
                fontsize=12.5, color="black", zorder=6)

def _act(ax, xy, color="black", r=0.062, label=None):
    """Backup-diagram action node: filled dot.  ``label`` goes to its left."""
    ax.add_patch(Circle(xy, r, fc=color, ec=color, lw=1.0, zorder=5))
    if label:
        ax.text(xy[0] - r - 0.09, xy[1], label, ha="right", va="center",
                fontsize=12.5, color="black", zorder=6)

def _edge(ax, p, q, color=GRAY, lw=1.3, label=None, lpos=0.5, off=(0.0, 0.0),
          fontsize=11.5, lcolor="black", bbox=False):
    """Backup-diagram edge with an optional label offset off the line."""
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=3,
            solid_capstyle="round")
    if label:
        x = p[0] + lpos * (q[0] - p[0]) + off[0]
        y = p[1] + lpos * (q[1] - p[1]) + off[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                color=lcolor, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2)
                if bbox else None)

def _max_arc(ax, root, width=0.9, height=0.5, color=ORANGE, label="max",
             lpos=(0.0, -0.42), fontsize=12.5, ha="center", va="center"):
    """The Sutton-Barto arc marking a maximum over the child edges."""
    ax.add_patch(Arc(root, width, height, theta1=200.0, theta2=340.0,
                     color=color, lw=2.0, zorder=6))
    if label:
        ax.text(root[0] + lpos[0], root[1] + lpos[1], label, ha=ha, va=va,
                fontsize=fontsize, color=color, zorder=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

def _black_axes(ax, labelsize=11):
    """The plot look the checklist asks for: black spines, ticks, tick labels."""
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="black", labelsize=labelsize)

def _leader(ax, text, point, textpos, color, fontsize=13, ha="center",
            va="center"):
    """A label offset away from the thing it names, with a short arrow back."""
    ax.annotate(text, xy=point, xytext=textpos, color=color, fontsize=fontsize,
                ha=ha, va=va,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                shrinkA=3, shrinkB=4))

def _clone_visitation(eps=0.2, horizon=12, n=4000, seed=0):
    """Visitation probabilities on the lake for the expert and for a clone.

    Returns ``(p_expert, p_clone)``, each the probability that the state is
    entered at some point within ``horizon`` steps.  The clone reproduces the
    expert's action with probability ``1 - eps`` on the states the expert
    visits (its training set) and acts uniformly at random anywhere else,
    which is all a fit with no data there can do.
    """
    P, _, terminal, n_states, n_actions = _frozen_lake_mdp(slippery=False)
    succ = np.argmax(P, axis=2)                     # deterministic successor
    out = []
    for clone in (False, True):
        rng = np.random.default_rng(seed)
        seen = np.zeros((n, n_states), bool)
        s = np.zeros(n, int)
        seen[:, 0] = True
        for _ in range(horizon):
            a = np.zeros(n, int)
            for i in range(n):
                si = int(s[i])
                if terminal[si]:
                    continue
                if si in LAKE_EXPERT:
                    a[i] = LAKE_EXPERT[si]
                    if clone and rng.random() < eps:  # the clone's error rate
                        wrong = [x for x in range(n_actions) if x != a[i]]
                        a[i] = wrong[int(rng.integers(3))]
                elif clone:                           # never trained here
                    a[i] = int(rng.integers(n_actions))
            s = np.where(terminal[s], s, succ[s, a])
            seen[np.arange(n), s] = True
        out.append(seen.mean(axis=0))
    return out[0], out[1]

def _chain_returns(eps=0.01, horizon=64, n=20000, seed=3):
    """Return against horizon on the chain, for behaviour cloning and DAgger.

    The chain has one state per position: the expert steps right and collects
    one unit of reward per step, so its return is exactly ``T``.  A fitted
    policy errs with probability ``eps`` per step.  Behaviour cloning has never
    seen an off-chain state, so its first mistake costs it every remaining
    step; DAgger has been trained on the states its own rollouts reach, so it
    steps back onto the chain and loses only the step it got wrong.  Returns
    the two cumulative-reward curves averaged over ``n`` rollouts, so the entry
    at index ``T-1`` is the mean return at horizon ``T``.
    """
    rng = np.random.default_rng(seed)
    err_bc = rng.random((n, horizon)) < eps
    err_dag = rng.random((n, horizon)) < eps
    on_bc = np.cumprod(~err_bc, axis=1)              # lost for good
    on_dag = (~err_dag).astype(float)                # recovers immediately
    return np.cumsum(on_bc, axis=1).mean(0), np.cumsum(on_dag, axis=1).mean(0)

def _bandit_regret(n_pulls=2000, n_seeds=20, arms=BANDIT_ARMS,
                   kappa=BANDIT_KAPPA):
    """Mean cumulative regret of five exploration rules on a Bernoulli bandit.

    A bandit is an MDP with one state, so the rules the lake uses apply
    unchanged: greedy, fixed and annealed epsilon, UCB with the time-dependent
    index ``mean + kappa * sqrt(log t / n)`` (each arm played once first, as in
    the notebook), and Thompson sampling.  Regret is measured against the best
    arm's mean and averaged over ``n_seeds`` independent runs; seed streams
    match qlearning.md's cells exactly, so the drawn curves are the printed
    numbers.
    """
    n_arms = len(arms)
    best = arms.max()
    curves = {}
    for algo_id, algo in enumerate(("greedy", "eps", "anneal", "ucb",
                                    "thompson")):
        regret = np.zeros((n_seeds, n_pulls))
        for seed in range(n_seeds):
            rng = np.random.default_rng([2026, algo_id, seed])
            count = np.zeros(n_arms)
            total = np.zeros(n_arms)
            gap = np.empty(n_pulls)
            for t in range(n_pulls):
                q = np.where(count > 0, total / np.maximum(count, 1), 0.0)
                if algo == "thompson":
                    a = int(np.argmax(rng.beta(1 + total, 1 + count - total)))
                elif algo == "ucb":
                    if (count == 0).any():   # play each arm once first
                        a = int(np.argmax(count == 0))
                    else:
                        a = int(np.argmax(q + kappa
                                          * np.sqrt(np.log(t) / count)))
                else:
                    e = {"greedy": 0.0, "eps": 0.1}.get(
                        algo, max(0.02, 1.0 / np.sqrt(t + 1)))
                    if rng.random() < e:
                        a = int(rng.integers(n_arms))
                    else:                            # ties broken uniformly
                        top = np.flatnonzero(q == q.max())
                        a = int(top[rng.integers(len(top))])
                count[a] += 1
                total[a] += float(rng.random() < arms[a])
                gap[t] = best - arms[a]
            regret[seed] = np.cumsum(gap)
        curves[algo] = regret.mean(axis=0)
    return curves

def _qlearning_curves(n_episodes=256, n_seeds=20, alpha=0.9, gamma=0.95,
                      cap=100):
    """Coverage and return per episode for three exploration schedules.

    Tabular Q-Learning on the deterministic lake with the chapter's settings
    (``alpha = 0.9``, ``gamma = 0.95``, ties in the argmax broken uniformly at
    random).  Coverage counts the state-action pairs tried at least once, as a
    fraction of the 44 pairs that can be tried at all (the five terminal states
    are never acted from).  Returns ``{schedule: (coverage, return)}``, each
    averaged over ``n_seeds`` runs.
    """
    P, R, terminal, n_states, n_actions = _frozen_lake_mdp(slippery=False)
    succ = np.argmax(P, axis=2)
    live = ~terminal
    n_pairs = int(live.sum()) * n_actions
    out = {}
    for sched_id, sched in enumerate(("greedy", "uniform", "annealed")):
        cover = np.zeros((n_seeds, n_episodes))
        ret = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            rng = np.random.default_rng([31337, sched_id, seed])
            Q = np.zeros((n_states, n_actions))
            tried = np.zeros((n_states, n_actions), bool)
            for ep in range(n_episodes):
                e = {"greedy": 0.0, "uniform": 1.0}.get(
                    sched, 0.9 + (0.05 - 0.9) * ep / (n_episodes - 1))
                s, g = 0, 0.0
                for _ in range(cap):
                    if rng.random() < e:
                        a = int(rng.integers(n_actions))
                    else:
                        top = np.flatnonzero(Q[s] == Q[s].max())
                        a = int(top[rng.integers(len(top))])
                    tried[s, a] = True
                    s2, r = int(succ[s, a]), R[s, a]
                    y = r if terminal[s2] else r + gamma * Q[s2].max()
                    Q[s, a] += alpha * (y - Q[s, a])
                    g += r
                    s = s2
                    if terminal[s]:
                        break
                cover[seed, ep] = tried[live].sum() / n_pairs
                ret[seed, ep] = g
        out[sched] = (cover.mean(axis=0), ret.mean(axis=0))
    return out

def _moving_average(x, w=25):
    """Trailing mean over ``w`` points, same length as ``x``."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(idx - w + 1, 0)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)

def _toy_reward(a):
    return TOY_FLOOR + 2.0 * np.exp(-(a - TOY_PEAK) ** 2 / (2 * TOY_WIDTH ** 2))

def _toy_density(a, mu=TOY_MU):
    return np.exp(-(a - mu) ** 2 / 2) / np.sqrt(2 * np.pi)

def _toy_moments():
    """Exact moments of the one-step problem, by quadrature on a fine grid.

    With ``X = R(a)(a - mu)`` and ``Y = a - mu`` (so ``Var(Y) = 1``), returns
    ``(E[R], dJ/dmu, Var(X), Cov(X, Y))``.  The variance of the estimator with
    baseline ``b`` is then ``Var(X) - 2 b Cov(X, Y) + b^2``, minimal at
    ``b = Cov(X, Y)``.
    """
    a = np.linspace(-9.0, 12.0, 400001)
    w = _toy_density(a) * (a[1] - a[0])
    R, y = _toy_reward(a), a - TOY_MU
    e_r = float((w * R).sum())
    grad = float((w * R * y).sum())
    var_x = float((w * (R * y) ** 2).sum()) - grad ** 2
    cov = float((w * R * y * y).sum())               # E[X Y], and E[Y] = 0
    return e_r, grad, var_x, cov

def _path_arrow(ax, pts, color=ORANGE, lw=1.9, ls="-", mut=15):
    """Right-angle routed poly-line with one arrowhead at the far end."""
    pts = [tuple(map(float, p)) for p in pts]
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=4)
    fl.arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, mut=mut)

def _sigma(t):
    """The logistic map, and the only nonlinearity F13 needs."""
    return 1.0 / (1.0 + np.exp(-np.asarray(t, float)))

def _softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()

def _prob_bars(ax, p, color, width=0.66, values=False, fontsize=11):
    """One distribution over a small action set as bars, optionally with each
    height printed above it (F24; F13(b) draws grouped pairs by hand)."""
    x = np.arange(len(p))
    ax.bar(x, p, width, color=color, ec="black", lw=0.9, zorder=3)
    if values:
        for xi, v in zip(x, p):
            ax.text(xi, v + 0.035, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=fontsize, color="black", zorder=4)

def _elbow(ax, pts, color, lw=1.6, ls="--", mut=15):
    """A right-angled poly-line with an arrowhead on its last segment (the
    long feedback edges of the data-flow figures)."""
    pts = [np.asarray(p, float) for p in pts]
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls,
                zorder=3, solid_capstyle="round")
    fl.arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, mut=mut)

def _tiles(ax, xs, y, w, h, c0=LIGHT, c1=BLUE, labels=None, fontsize=11,
           vertical=False):
    """A run of small rectangles tinted ``c0`` (oldest) to ``c1`` (newest):
    the replay buffer of F16 and the batch stack of F19."""
    cmap = LinearSegmentedColormap.from_list("age", [c0, c1])
    n = len(xs)
    for i, u in enumerate(xs):
        color = cmap(i / max(n - 1, 1))
        cx, cy = (y, u) if vertical else (u, y)
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=color,
                               ec="black", lw=0.8, zorder=4))
        if labels is not None and labels[i]:
            ax.text(cx, cy, labels[i], ha="center", va="center",
                    fontsize=fontsize, color="black", zorder=5)

def _strike_width(fig, ax, text, fontsize):
    """Width of ``text`` as a fraction of ``ax``'s width, from the font metrics
    (no draw pass, so it is stable across runs).  Used to strike out the items
    of F25's "simplifies" column at exactly their own length."""
    pts = TextPath((0, 0), text, prop=FontProperties(size=fontsize)).get_extents()
    axes_pt = fig.get_size_inches()[0] * ax.get_position().width * 72.0
    return pts.width / axes_pt

def _triad_anchors(n=1101):
    """For each of the seven Venn regions, the interior point that is farthest
    from every arc, plus the region's area, by sampling the three discs.  The
    deepest point rather than the centroid: the pairwise regions are crescents,
    whose centroid can sit closer to an arc than their fat middle does, and a
    label must not touch an arc."""
    g = np.linspace(-3.8, 3.9, n)
    X, Y = np.meshgrid(g, g)
    dist = [np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) for cx, cy in TRIAD_C]
    inside = [d <= TRIAD_R for d in dist]
    clear = np.minimum.reduce([np.abs(d - TRIAD_R) for d in dist])
    out = {}
    for key in ("100", "010", "001", "110", "101", "011", "111"):
        m = np.ones_like(X, dtype=bool)
        for bit, ins in zip(key, inside):
            m &= ins if bit == "1" else ~ins
        j = np.argmax(np.where(m, clear, -1.0))
        out[key] = (np.array([X.flat[j], Y.flat[j]]),
                    m.sum() * (g[1] - g[0]) ** 2)
    return out

def _max_bias(k, n, rng):
    """Measured bias of the single and the double estimator of ``max_a Q(a)``
    when every true value is zero and every estimate has unit noise."""
    QA = rng.standard_normal((n, k))
    QB = rng.standard_normal((n, k))
    single = QA.max(axis=1)
    double = QB[np.arange(n), QA.argmax(axis=1)]   # one set picks, the other scores
    return single.mean(), double.mean()

def _offline_experiment():
    """The whole of F20, computed: a uniform-policy dataset on the slippery
    lake, the tabular offline fit it supports, and the exact optimum to score
    it against.  Returns counts, |Qhat - Q*|, and the greedy policy's pairs."""
    P, R, terminal, S, A = _frozen_lake_mdp(slippery=True)
    goal = np.array([MAP4[s // 4][s % 4] == "G" for s in range(S)])

    V = np.zeros(S)                                  # exact optimum, 400 sweeps
    for _ in range(400):
        V = np.where(terminal, 0.0, (R + OFFLINE_GAMMA * P @ V).max(axis=1))
    Qstar = np.where(terminal[:, None], 0.0, R + OFFLINE_GAMMA * P @ V)

    rng = np.random.default_rng(0)                   # the dataset
    data = []
    for _ in range(OFFLINE_EPISODES):
        s = 0
        for _ in range(100):
            a = int(rng.integers(A))
            s2 = int(rng.choice(S, p=P[s, a]))
            data.append((s, a, float(goal[s2]), s2, bool(terminal[s2])))
            s = s2
            if terminal[s]:
                break
    counts = np.zeros((S, A))
    for s, a, _, _, _ in data:
        counts[s, a] += 1

    Q = np.zeros((S, A))                             # the offline fit
    for _ in range(OFFLINE_SWEEPS):
        for s, a, r, s2, done in data:
            target = r if done else r + OFFLINE_GAMMA * Q[s2].max()
            Q[s, a] += OFFLINE_ALPHA * (target - Q[s, a])

    err = np.abs(Q - Qstar)
    greedy = [(s, int(Q[s].argmax())) for s in range(S) if not terminal[s]]
    return counts, err, greedy, terminal, len(data), V[0], Q[0].max()


# --- figures (F1..F25 in program order) ---

def fig_agent_env():          # F1  -> mdl-rl-agent-env
    """The loop and the trajectory as one object: two boxes exchanging $a_t$
    and $(r_t, s_{t+1})$, and underneath the same exchange written out along a
    time line.  Everything the agent emits is BLUE, everything the environment
    emits is ORANGE, in both halves of the picture."""
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.axis("off")
    ax.set_aspect("equal")

    # --- the loop ----------------------------------------------------------- #
    _box(ax, 2.0, 2.05, 2.6, 1.25, "agent\n$\\pi(a \\mid s)$", BLUE,
         fontsize=14, fc="#e8f1f8")
    _box(ax, 7.0, 2.05, 2.6, 1.25,
         "environment\n$P(s' \\mid s,a)$,  $r(s,a)$", GRAY,
         fontsize=14, fc="#f3f3f3")

    _arc_arrow(ax, (3.32, 2.28), (5.68, 2.28), -0.42, BLUE)     # action, over
    _arc_arrow(ax, (5.68, 1.82), (3.32, 1.82), -0.42, ORANGE)   # answer, under
    ax.text(4.5, 2.86, "$a_t$", ha="center", va="bottom", fontsize=14,
            color=BLUE)
    ax.text(4.5, 1.24, "$r_t,\\ s_{t+1}$", ha="center", va="top", fontsize=14,
            color=ORANGE)

    # --- the same thing as a trajectory ------------------------------------ #
    y = 0.25
    fl.arrow(ax, (0.6, y), (8.75, y), color="black", lw=1.1, mut=12)
    for k, x0 in enumerate([1.1, 3.1, 5.1, 7.1]):
        ax.plot([x0, x0], [y, y - 0.12], color="black", lw=1.1)
        ax.text(x0, y - 0.22, f"$t={k}$" if k == 0 else f"${k}$",
                ha="center", va="top", fontsize=12, color="black")
        ax.text(x0, y + 0.16, f"$s_{k}$", ha="center", va="bottom",
                fontsize=14, color=ORANGE)
        if k < 3:
            ax.text(x0 + 0.62, y + 0.16, f"$a_{k}$", ha="center", va="bottom",
                    fontsize=14, color=BLUE)
            ax.text(x0 + 1.24, y + 0.16, f"$r_{k}$", ha="center", va="bottom",
                    fontsize=14, color=ORANGE)
    ax.text(7.72, y + 0.16, "$\\ldots$", ha="center", va="bottom",
            fontsize=14, color="black")
    ax.text(8.72, y + 0.42, "one trajectory $\\tau$", ha="right", va="bottom",
            fontsize=13, color="black")

    ax.set_xlim(0.30, 9.10)
    ax.set_ylim(-0.31, 3.17)
    fl.save(fig, "mdl-rl-agent-env")

def fig_roadmap():            # F2  -> mdl-rl-roadmap
    """Both reinforcement-learning chapters on one pair of axes: what is
    learned against which data may drive the update.  The second chapter's
    entries are greyed, so the same rendering serves as a preview there and as
    a recapitulation here.  Section numbers are deliberately absent (they rot;
    the prose carries the :numref: cross-references).  DAgger lives in the
    on-policy column: its update data are the learner's own rollouts,
    relabeled by the expert, so only behaviour cloning is offline."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axis("off")

    XS = {"on": 1.65, "off": 5.15, "offline": 8.65}      # column centres
    YS = {"value": 1.42, "policy": 3.30, "both": 5.05}   # row centres
    W, LH = 2.75, 0.31                                   # box width, line height
    X0, Y0 = 0.00, 0.42                                  # axis corner

    # --- the two axes ------------------------------------------------------ #
    fl.arrow(ax, (X0, Y0), (10.30, Y0), color="black", lw=1.3, mut=14)
    fl.arrow(ax, (X0, Y0), (X0, 5.88), color="black", lw=1.3, mut=14)
    for key, name in [("on", "on-policy"), ("off", "off-policy"),
                      ("offline", "offline")]:
        ax.text(XS[key], Y0 - 0.14, name, ha="center", va="top", fontsize=13,
                color="black")
    ax.text(5.15, Y0 - 0.66, "which data may drive the update", ha="center",
            va="top", fontsize=14, color="black", style="italic")
    for key, name in [("value", "value\nfunction"), ("policy", "policy"),
                      ("both", "both")]:
        ax.text(X0 - 0.14, YS[key], name, ha="right", va="center", fontsize=13,
                color="black")
    ax.text(-1.62, 3.20, "what is learned", ha="center", va="center",
            fontsize=14, color="black", style="italic", rotation=90)

    # --- the boxes: (row, col) -> edge colour, dashed?, [(text, colour)] --- #
    CH2 = GRAY                                    # the second chapter, greyed
    cells = [
        ("value", "on", GRAY, True, [("SARSA", CH2)]),
        ("value", "off", GREEN, False, [("value iteration", "black"),
                                        ("Q-learning", "black"),
                                        ("DQN", CH2)]),
        ("value", "offline", GRAY, True, [("Q-learning", CH2),
                                          ("+ pessimism", CH2)]),
        ("policy", "on", BLUE, False, [("DAgger", "black"),
                                       ("REINFORCE", "black"),
                                       ("+ baselines", "black"),
                                       ("regularized PO", CH2),
                                       ("GRPO", CH2)]),
        ("policy", "offline", BLUE, False, [("behaviour cloning", "black")]),
        ("both", "on", GRAY, True, [("actor-critic", CH2), ("PPO", CH2)]),
        ("both", "off", GRAY, True, [("SAC", CH2)]),
        ("both", "offline", GRAY, True, [("Decision", CH2),
                                         ("Transformer", CH2)]),
    ]
    for row, col, color, dashed, lines in cells:
        cx, cy = XS[col], YS[row]
        h = 0.42 + LH * len(lines)
        _box(ax, cx, cy, W, h, "", color, ls=(0, (4, 2.4)) if dashed else "-",
             lw=1.5 if dashed else 1.7)
        top = cy + LH * (len(lines) - 1) / 2
        for i, (txt, tc) in enumerate(lines):
            ax.text(cx, top - i * LH, txt, ha="center", va="center",
                    fontsize=12.5, color=tc, zorder=4)

    # --- the one bridge between the two families --------------------------- #
    fl.arrow(ax, (3.20, YS["policy"]), (5.90, YS["policy"]), color=GRAY,
             lw=1.6, mut=14)
    ax.text(4.90, YS["policy"] + 0.14,
            "importance ratios,\nwith a variance\nbudget",
            ha="center", va="bottom", fontsize=12, color="black")

    # --- the two families, bracketed and named in the right margin --------- #
    for y0, y1, note in [(0.60, 2.20, "learn $Q$,\nact greedily"),
                         (2.34, 5.72, "learn $\\pi_\\theta$\ndirectly")]:
        ax.add_patch(Rectangle((0.10, y0), 10.00, y1 - y0, fc="none", ec=LIGHT,
                               lw=1.1, ls="--", zorder=0))
        ax.text(10.52, (y0 + y1) / 2, note, ha="center", va="center",
                fontsize=12, color="black", rotation=90)

    # --- which chapter is which ------------------------------------------- #
    _box(ax, 0.30, 6.45, 0.52, 0.30, "", BLUE)
    ax.text(0.66, 6.45, "Reinforcement Learning", ha="left", va="center",
            fontsize=13, color="black")
    _box(ax, 4.30, 6.45, 0.52, 0.30, "", GRAY, ls=(0, (4, 2.4)), lw=1.5)
    ax.text(4.66, 6.45, "Deep Reinforcement Learning", ha="left", va="center",
            fontsize=13, color=GRAY)

    ax.set_xlim(-1.88, 10.95)
    ax.set_ylim(-0.72, 6.78)
    # Fill the figure: the default subplot margins would cost a fifth of the
    # width, and every box here is sized to hold its text.
    fig.subplots_adjust(left=0.008, right=0.992, top=0.992, bottom=0.008)
    fl.save(fig, "mdl-rl-roadmap")

def fig_gridworld():          # F3  -> mdl-rl-gridworld
    """(a) the map and the shortest path; (b) one action on slippery ice, with
    the three probabilities read straight out of ``P[9, DOWN]`` rather than
    drawn by hand."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1]})
    LIM_X, LIM_Y = (-0.74, 4.74), (4.98, -0.74)   # identical box for both panels

    # --- (a) the map, with the six-step path ------------------------------- #
    PATH = [0, 4, 8, 9, 13, 14, 15]
    Pd, _, _, _, _ = _frozen_lake_mdp(slippery=False)
    for s, s2 in zip(PATH[:-1], PATH[1:]):        # the path is really walkable
        assert Pd[s, :, s2].max() == 1.0, (s, s2)
    _grid(axa, MAP4, path=PATH)
    axa.text(2.0, 4.42, "shortest path: 6 moves, return $\\gamma^5$",
             ha="center", va="center", fontsize=13, color="black")

    # --- (b) one action, up close, on slippery ice ------------------------- #
    Ps, _, _, _, _ = _frozen_lake_mdp(slippery=True)
    s, a = 9, DOWN
    dest = np.flatnonzero(Ps[s, a])
    _grid(axb, MAP4, faint=True)
    r, c = divmod(s, 4)
    axb.add_patch(Rectangle((c, r), 1, 1, fc="none", ec="black", lw=2.0,
                            zorder=6))
    s_intended = s + 4                             # "down" one row
    # Every probability label sits outside its destination cell, offset off the
    # arrow's own line: the two slides into the left and right margins, the
    # intended one sideways past its arrowhead.
    LBL = {8: ((-0.14, 2.50), "right", ORANGE),
           10: ((4.14, 2.50), "left", ORANGE),
           13: ((2.16, 3.66), "left", BLUE)}
    for s2 in dest:
        r2, c2 = divmod(s2, 4)
        color = BLUE if s2 == s_intended else ORANGE
        axb.add_patch(Rectangle((c2, r2), 1, 1, fc=color, alpha=0.18, ec="none",
                                zorder=2))
        fl.arrow(axb, (c + 0.5, r + 0.5), (c2 + 0.5, r2 + 0.5), color=color,
                 lw=2.3, mut=14)   # equal widths: all three have probability 1/3
        (lx, ly), ha, lc = LBL[int(s2)]
        axb.text(lx, ly, f"${Ps[s, a, s2]:.3g}".replace("0.333", "1/3") + "$",
                 ha=ha, va="center", fontsize=13, color=lc)
    axb.text(2.0, -0.34, "$s = 9$, action down, on slippery ice", ha="center",
             va="center", fontsize=13, color="black")
    axb.text(2.0, 4.42, "$\\sum_{s'} P(s' \\mid s, a) = 1$", ha="center",
             va="center", fontsize=13, color="black")

    for ax in (axa, axb):
        ax.set_xlim(*LIM_X)
        ax.set_ylim(*LIM_Y)
    fig.subplots_adjust(wspace=0.02)
    fl.save(fig, "mdl-rl-gridworld")

def fig_return_discount():    # O1  -> mdl-rl-return-discount
    """What $\\gamma$ buys: (a) the weight a reward $t$ steps away still
    carries, with the step at which it has fallen to 5%; (b) that step count as
    a horizon, $1/(1-\\gamma)$."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})
    GAMMAS = [(0.5, GRAY), (0.9, BLUE), (0.99, ORANGE)]
    THRESH = 0.05

    def cross(g):             # first t with gamma^t <= 0.05, computed not guessed
        return int(np.ceil(np.log(THRESH) / np.log(g)))

    # --- (a) the discount weights ------------------------------------------ #
    t = np.arange(0, 61)
    axa.axhline(THRESH, color="black", lw=1.1, ls="--", zorder=1)
    for g, color in GAMMAS:
        axa.plot(t, g ** t, color=color, lw=2.2,
                 label=f"$\\gamma = {g:g}$", zorder=3)
        tc = cross(g)
        if tc <= t[-1]:
            axa.plot([tc, tc], [0, THRESH], color=color, lw=1.2, ls="--",
                     zorder=2)
            axa.plot([tc], [g ** tc], "o", color=color, ms=6, zorder=4)
            axa.text(tc + 1.2, 0.105, f"$t = {tc}$", ha="left", va="center",
                     fontsize=13, color="black")
    axa.text(59.4, 0.075, "$\\gamma^t = 0.05$", ha="right", va="bottom",
             fontsize=13, color="black")
    axa.text(19, 0.42, f"$\\gamma = 0.99$ reaches $0.05$\nonly at "
                       f"$t = {cross(0.99)}$",
             ha="left", va="top", fontsize=13, color="black")
    axa.set_xlim(0, 61)
    axa.set_ylim(0, 1.06)
    axa.set_yticks([0.0, THRESH, 0.25, 0.5, 0.75, 1.0])
    axa.set_yticklabels(["0", "0.05", "0.25", "0.50", "0.75", "1"])
    axa.set_xlabel("steps into the future, $t$", fontsize=13)
    axa.set_ylabel("weight $\\gamma^t$", fontsize=13)
    axa.legend(fontsize=13, loc="upper right", handlelength=1.5)

    # --- (b) the same number as a horizon ---------------------------------- #
    g = np.linspace(0.5, 0.995, 400)
    axb.semilogy(g, 1.0 / (1.0 - g), color="black", lw=2.2, zorder=3)
    for gv, color in GAMMAS:
        h = 1.0 / (1.0 - gv)
        axb.plot([gv, gv], [1.4, h], color=LIGHT, lw=1.1, ls="--", zorder=1)
        axb.plot([gv], [h], "o", color=color, ms=8, zorder=4)
        # Always label above the marker, on whichever side is empty: the
        # curve leaves the first marker rightwards and the others upwards.
        if gv == GAMMAS[0][0]:
            axb.text(gv + 0.016, h * 1.22, f"${h:.0f}$", ha="left", va="bottom",
                     fontsize=13, color=color)
        else:
            axb.text(gv - 0.012, h * 1.35, f"${h:.0f}$", ha="right", va="bottom",
                     fontsize=13, color=color)
    axb.set_xlim(0.48, 1.0)
    axb.set_ylim(1.4, 400)
    axb.set_xticks([0.5, 0.7, 0.9, 0.99])
    axb.set_xticklabels(["0.5", "0.7", "0.9", "0.99"])
    axb.set_yticks([2, 10, 100])
    axb.set_yticklabels(["2", "10", "100"])
    axb.set_xlabel("$\\gamma$", fontsize=14)
    axb.set_ylabel("horizon $1/(1-\\gamma)$  (steps)", fontsize=13)

    for ax in (axa, axb):
        for s in ("left", "bottom"):
            ax.spines[s].set_color("black")
        ax.tick_params(colors="black", labelsize=11)
    fig.subplots_adjust(wspace=0.26)
    fl.save(fig, "mdl-rl-return-discount")

def fig_backups():            # F4  -> mdl-rl-backups
    """The four one-step backups of the chapter in one node vocabulary: open
    circles are states, filled dots are state-action pairs, the orange arc is a
    maximum, and the one blue branch of the last panel is a single sample."""
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.2))
    LIM_X, LIM_Y = (-1.32, 1.32), (-0.62, 2.66)   # identical box for all four
    ROOT, YA, YS = (0.0, 2.2), 1.35, 0.35         # root, middle row, leaf row
    DX1, DX2 = 0.55, 0.28                         # first and second fan-out
    FORM, EDGE = 12.0, 11.5                       # formula / edge-label sizes

    def tree_va(ax, faded=False):
        """The V-shaped tree: state root, two actions, four states."""
        c = LIGHT if faded else GRAY                  # edges
        nc = LIGHT if faded else "black"              # nodes
        lw = 1.1 if faded else 1.3
        _state(ax, ROOT, color=nc, lw=1.3 if faded else 1.6,
               label=None if faded else "$s$")
        for sx in (-DX1, DX1):
            _edge(ax, ROOT, (sx, YA), color=c, lw=lw)
            _act(ax, (sx, YA), color=nc)
            for dx in (-DX2, DX2):
                _edge(ax, (sx, YA), (sx + dx, YS), color=c, lw=lw)
                _state(ax, (sx + dx, YS), color=nc, lw=1.3 if faded else 1.6)

    def tree_qa(ax, faded=False):
        """The tree rooted at a state-action pair: dot, two states, four acts."""
        c = LIGHT if faded else GRAY
        nc = LIGHT if faded else "black"
        lw = 1.1 if faded else 1.3
        _act(ax, ROOT, color=nc, label=None if faded else "$(s,a)$")
        for sx in (-DX1, DX1):
            _edge(ax, ROOT, (sx, YA), color=c, lw=lw)
            _state(ax, (sx, YA), color=nc, lw=1.3 if faded else 1.6)
            for dx in (-DX2, DX2):
                _edge(ax, (sx, YA), (sx + dx, YS), color=c, lw=lw)
                _act(ax, (sx + dx, YS), color=nc)

    def formula(ax, text):
        ax.text(0.0, -0.36, text, ha="center", va="center", fontsize=FORM,
                color="black")

    # --- (a) the value function of a policy -------------------------------- #
    axa = axes[0]
    tree_va(axa)
    axa.text(0.0, 1.60, r"$\pi(a \mid s)$", ha="center", va="center",
             fontsize=EDGE, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axa.text(0.0, 0.95, r"$P(s' \mid s,a)$", ha="center", va="center",
             fontsize=EDGE, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axa.text(0.84, 0.88, "$r$", ha="left", va="center", fontsize=EDGE,
             color="black")
    formula(axa, r"$V^\pi(s)=\sum_a \pi \left[\, r + \gamma \sum_{s'} P\,"
                 r"V^\pi(s') \right]$")

    # --- (b) the action-value function ------------------------------------- #
    axb = axes[1]
    tree_qa(axb)
    axb.text(-0.42, 1.86, r"$P(s' \mid s,a)$", ha="right", va="center",
             fontsize=EDGE, color="black")
    axb.text(0.0, 0.95, r"$\pi(a' \mid s')$", ha="center", va="center",
             fontsize=EDGE, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    formula(axb, r"$Q^\pi(s,a)=r+\gamma\sum_{s'}P\sum_{a'}\pi\,"
                 r"Q^\pi(s',a')$")

    # --- (c) the Bellman optimality backup: the max arc -------------------- #
    axc = axes[2]
    tree_va(axc)
    _max_arc(axc, ROOT, width=0.9, height=0.5, lpos=(0.0, -0.42))
    axc.text(0.0, 0.95, r"$P(s' \mid s,a)$", ha="center", va="center",
             fontsize=EDGE, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    formula(axc, r"$V^*(s)=\max_a \left[\, r+\gamma\sum_{s'}P\,"
                 r"V^*(s') \right]$")

    # --- (d) the sampled backup: one branch, and still a max --------------- #
    axd = axes[3]
    tree_qa(axd, faded=True)
    _act(axd, ROOT, color="black", label="$(s,a)$")
    obs = (DX1, YA)                                # the transition we observed
    _edge(axd, ROOT, obs, color=BLUE, lw=2.4, label="$r$", lpos=0.55,
          off=(0.15, 0.07), fontsize=12.5)
    _state(axd, obs, color=BLUE, lw=2.0)
    for dx in (-DX2, DX2):
        _edge(axd, obs, (obs[0] + dx, YS), color=GRAY, lw=1.3)
        _act(axd, (obs[0] + dx, YS), color="black")
    _max_arc(axd, obs, width=0.74, height=0.42, lpos=(0.40, -0.05),
             fontsize=11.5, ha="left")
    axd.text(0.0, 2.56, "one sample instead of the sum", ha="center",
             va="center", fontsize=12.0, color="black")
    formula(axd, r"$y = r+\gamma\max_{a'} \hat Q(s',a')$")

    for ax in axes:
        ax.set_xlim(*LIM_X)
        ax.set_ylim(*LIM_Y)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.02)
    fl.save(fig, "mdl-rl-backups")

def fig_contraction():        # F5  -> mdl-rl-contraction
    """(a) the sup-norm balls of radius $\\gamma^k$ around $V^*$ on a two-state
    MDP, with the iterates walking into them; (b) the same statement measured
    on the slippery lake, against the $\\gamma^k$ bound."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.2, 4.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    # --- (a) a two-state MDP, drawn in its own value plane ------------------ #
    r = np.array([[1.0, 0.0], [-0.5, 0.4]])           # r[state, action]
    Pa = np.array([[[0.7, 0.3], [0.4, 0.6]],          # Pa[action, state, next]
                   [[0.2, 0.8], [0.9, 0.1]]])
    gam = 0.8

    def backup(V):
        return np.max([r[:, a] + gam * Pa[a] @ V for a in (0, 1)], axis=0)

    V = np.array([-2.0, 3.0])
    iters = [V.copy()]
    for _ in range(400):
        V = backup(V)
        iters.append(V.copy())
    iters = np.array(iters)
    Vst = iters[-1]                                   # the fixed point
    r0 = np.abs(iters[0] - Vst).max()
    err = np.abs(iters - Vst).max(axis=1)
    assert (err[1:9] <= gam ** np.arange(1, 9) * r0 + 1e-12).all()

    for k in range(6):                                # nested sup-norm balls
        h = gam ** k * r0
        axa.add_patch(Rectangle((Vst[0] - h, Vst[1] - h), 2 * h, 2 * h,
                                fc="none", ec=LIGHT, lw=1.0, zorder=1))
    axa.annotate("", xy=(Vst[0] + r0, Vst[1]), xytext=tuple(Vst),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=0.9,
                                 shrinkA=0, shrinkB=0, mutation_scale=9),
                 zorder=4)
    for k in range(1, 6):                             # ticks at the radii
        x = Vst[0] + gam ** k * r0
        axa.plot([x, x], [Vst[1] - 0.14, Vst[1] + 0.14], color="black", lw=0.9,
                 zorder=4)
    bb = dict(fc="white", ec="none", alpha=0.9, pad=1.4)
    axa.text(Vst[0] + r0, Vst[1] + 0.24, "$r_0$", ha="center", va="bottom",
             fontsize=12.5, color="black", bbox=bb)
    axa.text(Vst[0] + gam * r0, Vst[1] - 0.26, r"$\gamma r_0$", ha="center",
             va="top", fontsize=12.5, color="black", bbox=bb)

    axa.plot(iters[:7, 0], iters[:7, 1], "o", color=BLUE, ms=5.0, zorder=6)
    for p, q in zip(iters[:6], iters[1:7]):
        fl.arrow(axa, tuple(p), tuple(q), color=BLUE, lw=1.6, mut=11)
    axa.plot(*Vst, "o", color=GREEN, ms=9, zorder=7)
    axa.text(Vst[0] + 0.30, Vst[1] + 0.34, "$V^*$", ha="left", va="bottom",
             fontsize=14, color=GREEN)
    for i, (dx, dy, ha, va) in enumerate([(0.30, 0.30, "left", "bottom"),
                                          (0.22, -0.30, "left", "top"),
                                          (-0.24, 0.10, "right", "bottom")]):
        axa.text(iters[i, 0] + dx, iters[i, 1] + dy, f"$V_{i}$", ha=ha, va=va,
                 fontsize=13, color=BLUE)

    pad = 1.08 * r0
    axa.set_xlim(Vst[0] - pad, Vst[0] + pad)
    axa.set_ylim(Vst[1] - pad, Vst[1] + pad)
    axa.set_aspect("equal")
    axa.set_xlabel("$V(s_1)$", fontsize=13)
    axa.set_ylabel("$V(s_2)$", fontsize=13)

    # --- (b) the same rate, measured on the slippery lake ------------------ #
    Ps, Rs, _, _, _ = _frozen_lake_mdp(slippery=True)
    g, K = 0.95, 120
    Vstar = _value_iteration(Ps, Rs, g, 5000)[-1]
    hist = _value_iteration(Ps, Rs, g, K)
    e = np.abs(hist - Vstar).max(axis=1)
    k = np.arange(K + 1)
    assert (e[1:] <= g ** k[1:] * e[0] + 1e-15).all()
    rate = (e[K] / e[60]) ** (1.0 / (K - 60))         # measured late-stage rate

    axb.semilogy(k, e, color=BLUE, lw=2.0, marker="o", markevery=10, ms=5,
                 zorder=4, label="value iteration")
    axb.semilogy(k, g ** k * e[0], color=ORANGE, lw=1.8, ls="--", zorder=3,
                 label=r"$\gamma^k \|V_0 - V^*\|_\infty$")
    axb.text(4, 3.4e-4, "the bound is never violated;\nthe measured rate here is\n"
                        f"${rate:.2f}$ per sweep, faster than $\\gamma$",
             ha="left", va="center", fontsize=12, color="black",
             linespacing=1.45)
    axb.set_xlim(-2, K + 2)
    axb.set_ylim(1e-5, 2.0)
    axb.set_xlabel("iteration $k$", fontsize=13)
    axb.set_ylabel(r"$\|V_k - V^*\|_\infty$", fontsize=13)
    axb.legend(loc="upper right", fontsize=11.5)

    for ax in (axa, axb):
        for s in ("left", "bottom"):
            ax.spines[s].set_color("black")
        ax.tick_params(colors="black", labelsize=11)
    fig.subplots_adjust(wspace=0.30)
    fl.save(fig, "mdl-rl-contraction")

def fig_value_wavefront():    # F6  -> mdl-rl-value-wavefront
    """Six sweeps of exact value iteration on the deterministic lake: the
    orange outline is the set of cells that acquired a value this sweep, and
    the start cell waits until the sixth."""
    Pd, Rd, _, _, _ = _frozen_lake_mdp(slippery=False)
    gam = 0.95
    H = _value_iteration(Pd, Rd, gam, 6)
    # d(s) = moves to the goal, so V_k(s) = gamma^(d(s)-1) once k >= d(s):
    assert abs(H[6][0] - gam ** 5) < 1e-12 and abs(H[1][14] - 1.0) < 1e-12
    assert abs(H[3][9] - gam ** 2) < 1e-12

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    desc = [row.replace("S", "F") for row in MAP4]     # "S" drawn in the corner
    for i, ax in enumerate(axes.ravel()):
        kk = i + 1
        wave = np.flatnonzero((H[kk] > 1e-12) & (H[kk - 1] <= 1e-12)).tolist()
        _grid(ax, desc, values=H[kk], vmax=1.6, annotate=False, outline=wave)
        for s in range(16):
            if H[kk][s] > 1e-12:
                rr, cc = divmod(s, 4)
                ax.text(cc + 0.5, rr + 0.5, f"{H[kk][s]:.2f}", ha="center",
                        va="center", fontsize=12, color="black", zorder=4,
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=1.0))
        ax.text(0.10, 0.10, "S", ha="left", va="top", fontsize=12,
                fontweight="bold", color="black", zorder=4)
        ax.set_title(f"$k = {kk}$", fontsize=13, color="black", pad=6)
        ax.set_xlim(-0.10, 4.10)
        ax.set_ylim(4.10, -0.10)
    axes[1, 2].text(2.0, 4.30, r"start cell: $0 \to \gamma^5 = "
                    f"{gam ** 5:.3f}$", ha="center", va="top", fontsize=13,
                    color="black", clip_on=False)
    fig.subplots_adjust(wspace=0.04, hspace=0.16)
    fl.save(fig, "mdl-rl-value-wavefront")

def fig_gpi():                # F23 -> mdl-rl-gpi
    """Generalized policy iteration: the two lines a learner tries to stand on
    at once, the zigzag that alternates between them, and the four algorithms
    of these chapters read off as four choices of how far each step goes."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.axis("off")
    ax.set_xlim(0.02, 10.10)
    ax.set_ylim(0.02, 4.95)

    CX, CY = 8.0, 4.3                 # the corner: V^*, pi^*
    A, B = 1.8, 0.30                  # dV per dpi on line A, dpi per dV on B
    # A*B < 1 is the contraction: each pair of moves shrinks the gap to the
    # corner by that factor, which is why the zigzag closes on it.

    def on_a(y):                      # evaluation: move V onto V = V^pi
        return CX - A * (CY - y)

    def on_b(x):                      # improvement: move pi onto greedy(V)
        return CY - B * (CX - x)

    def slope_angle(m):
        """Degrees a data slope ``m`` makes on the page, so a label can lie
        along a line instead of cutting across it."""
        bb = ax.get_position()
        w, h = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        return np.degrees(np.arctan(m * (h / (y1 - y0)) / (w / (x1 - x0))))

    # --- the abstract axes ------------------------------------------------- #
    fl.arrow(ax, (0.55, 0.30), (9.95, 0.30), color="black", lw=1.2, mut=13)
    fl.arrow(ax, (0.55, 0.30), (0.55, 4.80), color="black", lw=1.2, mut=13)
    ax.text(5.2, 0.10, "the value estimate $V$", ha="center", va="top",
            fontsize=13, color="black", style="italic")
    ax.text(0.24, 2.7, r"the policy $\pi$", ha="center", va="center",
            fontsize=13, color="black", style="italic", rotation=90)

    # --- the two lines, and the corner where both hold --------------------- #
    ax.plot([on_a(0.70), CX], [0.70, CY], color=GRAY, lw=2.0, zorder=2)
    ax.plot([1.05, CX], [on_b(1.05), CY], color=GRAY, lw=2.0, zorder=2)
    ax.text(3.80, CY - (CX - 3.80) / A - 0.10, "$V = V^\\pi$", ha="center",
            va="top", fontsize=14, color="black",
            rotation=slope_angle(1.0 / A), rotation_mode="anchor")
    ax.text(3.60, on_b(3.60) + 0.10, r"$\pi = \mathrm{greedy}(V)$",
            ha="center", va="bottom", fontsize=14, color="black",
            rotation=slope_angle(B), rotation_mode="anchor")

    # --- the zigzag: evaluate all the way, then improve all the way --------- #
    x, y = 1.34, 1.30
    pts = [(x, y)]
    for _ in range(4):
        x = on_a(y)
        pts.append((x, y))
        y = on_b(x)
        pts.append((x, y))
    for p, q in zip(pts[:-1], pts[1:]):
        fl.arrow(ax, p, q, color=BLUE, lw=1.9, mut=12)
    ax.plot(*pts[0], "o", color=BLUE, ms=6, zorder=5)
    ax.text(1.34, 1.14, "$V_0,\\ \\pi_0$", ha="center", va="top", fontsize=13,
            color=BLUE)
    ax.plot(CX, CY, "o", color=GREEN, ms=9, zorder=5)
    ax.text(CX + 0.20, CY + 0.16, "$V^*,\\ \\pi^*$", ha="left", va="bottom",
            fontsize=14, color=GREEN)
    ax.text(3.50, pts[2][1] - 0.12, "evaluation", ha="center", va="top",
            fontsize=12.5, color="black")
    ax.text(pts[3][0] + 0.15, 0.5 * (pts[3][1] + pts[4][1]), "improvement",
            ha="left", va="center", fontsize=12.5, color="black")

    # --- the four algorithms as four step disciplines ---------------------- #
    # Each key row draws the step it takes, at the same scale, so the picture
    # and the name are one object rather than a legend bolted on.
    def step_glyph(x0, y0, color, frac, jitter=False, diagonal=False):
        w, h = 0.66, 0.32
        if diagonal:                                   # both moves at once
            for i in range(3):
                fl.arrow(ax, (x0 + i * 0.32, y0 + i * 0.12),
                         (x0 + i * 0.32 + 0.28, y0 + i * 0.12 + 0.11),
                         color=color, lw=1.8, mut=10)
            return
        if jitter:                                     # a sampled evaluation
            jx = np.linspace(x0, x0 + frac * w, 9)
            jy = y0 + 0.032 * np.array([0, 1, -1, 1, -1, 1, -1, 0, 0])
            ax.plot(jx, jy, color=color, lw=1.6, ls=":", zorder=3)
            fl.arrow(ax, (jx[-2], jy[-2]), (jx[-1], jy[-1]), color=color,
                     lw=1.6, mut=10)
        else:
            fl.arrow(ax, (x0, y0), (x0 + frac * w, y0), color=color, lw=1.9,
                     mut=11)
        if frac < 1.0:                                 # the sweep not finished
            ax.plot([x0 + frac * w, x0 + w], [y0, y0], color=LIGHT, lw=1.4,
                    ls="--", zorder=2)
        fl.arrow(ax, (x0 + w, y0), (x0 + w, y0 + h), color=color, lw=1.9,
                 mut=11)

    KEY = [(2.00, BLUE, 1.00, "policy iteration: evaluate to the line"),
           (1.52, GREEN, 0.45, "value iteration: one sweep, then improve"),
           (1.04, GRAY, 0.45, "Q-learning: one sampled backup"),
           (0.56, ORANGE, 0.0, "actor-critic: both moves at once")]
    for yk, color, frac, text in KEY:
        step_glyph(4.95, yk, color, frac, jitter=(color is GRAY),
                   diagonal=(color is ORANGE))
        ax.text(6.10, yk + 0.16, text, ha="left", va="center", fontsize=12.5,
                color="black")

    fl.save(fig, "mdl-rl-gpi")

def fig_compounding_error():  # F21 -> mdl-rl-compounding-error
    """Why cloning degrades faster than its error rate suggests: (a) the
    expert's and the clone's state distributions on one lake, split cell by
    cell, with the first mistake that takes the clone off the expert's states;
    (b) the return each method loses on the chain, against the two rates."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [0.80, 1.30]})

    # --- (a) two distributions on one lake --------------------------------- #
    EPS_A, HORIZON = 0.2, 12
    p_exp, p_clone = _clone_visitation(eps=EPS_A, horizon=HORIZON)
    for s in range(16):                          # the map, as a faint backdrop
        r, c = divmod(s, 4)
        axa.add_patch(Rectangle((c, r), 1, 1, fc="white", ec=LIGHT, lw=1.0,
                                zorder=1))
    axa.set_aspect("equal")
    axa.axis("off")

    def half(ax, x0, y0, p, color, upper, size=1.0):
        """Shade half of a cell: the upper-left triangle for the expert, the
        lower-right one for the clone.  Alpha carries the probability."""
        tri = ([(x0, y0), (x0 + size, y0), (x0, y0 + size)] if upper
               else [(x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])
        ax.add_patch(Polygon(tri, closed=True, fc=color, ec="none",
                             alpha=0.14 + 0.72 * p, zorder=1.5))

    for s in range(16):
        r, c = divmod(s, 4)
        if p_exp[s] >= 0.005:
            half(axa, c, r, p_exp[s], BLUE, True)
        if p_clone[s] >= 0.005:
            half(axa, c, r, p_clone[s], ORANGE, False)
        ch = MAP4[r][c]
        if ch in "HGS":                        # glyphs back on top of the shading
            axa.text(c + 0.5, r + 0.62, ch, ha="center", va="center",
                     fontsize=16, fontweight="bold", color="black", zorder=6)

    # the first mistake: from state 9 the expert goes down, a slip goes right
    fl.arrow(axa, (1.70, 2.44), (2.66, 2.44), color="black", lw=2.2, mut=15)
    axa.text(2.18, 2.17, "first mistake", ha="center", va="center",
             fontsize=13, color="black", zorder=7,
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))

    # the key, drawn as the cell it explains
    half(axa, 0.05, -1.03, 1.0, BLUE, True, size=0.48)
    half(axa, 0.05, -1.03, 1.0, ORANGE, False, size=0.48)
    axa.add_patch(Rectangle((0.05, -1.03), 0.48, 0.48, fc="none", ec=LIGHT,
                            lw=1.0))
    axa.text(0.66, -0.93, "expert", ha="left", va="center", fontsize=13,
             color=BLUE)
    axa.text(0.66, -0.63, "clone", ha="left", va="center", fontsize=13,
             color=ORANGE)
    axa.text(1.88, -0.78, "darker: visited\nmore often", ha="left",
             va="center", fontsize=12.5, color="black")
    axa.text(2.0, 4.42, f"per-step error $\\varepsilon = {EPS_A}$, "
                        f"horizon $T = {HORIZON}$",
             ha="center", va="center", fontsize=13, color="black")
    axa.set_xlim(-0.12, 4.32)
    axa.set_ylim(4.68, -1.12)

    # --- (b) the two rates, measured on the chain --------------------------- #
    EPS_B = 0.01
    ret_bc, ret_dag = _chain_returns(eps=EPS_B, horizon=64)
    Ts = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64])
    lost_bc = Ts - ret_bc[Ts - 1]
    lost_dag = Ts - ret_dag[Ts - 1]

    grid = np.logspace(np.log10(3.4), np.log10(84.0), 200)
    axb.plot(grid, EPS_B * grid ** 2 / 2, ls="--", color=GRAY, lw=1.5,
             zorder=2)
    axb.plot(grid, EPS_B * grid, ls=":", color=GRAY, lw=1.8, zorder=2)
    axb.plot(Ts, lost_bc, "o-", color=ORANGE, lw=2.2, ms=6, zorder=4)
    axb.plot(Ts, lost_dag, "s-", color=GREEN, lw=2.2, ms=5.5, zorder=4)

    axb.text(4.1, 26.0, "behaviour cloning:\none mistake costs every\n"
                        "later step (dashed: $\\varepsilon T^2/2$)",
             ha="left", va="top", fontsize=13, color=ORANGE)
    axb.text(80.0, 0.031, "DAgger:\none mistake costs one step\n"
                          "(dotted: $\\varepsilon T$)",
             ha="right", va="bottom", fontsize=13, color=GREEN)
    axb.set_title(f"$\\varepsilon = {EPS_B}$ per step, "
                  f"20,000 rollouts per point", fontsize=12.5, color="black")

    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(3.4, 88.0)
    axb.set_ylim(0.024, 40.0)
    axb.set_xticks([4, 8, 16, 32, 64])
    axb.set_xticklabels(["4", "8", "16", "32", "64"])
    axb.set_yticks([0.05, 0.2, 1.0, 5.0, 20.0])
    axb.set_yticklabels(["0.05", "0.2", "1", "5", "20"])
    axb.set_xlabel("horizon $T$", fontsize=13)
    axb.set_ylabel("return lost to the expert", fontsize=13)
    _black_axes(axb)
    fig.subplots_adjust(wspace=0.16)
    fl.save(fig, "mdl-rl-compounding-error")

def fig_exploration():        # F7  -> mdl-rl-exploration
    """Exploration, in three measurements: (a) one action-value row read as
    three behaviour policies; (b) cumulative regret of five rules on a
    ten-armed bandit, with the UCB confidence radius kappa*sqrt(log t / n)
    inset at the run's horizon t = 2000; (c) what three schedules see, and
    what they earn while seeing it."""
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(13.8, 4.1),
        gridspec_kw={"width_ratios": [0.86, 1.10, 1.04]})

    # --- (a) one Q row, three policies ------------------------------------- #
    Q = np.array([0.20, 0.90, 0.55, 0.10])
    eps, tau = 0.3, 0.3
    greedy = (Q == Q.max()).astype(float)
    egreedy = eps / len(Q) + (1 - eps) * greedy
    softmax = np.exp(Q / tau) / np.exp(Q / tau).sum()
    x = np.arange(len(Q))
    rows = ((-0.27, greedy, GRAY, "greedy"),
            (0.0, egreedy, BLUE, "$\\epsilon$-greedy, $\\epsilon = 0.3$"),
            (0.27, softmax, ORANGE, "softmax, temperature 0.3"))
    for dx, p, color, _ in rows:
        axa.bar(x + dx, p, width=0.26, color=color, alpha=0.85, zorder=3)
    for i, (_, _, color, name) in enumerate(rows):      # labelled, not legended
        axa.text(-0.52, 1.38 - 0.125 * i, name, ha="left", va="center",
                 fontsize=12.5, color=color)
    axa.set_xticks(x)
    axa.set_xticklabels(["$\\leftarrow$", "$\\downarrow$", "$\\rightarrow$",
                         "$\\uparrow$"], fontsize=17)
    axa.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axa.set_xlim(-0.58, 3.5)
    axa.set_ylim(0, 1.48)
    axa.set_ylabel("$\\pi_e(a \\mid s)$", fontsize=13)
    axa.set_title("$\\hat Q(s, \\cdot) = (0.20,\\ 0.90,\\ 0.55,\\ 0.10)$",
                  fontsize=12.5, color="black")
    _black_axes(axa)

    # --- (b) regret on the bandit ------------------------------------------ #
    curves = _bandit_regret()
    n = np.arange(1, len(curves["greedy"]) + 1)
    for key, color, label in (("greedy", GRAY, "greedy"),
                              ("eps", BLUE, "$\\epsilon = 0.1$"),
                              ("anneal", GREEN,
                               "annealed $\\epsilon = 1/\\sqrt{n}$"),
                              ("ucb", ORANGE, f"UCB, $\\kappa = {BANDIT_KAPPA}$"),
                              ("thompson", PURPLE, "Thompson")):
        axb.plot(n, curves[key], color=color, lw=2.0, label=label, zorder=3)
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(1, 2600)
    axb.set_ylim(0.16, 2200.0)
    axb.set_xticks([1, 10, 100, 1000])
    axb.set_xticklabels(["1", "10", "100", "1000"])
    axb.set_yticks([0.3, 3, 30, 300])
    axb.set_yticklabels(["0.3", "3", "30", "300"])
    axb.set_xlabel("pulls $n$", fontsize=13)
    axb.set_ylabel("cumulative regret", fontsize=13)
    axb.legend(loc="upper left", fontsize=11.5, title="mean of 20 seeds",
               title_fontsize=11.5, handlelength=1.4, labelspacing=0.30,
               borderpad=0.1)
    _black_axes(axb)

    ins = axb.inset_axes([0.57, 0.155, 0.39, 0.25])
    m = np.arange(1, 41)
    ins.plot(m, BANDIT_KAPPA * np.sqrt(np.log(2000) / m), color=ORANGE, lw=2.0)
    ins.set_xlim(0, 41)
    ins.set_ylim(0, 1.55)
    ins.set_xticks([1, 20, 40])
    ins.set_yticks([0, 1])
    ins.set_yticklabels(["0", "1"])
    ins.set_xlabel("$n(a)$", fontsize=11, labelpad=0.5)
    ins.text(0.97, 0.94, "$\\kappa\\sqrt{\\log t/n}$", transform=ins.transAxes,
             ha="right", va="top", fontsize=13, color=ORANGE)
    ins.text(0.97, 0.66, "at $t = 2000$", transform=ins.transAxes, ha="right",
             va="top", fontsize=10, color=ORANGE)
    _black_axes(ins, labelsize=9.5)

    # --- (c) coverage and reward on the lake ------------------------------- #
    lake = _qlearning_curves()
    ep = np.arange(1, 257)
    for key, color in (("greedy", GRAY), ("annealed", GREEN),
                       ("uniform", RED)):
        cover, ret = lake[key]
        axc.plot(ep, cover, color=color, lw=2.0, zorder=3)
        axc.plot(ep, _moving_average(ret), color=color, lw=2.0, ls="--",
                 zorder=3)
    axc.text(6, 1.30, "greedy still explores: ties are coin flips",
             ha="left", va="top", fontsize=12.5, color="black")
    axc.text(250, 0.055, "$\\epsilon = 1$ never exploits", ha="right",
             va="bottom", fontsize=12.5, color=RED)
    axc.set_xlim(0, 258)
    axc.set_ylim(-0.02, 1.32)
    axc.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axc.set_xlabel("episode", fontsize=13)
    axc.set_ylabel("state-action pairs\ntried (solid)", fontsize=12)
    _black_axes(axc)
    axd = axc.twinx()                       # only to name the dashed family
    axd.set_ylim(*axc.get_ylim())
    axd.set_yticks([])
    axd.set_ylabel("success rate\n(dashed)", fontsize=12, color="black")

    fig.subplots_adjust(wspace=0.30)
    fl.save(fig, "mdl-rl-exploration")

def fig_score_ascent():       # F8  -> mdl-rl-score-ascent
    """One REINFORCE step on the one-step problem: (a) twelve sampled actions,
    each pushing its own log-probability up in proportion to what it earned;
    (b) the policy that one step produces."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 4.0))
    LIM_X, LIM_Y = (-3.4, 5.2), (-0.055, 0.66)
    SCALE = 0.055

    a = np.linspace(*LIM_X, 601)
    samples = np.random.default_rng(0).normal(TOY_MU, 1.0, 12)
    rew = _toy_reward(samples)
    u_hat = float((rew * (samples - TOY_MU)).mean())    # the score estimate
    alpha = 0.55 / u_hat                   # one step size, set for visibility
    mu_new = TOY_MU + alpha * u_hat

    for ax in (axa, axb):
        ax.plot(a, _toy_reward(a) / TOY_RESCALE, color=GREEN, lw=2.0, zorder=2)
        ax.set_xlim(*LIM_X)
        ax.set_ylim(*LIM_Y)
        ax.set_yticks([])
        ax.set_xticks([-2, 0, 2, 4])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_position(("data", 0.0))
        ax.spines["bottom"].set_color("black")
        ax.set_xlabel("action $a$", fontsize=13)
        ax.tick_params(colors="black", labelsize=11)
    axa.text(TOY_PEAK, 0.515, f"$R(a)/{TOY_RESCALE:.0f}$", ha="center",
             va="bottom", fontsize=13, color=GREEN)

    # --- (a) the twelve pushes --------------------------------------------- #
    axa.plot(a, _toy_density(a), color=BLUE, lw=2.2, zorder=3)
    axa.text(-1.45, _toy_density(-1.45) + 0.025, "$\\pi_\\theta(a)$",
             ha="right", va="bottom", fontsize=13, color=BLUE)
    for ai, ri in zip(samples, rew):
        axa.plot([ai, ai], [0.0, 0.014], color="black", lw=1.3, zorder=4)
        fl.arrow(axa, (ai, 0.016), (ai, 0.016 + SCALE * ri), color=BLUE,
                 lw=1.8, mut=7 + 7 * ri / rew.max())
    axa.text(-3.3, 0.645, "every sample pushes up;\nonly the size differs\n"
                          "(arrow length $\\propto R(a_i)$)",
             ha="left", va="top", fontsize=13, color="black")

    # --- (b) the policy one step later ------------------------------------- #
    axb.plot(a, _toy_density(a), color=LIGHT, lw=2.2, ls="--", zorder=3,
             label="before")
    axb.plot(a, _toy_density(a, mu_new), color=BLUE, lw=2.4, zorder=4,
             label="after one step")
    fl.arrow(axb, (TOY_MU, 0.435), (mu_new, 0.435), color="black", lw=1.6,
             mut=13)
    axb.text(5.1, 0.645, f"$\\mu: 0 \\to {mu_new:.2f}$", ha="right", va="top",
             fontsize=13, color="black")
    axb.legend(loc="upper left", fontsize=12.5, handlelength=1.6,
               borderpad=0.1)

    fig.subplots_adjust(wspace=0.10)
    fl.save(fig, "mdl-rl-score-ascent")

def fig_variance_reduction():  # F9  -> mdl-rl-variance-reduction
    """Two ways to quiet the same estimator: (a) rewards collected before an
    action cannot have been caused by it; (b) the sampling distribution with
    and without a baseline; (c) the variance as a function of the baseline."""
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(13.4, 3.9),
        gridspec_kw={"width_ratios": [1.08, 1.0, 1.0]})

    # --- (a) causality ------------------------------------------------------ #
    axa.axis("off")
    for t in range(8):
        past = t < 3
        _box(axa, t, 0.52, 0.90, 0.55, f"$r_{t}$",
             GRAY if past else BLUE, fontsize=13,
             fc="#f0f0f0" if past else "#e8f1f8",
             lw=1.5 if past else 1.7,
             tc=GRAY if past else "black")
        if past:
            axa.plot([t - 0.34, t + 0.34], [0.52, 0.52], color=GRAY, lw=1.6,
                     zorder=5)
    fl.arrow(axa, (2.5, 1.34), (2.5, 0.88), color="black", lw=1.6, mut=13)
    axa.text(2.5, 1.42, "action $a_3$ taken here", ha="center", va="bottom",
             fontsize=13, color="black")
    axa.plot([2.56, 2.56, 7.44, 7.44], [0.14, 0.02, 0.02, 0.14], color=BLUE,
             lw=1.6)
    axa.text(5.15, -0.08, "$\\hat G_3 = \\sum_{t' \\geq 3} "
                          "\\gamma^{t'-3} r_{t'}$",
             ha="center", va="top", fontsize=13, color="black")
    axa.text(0.55, 0.94, "already collected:\nmean-zero, dropped",
             ha="center", va="bottom", fontsize=12.5, color="black")
    axa.set_xlim(-0.80, 7.80)
    axa.set_ylim(-0.86, 1.80)

    # --- (b) the two sampling distributions -------------------------------- #
    e_r, grad, var_x, cov = _toy_moments()
    rng = np.random.default_rng(1)
    a = rng.normal(TOY_MU, 1.0, 200_000)
    y = a - TOY_MU
    x_plain = _toy_reward(a) * y
    x_base = (_toy_reward(a) - e_r) * y
    lo, hi = np.percentile(np.concatenate([x_plain, x_base]), [1.0, 95.0])
    bins = np.linspace(lo, hi, 91)
    for data, color, label in (
            (x_plain, BLUE, f"no baseline (std {x_plain.std():.2f})"),
            (x_base, ORANGE, f"$b = \\bar R$ (std {x_base.std():.2f})")):
        axb.hist(data, bins=bins, density=True, color=color, alpha=0.40,
                 zorder=2)
        axb.hist(data, bins=bins, density=True, histtype="step", color=color,
                 lw=1.8, label=label, zorder=3)
    peak = max(np.histogram(d, bins=bins, density=True)[0].max()
               for d in (x_plain, x_base))
    axb.set_ylim(0.0, peak * 1.34)
    axb.axvline(grad, color="black", lw=1.4, ls="--", zorder=5)
    axb.annotate(f"$\\nabla_\\mu J = {grad:.2f}$", xy=(grad, 0.62),
                 xycoords=("data", "axes fraction"), xytext=(8, 0),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=13, color="black",
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axb.text(hi, peak * 0.50, "long tails clipped", ha="right",
             va="bottom", fontsize=11.5, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axb.set_xlim(lo, hi)
    axb.set_xlabel("single-sample gradient estimate", fontsize=13)
    axb.set_ylabel("density", fontsize=13)
    axb.legend(loc="upper right", fontsize=12, handlelength=1.3,
               borderpad=0.1, labelspacing=0.3)
    _black_axes(axb)

    # --- (c) the variance parabola ----------------------------------------- #
    b = np.linspace(-0.6, 2.6, 400)
    var = var_x - 2 * b * cov + b ** 2
    b_star, rho = cov, cov / np.sqrt(var_x)
    axc.plot(b, var, color=BLUE, lw=2.4, zorder=3)
    for bv, color in ((0.0, GRAY), (e_r, GREEN), (b_star, ORANGE)):
        vv = var_x - 2 * bv * cov + bv ** 2
        axc.plot([bv, bv], [0.0, vv], color=color, lw=1.5, ls="--", zorder=2)
        axc.plot([bv], [vv], "o", color=color, ms=7, zorder=4)
    _leader(axc, "no baseline", (0.0, var_x), (0.30, var_x * 1.10), GRAY,
            ha="left", va="bottom")
    axc.set_xticks([0.0, e_r, b_star, 2.0])
    axc.set_xticklabels(["$0$", "$\\bar R$", "$b^\\star$", "$2$"])
    for lbl, color in zip(axc.get_xticklabels(), (GRAY, GREEN, ORANGE, "black")):
        lbl.set_color(color)
        lbl.set_fontsize(13 if color != "black" else 11)
    axc.text(0.95, var_x * 1.40,
             f"at $b^\\star$: $1 - \\mathrm{{corr}}^2 = {1 - rho ** 2:.2f}$\n"
             f"of the variance at $b = 0$",
             ha="center", va="top", fontsize=12.5, color="black")
    axc.set_xlim(-0.58, 2.58)
    axc.set_ylim(0.0, var_x * 1.45)
    axc.set_xlabel("baseline $b$", fontsize=13)
    axc.set_ylabel("variance of the estimator", fontsize=13)
    _black_axes(axc)

    fig.subplots_adjust(wspace=0.26)
    fl.save(fig, "mdl-rl-variance-reduction")

def fig_table_vs_network():    # F10 -> mdl-rl-table-vs-network
    """What one update touches, computed rather than drawn.

    The model is a random-feature regressor: ``phi(x) = tanh(W x + b)``, scaled
    to unit norm, fitted to a target value curve by ridge least squares.  Both
    halves of the picture are then the *same* formula,

        dV(x) = alpha * delta * k(x, x0),   k(x, x0) = phi(x) . phi(x0),

    because a table is exactly the case of one-hot features: its kernel is the
    indicator of "same bin", the model's is a smooth function of the state.
    Unit-norm features make ``k(x0, x0) = 1`` for both, so the two updates move
    the visited state by exactly the same amount and the panels differ only in
    what happens *elsewhere*.
    """
    rng = np.random.default_rng(0)
    M = 64
    W = rng.normal(0.0, 6.0, M)
    B = rng.uniform(-3.0, 3.0, M)

    def phi(x):
        h = np.tanh(np.outer(np.atleast_1d(x), W) + B)
        return h / np.linalg.norm(h, axis=1, keepdims=True)

    def vstar(x):
        return 0.6 + 0.35 * np.sin(2 * np.pi * x) - 0.2 * x

    xf = np.linspace(0.0, 1.0, 200)
    Phi = phi(xf)
    w = np.linalg.solve(Phi.T @ Phi + 1e-6 * np.eye(M), Phi.T @ vstar(xf))
    assert np.abs(Phi @ w - vstar(xf)).max() < 5e-3      # the fit is a fit

    x0, alpha, delta = 0.40, 0.5, 1.0                    # one semi-gradient step
    NB = 16                                              # table resolution
    step = alpha * delta                                 # move at the visited x
    grid = np.linspace(0.0, 1.0, 601)
    k = phi(grid) @ phi(x0)[0]
    dv_net = step * k                                    # exact effect of the step
    assert abs(dv_net.max() - step) < 1e-9               # k(x0, x0) = 1

    edges = np.linspace(0.0, 1.0, NB + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    tab0 = vstar(centres)
    b0 = int(np.searchsorted(edges, x0) - 1)             # the bin holding x0
    tab1 = tab0.copy()
    tab1[b0] += step

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(12.6, 3.7))
    YL = (0.02, 1.62)

    def stepplot(ax, vals, color, lw, label=None, zorder=3):
        ax.step(np.append(edges, 1.0), np.append(np.append(vals, vals[-1]),
                                                 vals[-1]),
                where="post", color=color, lw=lw, label=label, zorder=zorder)

    # --- (a) the table: one entry moves ------------------------------------ #
    stepplot(axa, tab0, LIGHT, 3.4, "before", zorder=2)
    stepplot(axa, tab1, BLUE, 1.8, "after", zorder=3)
    axa.plot([x0, x0], [0.06, 0.22], color="black", lw=1.4, zorder=4)
    axa.text(x0 + 0.02, 0.13, "$x_0$", ha="left", va="center", fontsize=13,
             color="black")
    axa.annotate("one entry moves,\nnothing else does",
                 xy=(centres[b0], tab1[b0] + 0.02), xytext=(0.015, 1.60),
                 ha="left", va="top", fontsize=12.5, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=2, shrinkB=3))
    axa.set_title("table: 16 entries", fontsize=13, color="black", pad=6)
    axa.set_ylabel(r"$\hat V(x)$", fontsize=13)
    axa.legend(loc="upper right", fontsize=11.5, handlelength=1.4)

    # --- (b) the same update on the model: the whole curve moves ----------- #
    v0 = phi(grid) @ w
    axb.plot(grid, v0, color=LIGHT, lw=3.4, zorder=2)
    axb.plot(grid, v0 + dv_net, color=BLUE, lw=2.0, zorder=3)
    axb.plot([x0], [phi(x0)[0] @ w], "o", color=ORANGE, ms=8, zorder=5)
    axb.annotate("the only state\nthe update mentions",
                 xy=(x0, float(phi(x0)[0] @ w) - 0.03), xytext=(0.055, 0.30),
                 ha="left", va="bottom", fontsize=12.5, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=2, shrinkB=3))
    axb.set_title("random-feature model: 64 features", fontsize=13,
                  color="black", pad=6)

    # --- (c) the two changes, side by side --------------------------------- #
    axc.bar(centres[b0], step, width=1.0 / NB, color=LIGHT, ec=GRAY, lw=1.0,
            zorder=3, label="table")
    axc.fill_between(grid, 0.0, dv_net, color=ORANGE, alpha=0.30, zorder=2)
    axc.plot(grid, dv_net, color=ORANGE, lw=2.2, zorder=4, label="model")
    axc.text(0.015, 0.648, "one update,\nevery state", ha="left", va="top",
             fontsize=12.5, color="black", linespacing=1.4)
    axc.annotate(f"still $+{dv_net[-1]:.2f}$ at the far end",
                 xy=(0.995, dv_net[-1] - 0.015), xytext=(0.955, 0.115),
                 ha="right", va="center", fontsize=12.5, color="black",
                 bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.6),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=3, shrinkB=3))
    axc.set_ylim(0.0, 0.66)
    axc.set_ylabel(r"$\Delta \hat V(x)$", fontsize=13)
    axc.legend(loc="upper right", fontsize=11.5, handlelength=1.4)

    for ax in (axa, axb):
        ax.set_ylim(*YL)
    for ax in (axa, axb, axc):
        ax.set_xlim(-0.01, 1.01)
        ax.set_xlabel("state $x$", fontsize=13)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("black")
        ax.tick_params(colors="black", labelsize=11)
    fig.subplots_adjust(wspace=0.22)
    fl.save(fig, "mdl-rl-table-vs-network")

def fig_score_vs_pathwise():  # F22 -> mdl-rl-score-vs-pathwise
    """Two ways to differentiate an expected return, in F1's box vocabulary.

    Left: the score-function estimator, whose only contact with the environment
    is a number, so the environment box may be opaque.  Right: the pathwise
    estimator, whose gradient walks back *through* the sampled action and
    through a critic -- which therefore has to be differentiable in the action,
    and which in exchange may be trained on anything.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    LIMX, LIMY = (0.10, 9.05), (1.90, 7.00)   # identical box -> identical panels
    GRAD = dict(color=ORANGE, lw=1.9, ls=(0, (4.5, 2.6)))
    HEAD, BOXF, LBL = 13.0, 13.0, 12.0

    # ===== (a) score function ============================================== #
    axa.text(4.55, 6.80, "score function:  "
                         r"$\hat g = R\,\nabla_\theta \log \pi_\theta(a)$",
             ha="center", va="center", fontsize=HEAD, color="black")

    _box(axa, 1.75, 5.35, 2.9, 1.0, "policy\n$\\pi_\\theta(a \\mid s)$", BLUE,
         fontsize=BOXF, fc="#e8f1f8")
    _box(axa, 7.10, 5.35, 2.9, 1.5, "", GRAY, fc="#dcdcdc")
    axa.text(7.10, 5.66, "environment", ha="center", va="center",
             fontsize=BOXF, color="black", zorder=4)
    axa.text(7.10, 5.34, "$r(s,a)$", ha="center", va="center", fontsize=BOXF,
             color="black", zorder=4)
    axa.text(7.10, 4.98, "not differentiated", ha="center", va="center",
             fontsize=11.5, color="black", style="italic", zorder=4)
    for y in (5.55, 5.35, 5.15):              # the N sampled actions
        axa.plot([4.15], [y], "o", color=BLUE, ms=6.0, zorder=5)
    axa.text(4.32, 5.98, "$N$ samples $a^{(i)}$", ha="center", va="center",
             fontsize=LBL, color="black")

    fl.arrow(axa, (3.25, 5.35), (3.90, 5.35), color=BLUE, lw=1.9, mut=14)
    fl.arrow(axa, (4.42, 5.35), (5.65, 5.35), color=GRAY, lw=1.9, mut=14)

    _box(axa, 4.50, 2.55, 6.4, 0.95,
         r"$\hat g = \frac{1}{N}\sum_i R^{(i)}\,"
         r"\nabla_\theta \log \pi_\theta(a^{(i)} \mid s)$", ORANGE,
         fontsize=BOXF)
    fl.arrow(axa, (7.10, 4.57), (7.10, 3.08), color=ORANGE, lw=1.9, mut=14)
    axa.text(6.95, 3.85, "$R^{(i)}$", ha="right", va="center", fontsize=BOXF,
             color=ORANGE)
    fl.arrow(axa, (4.15, 5.02), (4.15, 3.08), color=BLUE, lw=1.9, mut=14)
    axa.text(4.00, 3.90, r"$\nabla_\theta \log \pi_\theta$", ha="right",
             va="center", fontsize=LBL, color=BLUE)
    _path_arrow(axa, [(1.28, 2.55), (0.32, 2.55), (0.32, 5.35), (0.42, 5.35)],
                **GRAD)
    axa.text(0.46, 4.30, r"$\theta \leftarrow \theta + \alpha \hat g$",
             ha="left", va="center", fontsize=LBL, color="black")

    # ===== (b) pathwise ==================================================== #
    axb.text(4.55, 6.80, "pathwise:  "
                         r"$\hat g = \nabla_\theta Q_w(s, a_\theta)$",
             ha="center", va="center", fontsize=HEAD, color="black")

    _box(axb, 1.75, 5.35, 2.9, 1.0,
         "policy\n$\\mu_\\theta(s),\\ \\sigma_\\theta(s)$", BLUE,
         fontsize=BOXF, fc="#e8f1f8")
    _box(axb, 6.00, 5.35, 4.2, 1.10, "", BLUE)
    axb.text(6.00, 5.56, r"$a = \mu_\theta(s) + \sigma_\theta(s)\,z$",
             ha="center", va="center", fontsize=BOXF, color="black", zorder=4)
    axb.text(6.00, 5.10, r"$z \sim \mathcal{N}(0, I)$, held fixed",
             ha="center", va="center", fontsize=11.5, color=GRAY, zorder=4)
    _box(axb, 6.10, 2.55, 4.6, 1.20, "", GREEN, fc="#eaf6ea")
    axb.text(6.10, 2.76, "critic $Q_w(s,a)$", ha="center", va="center",
             fontsize=BOXF, color="black", zorder=4)
    axb.text(6.10, 2.36, "must be differentiable in $a$", ha="center",
             va="center", fontsize=11.5, color="black", style="italic",
             zorder=4)
    _box(axb, 1.75, 2.55, 2.9, 0.95, "replay buffer\n(any past data)", GRAY,
         fontsize=LBL, ls=(0, (4, 2.4)), lw=1.5)

    fl.arrow(axb, (3.25, 5.55), (3.95, 5.55), color=BLUE, lw=1.9, mut=14)
    fl.arrow(axb, (7.00, 4.80), (7.00, 3.15), color=BLUE, lw=1.9, mut=14)
    axb.text(7.15, 3.98, "$a$", ha="left", va="center", fontsize=BOXF,
             color=BLUE)
    fl.arrow(axb, (3.25, 2.55), (3.72, 2.55), color=GRAY, lw=1.7, mut=13,
             ls=(0, (4, 2.4)))

    fl.arrow(axb, (5.00, 3.15), (5.00, 4.80), mut=14, **GRAD)
    axb.text(5.18, 3.98, r"$\partial Q_w / \partial a$", ha="left",
             va="center", fontsize=LBL, color="black")
    fl.arrow(axb, (3.95, 5.15), (3.25, 5.15), mut=14, **GRAD)
    axb.text(3.60, 4.62, r"$\partial a / \partial \theta$", ha="center",
             va="center", fontsize=11.5, color="black")
    axb.text(1.75, 3.55, "the gradient walks back\nthrough both boxes",
             ha="center", va="center", fontsize=11.5, color="black",
             linespacing=1.4)

    for ax in (axa, axb):
        ax.set_xlim(*LIMX)
        ax.set_ylim(*LIMY)
        ax.set_aspect("equal")
        ax.axis("off")
    # Between the two panels, just below them (axes coordinates, so it follows
    # the panels however tight bounding boxes shift them).
    axa.text(1.0, -0.06, "the second needs a $Q$ you can differentiate; "
                         "a $Q$ can be trained off-policy",
             transform=axa.transAxes, ha="center", va="top", fontsize=13,
             color="black")
    fig.subplots_adjust(wspace=0.04, bottom=0.12, top=0.99)
    fl.save(fig, "mdl-rl-score-vs-pathwise")

def fig_actor_critic_loop():  # F11 -> mdl-rl-actor-critic-loop
    """One transition, one number, two jobs: the same $\\delta_t$ leaves the
    temporal-difference box twice, once towards the critic and once towards the
    actor.  Same box vocabulary as F1/F22."""
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    ax.axis("off")
    ax.set_aspect("equal")
    GRAD = dict(color=ORANGE, lw=1.9, ls=(0, (4.5, 2.6)))

    _box(ax, 1.6, 3.75, 2.7, 1.05, "actor $\\pi_\\theta$", BLUE, fontsize=14,
         fc="#e8f1f8")
    _box(ax, 8.1, 3.75, 3.0, 1.05, "environment", GRAY, fontsize=14,
         fc="#f3f3f3")
    _box(ax, 1.6, 1.55, 2.7, 1.05, "critic $\\hat V_w$", GREEN, fontsize=14,
         fc="#eaf6ea")
    _box(ax, 6.6, 1.55, 4.6, 0.95,
         r"$\delta_t = r_t + \gamma \hat V_w(s_{t+1}) - \hat V_w(s_t)$",
         ORANGE, fontsize=13.5)

    # forward: the actor acts, the environment answers
    fl.arrow(ax, (2.98, 3.75), (6.58, 3.75), color=BLUE, lw=2.0, mut=16)
    ax.text(4.78, 3.94, "$a_t$", ha="center", va="bottom", fontsize=14,
            color=BLUE)
    fl.arrow(ax, (8.1, 3.20), (8.1, 2.06), color=GRAY, lw=2.0, mut=16)
    ax.text(8.28, 2.62, "$r_t,\\ s_{t+1}$", ha="left", va="center",
            fontsize=13.5, color="black")

    # the critic supplies the two predictions the error is made of
    fl.arrow(ax, (2.98, 1.55), (4.24, 1.55), color=GREEN, lw=2.0, mut=16)
    ax.text(3.61, 2.00, "$\\hat V_w(s_t)$\n$\\hat V_w(s_{t+1})$", ha="center",
            va="center", fontsize=11.5, color=GREEN, linespacing=1.4)

    # the one number, leaving twice
    _path_arrow(ax, [(5.10, 1.06), (5.10, 0.52), (1.60, 0.52), (1.60, 0.99)],
                **GRAD)
    _path_arrow(ax, [(5.10, 2.04), (5.10, 2.76), (1.60, 2.76), (1.60, 3.20)],
                **GRAD)
    for y in (0.62, 2.86):
        ax.text(3.30, y, "$\\delta_t$", ha="center", va="bottom", fontsize=14,
                color=ORANGE)

    ax.text(5.2, -0.20,
            r"actor:  $\theta \leftarrow \theta + \alpha_\theta\, \delta_t\,"
            r"\nabla_\theta \log \pi_\theta(a_t \mid s_t)$"
            "        "
            r"critic:  $w \leftarrow w + \alpha_w\, \delta_t\,"
            r"\nabla_w \hat V_w(s_t)$",
            ha="center", va="center", fontsize=13, color="black")
    ax.text(5.2, -0.78, "the critic must learn faster than the actor: "
                        r"$\alpha_w$ large, or several critic passes per batch",
            ha="center", va="center", fontsize=12, color="black")

    ax.set_xlim(0.05, 10.35)
    ax.set_ylim(-1.05, 4.45)
    fl.save(fig, "mdl-rl-actor-critic-loop")

def fig_td_mc_spectrum():     # F12 -> mdl-rl-td-mc-spectrum
    """The credit-assignment dial: the same target family drawn (left, in F4's
    node vocabulary) and measured (right).

    The measurement is a ten-state chain, deterministic step right, terminal
    after the last state, whose only payment is $1$ on the final transition
    plus per-step reward noise.  The critic is deliberately imperfect in the
    shape value learning actually produces: exact near termination and wrong
    far from it, which is the wavefront of :numref:`fig_rl_value_wavefront` seen
    from the side.  The error profile is a smooth deterministic taper rather
    than per-state noise, so the bias curve shows the trade instead of a seed
    accident; the randomness that *is* measured is the 20000 seeded rollouts.
    Both the bias and the variance of the depth-$n$ target are available in
    closed form here, and the rollouts reproduce them, so the panel is a
    measurement with a proof attached.

    Note that a critic whose error is the *same* at every state would make the
    mean squared error monotone in $n$ (bias$^2$ and variance would then both
    depend on the depth only through $\\gamma^{2n}$).  The interior optimum
    exists because deeper targets bootstrap from better-known states.
    """
    fig = plt.figure(figsize=(11.8, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0],
                          height_ratios=[2.8, 1.0], wspace=0.22, hspace=0.34)
    axa = fig.add_subplot(gs[0, 0])
    axl = fig.add_subplot(gs[1, 0])
    axb = fig.add_subplot(gs[:, 1])

    # ===== (a) four chains in F4's node vocabulary ========================= #
    XS = [0.95, 2.45, 3.95, 5.65]
    TOP, DY, GAP = 3.05, 0.33, 0.50
    NAMES = ["TD(0)", "2-step", "$n$-step", "Monte Carlo"]
    PAIRS = {0: 1, 1: 2, 2: 2, 3: 3}            # (action, state) pairs drawn
    r_lbl = {0: "$r_t$", 1: None, 2: None, 3: None}

    for j, x in enumerate(XS):
        axa.text(x, TOP + 0.34, NAMES[j], ha="center", va="center",
                 fontsize=12.5, color="black")
        y = TOP
        _state(axa, (x, y), label="$s_t$" if j == 0 else None)
        for d in range(PAIRS[j]):               # the reward edge, then a state
            _edge(axa, (x, y), (x, y - DY),
                  label=r_lbl[j] if d == 0 else None, off=(0.30, 0.0),
                  fontsize=12.0)
            _act(axa, (x, y - DY))
            _edge(axa, (x, y - DY), (x, y - 2 * DY))
            y -= 2 * DY
            if d < PAIRS[j] - 1:                # an interior state node
                _state(axa, (x, y))
        if j >= 2:                              # ... a dotted continuation ... #
            axa.plot([x, x], [y - 0.06, y - GAP + 0.06], color=LIGHT, lw=2.2,
                     ls=(0, (1.0, 2.0)), zorder=2)
            y -= GAP
            _edge(axa, (x, y), (x, y - DY))
            _act(axa, (x, y - DY))
            _edge(axa, (x, y - DY), (x, y - 2 * DY))
            y -= 2 * DY
        if j < 3:                               # the bootstrap node
            _state(axa, (x, y), color=GREEN, fill=GREEN, lw=1.6)
            axa.text(x + 0.19, y, r"$\hat V$", ha="left", va="center",
                     fontsize=12.5, color=GREEN)
        else:                                   # termination
            axa.add_patch(Rectangle((x - 0.11, y - 0.11), 0.22, 0.22,
                                    fc="black", ec="black", lw=1.0, zorder=5))
            axa.text(x + 0.19, y, "end", ha="left", va="center", fontsize=12.5,
                     color="black")

    fl.arrow(axa, (2.55, 3.78), (4.15, 3.78), color="black", lw=1.3, mut=13)
    fl.arrow(axa, (4.15, 3.78), (2.55, 3.78), color="black", lw=1.3, mut=13)
    axa.text(2.42, 3.78, "more bootstrapping", ha="right", va="center",
             fontsize=12, color="black")
    axa.text(4.28, 3.78, "more sampling", ha="left", va="center", fontsize=12,
             color="black")
    axa.set_xlim(-0.30, 6.75)
    axa.set_ylim(-0.32, 4.05)
    axa.set_aspect("equal")
    axa.axis("off")

    # ===== the lambda-return weight strip ================================== #
    lam, NW = 0.9, 10
    n = np.arange(1, NW + 1)
    wts = (1 - lam) * lam ** (n - 1)
    tail = lam ** NW
    assert abs(wts.sum() + tail - 1.0) < 1e-12
    axl.bar(n, wts, width=0.62, color=ORANGE, ec=ORANGE, lw=0.0, zorder=3)
    axl.bar([NW + 1.9], [tail], width=1.7, color=ORANGE, alpha=0.45,
            ec=ORANGE, lw=1.4, zorder=3)
    axl.axhline(0.0, color="black", lw=1.1, zorder=4)
    axl.text(1.0, wts[0] + 0.022, "$1 - \\lambda$", ha="center", va="bottom",
             fontsize=12, color="black")
    axl.text(NW + 1.9, tail + 0.022, "tail", ha="center", va="bottom",
             fontsize=12, color="black")
    axl.text(2.55, 0.29, r"$\lambda$-return: weight "
                         r"$(1-\lambda)\lambda^{\,n-1}$" "\n"
                         f"on the $n$-step target,  $\\lambda = {lam:g}$",
             ha="left", va="center", fontsize=12, color="black",
             linespacing=1.45)
    axl.set_xlim(0.2, 13.4)
    axl.set_ylim(0.0, 0.44)
    axl.set_xticks(list(n) + [NW + 1.9])
    axl.set_xticklabels([str(i) for i in n] + ["$>10$"])
    axl.set_yticks([])
    axl.set_xlabel("target depth $n$", fontsize=12.5)
    for s in ("left", "right", "top"):
        axl.spines[s].set_visible(False)
    axl.spines["bottom"].set_visible(False)
    axl.tick_params(axis="x", colors="black", length=0, labelsize=10.5)

    # ===== (b) the same family, measured =================================== #
    NS, GAM, SIG, AMP = 10, 0.97, 0.15, 0.5
    V = GAM ** (NS - 1 - np.arange(NS))                  # exact values
    e = AMP * (1.0 - np.arange(NS) / (NS - 1.0)) ** 2     # wrong far from the end
    Vhat = V + e                                         # the imperfect critic

    ns = np.arange(1, NS + 1)
    rng = np.random.default_rng(7)
    NR = 20000
    noise = rng.normal(0.0, SIG, (NR, NS))
    pay = np.zeros(NS)
    pay[NS - 1] = 1.0                                    # only the last pays
    disc = GAM ** np.arange(NS)
    partial = np.cumsum((pay + noise) * disc, axis=1)     # sum_{l<n} gamma^l r_l
    boot = np.where(ns < NS, GAM ** ns * Vhat[np.minimum(ns, NS - 1)], 0.0)
    G = partial + boot                                   # targets from state 0
    bias2 = (G.mean(axis=0) - V[0]) ** 2
    var = G.var(axis=0)
    mse = bias2 + var

    b2_th = (np.where(ns < NS, GAM ** ns * e[np.minimum(ns, NS - 1)], 0.0)) ** 2
    var_th = SIG ** 2 * (1 - GAM ** (2 * ns)) / (1 - GAM ** 2)
    assert np.abs(bias2 - b2_th).max() < 0.01            # the rollouts agree
    assert np.abs(var - var_th).max() < 0.01
    assert (np.diff(bias2) <= 1e-9).all() and (np.diff(var) > 0).all()
    nbest = int(np.argmin(mse)) + 1
    assert 1 < nbest < NS                               # the optimum is interior

    axb.plot(ns, bias2, "o-", color=ORANGE, lw=2.0, ms=5, zorder=3,
             label="bias$^2$")
    axb.plot(ns, var, "o-", color=BLUE, lw=2.0, ms=5, zorder=3,
             label="variance")
    axb.plot(ns, mse, "o-", color="black", lw=2.6, ms=5.5, zorder=4,
             label="mean squared error")
    axb.plot([nbest], [mse[nbest - 1]], "o", color="black", ms=10, zorder=5)
    axb.text(6.9, 0.036, "the interior optimum reflects\nthis critic's error taper",
             ha="center", va="bottom", fontsize=12, color="black",
             linespacing=1.4, zorder=6)
    axb.text(nbest, mse[nbest - 1] + 0.011, f"best depth\n$n = {nbest}$",
             ha="center", va="bottom", fontsize=12, color="black",
             linespacing=1.4, zorder=6,
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.6))
    axb.set_xticks(ns)
    axb.set_xlim(0.5, 10.5)
    axb.set_ylim(0.0, max(mse.max(), bias2.max()) * 1.28)
    axb.set_xlabel("target depth $n$", fontsize=13)
    axb.set_ylabel("error of the depth-$n$ target", fontsize=13)
    axb.annotate("TD(0)", xy=(1, 0), xytext=(1, -0.135), ha="left",
                 va="center", fontsize=11, color="black",
                 textcoords=("data", "axes fraction"))
    axb.annotate("Monte Carlo", xy=(10, 0), xytext=(10, -0.135), ha="right",
                 va="center", fontsize=11, color="black",
                 textcoords=("data", "axes fraction"))
    axb.legend(loc="upper center", fontsize=11.5)
    for s in ("left", "bottom"):
        axb.spines[s].set_color("black")
    axb.tick_params(colors="black", labelsize=11)

    fl.save(fig, "mdl-rl-td-mc-spectrum")

def fig_policy_vs_parameter():   # F13 -> mdl-rl-policy-vs-parameter
    """Two parameter steps of exactly the same size: (a) the map from $\\theta$
    to the probability it produces, so the reader sees *why* the two steps do
    different things; (b) the four action distributions, which is all an
    observer of the robot could measure.  Every number comes from $\\sigma$."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.1),
                                   gridspec_kw={"width_ratios": [1.42, 1.0]})
    DT = 2.0                                     # the one step size, both times
    STARTS = [(0.0, GREEN), (6.0, ORANGE)]
    # sigma' = sigma (1 - sigma): the slope is the whole story.
    dsig = _sigma(np.array([s for s, _ in STARTS]))
    dsig = dsig * (1.0 - dsig)
    assert abs(dsig[0] - 0.25) < 1e-12 and abs(dsig[1] - 0.0025) < 1e-4

    def dpi_txt(d):              # two digits where there is something to see
        return rf"$\Delta\pi = {d:.2f}$" if d > 0.05 else \
               rf"$\Delta\pi = {d:.3f}$"

    # --- (a) the sigmoid, with the two equal steps on it -------------------- #
    XL, YT = (-6.7, 11.4), 1.17
    RAIL = -1.30                                 # where the two Dpi brackets sit
    th = np.linspace(XL[0], XL[1], 800)
    axa.plot(th, _sigma(th), color=BLUE, lw=2.4, zorder=3)

    for t0, color in STARTS:
        p0, p1 = _sigma(t0), _sigma(t0 + DT)
        # the step itself: identical black double arrow, identical label
        axa.annotate("", xy=(t0 + DT, 0.06), xytext=(t0, 0.06),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.3,
                                     shrinkA=0, shrinkB=0, mutation_scale=11))
        axa.text(t0 + DT / 2, 0.115, rf"$\Delta\theta = {DT:g}$", ha="center",
                 va="bottom", fontsize=12, color="black")
        for t, p in ((t0, p0), (t0 + DT, p1)):
            axa.plot([t, t], [0.06, p], ls=":", lw=1.2, color=LIGHT, zorder=1)
            axa.plot([t], [p], "o", color=color, ms=7, zorder=5)
            axa.plot([RAIL, t], [p, p], ls=":", lw=1.1, color=LIGHT, zorder=1)
        # the induced change, bracketed in the left margin
        if p1 - p0 > 0.05:
            axa.annotate("", xy=(RAIL, p1), xytext=(RAIL, p0),
                         arrowprops=dict(arrowstyle="<->", color=color, lw=1.7,
                                         shrinkA=0, shrinkB=0,
                                         mutation_scale=10))
        else:                                    # too small to draw: two caps
            for p in (p0, p1):
                axa.plot([RAIL - 0.22, RAIL + 0.22], [p, p], color=color,
                         lw=1.7, zorder=4)
        axa.text(RAIL - 0.55, (p0 + p1) / 2, dpi_txt(float(p1 - p0)),
                 ha="right", va="center", fontsize=13, color=color)

    axa.text(XL[1] - 0.1, 1.055,
             rf"$\sigma'(0) = {dsig[0]:.2f}$,    "
             rf"$\sigma'(6) \approx {dsig[1]:.4f}$",
             ha="right", va="bottom", fontsize=12.5, color="black")
    axa.set_xlim(*XL)
    axa.set_ylim(0.0, YT)
    axa.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axa.set_xticks([-6, -3, 0, 3, 6, 9])
    axa.set_xlabel(r"$\theta$", fontsize=14)
    axa.set_ylabel(r"$\pi_\theta(a_1) = \sigma(\theta)$", fontsize=13)
    _black_axes(axa)

    # --- (b) the four distributions an observer could measure --------------- #
    W, GAP = 0.36, 0.19
    CENT = {0.0: (0.0, 1.0), 6.0: (2.5, 3.5)}    # the two x groups
    for t0, color in STARTS:
        p0, p1 = float(_sigma(t0)), float(_sigma(t0 + DT))
        before, after = [p0, 1.0 - p0], [p1, 1.0 - p1]   # a_1 and a_2
        for c, b, a in zip(CENT[t0], before, after):
            axb.bar(c - GAP, b, W, color=LIGHT, ec="black", lw=0.9, zorder=3)
            axb.bar(c + GAP, a, W, color=color, ec="black", lw=0.9, zorder=3)
    for t0, color, x, y in ((0.0, GREEN, 1.06, 0.68), (6.0, ORANGE, 3.56, 0.62)):
        d = float(_sigma(t0 + DT) - _sigma(t0))
        axb.text(x, y, dpi_txt(d), ha="center", va="center", fontsize=13,
                 color=color)
    for t0, x in ((0.0, 0.5), (6.0, 3.0)):
        axb.text(x, -0.155, rf"$\theta: {t0:g} \to {t0 + DT:g}$", ha="center",
                 va="center", fontsize=13, color="black")
    hb = Patch(fc=LIGHT, ec="black", lw=0.9)
    ha = (Patch(fc=GREEN, ec="black", lw=0.9),
          Patch(fc=ORANGE, ec="black", lw=0.9))
    axb.legend([hb, ha], ["before", "after"],
               handler_map={tuple: HandlerTuple(ndivide=None)},
               loc="upper center", bbox_to_anchor=(0.46, 1.02), ncol=2,
               fontsize=11.5, handlelength=1.4, columnspacing=1.4)
    axb.set_xlim(-0.62, 4.12)
    axb.set_ylim(0.0, 1.20)
    axb.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xticks([0.0, 1.0, 2.5, 3.5])
    axb.set_xticklabels([r"$a_1$", r"$a_2$", r"$a_1$", r"$a_2$"], fontsize=13)
    axb.set_ylabel(r"$\pi_\theta(a)$", fontsize=14)
    _black_axes(axb)
    fig.subplots_adjust(wspace=0.24)
    fl.save(fig, "mdl-rl-policy-vs-parameter")

def fig_trust_region():          # F14 -> mdl-rl-trust-region
    """(a) a surrogate that matches value and slope at $\\theta_{old}$ and then
    keeps promising, maximized inside and outside a trust region; (b) the same
    statement in two parameters, with a real softmax Fisher matrix: equal steps
    in $\\theta$ are wildly unequal steps in policy space."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.0, 4.3),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- (a) the local model ----------------------------------------------- #
    def J(t):
        t = np.asarray(t, float)
        return (1.6 * np.exp(-(t - 0.55) ** 2 / 0.45)
                - 0.9 * np.exp(-(t - 2.3) ** 2 / 0.35))

    def dJ(t):
        t = np.asarray(t, float)
        return (1.6 * np.exp(-(t - 0.55) ** 2 / 0.45) * (-2 * (t - 0.55) / 0.45)
                - 0.9 * np.exp(-(t - 2.3) ** 2 / 0.35) * (-2 * (t - 2.3) / 0.35))

    T_OLD, DELTA = 0.0, 0.7
    J0, S0 = float(J(T_OLD)), float(dJ(T_OLD))

    # The surrogate: matches J in value and slope at theta_old, with a
    # deliberately optimistic curvature, so it keeps rising through the stretch
    # where J has already turned over.  (program-figures.md writes +0.25*t**2,
    # which is convex and has no maximizer at all; a concave-but-optimistic
    # model gives the unconstrained step a well-defined place to land, which is
    # what the panel is about.)
    CURV = -0.35

    def L(t):
        t = np.asarray(t, float)
        return J0 + S0 * t + CURV * t ** 2

    XL, YL = (-1.08, 3.18), (-1.50, 4.28)
    t = np.linspace(XL[0], XL[1], 700)
    inside = t[np.abs(t - T_OLD) <= DELTA]
    T_IN = float(inside[np.argmax(L(inside))])            # argmax L in region
    T_OUT = float(t[np.argmax(L(t))])                     # argmax L, no region
    assert abs(L(T_OLD) - J(T_OLD)) < 1e-12               # value matches
    assert abs((L(1e-6) - L(-1e-6)) / 2e-6 - S0) < 1e-4   # slope matches
    assert J(T_IN) > J0 > J(T_OUT)                        # the whole point

    axa.axvspan(T_OLD - DELTA, T_OLD + DELTA, color=LIGHT, alpha=0.40, zorder=0)
    axa.text(T_OLD, 4.10, "trust region\n"
             r"$|\theta - \theta_{\mathrm{old}}| \leq \delta$", ha="center",
             va="top", fontsize=12.5, color="black", linespacing=1.4)
    axa.plot(t, J(t), color="black", lw=2.4, zorder=3)
    axa.plot(t, L(t), color=BLUE, lw=2.2, ls="--", zorder=3)
    axa.text(2.28, -1.24, r"$J(\theta)$", ha="center", va="center",
             fontsize=14, color="black")
    axa.text(XL[1] - 0.06, 3.90, r"$L(\theta)$: the surrogate", ha="right",
             va="bottom", fontsize=13, color=BLUE)

    for tt, color in ((T_IN, GREEN), (T_OUT, ORANGE)):
        axa.plot([tt, tt], [float(J(tt)), float(L(tt))], ls=":", lw=1.4,
                 color=color, zorder=2)
        axa.plot([tt], [float(L(tt))], "o", color=color, ms=7.5, zorder=5)
        axa.plot([tt], [float(J(tt))], "o", color=color, ms=7.5, zorder=5)
    # The two sentences live in the wedge between the curves, right-aligned in
    # one column that clears J underneath, L above and the orange drop line on
    # the right, and each carries the colour of the markers it explains.
    axa.text(T_OUT - 0.14, 2.35, "inside the region:\n" r"$J$ improves," "\n"
             rf"${J0:.2f} \to {float(J(T_IN)):.2f}$", ha="right", va="center",
             fontsize=12, color=GREEN, linespacing=1.45)
    axa.text(T_OUT - 0.14, 0.85, "outside it:\n" r"$J$ collapses," "\n"
             rf"${J0:.2f} \to {float(J(T_OUT)):.2f}$",
             ha="right", va="center", fontsize=12, color=ORANGE,
             linespacing=1.45)
    axa.plot([T_OLD], [J0], "o", color="black", ms=7.5, zorder=6)
    axa.plot([T_OLD, T_OLD], [YL[0], J0], ls=":", lw=1.1, color=LIGHT, zorder=1)
    axa.set_xlim(*XL)
    axa.set_ylim(*YL)
    axa.set_xticks([T_OLD - DELTA, T_OLD, T_OLD + DELTA])
    axa.set_xticklabels([r"$-\delta$", r"$\theta_{\mathrm{old}}$",
                         r"$+\delta$"], fontsize=13)
    axa.set_yticks([-1, 0, 1, 2, 3, 4])
    axa.set_xlabel(r"$\theta$", fontsize=14)
    axa.set_ylabel("objective", fontsize=13)
    _black_axes(axa)

    # --- (b) the KL ellipse of a real Fisher matrix ------------------------- #
    p = _softmax([2.5, 0.0, 0.0])                  # three actions, two free
    F = np.diag(p[:2]) - np.outer(p[:2], p[:2])    # the Fisher metric there
    DEL = 0.02
    lam, V = np.linalg.eigh(F)                     # ascending eigenvalues
    semi = np.sqrt(2 * DEL / lam)                  # KL ellipse semi-axes
    u_long, u_short = V[:, 0], V[:, 1]             # long axis <-> small lambda
    u_long = u_long * np.sign(u_long[1])           # fix signs: up-right,
    u_short = u_short * np.sign(u_short[0])        # and down-right
    ang = np.linspace(0.0, 2 * np.pi, 401)
    ell = (V * semi) @ np.vstack([np.cos(ang), np.sin(ang)])
    kl = 0.5 * np.einsum("ij,jk,ik->i", ell.T, F, ell.T)
    assert np.allclose(kl, DEL)                    # it really is a KL level set
    RHO = float(np.sqrt(semi[0] * semi[1]))        # equal area: pi ab = pi rho^2

    axb.plot(ell[0], ell[1], color=ORANGE, lw=2.2, zorder=4)
    axb.add_patch(Circle((0, 0), RHO, fc="none", ec=GRAY, lw=1.6,
                         ls=(0, (5, 3)), zorder=3))
    fl.axis_cross(axb, (-1.16, 1.16), (-1.16, 1.16), color="black", lw=1.0)
    axb.plot([0], [0], "o", color="black", ms=7, zorder=6)
    axb.text(-0.07, -0.12, r"$\theta_{\mathrm{old}}$", ha="right", va="top",
             fontsize=13, color="black")

    # The two equal steps, and their annotations placed radially past the tip of
    # the ellipse axis each one follows -- for an ellipse's own axes the radial
    # direction *is* the outward normal, and it is the only direction in which
    # the label cannot fall back onto the curve.
    for u, semi_i, name, va in ((u_long, semi[0], "barely moves", "bottom"),
                                (u_short, semi[1], "is rewritten", "top")):
        tip = RHO * u
        fl.arrow(axb, (0.0, 0.0), tuple(tip), color=BLUE, lw=2.2, mut=15)
        d_kl = 0.5 * float(tip @ F @ tip)
        far = u * (max(RHO, float(semi_i)) + (0.22 if va == "bottom" else 0.27))
        axb.text(far[0], far[1],
                 rf"same $\|\Delta\theta\|$: $D_{{\mathrm{{KL}}}} = {d_kl:.3f}$"
                 "\n" f"the policy {name}", ha="left", va=va, fontsize=12,
                 color="black", linespacing=1.5)
    axb.text(-0.62, 0.95, rf"$D_{{\mathrm{{KL}}}} \leq \delta = {DEL:g}$",
             ha="right", va="center", fontsize=12.5, color=ORANGE)
    axb.text(-0.85, 0.40, "equal-area ball\n" r"$\|\Delta\theta\| \leq \rho$",
             ha="right", va="center", fontsize=12.5, color=GRAY,
             linespacing=1.5)
    axb.text(1.22, -0.10, r"$\Delta\theta_1$", ha="left", va="top",
             fontsize=13, color="black")
    axb.text(0.0, 1.22, r"$\Delta\theta_2$", ha="center", va="bottom",
             fontsize=13, color="black")
    axb.set_aspect("equal")
    axb.set_xlim(-1.78, 1.92)
    axb.set_ylim(-1.22, 1.68)
    axb.axis("off")
    fig.subplots_adjust(wspace=0.20)
    fl.save(fig, "mdl-rl-trust-region")

def fig_ppo_clip():              # F15 -> mdl-rl-ppo-clip
    """One sample's contribution to the clipped objective against the ratio,
    for a good action and a bad one.  Identical $y$ limits, so the asymmetry
    between the two panels is the figure's whole content."""
    EPS = 0.2
    rho = np.linspace(0.0, 2.0, 801)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)

    for ax, adv, title in ((axa, +1.0, r"$\hat A_t > 0$"),
                           (axb, -1.0, r"$\hat A_t < 0$")):
        unc = rho * adv
        clp = np.clip(rho, 1 - EPS, 1 + EPS) * adv
        obj = np.minimum(unc, clp)
        ax.axvspan(1 - EPS, 1 + EPS, color=LIGHT, alpha=0.30, zorder=0)
        ax.plot(rho, unc, ls="--", color=LIGHT, lw=1.8, zorder=2,
                label=r"$\rho\,\hat A$ (unclipped)")
        ax.plot(rho, clp, ls=":", color=GRAY, lw=1.8, zorder=2,
                label=r"$\mathrm{clip}(\rho)\,\hat A$")
        ax.plot(rho, obj, color=BLUE, lw=3.0, zorder=4,
                label=r"$L^{\mathrm{CLIP}}$: the minimum")
        ax.plot([1.0], [adv], "o", color="black", ms=7, zorder=5)
        ax.set_xlim(0.0, 2.02)
        ax.set_ylim(-2.05, 1.45)
        ax.set_xticks([1 - EPS, 1.0, 1 + EPS])
        ax.set_xticklabels([r"$1-\epsilon$", r"$1$", r"$1+\epsilon$"],
                           fontsize=12.5)
        ax.set_yticks([-2, -1, 0, 1])
        ax.set_title(title, fontsize=13.5, color="black")
        ax.set_xlabel(r"ratio $\rho_t(\theta)$", fontsize=13)
        _black_axes(ax)
    axa.set_ylabel("clipped objective, one sample", fontsize=12.5)

    # (a) the flat stretch beyond 1 + eps, and where rho = 1 is
    axa.annotate("gradient $= 0$: no payoff\n" r"past $1+\epsilon$",
                 xy=(1.62, 1.185), xytext=(2.00, 0.42), ha="right",
                 va="center", fontsize=12, color=ORANGE, linespacing=1.45,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                                 shrinkA=6, shrinkB=3, mutation_scale=13))
    axa.annotate(r"$\rho = 1$: $\theta = \theta_{\mathrm{old}}$",
                 xy=(1.0, 1.0), xytext=(0.66, -0.34), ha="center", va="center",
                 fontsize=12, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=5, shrinkB=5, mutation_scale=12))
    axa.legend(loc="lower left", fontsize=11, handlelength=1.8)

    # (b) the flat stretch below 1 - eps, and the fall that never stops
    axb.annotate("gradient $= 0$:\nnothing more to gain",
                 xy=(0.24, -0.78), xytext=(0.06, 0.38), ha="left",
                 va="center", fontsize=12, color=ORANGE, linespacing=1.45,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                                 shrinkA=6, shrinkB=3, mutation_scale=13))
    axb.annotate("still free to fall:\n" r"the pessimistic $\min$",
                 xy=(1.78, -1.76), xytext=(1.99, -0.42), ha="right",
                 va="center", fontsize=12, color=ORANGE, linespacing=1.45,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                                 shrinkA=6, shrinkB=3, mutation_scale=13))
    axb.annotate(r"$\rho = 1$", xy=(1.0, -1.0), xytext=(0.52, -1.48),
                 ha="center", va="center", fontsize=12, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=5, shrinkB=5, mutation_scale=12))
    fig.subplots_adjust(wspace=0.10)
    fl.save(fig, "mdl-rl-ppo-clip")

def fig_kl_tilting():            # F24 -> mdl-rl-kl-tilting
    """The closed form of the KL-regularized optimum, drawn: the reference, the
    exponential tilt it is multiplied by, and the product.  One row per
    $\\beta$, identical $y$ limits everywhere, so the two rows can be compared
    by eye and the two limits read off the ends."""
    # The reference is deliberately not monotone in the reward: the tilt then
    # moves mass toward the better actions while $a_2$ stays the reference's own
    # unlikely action in $\pi^\star$ too, which is what "a product" looks like.
    P_REF = np.array([0.30, 0.10, 0.25, 0.20, 0.15])
    R = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    BETAS = [2.0, 0.2]
    assert abs(P_REF.sum() - 1.0) < 1e-12

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.3), sharex=True,
                             sharey=True)
    for row, beta in zip(axes, BETAS):
        tilt = np.exp(R / beta)
        star = P_REF * tilt
        star = star / star.sum()
        # the closed form is the normalized product, and its value is the
        # log-partition function -- checked, not asserted in prose
        assert abs(star.sum() - 1.0) < 1e-12
        _prob_bars(row[0], P_REF, GRAY)
        _prob_bars(row[1], tilt / tilt.sum(), ORANGE)
        _prob_bars(row[2], star, BLUE, values=True)
        # how far the tilt actually moved the policy: the quantity the section
        # sweeps as a frontier, so the two rows are two points on it
        kl = float(np.sum(star * np.log(star / P_REF)))
        row[2].text(-0.45, 1.02, r"$D_{\mathrm{KL}}(\pi^\star \Vert "
                    rf"\pi_{{\mathrm{{ref}}}}) = {kl:.2f}$", ha="left",
                    va="top", fontsize=12, color="black")

    for ax, title in zip(axes[0], [r"$\pi_{\mathrm{ref}}$",
                                   r"the tilt: $e^{r/\beta}$, normalized",
                                   r"$\pi^\star \propto \pi_{\mathrm{ref}}\,"
                                   r"e^{r/\beta}$"]):
        ax.set_title(title, fontsize=13.5, color="black", pad=8)
    axes[0][1].text(-0.42, 1.05, r"$r = (0,\ 0.5,\ 1,\ 2,\ 3)$", ha="left",
                    va="top", fontsize=12, color="black")
    for ax in axes.ravel():
        ax.set_xlim(-0.62, 4.62)
        ax.set_ylim(0.0, 1.14)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([rf"$a_{i}$" for i in range(1, 6)], fontsize=12.5)
        _black_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("probability", fontsize=12.5)

    fig.subplots_adjust(left=0.105, right=0.845, top=0.925, bottom=0.085,
                        wspace=0.10, hspace=0.20)
    for row, beta, note in zip(axes, BETAS,
                               [r"$\beta \to \infty$:" "\n"
                                r"$\pi^\star \to \pi_{\mathrm{ref}}$," "\n"
                                "the reference",
                                r"$\beta \to 0$:" "\n"
                                r"$\pi^\star \to$ a point mass" "\n"
                                "on the best action"]):
        bb = row[0].get_position()
        y = bb.y0 + bb.height / 2
        fig.text(0.020, y, rf"$\beta = {beta:g}$", ha="left", va="center",
                 fontsize=15, color="black", rotation=90)
        fig.text(0.862, y, note, ha="left", va="center", fontsize=12.5,
                 color="black", linespacing=1.5)
    fl.save(fig, "mdl-rl-kl-tilting")

def fig_dqn_dataflow():       # F16 -> mdl-rl-dqn-dataflow
    """Deep Q-Networks as a data flow, in the box vocabulary of F1/F11/F19.
    The two inventions are captioned inside the picture, and the gradient is
    drawn as the one solid path that returns to the online network."""
    fig, ax = plt.subplots(figsize=(10.1, 7.1))
    ax.axis("off")
    ax.set_aspect("equal")

    # --- acting: the behaviour policy and the world ------------------------- #
    _box(ax, 2.55, 4.35, 3.5, 1.05,
         "behaviour policy\n$\\epsilon$-greedy in $Q_w$", BLUE,
         fontsize=13.5, fc="#e8f1f8")
    _box(ax, 7.7, 4.35, 2.8, 1.05, "environment", GRAY, fontsize=13.5,
         fc="#f3f3f3")
    fl.arrow(ax, (4.35, 4.5), (6.25, 4.5), color=BLUE, lw=2.0, mut=15)
    ax.text(5.3, 4.62, "$a_t$", ha="center", va="bottom", fontsize=13.5,
            color=BLUE)
    fl.arrow(ax, (7.7, 3.79), (7.7, 3.32), color=GRAY, lw=1.8, mut=15)
    ax.text(7.86, 3.55, "$(s,a,r,s')$", ha="left", va="center", fontsize=12.5,
            color="black")

    # --- the first fix: the buffer ----------------------------------------- #
    ax.add_patch(FancyBboxPatch((2.3, 2.50), 6.8, 0.75,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec="black", lw=1.6, zorder=2))
    _tiles(ax, np.linspace(2.62, 8.78, 14), 2.875, 0.32, 0.5)
    ax.text(4.55, 3.36, "replay buffer $\\mathcal{D}$ (200k transitions)",
            ha="center", va="bottom", fontsize=13, color="black")
    fl.arrow(ax, (2.45, 3.05), (1.90, 3.42), color=ORANGE, lw=1.5, ls="--",
             mut=13)
    ax.text(1.82, 3.47, "evicted", ha="right", va="center", fontsize=12,
            color=ORANGE)
    ax.text(3.00, 2.30, "fix 1: uncorrelated, reusable data", ha="center",
            va="top", fontsize=12.5, color="black")

    # --- the minibatch ----------------------------------------------------- #
    fl.arrow(ax, (5.7, 2.48), (5.7, 1.98), color=BLUE, lw=2.0, mut=15)
    ax.text(5.92, 2.24, "scrambles time", ha="left", va="center", fontsize=12.5,
            color="black")
    _box(ax, 5.7, 1.5, 3.9, 0.85, "minibatch of 128,\nsampled uniformly", BLUE,
         fontsize=13, fc="#e8f1f8")

    # --- the two networks -------------------------------------------------- #
    _box(ax, 2.2, 0.15, 3.1, 1.05, "online $Q_w$\n$Q_w(s,a)$", BLUE,
         fontsize=13, fc="#e8f1f8")
    _box(ax, 8.05, 0.15, 3.9, 1.05,
         "target $Q_{w^-}$ (frozen)\n"
         "$y = r + \\gamma \\max_{a'} Q_{w^-}(s',a')$", ORANGE,
         fontsize=12, fc="#fdf1e3")
    fl.arrow(ax, (4.20, 1.12), (2.95, 0.72), color=BLUE, lw=1.8, mut=14)
    fl.arrow(ax, (7.20, 1.12), (8.05, 0.72), color=BLUE, lw=1.8, mut=14)

    # the second fix, and the only edge that ever writes into the frozen copy
    fl.arrow(ax, (3.85, 0.02), (6.05, 0.02), color=ORANGE, lw=1.6, ls="--",
             mut=14)
    ax.text(4.92, 0.15, "copy weights\nevery $C$ steps", ha="center",
            va="bottom", fontsize=12, color="black")
    ax.text(8.35, -1.10, "fix 2: targets stand still\nbetween syncs",
            ha="center", va="top", fontsize=12.5, color="black")

    # --- the loss, and the one path a gradient may take -------------------- #
    _box(ax, 5.0, -1.45, 3.4, 0.8,
         "Huber loss  $\\ell(Q_w(s,a) - y)$", "black",
         fontsize=12.5, lw=1.6)
    fl.arrow(ax, (2.80, -0.40), (3.90, -1.12), color=BLUE, lw=1.8, mut=14)
    fl.arrow(ax, (7.10, -0.40), (6.32, -1.10), color=ORANGE, lw=1.8, mut=14)
    ax.plot([6.60, 6.84], [-0.82, -0.60], color="black", lw=1.6, zorder=6)
    ax.plot([6.60, 6.84], [-0.60, -0.82], color="black", lw=1.6, zorder=6)
    ax.text(7.00, -0.71, "no gradient", ha="left", va="center", fontsize=11.5,
            color="black")
    _arc_arrow(ax, (3.30, -1.45), (2.20, -0.42), -0.32, "black", lw=2.0)
    ax.text(2.02, -1.28, "$\\nabla_w$", ha="right", va="center",
            fontsize=13.5, color="black")

    # --- and the loop closes: the online net is the behaviour policy -------- #
    _elbow(ax, [(0.65, 0.15), (0.32, 0.15), (0.32, 4.35), (0.80, 4.35)], BLUE)
    ax.text(0.18, 2.40, "acts with $Q_w$", ha="center", va="center",
            fontsize=12, color=BLUE, rotation=90)

    ax.set_xlim(-0.05, 10.15)
    ax.set_ylim(-2.05, 5.05)
    fl.save(fig, "mdl-rl-dqn-dataflow")

def fig_deadly_triad():       # F17 -> mdl-rl-deadly-triad
    """The deadly triad with every region named by an algorithm the book has
    already taught.  Label anchors are computed by sampling the three discs,
    not eyeballed; the labels carry white boxes so they never sit on an arc."""
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    ax.axis("off")
    ax.set_aspect("equal")

    names = [("function\napproximation", BLUE), ("bootstrapping", ORANGE),
             ("off-policy data", GREEN)]
    for (cx, cy), (name, color) in zip(TRIAD_C, names):
        ax.add_patch(Circle((cx, cy), TRIAD_R, fc=color, alpha=0.14,
                            ec="none", zorder=1))
        ax.add_patch(Circle((cx, cy), TRIAD_R, fc="none", ec=color, lw=2.0,
                            zorder=3))
    centre = TRIAD_C.mean(axis=0)
    for (cx, cy), (name, color) in zip(TRIAD_C, names):
        u = np.array([cx, cy]) - centre
        u = u / np.linalg.norm(u)
        p = np.array([cx, cy]) + u * (TRIAD_R + 0.26)
        ax.text(p[0], p[1], name, ha="center", va="center", fontsize=13.5,
                color=color, zorder=6)

    # region key -> (label, extra offset from the computed anchor)
    regions = {
        "100": ("REINFORCE\nwith a network", (0.0, 0.00)),
        "010": ("tabular\nSARSA", (0.0, 0.00)),
        "001": ("importance-sampled\nMonte Carlo", (0.0, 0.00)),
        "110": ("actor-critic,\nlinear TD", (0.0, 0.00)),
        "101": ("behaviour\ncloning", (0.0, 0.00)),
        "011": ("tabular\nQ-learning", (0.0, 0.00)),
    }
    anchors = _triad_anchors()
    for key, (text, off) in regions.items():
        p = anchors[key][0] + np.asarray(off, float)
        ax.text(p[0], p[1], text, ha="center", va="center", fontsize=12,
                color="black", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.6))
    p = anchors["111"][0]
    ax.text(p[0], p[1], "DQN\noffline\nQ-learning", ha="center", va="center",
            fontsize=12, fontweight="bold", color="black", zorder=7,
            bbox=dict(fc="white", ec="black", lw=1.2, alpha=0.92, pad=2.2))

    ax.text(0.0, -3.55, "drop any one of the three and the instability goes "
                        "away;\nDQN keeps all three and repairs the couplings "
                        "instead",
            ha="center", va="top", fontsize=12.5, color="black")

    ax.set_xlim(-4.90, 4.90)
    ax.set_ylim(-4.40, 4.32)
    fl.save(fig, "mdl-rl-deadly-triad")

def fig_max_bias():           # F18 -> mdl-rl-max-bias
    """A maximum over noisy estimates is biased upward: (a) the distribution of
    the largest of four estimates of the same zero value, (b) how the bias
    grows with the number of actions, and how the double estimator removes it.
    Both panels are seeded Monte Carlo, 200000 draws per point."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.12, 1.0]})
    rng = np.random.default_rng(3)
    N = 200_000

    # --- (a) four estimates of the same zero value ------------------------- #
    draws = rng.standard_normal((N, 4))
    mx = draws.max(axis=1)
    bias4 = mx.mean()
    x = np.linspace(-3.6, 4.4, 601)
    axa.fill_between(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=LIGHT,
                     alpha=0.55, lw=0.0, zorder=1)
    axa.hist(mx, bins=120, density=True, color=ORANGE, alpha=0.6, zorder=2)
    axa.axvline(0.0, color="black", lw=1.4, ls="--", zorder=4)
    axa.axvline(bias4, color=ORANGE, lw=2.0, zorder=4)
    axa.annotate("", xy=(bias4, 0.09), xytext=(0.0, 0.09),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.4,
                                 shrinkA=0, shrinkB=0))
    axa.text(bias4 / 2, 0.105, f"bias ${bias4:.2f}$", ha="center", va="bottom",
             fontsize=12.5, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.4))
    # the two distributions are named where they are, not in a legend box
    axa.text(-3.45, 0.30, "one estimate", ha="left", va="center", fontsize=12.5,
             color=GRAY)
    axa.text(2.55, 0.32, "$\\max$ of four", ha="left", va="center",
             fontsize=12.5, color=ORANGE)
    axa.text(-0.10, 0.610, "$\\max_a \\mathbb{E}[\\hat Q] = 0$", ha="right",
             va="center", fontsize=12.5, color="black")
    axa.text(bias4 + 0.10, 0.610,
             f"$\\mathbb{{E}}[\\max_a \\hat Q] = {bias4:.2f}$", ha="left",
             va="center", fontsize=12.5, color=ORANGE)
    axa.set_xlim(-3.6, 4.4)
    axa.set_ylim(0.0, 0.665)
    axa.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axa.set_xlabel("estimated value", fontsize=13)
    axa.set_ylabel("density", fontsize=13)
    _black_axes(axa)

    # --- (b) the bias against the number of actions ------------------------ #
    ks = np.array([2, 3, 4, 6, 8, 12, 16])
    single, double = np.array([_max_bias(int(k), N, rng) for k in ks]).T
    axb.axhline(0.0, color="black", lw=1.2, ls="--", zorder=1)
    axb.plot(ks, single, "o-", color=BLUE, lw=2.0, ms=6, zorder=3,
             label="single estimator (the DQN target)")
    axb.plot(ks, double, "s-", color=ORANGE, lw=2.0, ms=6, zorder=3,
             label="double estimator (Double DQN)")
    _leader(axb, "the case on the left", (4, single[2]), (6.4, 0.60), "black",
            fontsize=11.5, ha="left")
    axb.text(16.6, 0.075, f"double estimator: bias below "
                          f"${abs(double).max():.3f}$ throughout", ha="right",
             va="bottom", fontsize=11.5, color=ORANGE)
    axb.set_xlim(1.2, 16.8)
    axb.set_ylim(-0.14, 2.06)
    axb.set_xticks(ks)
    axb.set_xlabel("number of actions", fontsize=13)
    axb.set_ylabel("bias of the maximum", fontsize=13)
    axb.legend(loc="upper left", fontsize=11.5, handlelength=1.6)
    _black_axes(axb)

    fig.subplots_adjust(wspace=0.24)
    fl.save(fig, "mdl-rl-max-bias")
    return bias4, ks, single, double

def fig_data_rules():         # F19 -> mdl-rl-data-rules
    """Three data regimes in one vocabulary: a policy, a world, a data object,
    an update, and the dashed arrow that carries the update back to the policy.
    The panels share their geometry exactly, so what differs between them is
    what the section is about."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9))
    XL, YL = (-0.35, 6.85), (0.55, 7.15)
    PBOX, EBOX = (1.75, 5.50), (5.35, 5.50)      # policy, environment
    DATA = (4.50, 3.95)                          # the data object
    UBOX = (2.50, 1.85)                          # the update
    titles = ("on-policy: REINFORCE, actor-critic, PPO",
              "off-policy: Q-learning, DQN",
              "offline: no interaction at all")
    updates = ("update\nan expectation\nunder $\\pi_\\theta$",
               "update\nthe target\n$r + \\gamma \\max_{a'} Q(s',a')$",
               "update\nthe same target,\non $\\mathcal{D}$ only")

    for ax, title, upd in zip(axes, titles, updates):
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(*XL)
        ax.set_ylim(*YL)
        ax.text(3.25, 7.10, title, ha="center", va="top", fontsize=13.5,
                color="black")
        _box(ax, *PBOX, 2.7, 0.8, "policy $\\pi_\\theta$", BLUE, fontsize=13,
             fc="#e8f1f8")
        _box(ax, *UBOX, 3.7, 1.15, upd, GREEN, fontsize=12, fc="#eaf4ea")
        # the update's product goes back to the policy: same edge in all three
        _elbow(ax, [(0.65, UBOX[1]), (0.05, UBOX[1]), (0.05, PBOX[1]),
                    (0.40, PBOX[1])], GREEN)
        ax.text(-0.12, 3.70, "new policy", ha="center", va="center",
                fontsize=11.5, color=GREEN, rotation=90)

    axa, axb, axc = axes

    # --- (a) on-policy: the batch expires when the parameters move ---------- #
    _box(axa, *EBOX, 2.4, 0.8, "environment", GRAY, fontsize=13, fc="#f3f3f3")
    fl.arrow(axa, (3.15, 5.50), (4.10, 5.50), color=BLUE, lw=1.8, mut=14)
    axa.text(3.62, 5.62, "$a_t$", ha="center", va="bottom", fontsize=12.5,
             color=BLUE)
    fl.arrow(axa, (5.35, 5.08), (5.35, 4.45), color=GRAY, lw=1.8, mut=14)
    _box(axa, *DATA, 1.9, 0.8, "one batch\n$\\sim \\pi_\\theta$", BLUE,
         fontsize=12, fc="#e8f1f8")
    fl.arrow(axa, (3.44, 3.80), (4.00, 2.50), color=BLUE, lw=1.8, mut=14)
    fl.arrow(axa, (5.50, 3.62), (6.05, 3.20), color=GRAY, lw=1.5, ls="--",
             mut=13)
    axa.text(6.28, 3.02, "$\\times$", ha="center", va="center", fontsize=19,
             color=GRAY)
    axa.text(6.62, 2.72, "used once,\nthen stale", ha="right", va="top",
             fontsize=11.5, color=GRAY)
    axa.text(3.25, 1.15, "PPO's ratios buy a few\nepochs of extra life",
             ha="center", va="top", fontsize=11.5, color="black")

    # --- (b) off-policy: every version's data still counts ------------------ #
    _box(axb, *EBOX, 2.4, 0.8, "environment", GRAY, fontsize=13, fc="#f3f3f3")
    fl.arrow(axb, (3.15, 5.50), (4.10, 5.50), color=BLUE, lw=1.8, mut=14)
    axb.text(3.62, 5.62, "$a_t$", ha="center", va="bottom", fontsize=12.5,
             color=BLUE)
    fl.arrow(axb, (5.35, 5.08), (5.35, 4.90), color=GRAY, lw=1.8, mut=12)
    ys = DATA[1] + 0.39 * np.arange(-2, 3)          # ascending: oldest at the
    _tiles(axb, ys, DATA[0], 1.9, 0.30,             # bottom, newest on top
           labels=["$\\pi_1$", "$\\pi_2$", "$\\ldots$", "$\\pi_{k-1}$",
                   "$\\pi_k$"], vertical=True)
    for y in ys:
        fl.arrow(axb, (3.44, y), (4.00, 2.50), color=BLUE, lw=1.3, mut=12)
    axb.text(6.62, 4.74, "newest", ha="right", va="center", fontsize=11,
             color=GRAY)
    axb.text(6.62, 3.16, "oldest", ha="right", va="center", fontsize=11,
             color=GRAY)
    axb.text(3.25, 1.15, "the target does not mention\n"
                         "who collected the data",
             ha="center", va="top", fontsize=11.5, color="black")

    # --- (c) offline: the arrow to the world is gone ------------------------ #
    _box(axc, *EBOX, 2.4, 0.8, "environment", LIGHT, fontsize=13, tc=GRAY)
    # the wall: it separates the policy from the world, and only the deploy
    # arrow ever crosses it
    axc.plot([4.05, 4.05], [2.70, 6.58], color="black", lw=3.0,
             ls=(0, (5.0, 3.0)), zorder=5)
    fl.arrow(axc, (4.75, 4.30), (4.14, 4.30), color="black", lw=1.2, mut=13)
    axc.text(4.85, 4.30, "no interaction", ha="left", va="center",
             fontsize=12.5, color="black")
    fl.arrow(axc, (3.15, 5.50), (3.72, 5.50), color=LIGHT, lw=1.8, ls="--",
             mut=14)
    axc.text(4.05, 5.50, "$\\times$", ha="center", va="center", fontsize=19,
             color="black", zorder=6,
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))
    _box(axc, 2.30, DATA[1], 2.6, 0.9,
         "$\\mathcal{D} \\sim \\pi_\\beta$\ncollected once", GRAY, fontsize=12,
         fc="#f3f3f3")
    fl.arrow(axc, (3.10, 3.48), (3.30, 2.50), color=GRAY, lw=1.8, mut=14)
    _arc_arrow(axc, (0.85, 2.48), (1.30, 3.48), 0.42, GRAY, lw=1.6, mut=14)
    axc.text(1.62, 2.96, "sweep", ha="left", va="center", fontsize=11.5,
             color=GRAY)
    _elbow(axc, [(1.75, 5.92), (1.75, 6.34), (5.90, 6.34)], "black", lw=1.6)
    axc.text(2.95, 6.44, "deploy", ha="center", va="bottom", fontsize=11.5,
             color="black")
    axc.text(4.30, 6.42, "?", ha="left", va="bottom", fontsize=15,
             color="black", fontweight="bold")
    axc.text(3.25, 1.15, "errors are never tested: no mistake\n"
                         "the learned policy makes is discovered",
             ha="center", va="top", fontsize=11.5, color="black")

    fig.subplots_adjust(wspace=0.04)
    fl.save(fig, "mdl-rl-data-rules")

def fig_distribution_shift():  # F20 -> mdl-rl-distribution-shift
    """Distribution shift, measured.  (a) the dataset's support, pair by pair,
    with the pairs the learned greedy policy actually queries marked; (b) the
    error of each learned value against that support, with the pessimism
    penalty of the same shape drawn through the cloud -- the exploration bonus
    of :numref:`fig_rl_exploration` with the sign flipped."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.1),
                                   gridspec_kw={"width_ratios": [1.1, 1.0]})
    counts, err, greedy, terminal, n_trans, v0, q0 = _offline_experiment()
    live = ~terminal                                  # 11 cells, 44 pairs
    pairs = [(s, a) for s in range(16) if live[s] for a in range(4)]
    n = np.array([counts[s, a] for s, a in pairs])
    e = np.array([err[s, a] for s, a in pairs])
    order = np.argsort(-n)
    rank = {pairs[j]: i + 1 for i, j in enumerate(order)}
    kappa = float(np.median(e * np.sqrt(n)))          # the penalty that fits

    # --- (a) how much data each pair has ----------------------------------- #
    axa.bar(np.arange(1, len(n) + 1), n[order], width=0.86, color=BLUE,
            zorder=2, label="pairs in the dataset")
    qx = np.array([rank[p] for p in greedy])
    qy = np.array([counts[s, a] for s, a in greedy])
    axa.plot(qx, qy, "v", color=ORANGE, ms=9, ls="none", zorder=4,
             label="queried by the learned policy")
    axa.set_yscale("log")
    axa.set_xlim(0, len(n) + 1)
    axa.set_ylim(1.0, 1600.0)
    axa.set_yticks([1, 10, 100, 1000])
    axa.set_yticklabels(["1", "10", "100", "1000"])
    axa.set_xlabel("state-action pairs, sorted by dataset support", fontsize=13)
    axa.set_ylabel("times tried, $n(s,a)$", fontsize=13)
    axa.legend(loc="upper right", fontsize=11.5, handlelength=1.4,
               labelspacing=0.35)
    _leader(axa, "the learned policy\nasks here too", (38.6, 11.0),
            (23.5, 130.0), "black", fontsize=12, ha="left")

    # --- (b) and how wrong each value is ----------------------------------- #
    axb.plot(n, e, "o", color=BLUE, ms=5, alpha=0.6, ls="none", zorder=2,
             label="all 44 pairs")
    axb.plot([counts[s, a] for s, a in greedy], [err[s, a] for s, a in greedy],
             "v", color=ORANGE, ms=9, ls="none", zorder=4,
             label="queried by the learned policy")
    grid = np.logspace(np.log10(1.6), np.log10(700.0), 200)
    axb.plot(grid, kappa / np.sqrt(grid), color=ORANGE, lw=2.4, zorder=3,
             label="fitted count penalty $\\kappa/\\sqrt{n}$")
    axb.set_xscale("log")
    axb.set_xlim(1.6, 700.0)
    axb.set_ylim(0.0, 0.50)
    axb.set_xticks([2, 10, 100, 500])
    axb.set_xticklabels(["2", "10", "100", "500"])
    axb.set_xlabel("$n(s,a)$", fontsize=13)
    axb.set_ylabel("$|\\hat Q - Q^\\star|$", fontsize=13)
    axb.legend(loc="upper right", fontsize=11.5, handlelength=1.5,
               labelspacing=0.35)
    axb.text(560.0, 0.305, "a fitted envelope:\npoints on either side", ha="right",
             va="center", fontsize=12, color="black")

    for ax in (axa, axb):
        _black_axes(ax)
    fig.subplots_adjust(wspace=0.26)
    fl.save(fig, "mdl-rl-distribution-shift")
    return counts, err, greedy, kappa, n_trans, v0, q0

def fig_token_mdp():          # F25 -> mdl-rl-token-mdp
    """A response is a trajectory: (a) the token Markov decision process, whose
    transitions all have probability 1 and whose reward arrives only at the
    end; (b) the same object read as a one-step bandit.  The two columns say
    which of the two chapters' machinery the collapse takes with it."""
    fig = plt.figure(figsize=(11.8, 4.8))
    gs = GridSpec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[3.1, 1.7],
                  figure=fig, wspace=0.06, hspace=0.10)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), \
        fig.add_subplot(gs[:, 1])
    for ax in (axa, axb, axc):
        ax.axis("off")

    # --- (a) the token MDP -------------------------------------------------- #
    axa.set_aspect("equal")
    Y = 2.05
    xs = [0.75, 2.65, 4.55, 7.15]
    slabels = ["$x$", "$(x, y_1)$", "$(x, y_{1:2})$", "$(x, y_{1:T-1})$"]
    tokens = ["$y_1$", "$y_2$", None, "$y_T = \\mathrm{EOS}$"]
    xe = 9.35                                        # the terminal node
    for i, (x, lab) in enumerate(zip(xs, slabels)):
        _state(axa, (x, Y), r=0.15)
        axa.text(x, Y - 0.44, lab, ha="center", va="top", fontsize=12.5,
                 color="black")
    axa.add_patch(Rectangle((xe - 0.17, Y - 0.17), 0.34, 0.34, fc="black",
                            ec="black", lw=1.4, zorder=5))
    axa.text(xe, Y - 0.44, "$\\mathrm{EOS}$: terminal", ha="center", va="top",
             fontsize=12.5, color="black")

    ends = xs[1:] + [xe]
    for i, (x0, x1) in enumerate(zip(xs, ends)):
        if tokens[i] is None:                        # the elided middle
            axa.plot([x0 + 0.24, x1 - 0.24], [Y, Y], color=LIGHT, lw=1.6,
                     ls=(0, (1.6, 2.4)), zorder=2)
            axa.text((x0 + x1) / 2, Y + 0.10, "$\\ldots$", ha="center",
                     va="bottom", fontsize=13, color="black")
            continue
        fl.arrow(axa, (x0 + 0.19, Y), (x1 - 0.21, Y), color=GRAY, lw=1.6,
                 mut=14)
        # on the last edge the token label keeps to the left half, so that the
        # reward -- the only one in the whole chain -- owns the right half
        lx = x0 + (0.32 if i == len(xs) - 1 else 0.5) * (x1 - x0)
        axa.text(lx, Y + 0.14, tokens[i], ha="center", va="bottom",
                 fontsize=13, color=BLUE)
        axa.text(lx, Y - 0.12, "$1$", ha="center", va="top",
                 fontsize=13, color=GRAY)
    rx = xs[-1] + 0.66 * (xe - xs[-1])
    axa.text(rx, Y + 0.60, "$r(x, y)$", ha="center", va="bottom",
             fontsize=13.5, color=ORANGE)
    fl.arrow(axa, (rx, Y + 0.56), (rx, Y + 0.12), color=ORANGE, lw=1.6, mut=13)
    axa.text(0.75, Y + 0.34, "prompt", ha="center", va="bottom", fontsize=12.5,
             color="black")
    axa.text(4.9, 1.12, "state $=$ prefix,   action $=$ token,   "
                        "transition $=$ concatenation", ha="center", va="top",
             fontsize=13, color="black")
    axa.text(4.9, 0.68, "every transition has probability $1$: all the "
                        "randomness is the policy's,\nand $r_t = 0$ until "
                        "the response ends",
             ha="center", va="top", fontsize=12.5, color=GRAY)
    axa.set_xlim(-0.05, 10.15)
    axa.set_ylim(0.05, 3.15)

    # --- (b) the same object as one step ------------------------------------ #
    axb.set_aspect("equal")
    Yb, xb0, xb1 = 1.25, 3.25, 6.55
    _state(axb, (xb0, Yb), r=0.15, label="$x$")
    axb.add_patch(Rectangle((xb1 - 0.17, Yb - 0.17), 0.34, 0.34, fc="black",
                            ec="black", lw=1.4, zorder=5))
    axb.text(xb1, Yb - 0.40, "$\\mathrm{EOS}$", ha="center", va="top",
             fontsize=12.5, color="black")
    fl.arrow(axb, (xb0 + 0.19, Yb), (xb1 - 0.21, Yb), color=GRAY, lw=1.6,
             mut=14)
    axb.text((xb0 + xb1) / 2, Yb + 0.14,
             "$y \\sim \\pi_\\theta(\\cdot \\mid x)$", ha="center",
             va="bottom", fontsize=13, color=BLUE)
    axb.text((xb0 + xb1) / 2, Yb - 0.12, "$1$", ha="center", va="top",
             fontsize=13, color=GRAY)
    axb.text(xb1 + 0.34, Yb, "$r(x, y)$", ha="left", va="center",
             fontsize=13.5, color=ORANGE)
    axb.text(4.9, 0.52, "$\\nabla_\\theta \\log \\pi_\\theta(y \\mid x) = "
                        "\\sum_t \\nabla_\\theta \\log \\pi_\\theta"
                        "(y_t \\mid x, y_{<t})$: the two views, one gradient",
             ha="center", va="top", fontsize=13, color="black")
    axb.set_xlim(-0.05, 10.15)
    axb.set_ylim(-0.15, 1.85)

    # --- the two columns ---------------------------------------------------- #
    x0, dy = 0.07, 0.0455
    y = 0.975
    for head, color, items, struck in (("simplifies", RED, COLLAPSE, True),
                                       ("survives", GREEN, SURVIVE, False)):
        axc.text(x0 - 0.05, y, head, ha="left", va="center", fontsize=13.5,
                 fontweight="bold", color=color, transform=axc.transAxes)
        y -= 1.35 * dy
        for item in items:
            axc.text(x0, y, item, ha="left", va="center", fontsize=12,
                     color=GRAY if struck else "black",
                     transform=axc.transAxes)
            if struck:
                w = _strike_width(fig, axc, item, 12)
                axc.plot([x0 - 0.006, x0 + w + 0.006], [y, y], color=RED,
                         lw=1.3, transform=axc.transAxes, zorder=5,
                         clip_on=False)
            else:
                axc.plot([x0 - 0.028], [y], marker="s", ms=4.5, color=GREEN,
                         transform=axc.transAxes, clip_on=False)
            y -= dy
        y -= 0.9 * dy
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)

    fl.save(fig, "mdl-rl-token-mdp")


FIGURES = [
    fig_agent_env,
        fig_roadmap,
        fig_gridworld,
        fig_return_discount,
        fig_backups,
        fig_contraction,
        fig_value_wavefront,
        fig_gpi,
        fig_compounding_error,
        fig_exploration,
        fig_score_ascent,
        fig_variance_reduction,
        fig_table_vs_network,
        fig_score_vs_pathwise,
        fig_actor_critic_loop,
        fig_td_mc_spectrum,
        fig_policy_vs_parameter,
        fig_trust_region,
        fig_ppo_clip,
        fig_kl_tilting,
        fig_dqn_dataflow,
        fig_deadly_triad,
        fig_max_bias,
        fig_data_rules,
        fig_distribution_shift,
        fig_token_mdp,
]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):36s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == '__main__':
    main()
