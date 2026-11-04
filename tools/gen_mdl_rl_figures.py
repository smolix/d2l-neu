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

from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Patch, Rectangle)
from matplotlib.ticker import NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402  (applies the shared rcParams)

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
PURPLE, RED = fl.T.PURPLE.base, fl.T.RED.base   # token palette, not literals

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
NOT_NEEDED = ["discounting", "replay", "Q-learning", "a transition model",
              "the deadly triad"]
OPTIONAL = ["a learned critic", "Monte Carlo restarts"]

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

def _letter(ax, text, x=0.0, y=1.0, ha="left", va="bottom", fontsize=14):
    """Panel letter "(a)", "(b)", ... in axes coordinates (top-left by
    default), so the caption's letters can be found on the figure."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, color="black", fontweight="bold", zorder=8)

CLONE_EPS = 0.18186694      # the clone of imitation.md: training error at stop

def _clone_entry(eps=CLONE_EPS, horizon=100):
    """Exact entry probabilities on the calm lake for the expert and for the
    clone of the imitation section, by propagation (no sampling).

    The clone puts ``1 - eps`` on the expert's action at the six states the
    expert demonstrates and ``eps / 3`` on each other action there (Adam's
    per-parameter normalization makes every demonstrated row the same), and
    is exactly uniform at every other state, where it has no data.  Entry
    probability of ``s`` = mass absorbed at ``s`` when ``s`` is made
    absorbing, after ``horizon`` steps (the time limit).  Returns
    ``(p_expert, p_clone)``.
    """
    P, _, terminal, n_states, n_actions = _frozen_lake_mdp(slippery=False)
    out = []
    for clone in (False, True):
        pi = np.full((n_states, n_actions), 1.0 / n_actions)
        for s, a in LAKE_EXPERT.items():
            row = np.full(n_actions, eps / (n_actions - 1) if clone else 0.0)
            row[a] = 1.0 - eps if clone else 1.0
            pi[s] = row
        if not clone:                         # the expert never leaves its path
            for s in range(n_states):
                if s not in LAKE_EXPERT:
                    pi[s] = np.eye(n_actions)[0]
        M = np.einsum("sa,sat->st", pi, P)    # state-to-state kernel
        entry = np.zeros(n_states)
        for s in range(n_states):
            Ms = M.copy()
            Ms[s] = np.eye(n_states)[s]       # s absorbing
            d = np.eye(n_states)[0]
            for _ in range(horizon):
                d = d @ Ms
            entry[s] = d[s]
        out.append(entry)
    return out[0], out[1]

def _ucb_snapshot(n_pulls=2000, arms=BANDIT_ARMS, kappa=BANDIT_KAPPA,
                  seed=(2026, 3, 0)):
    """One run of the exploration section's UCB rule (each arm once, then the
    largest ``mean + kappa * sqrt(log t / n)``) on its ten Bernoulli arms.
    Returns the tallies ``(wins, count)`` after ``n_pulls`` pulls."""
    rng = np.random.default_rng(list(seed))
    wins, count = np.zeros(len(arms)), np.zeros(len(arms))
    for t in range(n_pulls):
        if (count == 0).any():
            a = int(np.argmax(count == 0))
        else:
            a = int(np.argmax(wins / count + kappa * np.sqrt(np.log(t) / count)))
        count[a] += 1
        wins[a] += float(rng.random() < arms[a])
    return wins, count

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
    entries are grayed, so the same rendering serves as a preview there and as
    a recapitulation here.  Section numbers are deliberately absent (they rot;
    the prose carries the :numref: cross-references).  DAgger lives in the
    on-policy column: its update data are the learner's own rollouts,
    relabeled by the expert, so only behavior cloning is offline."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axis("off")

    XS = {"on": 1.65, "off": 5.15, "offline": 8.65}      # column centres
    YS = {"value": 1.42, "policy": 3.36, "both": 5.05}   # row centres
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
    CH2 = GRAY                                    # the second chapter, grayed
    cells = [
        ("value", "on", BLUE, False, [("SARSA", "black")]),
        ("value", "off", BLUE, False, [("value iteration", "black"),
                                        ("Q-learning", "black"),
                                        ("DQN", CH2)]),
        ("value", "offline", GRAY, True, [("Q-learning", CH2),
                                          ("+ pessimism", CH2)]),
        ("policy", "on", BLUE, False, [("DAgger", "black"),
                                       ("REINFORCE", "black"),
                                       ("+ baselines", "black"),
                                       ("regularized PO", CH2),
                                       ("GRPO", CH2)]),
        ("policy", "offline", BLUE, False, [("behavior cloning", "black")]),
        ("both", "on", GRAY, True, [("actor-critic", CH2), ("PPO", CH2)]),
        ("both", "off", GRAY, True, [("SAC", CH2)]),
        ("model", "off", GRAY, True, [("learned model", CH2),
                                      ("+ planning", CH2)]),
        ("both", "offline", GRAY, True, [("Decision", CH2),
                                         ("Transformer", CH2)]),
    ]
    # The learned-model box sits in the off-policy column between SAC and the
    # bridge arrow: it learns from its own replay buffer, and what it learns
    # (a model of the dynamics) is neither a value function nor a policy.
    YS = dict(YS, model=3.95)
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
    YB = YS["policy"] - 0.20                     # clear of the model box
    fl.arrow(ax, (3.20, YB), (5.90, YB), color=GRAY, lw=1.6, mut=14)
    ax.text(4.62, YB - 0.14, "reusing recent batches,\nwith corrections",
            ha="center", va="top", fontsize=12, color="black")

    # --- the two families, bracketed and named in the right margin --------- #
    for y0, y1, note in [(0.60, 2.18, "learn $Q$,\nact greedily"),
                         (2.27, 5.72, "learn $\\pi_\\theta$\ndirectly")]:
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
    axa.text(2.0, 4.42, "shortest path: 6 moves",
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
    # Every probability sits inside the cell it belongs to, in the free
    # lower-right corner (where S/H/G glyphs sit elsewhere), below the arrow
    # that enters the cell, so none can be read as belonging to a neighbour.
    LBL = {8: ((0.92, 2.81), "right", ORANGE),
           10: ((2.92, 2.81), "right", ORANGE),
           13: ((1.92, 3.81), "right", BLUE)}
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

    for ax, letter in ((axa, "(a)"), (axb, "(b)")):
        ax.set_xlim(*LIM_X)
        ax.set_ylim(*LIM_Y)
        ax.text(-0.66, -0.34, letter, ha="left", va="center", fontsize=14,
                fontweight="bold", color="black")
    fig.subplots_adjust(wspace=0.02)
    fl.save(fig, "mdl-rl-gridworld")

def fig_termination():        # F27 -> mdl-rl-termination
    """Two ways an episode stops.  Top: it enters a terminal state (a hole);
    nothing follows, so the value after it is zero and the last target is
    the reward alone.  Bottom: the time limit cuts the recording at step 100
    in an ordinary cell; the process would have gone on, so the last target
    bootstraps from the value of the state at the cut."""
    fig, ax = plt.subplots(figsize=(10.4, 3.7))
    ax.axis("off")
    ax.set_aspect("equal")
    RAD = 0.31
    YA, YB = 2.50, 0.55

    def node(x, y, text, ec="black", ls="-", tc="black"):
        ax.add_patch(Circle((x, y), RAD, fc="white", ec=ec, lw=1.6, ls=ls,
                            zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=12,
                color=tc, zorder=4)

    def link(x0, x1, y, color="black", ls="-"):
        fl.arrow(ax, (x0 + RAD + 0.04, y), (x1 - RAD - 0.04, y), color=color,
                 lw=1.6, mut=12, ls=ls)

    def row(y, names):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        for x, name in zip(xs, names):
            if name == "...":
                ax.text(x, y, "$\\cdots$", ha="center", va="center",
                        fontsize=14, color="black")
            elif name is not None:
                node(x, y, name)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            link(x0, x1, y)

    # --- terminated: the episode ends in a hole ---------------------------- #
    row(YA, ["$s_0$", "$s_1$", "$s_2$", "...", "$s_7$", None])
    ax.add_patch(Rectangle((5.0 - RAD, YA - RAD), 2 * RAD, 2 * RAD, fc=GRAY,
                           alpha=0.55, ec="black", lw=1.4, zorder=3))
    ax.text(5.0, YA, "H", ha="center", va="center", fontsize=14,
            fontweight="bold", color="white", zorder=4)
    ax.text(5.0, YA + RAD + 0.06, "$s_8$", ha="center", va="bottom",
            fontsize=12.5, color="black")
    ax.text(5.62, YA, "a terminal state:\nno future, $V(s_8) = 0$", ha="left",
            va="center", fontsize=13, color="black", linespacing=1.3)
    ax.text(5.0 + RAD, YA - 0.44, "$y_7 = r_7$", ha="right", va="top",
            fontsize=13, color="black")

    # --- truncated: the recording is cut at step 100 ----------------------- #
    row(YB, ["$s_0$", "$s_1$", "$s_2$", "...", "$s_{99}$", "$s_{100}$"])
    XC = 5.62                                   # the cut
    ax.plot([XC, XC], [YB - 0.62, YB + 0.62], color="black", lw=1.6,
            ls=(0, (4, 2.5)), zorder=2)
    ax.text(XC, YB + 0.68, "time limit", ha="center", va="bottom",
            fontsize=12.5, color="black")
    for x0, x1, name in ((5.0, 6.2, "$s_{101}$"), (6.2, 7.4, "$s_{102}$")):
        node(x1, YB, name, ec=LIGHT, ls=(0, (3, 2)), tc=GRAY)
        link(x0, x1, YB, color=LIGHT, ls=(0, (3, 2)))
    ax.text(8.05, YB, "$\\cdots$", ha="center", va="center", fontsize=14,
            color=GRAY)
    # the unrecorded future, summarized by one number
    ax.plot([5.95, 5.95, 8.30, 8.30], [YB - 0.44, YB - 0.52, YB - 0.52,
                                       YB - 0.44], color=ORANGE, lw=1.6)
    ax.text(7.12, YB - 0.60, "not recorded; worth $V(s_{100})$",
            ha="center", va="top", fontsize=13, color=ORANGE)
    ax.text(5.0 + RAD, YB - 0.44, "$y_{99} = r_{99} + \\gamma V(s_{100})$",
            ha="right", va="top", fontsize=13, color="black")

    for y, letter, name in ((YA, "(a)", "terminated"),
                            (YB, "(b)", "truncated")):
        ax.text(-0.45, y, name, ha="right", va="center", fontsize=13,
                color="black", style="italic")
        ax.text(-2.35, y, letter, ha="left", va="center", fontsize=14,
                color="black", fontweight="bold")
    ax.set_xlim(-2.40, 8.45)
    ax.set_ylim(-0.50, 3.10)
    fl.save(fig, "mdl-rl-termination")

def fig_return_discount():    # O1  -> mdl-rl-return-discount
    """What $\\gamma$ buys: (a) the weight a reward $t$ steps away still
    carries, with the step at which it has fallen to 5%; (b) that step count as
    a horizon, $1/(1-\\gamma)$.  The chapter's own $\\gamma = 0.95$ is drawn
    with the three reference values."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 3.9),
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})
    GAMMAS = [(0.5, GRAY), (0.9, BLUE), (0.95, GREEN), (0.99, ORANGE)]
    THRESH = 0.05

    def cross(g):             # first t with gamma^t <= 0.05, computed not guessed
        return int(np.ceil(np.log(THRESH) / np.log(g)))

    assert [cross(g) for g, _ in GAMMAS] == [5, 29, 59, 299]

    # --- (a) the discount weights ------------------------------------------ #
    t = np.linspace(0, 68, 681)          # gamma^t is smooth in t; no kinks
    axa.axhline(THRESH, color="black", lw=1.1, ls="--", zorder=1)
    for g, color in GAMMAS:
        axa.plot(t, g ** t, color=color, lw=2.2,
                 label=f"$\\gamma = {g:g}$", zorder=3)
        tc = cross(g)
        if tc <= t[-1]:
            axa.plot([tc, tc], [0, THRESH], color=color, lw=1.2, ls="--",
                     zorder=2)
            axa.plot([tc], [g ** tc], "o", color=color, ms=6, zorder=4)
            axa.text(tc + 1.0, 0.105, f"$t = {tc}$", ha="left", va="center",
                     fontsize=13, color="black")
    axa.text(24, 0.56, f"$\\gamma = 0.99$ reaches $0.05$\nonly at "
                       f"$t = {cross(0.99)}$",
             ha="left", va="top", fontsize=13, color="black")
    axa.set_xlim(0, 68)
    axa.set_ylim(0, 1.13)
    # the dashed line is named by its own tick; no "0" label to crowd it
    axa.set_yticks([0.0, THRESH, 0.25, 0.5, 0.75, 1.0])
    axa.set_yticklabels(["", "0.05", "0.25", "0.50", "0.75", "1"])
    axa.set_xlabel("steps into the future, $t$", fontsize=13)
    axa.set_ylabel("weight $\\gamma^t$", fontsize=13)
    axa.legend(fontsize=12.5, loc="upper right", ncol=2, handlelength=1.3,
               columnspacing=1.2, handletextpad=0.5, borderaxespad=0.1,
               labelspacing=0.3, frameon=False)
    _letter(axa, "(a)", x=-0.16, y=1.0, va="top")

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
    axb.yaxis.set_minor_formatter(NullFormatter())
    axb.set_xlabel("$\\gamma$", fontsize=14)
    axb.set_ylabel("horizon $1/(1-\\gamma)$  (steps)", fontsize=13)
    _letter(axb, "(b)", x=-0.25, y=1.0, va="top")

    for ax in (axa, axb):
        for s in ("left", "bottom"):
            ax.spines[s].set_color("black")
        ax.tick_params(colors="black", labelsize=12)
    fig.subplots_adjust(wspace=0.28)
    fl.save(fig, "mdl-rl-return-discount")

def fig_backups():            # F4  -> mdl-rl-backups
    """The four one-step backups of the chapter in one node vocabulary: open
    circles are states, filled dots are state-action pairs, the orange arc is a
    maximum, and the one blue branch of the last panel is a single sample."""
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 4.2))
    LIM_X, LIM_Y = (-1.32, 1.32), (-0.62, 2.42)   # identical box for all four
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
    formula(axa, r"$V^\pi(s)=\sum_a \pi \left[\, r + \gamma\, \sum_{s'} \, P\,"
                 r"V^\pi(s') \right]$")

    # --- (b) the action-value function ------------------------------------- #
    axb = axes[1]
    tree_qa(axb)
    axb.text(-0.42, 1.86, r"$P(s' \mid s,a)$", ha="right", va="center",
             fontsize=EDGE, color="black")
    axb.text(0.0, 0.95, r"$\pi(a' \mid s')$", ha="center", va="center",
             fontsize=EDGE, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    formula(axb, r"$Q^\pi(s,a)=r+\gamma\,\sum_{s'}\,P\,\sum_{a'}\,\pi\,"
                 r"Q^\pi(s',a')$")

    # --- (c) the Bellman optimality backup: the max arc -------------------- #
    axc = axes[2]
    tree_va(axc)
    _max_arc(axc, ROOT, width=0.9, height=0.5, lpos=(0.0, -0.42))
    axc.text(0.0, 0.95, r"$P(s' \mid s,a)$", ha="center", va="center",
             fontsize=EDGE, color="black",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    formula(axc, r"$V^*(s)=\max_a \left[\, r+\gamma\,\sum_{s'}\,P\,"
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
    formula(axd, r"$y = r+\gamma\,\max_{a'}\, \hat Q(s',a')$")

    for ax, letter in zip(axes, ("(a)", "(b)", "(c)", "(d)")):
        ax.set_xlim(*LIM_X)
        ax.set_ylim(*LIM_Y)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(LIM_X[0] + 0.02, ROOT[1], letter, ha="left", va="center",
                fontsize=14, fontweight="bold", color="black")
    fig.subplots_adjust(wspace=0.08)        # room between the four formulas
    fl.save(fig, "mdl-rl-backups")

def fig_contraction():        # F5  -> mdl-rl-contraction
    """The sup-norm balls of radius $\\gamma^k r_0$ around $V^*$ on a
    two-state MDP small enough to state in one sentence, with the iterates of
    value iteration walking into them.

    The MDP: in each of the two states the agent may stay or switch; staying
    in $s_1$ pays 1, every other choice pays 0; $\\gamma = 0.8$.  So
    $V^*(s_1) = 1/(1-\\gamma) = 5$ and $V^*(s_2) = \\gamma V^*(s_1) = 4$."""
    fig, axa = plt.subplots(figsize=(5.4, 4.6))
    STAY, SWITCH = 0, 1
    r = np.array([[1.0, 0.0], [0.0, 0.0]])            # r[state, action]
    nxt = np.array([[0, 1], [1, 0]])                  # nxt[state, action]
    gam = 0.8

    def backup(V):
        return (r + gam * V[nxt]).max(axis=1)

    V = np.array([-2.0, 3.0])
    iters = [V.copy()]
    for _ in range(400):
        V = backup(V)
        iters.append(V.copy())
    iters = np.array(iters)
    Vst = iters[-1]                                   # the fixed point
    assert np.allclose(Vst, [5.0, 4.0])
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
             fontsize=13, color="black", bbox=bb)
    axa.text(Vst[0] + gam * r0, Vst[1] - 0.26, r"$\gamma r_0$", ha="center",
             va="top", fontsize=13, color="black", bbox=bb)

    # the first two sweeps as arrows; after V_2 the steps are short, so the
    # rest of the walk is one thin trail of small dots that runs into V*
    axa.plot(iters[:3, 0], iters[:3, 1], "o", color=BLUE, ms=5.0, zorder=6)
    for p, q in zip(iters[:2], iters[1:3]):
        fl.arrow(axa, tuple(p), tuple(q), color=BLUE, lw=1.6, mut=11)
    tail = iters[2:40]
    axa.plot(tail[:, 0], tail[:, 1], color=BLUE, lw=1.2, zorder=5)
    axa.plot(tail[1:16, 0], tail[1:16, 1], "o", color=BLUE, ms=2.8, zorder=6)
    axa.plot(*Vst, "o", color=GREEN, ms=9, zorder=7)
    axa.text(Vst[0] - 0.10, Vst[1] + 0.40, "$V^*$", ha="right", va="bottom",
             fontsize=14, color=GREEN)
    for i, (dx, dy, ha, va) in enumerate([(0.0, 0.40, "center", "bottom"),
                                          (0.0, 0.40, "center", "bottom"),
                                          (0.10, -0.40, "left", "top")]):
        axa.text(iters[i, 0] + dx, iters[i, 1] + dy, f"$V_{i}$", ha=ha, va=va,
                 fontsize=13, color=BLUE)

    pad = 1.08 * r0
    axa.set_xlim(Vst[0] - pad, Vst[0] + pad)
    axa.set_ylim(Vst[1] - pad, Vst[1] + pad)
    axa.set_aspect("equal")
    axa.set_xlabel("$V(s_1)$", fontsize=13)
    axa.set_ylabel("$V(s_2)$", fontsize=13)
    for side in ("left", "bottom"):
        axa.spines[side].set_color("black")
    axa.tick_params(colors="black", labelsize=12)
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
                ax.text(cc + 0.5, rr + 0.42, f"{H[kk][s]:.2f}", ha="center",
                        va="center", fontsize=12, color="black", zorder=4,
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=1.0))
        # S takes the corner _grid gives every H/G glyph, clear of the value
        ax.text(0.79, 0.79, "S", ha="center", va="center", fontsize=16,
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
    """Generalized policy iteration as four staircases on the same two lines.

    The horizontal axis is the value estimate, the vertical one the policy;
    the line $V = V^\\pi$ holds the estimates that are exact for their policy,
    the line $\\pi = \\mathrm{greedy}(V)$ the policies that are greedy for
    their estimate, and both hold only at the corner $(V^*, \\pi^*)$.  Each
    panel walks from the same start with one discipline: policy iteration
    evaluates all the way to the first line before it improves; value
    iteration takes one sweep toward it; a sampled backup takes a short,
    noisy step; actor-critic moves both coordinates a little at once.  The
    paths are computed from these rules, not drawn by hand."""
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.7))
    CX, CY = 8.0, 4.3                 # the corner: V^*, pi^*
    A, B = 1.8, 0.30                  # dV per dpi on line A, dpi per dV on B
    X0, Y0 = 1.34, 1.30               # one start for all four

    def on_a(y):                      # evaluation: move V onto V = V^pi
        return CX - A * (CY - y)

    def on_b(x):                      # improvement: move pi onto greedy(V)
        return CY - B * (CX - x)

    def staircase(frac, n, noise=0.0, seed=0):
        """Alternate an evaluation move that covers ``frac`` of the way to
        line A (plus optional noise) with a full improvement move."""
        rng = np.random.default_rng(seed)
        x, y = X0, Y0
        pts = [(x, y)]
        for _ in range(n):
            x = x + frac * (on_a(y) - x) + noise * rng.standard_normal()
            pts.append((x, y))
            y = on_b(x)
            pts.append((x, y))
        return np.array(pts)

    def both(frac, n):
        """Move V toward line A and pi toward line B in the same step."""
        x, y = X0, Y0
        pts = [(x, y)]
        for _ in range(n):
            x, y = x + frac * (on_a(y) - x), y + frac * (on_b(x) - y)
            pts.append((x, y))
        return np.array(pts)

    paths = [("policy iteration", staircase(1.0, 4), BLUE, 1.9, 12),
             ("value iteration", staircase(0.45, 9), GREEN, 1.7, 10),
             ("sampled backups (Q-learning)",
              staircase(0.20, 30, noise=0.22, seed=0), PURPLE, 1.4, 0),
             ("actor-critic", both(0.15, 80), ORANGE, 1.6, 0)]
    for ax, (title, pts, color, lw, mut) in zip(axes, paths):
        ax.axis("off")
        ax.set_xlim(0.30, 9.10)
        ax.set_ylim(0.15, 4.95)
        fl.arrow(ax, (0.55, 0.30), (9.0, 0.30), color="black", lw=1.1, mut=11)
        fl.arrow(ax, (0.55, 0.30), (0.55, 4.85), color="black", lw=1.1, mut=11)
        ax.text(8.95, 0.48, "$V$", ha="right", va="bottom", fontsize=14,
                color="black")
        ax.text(0.75, 4.80, "$\\pi$", ha="left", va="top", fontsize=14,
                color="black")
        ax.plot([on_a(0.70), CX], [0.70, CY], color=GRAY, lw=2.0, zorder=2)
        ax.plot([1.05, CX], [on_b(1.05), CY], color=GRAY, lw=2.0, zorder=2)
        if mut:                       # few, long moves: each one an arrow
            for p, q in zip(pts[:-1], pts[1:]):
                if np.hypot(*(q - p)) > 0.25:
                    fl.arrow(ax, tuple(p), tuple(q), color=color, lw=lw,
                             mut=mut)
                else:
                    ax.plot(*zip(p, q), color=color, lw=lw, zorder=3)
        else:                         # many short moves: one polyline
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=3,
                    solid_joinstyle="miter")
        ax.plot(X0, Y0, "o", color=color, ms=6, zorder=5)
        ax.plot(CX, CY, "o", color=GREEN, ms=8, zorder=5)
        ax.set_title(title, fontsize=13, color="black", pad=2)
    fig.subplots_adjust(wspace=0.06)
    axa = axes[0]

    def slope_angle(m):
        """Degrees a data slope ``m`` makes on the page, so a label can lie
        along its line instead of cutting across it."""
        bb = axa.get_position()
        w, h = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
        (x0, x1), (y0, y1) = axa.get_xlim(), axa.get_ylim()
        return np.degrees(np.arctan(m * (h / (y1 - y0)) / (w / (x1 - x0))))

    # line labels on the empty side of each line: above B, right of A
    axa.text(1.25, on_b(1.25) + 0.16, r"$\pi = \mathrm{greedy}(V)$",
             ha="left", va="bottom", fontsize=13, color="black",
             rotation=slope_angle(B), rotation_mode="anchor")
    axa.text(on_a(0.85) + 1.30, 0.85, "$V = V^\\pi$", ha="left", va="top",
             fontsize=13, color="black", rotation=slope_angle(1.0 / A),
             rotation_mode="anchor")
    axa.text(CX - 0.15, CY + 0.20, "$V^*, \\pi^*$", ha="right", va="bottom",
             fontsize=13, color=GREEN)
    axa.text(0.80, Y0 + 0.22, "$V_0, \\pi_0$", ha="left", va="bottom",
             fontsize=13, color=BLUE)
    fl.save(fig, "mdl-rl-gpi")

def fig_compounding_error():  # F21 -> mdl-rl-compounding-error
    """Where the clone of the imitation section goes: the calm lake, the
    expert's six-move path, and in every cell the exact probability that the
    deployed (sampling) clone enters it within the 100-step limit.  The black
    arrows are the clone's exits from the expert's cells into cells it has no
    data for; the holes next to the path take the rest of its loss."""
    p_exp, p_clone = _clone_entry()
    # the numbers of the imitation section's own cells (exp-imitation report)
    assert abs(p_clone[15] - 0.660) < 5e-4 and abs(p_clone[10] - 0.105) < 5e-4
    assert abs(p_clone[5] - 0.171) < 5e-4 and abs(p_clone[1] - 0.077) < 5e-4
    assert np.allclose(p_exp[[0, 4, 8, 9, 13, 14, 15]], 1.0)

    fig, ax = plt.subplots(figsize=(5.0, 5.2))
    cmap = plt.get_cmap("Oranges")
    for s in range(16):
        r, c = divmod(s, 4)
        ch = MAP4[r][c]
        fc = cmap(0.05 + 0.50 * p_clone[s])
        ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec="black", lw=1.2,
                               zorder=1))
        ax.text(c + 0.08, r + 0.08, f"{s}", ha="left", va="top", fontsize=12,
                color=GRAY, zorder=2)
        ax.text(c + 0.95, r + 0.08, f"{p_clone[s]:.2f}", ha="right", va="top",
                fontsize=12.5, color="black", zorder=2)
        if ch in "HGS":
            ax.text(c + 0.77, r + 0.75, ch, ha="center", va="center",
                    fontsize=16, fontweight="bold", color="black", zorder=2)
    PATH = [0, 4, 8, 9, 13, 14, 15]
    for s, s2 in zip(PATH[:-1], PATH[1:]):
        (r, c), (r2, c2) = divmod(s, 4), divmod(s2, 4)
        p0, p1 = np.array([c + 0.5, r + 0.5]), np.array([c2 + 0.5, r2 + 0.5])
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        fl.arrow(ax, p0 + 0.09 * u, p1 - 0.09 * u, color=BLUE, lw=2.4, mut=14)
    # the clone's exits into cells without data: 0 -> 1, 9 -> 10, 14 -> 10
    for s, s2 in ((0, 1), (9, 10), (14, 10)):
        (r, c), (r2, c2) = divmod(s, 4), divmod(s2, 4)
        p0 = np.array([c + 0.5, r + 0.5])
        p1 = np.array([c2 + 0.5, r2 + 0.5])
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        # on the centre line; the vertical one nudged left, into the gap
        # between the cell's index and its probability
        sh = np.array([-0.08, 0.0]) if u[0] == 0 else np.zeros(2)
        fl.arrow(ax, p0 + 0.12 * u + sh, p1 - 0.12 * u + sh, color="black",
                 lw=1.8, mut=13, ls=(0, (3, 2)))
    ax.text(2.0, 4.32, "blue: the expert's path; dashed: the clone leaves it",
            ha="center", va="center", fontsize=12.5, color="black")
    ax.set_xlim(-0.04, 4.04)
    ax.set_ylim(4.50, -0.04)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-rl-compounding-error")

def fig_exploration():        # F7  -> mdl-rl-exploration
    """One action-value row read as three behavior policies: greedy,
    epsilon-greedy and softmax.  (The regret curves and the coverage
    comparison that used to sit beside it are computed in the notebook.)"""
    fig, axa = plt.subplots(figsize=(5.6, 3.9))
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
        axa.text(1.55, 0.98 - 0.125 * i, name, ha="left", va="center",
                 fontsize=13, color=color)
    axa.set_xticks(x)
    axa.set_xticklabels(["$\\leftarrow$", "$\\downarrow$", "$\\rightarrow$",
                         "$\\uparrow$"], fontsize=17)
    axa.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axa.set_xlim(-0.55, 3.5)
    axa.set_ylim(0, 1.08)
    axa.set_ylabel("$\\pi(a \\mid s)$", fontsize=13)
    for xi, q in zip(x, Q):                               # the row itself
        axa.text(xi, -0.13, f"$\\hat Q = {q:.2f}$", ha="center", va="top",
                 fontsize=12.5, color="black")
    _black_axes(axa, labelsize=12)
    fl.save(fig, "mdl-rl-exploration")

def fig_ucb():                # F26 -> mdl-rl-ucb
    """Optimism in two pictures: (a) the UCB bonus as a function of an arm's
    count at the end of the run, $t = 2000$, and at $t = 100$; (b) one run of
    the section's UCB rule after 2,000 pulls: each arm's empirical mean with
    its bonus stacked on top.  The rule plays the arm whose bar reaches
    highest, so rarely tried arms stay in contention until their counts make
    the bonus small."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                   gridspec_kw={"width_ratios": [0.85, 1.3]})
    k = BANDIT_KAPPA
    # --- (a) the bonus ------------------------------------------------------ #
    m = np.linspace(1, 60, 400)
    for t, color, ls, lab in ((2000, ORANGE, "-", "$t = 2000$"),
                              (100, GRAY, "--", "$t = 100$")):
        axa.plot(m, k * np.sqrt(np.log(t) / m), color=color, lw=2.2, ls=ls,
                 label=lab)
    axa.set_xlim(0, 61)
    axa.set_ylim(0, 1.5)
    axa.set_yticks([0, 0.5, 1.0, 1.5])
    axa.set_xlabel("count $n(a)$", fontsize=13)
    axa.set_ylabel("bonus $\\kappa\\sqrt{\\log t / n(a)}$", fontsize=13)
    axa.legend(loc="upper right", fontsize=12.5, handlelength=1.6,
               frameon=False)
    _black_axes(axa, labelsize=12)
    _letter(axa, "(a)", x=-0.26, y=1.0, va="top")

    # --- (b) one run, at its end -------------------------------------------- #
    T = 2000
    wins, count = _ucb_snapshot(T)
    mean = wins / count
    bonus = k * np.sqrt(np.log(T) / count)
    top = mean + bonus
    x = np.arange(len(BANDIT_ARMS))
    axb.bar(x, mean, width=0.6, color=BLUE, alpha=0.85, zorder=3,
            label="estimate $\\hat\\mu(a)$")
    axb.bar(x, bonus, bottom=mean, width=0.6, color=ORANGE, alpha=0.35,
            ec=ORANGE, lw=1.2, zorder=3, label="bonus")
    axb.plot(x, BANDIT_ARMS, "_", color="black", ms=22, mew=2.2, zorder=4,
             label="true mean $\\mu_a$")
    axb.axhline(top.max(), color="black", lw=1.0, ls="--", zorder=2)
    axb.text(9.55, top.max() + 0.03, "largest index: played next",
             ha="right", va="bottom", fontsize=12.5, color="black")
    axb.set_xticks(x)
    axb.set_xticklabels([f"{n:.0f}" for n in count])
    axb.set_xlim(-0.6, 9.6)
    axb.set_ylim(0, 1.30)
    axb.set_yticks([0, 0.5, 1.0])
    axb.set_xlabel(f"pulls $n(a)$ of each arm after $t = {T}$", fontsize=13)
    axb.legend(loc="upper right", ncol=3, fontsize=12, handlelength=1.2,
               columnspacing=1.0, frameon=False, borderaxespad=0.0)
    _black_axes(axb, labelsize=12)
    _letter(axb, "(b)", x=-0.15, y=1.0, va="top")
    fig.subplots_adjust(wspace=0.30)
    fl.save(fig, "mdl-rl-ucb")

def fig_score_ascent():       # F8  -> mdl-rl-score-ascent
    """One REINFORCE step on the one-step problem: (a) twelve sampled actions,
    each pushing its own log-probability up in proportion to what it earned;
    (b) the policy that one step produces."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 4.0))
    LIM_X, LIM_Y = (-3.4, 5.2), (-0.055, 0.66)
    SCALE = 0.065

    a = np.linspace(*LIM_X, 601)
    samples = np.random.default_rng(0).normal(TOY_MU, 1.0, 12)
    rew = _toy_reward(samples)
    u_hat = float((rew * (samples - TOY_MU)).mean())    # the score estimate
    eta = 1.0                              # step size: the estimate itself
    mu_new = TOY_MU + eta * u_hat

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
        # one small fixed head, so the shaft carries the length
        fl.arrow(axa, (ai, 0.016), (ai, 0.016 + SCALE * ri), color=BLUE,
                 lw=1.6, mut=6)
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

    for ax, letter in ((axa, "(a)"), (axb, "(b)")):
        _letter(ax, letter, x=0.0, y=1.02)
    fig.subplots_adjust(wspace=0.10)
    fl.save(fig, "mdl-rl-score-ascent")

def fig_trajectory_chain():   # F28 -> mdl-rl-trajectory-chain
    """The trajectory likelihood as a chain of factors,
    $P(\\tau;\\theta) = \\mu_0(s_0)\\prod_t \\pi_\\theta(a_t\\mid s_t)
    P(s_{t+1}\\mid s_t,a_t)$.  The policy factors (blue, above the chain)
    depend on $\\theta$; the start distribution and the kernel (gray, below)
    do not, so their logarithms have zero gradient and the score of a
    trajectory is the sum of the policy's scores."""
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    ax.axis("off")
    ax.set_aspect("equal")
    RS, RA = 0.27, 0.09                      # state circle, action dot
    Y = 0.0
    xs = [0.0, 1.75, 3.5, 5.25, 7.0]         # s0 a0 s1 a1 s2
    kinds = ["s", "a", "s", "a", "s"]
    names = ["$s_0$", "$a_0$", "$s_1$", "$a_1$", "$s_2$"]
    for x, kind, name in zip(xs, kinds, names):
        if kind == "s":
            ax.add_patch(Circle((x, Y), RS, fc="white", ec="black", lw=1.6,
                                zorder=3))
            ax.text(x, Y, name, ha="center", va="center", fontsize=13,
                    color="black", zorder=4)
        else:
            ax.add_patch(Circle((x, Y), RA, fc="black", ec="black", zorder=3))
            ax.text(x, Y + 0.18, name, ha="center", va="bottom", fontsize=13,
                    color="black")

    def edge(x0, k0, x1, k1, color):
        r0 = RS if k0 == "s" else RA
        r1 = RS if k1 == "s" else RA
        fl.arrow(ax, (x0 + r0 + 0.05, Y), (x1 - r1 - 0.06, Y), color=color,
                 lw=2.0, mut=13)

    # mu_0 enters the chain from the left, like the kernel: no theta
    fl.arrow(ax, (-1.35, Y), (-RS - 0.06, Y), color=GRAY, lw=2.0, mut=13)
    ax.text(-0.72, Y - 0.24, "$\\mu_0(s_0)$", ha="center", va="top",
            fontsize=13, color=GRAY)
    LBL = [("$\\pi_\\theta(a_0 \\mid s_0)$", BLUE, True),
           ("$P(s_1 \\mid s_0, a_0)$", GRAY, False),
           ("$\\pi_\\theta(a_1 \\mid s_1)$", BLUE, True),
           ("$P(s_2 \\mid s_1, a_1)$", GRAY, False)]
    for i, (text, color, above) in enumerate(LBL):
        edge(xs[i], kinds[i], xs[i + 1], kinds[i + 1], color)
        xm = 0.5 * (xs[i] + xs[i + 1])
        ax.text(xm - (0.12 if above else 0.0), Y + 0.20 if above else Y - 0.24,
                text, ha="center",
                va="bottom" if above else "top", fontsize=13, color=color)
    fl.arrow(ax, (xs[-1] + RS + 0.05, Y), (8.05, Y), color=BLUE, lw=2.0,
             mut=13)
    ax.text(8.35, Y, "$\\cdots$", ha="center", va="center", fontsize=15,
            color="black")

    # the two rows, named in the left margin
    ax.text(-1.65, Y + 0.33, "depends on $\\theta$:", ha="right",
            va="center", fontsize=13, color=BLUE)
    ax.text(-1.65, Y - 0.37, "independent of $\\theta$:", ha="right",
            va="center", fontsize=13, color=GRAY)
    ax.text(3.6, Y - 1.12,
            "$\\nabla_\\theta \\log P(\\tau;\\theta) = \\sum_t "
            "\\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t)$:"
            "  the gray factors drop out",
            ha="center", va="center", fontsize=13, color="black")
    ax.set_xlim(-3.75, 8.75)
    ax.set_ylim(-1.40, 0.70)
    fl.save(fig, "mdl-rl-trajectory-chain")

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
    # the action marker splits the strip: its label goes right, over the
    # rewards it can affect; the dropped past gets its note on the left
    fl.arrow(axa, (2.5, 1.50), (2.5, 0.88), color="black", lw=1.6, mut=13)
    axa.text(2.68, 1.34, "action $a_3$ taken here", ha="left", va="center",
             fontsize=13, color="black")
    axa.plot([2.56, 2.56, 7.44, 7.44], [0.14, 0.02, 0.02, 0.14], color=BLUE,
             lw=1.6)
    axa.text(5.15, -0.08, "$\\hat G_3 = \\sum_{t' \\geq 3} "
                          "\\gamma^{t'-3} r_{t'}$",
             ha="center", va="top", fontsize=13, color="black")
    axa.text(2.30, 0.94, "already collected:\nzero expected product\n"
                         "with the score of $a_3$",
             ha="right", va="bottom", fontsize=12.5, color="black")
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
    # stops below the legend rows so no entry is struck through
    axb.axvline(grad, ymax=0.76, color="black", lw=1.4, ls="--", zorder=5)
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
    # R-bar keeps its orange from (b); the optimum gets its own colour
    for bv, color in ((0.0, GRAY), (e_r, ORANGE), (b_star, GREEN)):
        vv = var_x - 2 * bv * cov + bv ** 2
        axc.plot([bv, bv], [0.0, vv], color=color, lw=1.5, ls="--", zorder=2)
        axc.plot([bv], [vv], "o", color=color, ms=7, zorder=4)
    _leader(axc, "no baseline", (0.0, var_x), (0.34, var_x * 1.04), GRAY,
            ha="left", va="bottom")
    axc.set_xticks([0.0, e_r, b_star, 2.0])
    axc.set_xticklabels(["$0$", "$\\bar R$", "$b^{\\!*}$", "$2$"])
    for lbl, color in zip(axc.get_xticklabels(), (GRAY, ORANGE, GREEN, "black")):
        lbl.set_color(color)
        lbl.set_fontsize(13)
    axc.text(0.95, var_x * 1.40,
             f"at $b^{{\\!*}}$, $1 - \\mathrm{{corr}}^2 = {1 - rho ** 2:.2f}$\n"
             f"of the variance at $b = 0$",
             ha="center", va="top", fontsize=12.5, color="black")
    axc.set_xlim(-0.58, 2.58)
    axc.set_ylim(0.0, var_x * 1.45)
    axc.set_xlabel("baseline $b$", fontsize=13)
    axc.set_ylabel("variance of the estimator", fontsize=13)
    _black_axes(axc)

    for ax, letter, x in ((axa, "(a)", -0.12), (axb, "(b)", 0.0),
                          (axc, "(c)", 0.0)):
        _letter(ax, letter, x=x, y=1.02)
    fig.subplots_adjust(wspace=0.26)
    fl.save(fig, "mdl-rl-variance-reduction")

def fig_table_vs_network():    # F10 -> mdl-rl-table-vs-network
    """What one value update touches, computed rather than drawn.

    The model is a one-hidden-layer network, ``V(x) = w2 . tanh(w1 u + b1) +
    b2`` with 64 tanh units on the centred input ``u = 4 (x - 1/2)``, its
    first layer drawn at random and its output layer fitted to a target value
    curve by ridge least squares.  One gradient step of size ``eta_V`` on the
    squared error at ``x0`` changes the estimate everywhere, to first order, by

        dV(x) = eta_V * delta * k(x, x0),   k(x, x0) = grad V(x) . grad V(x0),

    the gradient taken over *all* parameters (both layers), the tangent kernel
    of the network.  A table is the case of one-hot features, whose kernel is
    the indicator of "same bin".  ``eta_V`` is set so the visited state moves
    by the same +0.50 under both; the panels then differ only in what happens
    elsewhere: the network's change is a bump around ``x0`` that decays and
    turns negative on the far side, the sign structure the section's
    coupling probe measures on CartPole.
    """
    rng = np.random.default_rng(0)
    M = 64
    W1 = rng.uniform(-4.0, 4.0, M)
    B1 = rng.uniform(-2.0, 2.0, M)

    def u(x):
        return 4.0 * (np.atleast_1d(x) - 0.5)

    def hidden(x):
        return np.tanh(np.outer(u(x), W1) + B1)

    def vstar(x):
        return 0.6 + 0.35 * np.sin(2 * np.pi * x) - 0.2 * x

    xf = np.linspace(0.0, 1.0, 200)
    A = np.c_[hidden(xf), np.ones(len(xf))]
    wb = np.linalg.solve(A.T @ A + 1e-6 * np.eye(M + 1), A.T @ vstar(xf))
    W2, B2 = wb[:M], wb[M]
    assert np.abs(A @ wb - vstar(xf)).max() < 5e-3       # the fit is a fit

    def value(x):
        return hidden(x) @ W2 + B2

    def grad(x):                                         # dV/d(all parameters)
        h = hidden(x)
        g_pre = W2 * (1.0 - h ** 2)
        return np.c_[h, np.ones((len(h), 1)), g_pre * u(x)[:, None], g_pre]

    x0, delta = 0.40, 1.0                                # one semi-gradient step
    NB = 16                                              # table resolution
    step = 0.5                                           # move at the visited x
    grid = np.linspace(0.0, 1.0, 601)
    g0 = grad(x0)[0]
    eta_v = step / (delta * (g0 @ g0))                   # same +0.50 at x0
    dv_net = eta_v * delta * (grad(grid) @ g0)           # first-order change
    assert abs(dv_net.max() - step) < 0.02 and dv_net[-1] < 0

    edges = np.linspace(0.0, 1.0, NB + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    tab0 = vstar(centres)
    b0 = int(np.searchsorted(edges, x0) - 1)             # the bin holding x0
    tab1 = tab0.copy()
    tab1[b0] += step

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(12.6, 3.7))
    YL = (-0.06, 1.62)

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
    v0 = value(grid)
    axb.plot(grid, v0, color=LIGHT, lw=3.4, zorder=2, label="before")
    axb.plot(grid, v0 + dv_net, color=BLUE, lw=2.0, zorder=3, label="after")
    axb.plot([x0], [value(x0)[0]], "o", color=ORANGE, ms=8, zorder=5)
    axb.annotate("the only state\nthe update mentions",
                 xy=(x0, float(value(x0)[0]) - 0.03), xytext=(0.03, 0.08),
                 ha="left", va="bottom", fontsize=12.5, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=2, shrinkB=3))
    axb.set_title("network: 64 tanh units", fontsize=13, color="black",
                  pad=6)
    axb.legend(loc="upper right", fontsize=11.5, handlelength=1.4)

    # --- (c) the two changes, side by side --------------------------------- #
    axc.axhline(0.0, color="black", lw=1.0, zorder=1)
    axc.bar(centres[b0], step, width=1.0 / NB, color=LIGHT, ec=GRAY, lw=1.0,
            zorder=3, label="table")
    axc.fill_between(grid, 0.0, dv_net, color=ORANGE, alpha=0.30, zorder=2)
    axc.plot(grid, dv_net, color=ORANGE, lw=2.2, zorder=4, label="network")
    cross = grid[np.flatnonzero((dv_net[:-1] > 0) & (dv_net[1:] <= 0))[0]]
    axc.text(0.03, -0.07, "one update,\nevery state", ha="left", va="top",
             fontsize=12.5, color="black", linespacing=1.4)
    axc.annotate(f"${dv_net[-1]:+.2f}$ at the far end",
                 xy=(0.99, dv_net[-1] - 0.01), xytext=(0.42, -0.34),
                 ha="left", va="center", fontsize=12.5, color="black",
                 bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.6),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=3, shrinkB=3))
    assert 0.6 < cross < 0.85
    axc.set_ylim(-0.44, 0.62)
    axc.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    axc.set_ylabel(r"$\Delta \hat V(x)$", fontsize=13)
    axc.legend(loc="upper right", fontsize=11.5, handlelength=1.4)

    for ax in (axa, axb):
        ax.set_ylim(*YL)
    for ax, letter in ((axa, "(a)"), (axb, "(b)"), (axc, "(c)")):
        ax.set_xlim(-0.01, 1.01)
        ax.set_xlabel("state $x$", fontsize=13)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("black")
        ax.tick_params(colors="black", labelsize=12)
        _letter(ax, letter, x=-0.20, y=1.02)
    fig.subplots_adjust(wspace=0.32)
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
    LIMX, LIMY = (-0.22, 9.05), (1.90, 7.00)  # identical box -> identical panels
    GRAD = dict(color=ORANGE, lw=1.9, ls=(0, (4.5, 2.6)))
    HEAD, BOXF, LBL = 13.0, 13.0, 12.0

    # ===== (a) score function ============================================== #
    axa.text(4.55, 6.80, "score function:  "
                         r"$\hat g = R\,\nabla_\theta \log \pi_\theta(a)$",
             ha="center", va="center", fontsize=HEAD, color="black")

    _box(axa, 1.75, 5.35, 2.9, 1.0, "policy\n$\\pi_\\theta(a \\mid s)$", BLUE,
         fontsize=BOXF, fc="#e8f1f8")
    _box(axa, 7.10, 5.35, 2.9, 1.5, "", GRAY, fc="#dcdcdc")
    axa.text(7.10, 5.75, "environment", ha="center", va="center",
             fontsize=BOXF, color="black", zorder=4)
    axa.text(7.10, 5.34, "$r(s,a)$", ha="center", va="center", fontsize=BOXF,
             color="black", zorder=4)
    axa.text(7.10, 4.93, "not differentiated", ha="center", va="center",
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
    # the loop runs clear of the policy box and enters it at mid-height
    _path_arrow(axa, [(1.28, 2.55), (-0.02, 2.55), (-0.02, 5.35), (0.29, 5.35)],
                **GRAD)
    axa.text(0.46, 4.30, r"$\theta \leftarrow \theta + \eta \hat g$",
             ha="left", va="center", fontsize=LBL, color="black")

    # ===== (b) pathwise ==================================================== #
    axb.text(4.55, 6.80, "pathwise:  "
                         r"$\hat g = \nabla_\theta Q_w(s, a_\theta)$",
             ha="center", va="center", fontsize=HEAD, color="black")

    _box(axb, 1.75, 5.35, 2.9, 1.0,
         "policy\n$\\mu_\\theta(s),\\ \\sigma_\\theta$", BLUE,
         fontsize=BOXF, fc="#e8f1f8")
    _box(axb, 6.00, 5.35, 4.2, 1.10, "", BLUE)
    axb.text(6.00, 5.56, r"$a = \mu_\theta(s) + \sigma_\theta\,z$",
             ha="center", va="center", fontsize=BOXF, color="black", zorder=4)
    axb.text(6.00, 5.10, r"$z \sim \mathcal{N}(0, I)$, held fixed",
             ha="center", va="center", fontsize=11.5, color=GRAY, zorder=4)
    _box(axb, 6.10, 2.55, 4.6, 1.20, "", GREEN, fc="#eaf6ea")
    axb.text(6.10, 2.76, "critic $Q_w(s,a)$", ha="center", va="center",
             fontsize=BOXF, color="black", zorder=4)
    axb.text(6.10, 2.36, "must be differentiable in $a$", ha="center",
             va="center", fontsize=11.5, color="black", style="italic",
             zorder=4)
    _box(axb, 1.75, 2.55, 2.9, 0.95, "replay buffer\n(past data)", GRAY,
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
    axb.text(3.42, 4.55, r"$\partial a / \partial \theta$", ha="left",
             va="center", fontsize=11.5, color="black")
    axb.text(1.75, 3.66, "backpropagation passes\nthrough both boxes",
             ha="center", va="center", fontsize=11.5, color="black",
             linespacing=1.2)

    for ax in (axa, axb):
        ax.set_xlim(*LIMX)
        ax.set_ylim(*LIMY)
        ax.set_aspect("equal")
        ax.axis("off")
    # Between the two panels, just below them (axes coordinates, so it follows
    # the panels however tight bounding boxes shift them).
    axa.text(1.0, -0.06, "the pathwise estimator needs a $Q$ that is "
                         "differentiable in $a$; $Q$ can be trained off-policy",
             transform=axa.transAxes, ha="center", va="top", fontsize=13,
             color="black")
    for ax, letter in ((axa, "(a)"), (axb, "(b)")):
        ax.text(-0.15, 6.80, letter, ha="left", va="center", fontsize=14,
                fontweight="bold", color="black")
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
    _box(ax, 6.75, 1.55, 4.6, 0.95,
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
    fl.arrow(ax, (2.98, 1.55), (4.39, 1.55), color=GREEN, lw=2.0, mut=16)
    ax.text(3.68, 1.68, "$\\hat V_w(s_t)$\n$\\hat V_w(s_{t+1})$", ha="center",
            va="bottom", fontsize=13, color=GREEN, linespacing=1.25)

    # the one number, leaving twice
    _path_arrow(ax, [(5.10, 1.06), (5.10, 0.52), (1.60, 0.52), (1.60, 0.99)],
                **GRAD)
    _path_arrow(ax, [(5.10, 2.04), (5.10, 2.76), (1.60, 2.76), (1.60, 3.20)],
                **GRAD)
    for y in (0.62, 2.86):
        ax.text(3.30, y, "$\\delta_t$", ha="center", va="bottom", fontsize=14,
                color=ORANGE)

    ax.text(5.2, -0.20,
            r"actor:  $\theta \leftarrow \theta + \eta\, \delta_t\,"
            r"\nabla_\theta \log \pi_\theta(a_t \mid s_t)$"
            "        "
            r"critic:  $w \leftarrow w + \alpha\, \delta_t\,"
            r"\nabla_w \hat V_w(s_t)$",
            ha="center", va="center", fontsize=13, color="black")
    ax.text(5.2, -0.78, "the critic stays ahead of the actor: "
                        "several critic passes per batch, one actor step",
            ha="center", va="center", fontsize=12, color="black")

    ax.set_xlim(0.05, 10.35)
    ax.set_ylim(-1.05, 4.45)
    fl.save(fig, "mdl-rl-actor-critic-loop")

def fig_td_mc_spectrum():     # F12 -> mdl-rl-td-mc-spectrum
    """The credit-assignment dial: the same target family drawn (left, in F4's
    node vocabulary) and measured (right).

    The measurement is a ten-state chain, deterministic step right, terminal
    after the last state, whose only payment is $1$ on the final transition
    plus per-step reward noise.  The critic is deliberately imperfect, with an
    error that shrinks toward termination: exact near the end and wrong far
    from it, which is the wavefront of :numref:`fig_rl_value_wavefront` seen
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
        big = p1 - p0 > 0.05
        for t, p in ((t0, p0), (t0 + DT, p1)):
            axa.plot([t, t], [0.06, p], ls=":", lw=1.2, color=LIGHT, zorder=1)
            axa.plot([t], [p], "o", color=color, ms=7, zorder=5)
            if big:
                axa.plot([RAIL, t], [p, p], ls=":", lw=1.1, color=LIGHT,
                         zorder=1)
        if big:              # the induced change, bracketed in the left margin
            axa.annotate("", xy=(RAIL, p1), xytext=(RAIL, p0),
                         arrowprops=dict(arrowstyle="<->", color=color, lw=1.7,
                                         shrinkA=0, shrinkB=0,
                                         mutation_scale=10))
            axa.text(RAIL - 0.55, (p0 + p1) / 2, dpi_txt(float(p1 - p0)),
                     ha="right", va="center", fontsize=13, color=color)
        else:                # too small to bracket: named beside the points
            axa.text(t0 + DT + 0.35, p1 - 0.075, dpi_txt(float(p1 - p0)),
                     ha="left", va="top", fontsize=13, color=color)

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
    for t0, color, x, y in ((0.0, GREEN, 1.17, 0.68), (6.0, ORANGE, 3.68, 0.30)):
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
    axb.set_xlim(-0.62, 4.45)
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
    # In one parameter the KL region is an interval; its half-width in
    # theta is the delta of the ticks (and the radius of panel (b)'s ball).
    axa.text(T_OLD, 4.10, "trust region\n"
             r"$D_{\mathrm{KL}} \leq \Delta_{\mathrm{KL}}$", ha="center",
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
    axa.text(T_OUT - 0.14, 2.45, "inside the region:\n" r"$J$ improves," "\n"
             rf"${J0:.2f} \to {float(J(T_IN)):.2f}$", ha="right", va="center",
             fontsize=12, color=GREEN, linespacing=1.2)
    axa.text(T_OUT - 0.14, 0.95, "outside it:\n" r"$J$ collapses," "\n"
             rf"${J0:.2f} \to {float(J(T_OUT)):.2f}$",
             ha="right", va="center", fontsize=12, color=ORANGE,
             linespacing=1.2)
    axa.plot([T_OLD], [J0], "o", color="black", ms=7.5, zorder=6)
    axa.plot([T_OLD, T_OLD], [YL[0], J0], ls=":", lw=1.1, color=LIGHT, zorder=1)
    axa.set_xlim(*XL)
    axa.set_ylim(*YL)
    axa.set_xticks([T_OLD - DELTA, T_OLD, T_OLD + DELTA])
    axa.set_xticklabels([r"$\theta_{\mathrm{old}} - \delta$",
                         r"$\theta_{\mathrm{old}}$",
                         r"$\theta_{\mathrm{old}} + \delta$"], fontsize=13)
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
    u_long = -u_long * np.sign(u_long[1])          # fix signs: down-left,
    u_short = -u_short * np.sign(u_short[0])       # and up-left
    ang = np.linspace(0.0, 2 * np.pi, 401)
    ell = (V * semi) @ np.vstack([np.cos(ang), np.sin(ang)])
    kl = 0.5 * np.einsum("ij,jk,ik->i", ell.T, F, ell.T)
    assert np.allclose(kl, DEL)                    # a level set of the quadratic
    RHO = float(np.sqrt(semi[0] * semi[1]))        # equal area: pi ab = pi rho^2

    axb.plot(ell[0], ell[1], color=ORANGE, lw=2.2, zorder=4)
    axb.add_patch(Circle((0, 0), RHO, fc="none", ec=GRAY, lw=1.6,
                         ls=(0, (5, 3)), zorder=3))
    fl.axis_cross(axb, (-1.16, 1.16), (-1.16, 1.16), color="black", lw=1.0)
    axb.plot([0], [0], "o", color="black", ms=7, zorder=6)
    axb.text(0.09, 0.07, r"$\theta_{\mathrm{old}}$", ha="left", va="bottom",
             fontsize=13, color="black")

    # The two equal steps, and their annotations placed radially past the tip of
    # the ellipse axis each one follows -- for an ellipse's own axes the radial
    # direction *is* the outward normal, and it is the only direction in which
    # the label cannot fall back onto the curve.  The labels quote the exact
    # divergence KL(pi_old || pi_new) of each drawn step; the ellipse is its
    # second-order approximation.
    base = np.array([2.5, 0.0, 0.0])

    def exact_kl(step):
        q = _softmax(base + np.r_[step, 0.0])
        return float((p * np.log(p / q)).sum())

    d_kls = [exact_kl(RHO * u) for u in (u_long, u_short)]
    assert abs(d_kls[0] - 0.009) < 5e-4 and abs(d_kls[1] - 0.062) < 5e-4
    assert abs(d_kls[1] / d_kls[0] - 7.0) < 0.1    # "about seven times"
    for u, semi_i, d_kl, name, va in (
            (u_long, semi[0], d_kls[0], "barely moves", "top"),
            (u_short, semi[1], d_kls[1], "about seven times as far",
             "bottom")):
        tip = RHO * u
        fl.arrow(axb, (0.0, 0.0), tuple(tip), color=BLUE, lw=2.2, mut=15)
        far = u * (max(RHO, float(semi_i)) + (0.22 if va == "top" else 0.27))
        axb.text(far[0], far[1],
                 rf"$D_{{\mathrm{{KL}}}} = {d_kl:.3f}$:" "\n" + name,
                 ha="right", va=va, fontsize=12,
                 color="black", linespacing=1.4)
    axb.text(0.98, 1.10, r"$\Delta\theta^{\!\top}\! F\,\Delta\theta\, /\, 2 \leq "
             rf"{DEL:g}$", ha="left", va="bottom", fontsize=12.5,
             color=ORANGE)
    axb.text(0.95, -0.50, "equal-area ball\n" r"$\|\Delta\theta\| \leq \delta$",
             ha="left", va="top", fontsize=12.5, color="black",
             linespacing=1.5)
    axb.text(1.22, -0.10, r"$\Delta\theta_1$", ha="left", va="top",
             fontsize=13, color="black")
    axb.text(0.0, 1.22, r"$\Delta\theta_2$", ha="center", va="bottom",
             fontsize=13, color="black")
    axb.set_aspect("equal")
    axb.set_xlim(-2.25, 2.75)
    axb.set_ylim(-1.35, 1.50)
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
        # the point rho = 1 is annotated in both panels, so the band's two
        # edges carry the only tick labels (three would collide)
        ax.set_xticks([1 - EPS, 1 + EPS])
        ax.set_xticklabels([r"$1-\epsilon_{\mathrm{clip}}$",
                            r"$1+\epsilon_{\mathrm{clip}}$"], fontsize=12.5)
        ax.set_yticks([-2, -1, 0, 1])
        ax.set_title(title, fontsize=13.5, color="black")
        ax.set_xlabel(r"ratio $\rho_t(\theta)$", fontsize=13)
        _black_axes(ax)
    axa.set_ylabel("clipped objective, one sample", fontsize=12.5)

    # (a) the flat stretch beyond 1 + eps, and where rho = 1 is
    axa.annotate("zero gradient: no gain\n"
                 r"past $1+\epsilon_{\mathrm{clip}}$",
                 xy=(1.62, 1.185), xytext=(2.00, 0.42), ha="right",
                 va="center", fontsize=12, color=ORANGE, linespacing=1.25,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                                 shrinkA=6, shrinkB=3, mutation_scale=13))
    axa.annotate(r"$\rho = 1$: $\theta = \theta_{\mathrm{old}}$",
                 xy=(1.0, 1.0), xytext=(0.66, -0.34), ha="center", va="center",
                 fontsize=12, color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                 shrinkA=5, shrinkB=5, mutation_scale=12))
    axa.legend(loc="lower left", fontsize=11, handlelength=1.8)

    # (b) the flat stretch below 1 - eps, and the fall that never stops
    axb.annotate("zero gradient:\nnothing more to gain",
                 xy=(0.24, -0.78), xytext=(0.06, 0.38), ha="left",
                 va="center", fontsize=12, color=ORANGE, linespacing=1.25,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                                 shrinkA=6, shrinkB=3, mutation_scale=13))
    axb.annotate("still free to fall:\n" r"the pessimistic $\min$",
                 xy=(1.78, -1.76), xytext=(1.99, -0.42), ha="right",
                 va="center", fontsize=12, color=ORANGE, linespacing=1.25,
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

    # --- acting: the behavior policy and the world ------------------------- #
    _box(ax, 2.55, 4.35, 3.5, 1.05,
         "behavior policy\n$\\epsilon$-greedy in $Q_w$", BLUE,
         fontsize=13.5, fc="#e8f1f8")
    _box(ax, 7.7, 4.35, 2.8, 1.05, "environment", GRAY, fontsize=13.5,
         fc="#f3f3f3")
    fl.arrow(ax, (4.35, 4.5), (6.25, 4.5), color=BLUE, lw=2.0, mut=15)
    ax.text(5.3, 4.62, "$a_t$", ha="center", va="bottom", fontsize=13.5,
            color=BLUE)
    fl.arrow(ax, (7.7, 3.79), (7.7, 3.32), color=GRAY, lw=1.8, mut=15)
    ax.text(7.86, 3.55, "$(s,a,r,s',\\mathrm{term})$", ha="left", va="center", fontsize=12.5,
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
    _box(ax, 8.17, 0.15, 4.14, 1.05,
         "target $Q_{w^-}$ (frozen)\n"
         "$y = r + \\gamma (1 - \\mathrm{term})\\, \\mathrm{max}_{a'}\\, "
         "Q_{w^-}(s',a')$", ORANGE,
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

    # --- and the loop closes: the online net is the behavior policy -------- #
    _elbow(ax, [(0.65, 0.15), (0.32, 0.15), (0.32, 4.35), (0.80, 4.35)], BLUE)
    ax.text(0.09, 2.40, "acts with $Q_w$", ha="center", va="center",
            fontsize=12, color=BLUE, rotation=90)

    ax.set_xlim(-0.05, 10.30)
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
    # Each name hangs off the disc's outer point, aligned away from it (the
    # corner nearest the arc touches the anchor), so no glyph meets an arc.
    centre = TRIAD_C.mean(axis=0)
    aligns = [("right", "top"), ("left", "top"), ("center", "bottom")]
    for (cx, cy), (name, color), (ha, va) in zip(TRIAD_C, names, aligns):
        u = np.array([cx, cy]) - centre
        u = u / np.linalg.norm(u)
        p = np.array([cx, cy]) + u * (TRIAD_R + 0.20)
        ax.text(p[0], p[1], name, ha=ha, va=va, fontsize=13.5,
                color=color, zorder=6, multialignment="center")

    # region key -> (label, extra offset from the computed anchor)
    regions = {
        "100": ("REINFORCE\nwith a network", (0.0, 0.00)),
        "010": ("tabular\nSARSA", (0.0, 0.00)),
        "001": ("importance-sampled\nMonte Carlo", (0.0, 0.00)),
        "110": ("actor-critic,\nlinear TD", (0.0, 0.00)),
        "101": ("behavior\ncloning", (0.0, 0.00)),
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

    ax.text(0.0, -3.55, "drop one ingredient and a convergence argument "
                        "becomes available,\neach under its own conditions",
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
             fontsize=12.5, color="black", zorder=6,
             bbox=dict(fc="white", ec="none", alpha=0.92, pad=1.4))
    # the two distributions are named where they are, not in a legend box
    axa.text(-3.45, 0.30, "one estimate", ha="left", va="center", fontsize=12.5,
             color="black")
    axa.text(2.55, 0.32, "max of four", ha="left", va="center",
             fontsize=12.5, color=ORANGE)
    axa.text(-0.16, 0.610, "$\\max_a \\mathbb{E}[\\hat Q] = 0$", ha="right",
             va="center", fontsize=12.5, color="black")
    axa.text(bias4 + 0.16, 0.610,
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
    axb.plot(ks, single, "o-", color=BLUE, lw=2.0, ms=6, zorder=3)
    axb.plot(ks, double, "s-", color=ORANGE, lw=2.0, ms=6, zorder=3)
    # each curve is named beside itself, not in a legend box
    axb.text(6.4, 1.06, "single estimator (the DQN target)", ha="left",
             va="center", fontsize=12, color=BLUE)
    _leader(axb, "the case on the left", (4, single[2]), (6.4, 0.60), "black",
            fontsize=11.5, ha="left")
    assert abs(double).max() < 0.005              # zero within MC error
    axb.text(16.6, 0.085, "independent double estimator:\nbias "
                          "indistinguishable from zero", ha="right",
             va="bottom", fontsize=12, color=ORANGE)
    axb.set_xlim(1.2, 16.8)
    axb.set_ylim(-0.14, 2.06)
    axb.set_xticks(ks)
    axb.set_xlabel("number of actions", fontsize=13)
    axb.set_ylabel("bias of the maximum", fontsize=13)
    _black_axes(axb)

    fig.subplots_adjust(wspace=0.24)
    fl.save(fig, "mdl-rl-max-bias")
    return bias4, ks, single, double

def fig_data_rules():         # F19 -> mdl-rl-data-rules
    """Three data regimes in one vocabulary: a policy, a world, a data object,
    an update, and the dashed arrow that carries the update back to the policy.
    The panels share their geometry exactly, so what differs between them is
    what the section is about."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.3))
    XL, YL = (-0.40, 6.85), (0.30, 7.25)
    PBOX, EBOX = (1.75, 5.55), (5.35, 5.55)      # policy, environment
    DATA = (5.35, 3.95)                          # the data object
    UBOX, UW, UH = (2.55, 1.95), 4.0, 1.0        # the update
    UTOP = UBOX[1] + UH / 2
    titles = ("on-policy: REINFORCE, actor-critic, PPO",
              "off-policy: Q-learning, DQN",
              "offline: no interaction at all")
    updates = ("update an expectation\nunder $\\pi_\\theta$",
               "update the target\n"
               "$r + \\gamma\\, \\mathrm{max}_{a'}\\, Q(s',a')$",
               "update the same target,\non $\\mathcal{D}$ only")

    for ax, title, upd in zip(axes, titles, updates):
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(*XL)
        ax.set_ylim(*YL)
        ax.text(3.25, 7.22, title, ha="center", va="top", fontsize=13.5,
                color="black")
        _box(ax, *PBOX, 2.7, 0.8, "policy $\\pi_\\theta$", BLUE, fontsize=13,
             fc="#e8f1f8")
        _box(ax, *UBOX, UW, UH, upd, GREEN, fontsize=12.5, fc="#eaf4ea")
        # the update's product goes back to the policy: same edge in all three
        x_left = UBOX[0] - UW / 2
        _elbow(ax, [(x_left, UBOX[1]), (0.12, UBOX[1]), (0.12, PBOX[1]),
                    (0.40, PBOX[1])], GREEN)
        ax.text(-0.17, 3.75, "new policy", ha="center", va="center",
                fontsize=12, color=GREEN, rotation=90)

    axa, axb, axc = axes
    notes = ("PPO's ratios allow a few\nepochs of reuse",
             "the target does not mention\nwho collected the data",
             "errors are never tested: no mistake\n"
             "the learned policy makes is discovered")
    for ax, note in zip(axes, notes):
        ax.text(3.25, UBOX[1] - UH / 2 - 0.14, note, ha="center", va="top",
                fontsize=12, color="black")

    # --- (a) on-policy: the batch expires when the parameters move ---------- #
    _box(axa, *EBOX, 2.4, 0.8, "environment", GRAY, fontsize=13, fc="#f3f3f3")
    fl.arrow(axa, (3.10, 5.55), (4.15, 5.55), color=BLUE, lw=1.8, mut=14)
    axa.text(3.62, 5.67, "$a_t$", ha="center", va="bottom", fontsize=13,
             color=BLUE)
    fl.arrow(axa, (5.35, 5.15), (5.35, 4.25), color=GRAY, lw=1.8, mut=14)
    _box(axa, *DATA, 2.8, 0.6, "one batch $\\sim \\pi_\\theta$", BLUE,
         fontsize=12.5, fc="#e8f1f8")
    fl.arrow(axa, (4.75, 3.65), (4.05, UTOP), color=BLUE, lw=1.8, mut=14)
    fl.arrow(axa, (6.10, 3.65), (6.28, 3.20), color=GRAY, lw=1.5, ls="--",
             mut=13)
    axa.text(6.32, 3.02, "$\\times$", ha="center", va="center", fontsize=19,
             color=GRAY)
    axa.text(6.80, 2.74, "used once,\nthen stale", ha="right", va="top",
             fontsize=12, color="black")

    # --- (b) off-policy: every version's data still counts ------------------ #
    _box(axb, *EBOX, 2.4, 0.8, "environment", GRAY, fontsize=13, fc="#f3f3f3")
    fl.arrow(axb, (3.10, 5.55), (4.15, 5.55), color=BLUE, lw=1.8, mut=14)
    axb.text(3.62, 5.67, "$a_t$", ha="center", va="bottom", fontsize=13,
             color=BLUE)
    TX = 4.55                                       # the stack's centre
    ys = DATA[1] - 0.10 + 0.46 * np.arange(-2, 3)   # ascending: oldest at the
    fl.arrow(axb, (5.35, 5.15), (5.35, ys[-1] + 0.20), color=GRAY, lw=1.8,
             mut=12)
    # a mid-tone newest tile keeps the black labels readable on every tile
    mid = tuple(0.5 + 0.5 * np.array(to_rgb(BLUE)))
    _tiles(axb, ys, TX, 1.9, 0.40, c1=mid,          # bottom, newest on top
           vertical=True)
    for y, lab in zip(ys, ["$\\pi_1$", "$\\pi_2$", "$\\ldots$",
                           "$\\pi_{k-1}$", "$\\pi_k$"]):
        axb.text(TX, y + 0.05, lab, ha="center", va="center", fontsize=11,
                 color="black", zorder=5)
    # every version's tile feeds one merged draw into the update
    merge = (3.05, 3.00)
    for y in ys:
        axb.plot([TX - 0.95, merge[0]], [y, merge[1]], color=BLUE, lw=1.2,
                 zorder=3, solid_capstyle="round")
    fl.arrow(axb, merge, (merge[0], UTOP), color=BLUE, lw=1.8, mut=14)
    axb.text(5.65, ys[-1], "newest", ha="left", va="center", fontsize=12,
             color="black")
    axb.text(5.65, ys[0], "oldest", ha="left", va="center", fontsize=12,
             color="black")

    # --- (c) offline: the arrow to the world is gone ------------------------ #
    _box(axc, *EBOX, 2.4, 0.8, "environment", LIGHT, fontsize=13, tc=GRAY)
    # the wall: it separates the policy from the world, and only the deploy
    # arrow ever crosses it
    axc.plot([4.05, 4.05], [2.75, 6.70], color="black", lw=3.0,
             ls=(0, (5.0, 3.0)), zorder=5)
    fl.arrow(axc, (4.75, 4.35), (4.14, 4.35), color="black", lw=1.2, mut=13)
    axc.text(4.85, 4.35, "no interaction", ha="left", va="center",
             fontsize=12.5, color="black")
    fl.arrow(axc, (3.10, 5.55), (3.72, 5.55), color=LIGHT, lw=1.8, ls="--",
             mut=14)
    axc.text(4.05, 5.55, "$\\times$", ha="center", va="center", fontsize=19,
             color="black", zorder=6,
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))
    _box(axc, 2.10, DATA[1], 2.6, 0.9,
         "$\\mathcal{D} \\sim \\pi_e$\ncollected once", GRAY,
         fontsize=12.5, fc="#f3f3f3")
    fl.arrow(axc, (2.95, 3.50), (3.15, UTOP), color=GRAY, lw=1.8, mut=14)
    _arc_arrow(axc, (0.95, UTOP), (1.25, 3.50), 0.42, GRAY, lw=1.6, mut=14)
    axc.text(1.55, 2.97, "sweep", ha="left", va="center", fontsize=12,
             color="black")
    # deploy: the only edge that crosses the wall, into the world at last
    _elbow(axc, [(1.75, 5.95), (1.75, 6.60), (5.35, 6.60), (5.35, 5.97)],
           "black", lw=1.6)
    axc.text(2.95, 6.50, "deploy", ha="center", va="top", fontsize=12,
             color="black")
    axc.text(5.50, 6.29, "?", ha="left", va="center", fontsize=15,
             color="black", fontweight="bold")

    fig.subplots_adjust(wspace=0.04)
    fl.save(fig, "mdl-rl-data-rules")

def fig_token_mdp():          # F25 -> mdl-rl-token-mdp
    """A response is a trajectory: (a) the token Markov decision process, whose
    transitions all have probability 1 and whose reward arrives only at the
    end; (b) the same object read as a one-step bandit.  The right column says
    what the response-level update does without, and what per-token credit
    may still use.  Tokens are actions $a_t$; $y$ is the complete response."""
    fig = plt.figure(figsize=(11.8, 4.0))
    gs = GridSpec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[3.1, 2.27],
                  figure=fig, wspace=0.06, hspace=0.10, top=0.98, bottom=0.02)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), \
        fig.add_subplot(gs[:, 1])
    for ax in (axa, axb, axc):
        ax.axis("off")
    axa.set_anchor("S")
    axb.set_anchor("N")

    # --- (a) the token MDP -------------------------------------------------- #
    axa.set_aspect("equal")
    Y = 2.05
    xs = [0.75, 2.65, 4.55, 7.15]
    slabels = ["$x$", "$(x, a_1)$", "$(x, a_{1:2})$", "$(x, a_{1:T-1})$"]
    tokens = ["$a_1$", "$a_2$", None, "$a_T = \\mathrm{EOS}$"]
    xe = 9.35                                        # the terminal node
    for i, (x, lab) in enumerate(zip(xs, slabels)):
        _state(axa, (x, Y), r=0.15)
        axa.text(x, Y - 0.44, lab, ha="center", va="top", fontsize=12.5,
                 color="black")
    axa.add_patch(Rectangle((xe - 0.17, Y - 0.17), 0.34, 0.34, fc="black",
                            ec="black", lw=1.4, zorder=5))
    axa.text(xe, Y - 0.44, "terminal", ha="center", va="top",
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
                 fontsize=13, color="black")
    rx = xs[-1] + 0.80 * (xe - xs[-1])         # clear of the token label
    axa.text(rx, Y + 0.60, "$r(x, y)$", ha="center", va="bottom",
             fontsize=13.5, color=ORANGE)
    fl.arrow(axa, (rx, Y + 0.56), (rx, Y + 0.12), color=ORANGE, lw=1.6, mut=13)
    axa.text(0.75, Y + 0.34, "prompt", ha="center", va="bottom", fontsize=12.5,
             color="black")
    axa.text(4.9, 1.12, "state $s_t =$ prefix,   action $a_t =$ token,   "
                        "transition $=$ concatenation", ha="center", va="top",
             fontsize=13, color="black")
    axa.text(4.9, 0.68, "every transition has probability $1$: all the "
                        "randomness is the policy's,\nand the reward arrives "
                        "only when the response ends",
             ha="center", va="top", fontsize=12.5, color="black")
    axa.set_xlim(-0.05, 10.15)
    axa.set_ylim(0.05, 3.15)

    # --- (b) the same object as one step ------------------------------------ #
    axb.set_aspect("equal")
    Yb, xb0, xb1 = 1.25, 2.05, 7.25
    _state(axb, (xb0, Yb), r=0.15, label="$x$")
    axb.add_patch(Rectangle((xb1 - 0.17, Yb - 0.17), 0.34, 0.34, fc="black",
                            ec="black", lw=1.4, zorder=5))
    fl.arrow(axb, (xb0 + 0.19, Yb), (xb1 - 0.21, Yb), color=GRAY, lw=1.6,
             mut=14)
    axb.text((xb0 + xb1) / 2, Yb + 0.14,
             "$y = (a_1, \\ldots, a_T) \\sim \\pi_\\theta(\\cdot \\mid x)$",
             ha="center", va="bottom", fontsize=13, color=BLUE)
    axb.text((xb0 + xb1) / 2, Yb - 0.12, "$1$", ha="center", va="top",
             fontsize=13, color="black")
    axb.text(xb1 + 0.34, Yb, "$r(x, y)$", ha="left", va="center",
             fontsize=13.5, color=ORANGE)
    axb.text(4.9, 0.30, "$\\nabla_\\theta \\log \\pi_\\theta(y \\mid x) = "
                        "\\sum_t \\nabla_\\theta \\log \\pi_\\theta"
                        "(a_t \\mid s_t)$: the two views, one gradient",
             ha="center", va="top", fontsize=13, color="black")
    axb.set_xlim(-0.05, 10.15)
    axb.set_ylim(-0.42, 1.85)

    # --- the right column: not needed, and optional ------------------------- #
    # The equal-aspect panels shrink inside their cells; the column's axes is
    # fitted to their joint height so the tight crop leaves no empty band.
    axa.apply_aspect()
    axb.apply_aspect()
    top, bot = axa.get_position().y1, axb.get_position().y0
    pc = axc.get_position()
    axc.set_position([pc.x0, bot, pc.width, top - bot])
    x0, dy = 0.07, 0.072
    rows = [("head", "not needed", RED)]
    rows += [("item", t, RED) for t in NOT_NEEDED]
    rows += [("gap", None, None), ("head", "optional", GREEN),
             ("text", "a prefix value $V(s_t)$", None),
             ("text", "for per-token credit, from", None)]
    rows += [("sub", t, GREEN) for t in OPTIONAL]
    y = 0.5 + dy * (len(rows) - 1) / 2               # centred on the left half
    for kind, text, color in rows:
        if kind == "head":
            axc.text(x0 - 0.05, y, text, ha="left", va="center",
                     fontsize=13.5, fontweight="bold", color=color,
                     transform=axc.transAxes)
        elif kind in ("item", "sub", "text"):
            dx = 0.05 if kind == "sub" else 0.0
            axc.text(x0 + dx, y, text, ha="left", va="center", fontsize=12,
                     color="black", transform=axc.transAxes)
            if kind != "text":
                axc.plot([x0 + dx - 0.028], [y], marker="s", ms=4.5,
                         color=color, transform=axc.transAxes, clip_on=False)
        y -= dy
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)

    fl.save(fig, "mdl-rl-token-mdp")


def fig_squashed_density():   # F8 (ch. 15) -> mdl-rl-squashed-density
    """Squashing a Gaussian through $a = 2\\tanh u$: (a) the Gaussian of $u$,
    (b) the density of $a$ with the Jacobian term and without it.  The areas
    are computed by the same trapezoid rule as the quadrature cell of the SAC
    section, so the figure and the printout quote the same numbers."""
    MU, SIG, C = 0.7, 0.8, 2.0
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 3.7))

    # --- (a) the Gaussian before the squash --------------------------------- #
    u = np.linspace(-2.6, 4.0, 801)
    pu = np.exp(-0.5 * ((u - MU) / SIG) ** 2) / (SIG * np.sqrt(2 * np.pi))
    axa.fill_between(u, pu, color=BLUE, alpha=0.15, lw=0.0, zorder=1)
    axa.plot(u, pu, color=BLUE, lw=2.4, zorder=3)
    axa.text(MU + 1.05, 0.40, f"$u \\sim \\mathcal{{N}}({MU:g},\\, {SIG:g}^2)$",
             ha="left", va="center", fontsize=13, color="black")
    axa.set_xlim(u[0], u[-1])
    axa.set_ylim(0.0, 0.62)
    axa.set_yticks([0.0, 0.2, 0.4, 0.6])
    axa.set_xlabel("$u$", fontsize=14)
    axa.set_ylabel("density", fontsize=13)
    _black_axes(axa)

    # --- (b) the density of the squashed action ----------------------------- #
    a = np.linspace(-C + 1e-6, C - 1e-6, 200_001)
    ua = np.arctanh(a / C)
    pa_naive = np.exp(-0.5 * ((ua - MU) / SIG) ** 2) / (SIG * np.sqrt(2 * np.pi))
    pa = pa_naive / (C * (1.0 - np.tanh(ua) ** 2))     # divide by da/du
    area, area_naive = np.trapezoid(pa, a), np.trapezoid(pa_naive, a)
    assert abs(area - 1.0) < 1e-4 and abs(area_naive - 1.13) < 0.005
    for x in (-C, C):
        axb.axvline(x, color=LIGHT, lw=1.2, ls="--", zorder=0)
    axb.fill_between(a, pa, color=BLUE, alpha=0.15, lw=0.0, zorder=1)
    axb.plot(a, pa, color=BLUE, lw=2.4, zorder=3)
    axb.plot(a, pa_naive, color=ORANGE, lw=2.2, ls=(0, (5, 2.5)), zorder=2)
    axb.text(-1.85, 0.66, f"with the Jacobian:\narea ${area:.2f}$",
             ha="left", va="center", fontsize=12.5, color=BLUE,
             linespacing=1.3)
    axb.text(-1.85, 0.32, f"without it:\narea ${area_naive:.2f}$",
             ha="left", va="center", fontsize=12.5, color=ORANGE,
             linespacing=1.3)
    axb.set_xlim(-2.2, 2.2)
    axb.set_ylim(0.0, 1.00)
    axb.set_xticks([-2, -1, 0, 1, 2])
    axb.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xlabel("$a = 2 \\tanh u$", fontsize=14)
    _black_axes(axb)
    fig.subplots_adjust(wspace=0.22)
    fl.save(fig, "mdl-rl-squashed-density")


def _between_rows(p, d, y_from, y_to):
    """Tail and tip of the segment of the line ``p + t d`` that runs from the
    horizontal edge ``y = y_from`` to the edge ``y = y_to``."""
    tail = p + (y_from - p[1]) / d[1] * d
    tip = p + (y_to - p[1]) / d[1] * d
    return tuple(tail), tuple(tip)

def fig_sac_update():         # F9 (ch. 15) -> mdl-rl-sac-update
    """One SAC update as a graph, in the box vocabulary of the DQN data flow.
    Top row: the soft target, from the target critics at a fresh action at
    $s'$.  Middle row: the critics regress onto it.  Bottom row: the actor's
    loss reaches the actor through the fresh action $\tilde a$ (the pathwise
    gradient, red), and the temperature has a loss of its own.  The Polyak
    arrow is the only edge that writes into the target critics."""
    fig, ax = plt.subplots(figsize=(12.0, 5.3))
    ax.axis("off")
    ax.set_aspect("equal")
    GRAD = dict(color=RED, lw=1.8, ls=(0, (4.5, 2.6)))
    XB, XA, XC, XL = 0.95, 3.3, 6.95, 11.3   # buffer, actor, critics, losses
    YT, YM, YB = 4.85, 2.75, 0.65            # target, critic, actor rows
    BW, BH, LW = 2.7, 0.95, 4.3

    # --- the boxes ---------------------------------------------------------- #
    _box(ax, XB, YM, 1.75, 1.15, "replay\nbuffer", GRAY, fontsize=13.5,
         fc="#f3f3f3")
    _box(ax, XA, YT, 2.2, BH, "actor $\\pi_\\theta$", BLUE, fontsize=14,
         fc="#e8f1f8")
    _box(ax, XC, YT, BW, BH, "target critics\n$Q_{w_1^-},\\ Q_{w_2^-}$",
         ORANGE, fontsize=13, fc="#fdf1e3")
    _box(ax, XC, YM, BW, BH, "critics\n$Q_{w_1},\\ Q_{w_2}$", GREEN,
         fontsize=13, fc="#eaf6ea")
    _box(ax, XL, YT, LW, 1.3,
         "soft target $y$\n"
         "$r + \\gamma (1 - \\mathrm{term})\\,(\\min_j Q_{w_j^-}(s', \\tilde a')$\n"
         "$-\\ \\beta \\log \\pi_\\theta(\\tilde a' \\mid s'))$",
         ORANGE, fontsize=12.5)
    _box(ax, XL, YM, LW, BH,
         "critic loss\n$\\sum_j (Q_{w_j}(s, a) - y)^2$", GREEN, fontsize=12.5)
    _box(ax, XL, YB, LW, BH,
         "actor loss\n$\\beta \\log \\pi_\\theta(\\tilde a \\mid s) - \\min_j"
         " Q_{w_j}(s, \\tilde a)$", BLUE, fontsize=12.5)
    TW = BW + 0.4
    _box(ax, XC, YB, TW, BH,
         "temperature loss\n$-\\log \\beta\\, (\\log \\pi_\\theta"
         "(\\tilde a \\mid s) + \\bar{\\mathcal{H}})$", PURPLE, fontsize=12.5)

    # --- data into the update ----------------------------------------------- #
    fl.arrow(ax, (XB + 0.875, YM), (XC - BW / 2, YM), color=GRAY, lw=1.9,
             mut=15)
    ax.text(3.0, YM - 0.10, "$(s, a)$", ha="center", va="top",
            fontsize=13, color="black")
    _elbow(ax, [(XB, YM + 0.575), (XB, YT), (XA - 1.1, YT)], GRAY, lw=1.9,
           ls="-")
    ax.text(XB + 0.12, YT - 0.40, "$s,\\ s'$", ha="left", va="center",
            fontsize=13, color="black")
    ax.text(XB, YM - 0.70, "minibatch\n$(s, a, r, s', \\mathrm{term})$",
            ha="center", va="top", fontsize=12, color="black",
            linespacing=1.3)

    # --- the actor's two fresh actions -------------------------------------- #
    fl.arrow(ax, (XA + 1.1, YT), (XC - BW / 2, YT), color=BLUE, lw=1.9,
             mut=15)
    ax.text((XA + 1.1 + XC - BW / 2) / 2, YT + 0.10, "$\\tilde a'$ at $s'$",
            ha="center", va="bottom", fontsize=13, color=BLUE)
    p0 = np.array([XA + 0.55, YT - BH / 2])
    p1 = np.array([XC - 0.80, YM + BH / 2])
    fl.arrow(ax, tuple(p0), tuple(p1), color=BLUE, lw=1.9, mut=15)
    n = (p1 - p0) / np.linalg.norm(p1 - p0)
    nrm = np.array([-n[1], n[0]])                 # points up and to the right
    mid = (p0 + p1) / 2
    ax.text(*(mid - 0.30 * nrm), "$\\tilde a$ at $s$", ha="left", va="top",
            fontsize=13, color=BLUE)
    # the pathwise gradient runs back along the same edge, offset to its
    # upper side and cut at the two box edges it runs between
    fl.arrow(ax, *_between_rows(p0 + 0.24 * nrm, n, YM + BH / 2, YT - BH / 2),
             **GRAD, mut=14)
    ax.text(3.62, 3.80, "pathwise\ngradient", ha="right", va="center",
            fontsize=12.5, color=RED, linespacing=1.2)

    # --- into the losses ----------------------------------------------------- #
    fl.arrow(ax, (XC + BW / 2, YT), (XL - LW / 2, YT), color=ORANGE, lw=1.9,
             mut=15)
    ax.text((XC + BW / 2 + XL - LW / 2) / 2, YT + 0.10, "$\\min_j$",
            ha="center", va="bottom", fontsize=13, color="black")
    fl.arrow(ax, (XL, YT - 0.65), (XL, YM + BH / 2), color=ORANGE, lw=1.9,
             mut=15)
    ax.text(XL + 0.12, (YT - 0.65 + YM + BH / 2) / 2, "$y$", ha="left",
            va="center", fontsize=13.5, color="black")
    fl.arrow(ax, (XC + BW / 2, YM), (XL - LW / 2, YM), color=GREEN, lw=1.9,
             mut=15)
    q0 = np.array([XC + 0.80, YM - BH / 2])
    q1 = np.array([XL - 1.30, YB + BH / 2])
    fl.arrow(ax, tuple(q0), tuple(q1), color=GREEN, lw=1.9, mut=15)
    m = (q1 - q0) / np.linalg.norm(q1 - q0)
    mrm = np.array([m[1], -m[0]])                # points down and to the left
    fl.arrow(ax, *_between_rows(q0 + 0.24 * mrm, m, YB + BH / 2, YM - BH / 2),
             **GRAD, mut=14)
    # the actor loss hands its log-probability to the temperature
    fl.arrow(ax, (XL - LW / 2, YB), (XC + TW / 2, YB), color=BLUE, lw=1.6,
             mut=14)

    # --- the slow copy -------------------------------------------------------- #
    fl.arrow(ax, (XC + 0.95, YM + BH / 2), (XC + 0.95, YT - BH / 2),
             color=ORANGE, lw=1.6, ls=(0, (4.5, 2.6)), mut=14)
    ax.text(XC + 1.07, (YT + YM) / 2, "Polyak,\nrate $\\nu$", ha="left",
            va="center", fontsize=12, color="black", linespacing=1.2)

    ax.set_xlim(0.0, 13.5)
    ax.set_ylim(0.1, 5.6)
    fl.save(fig, "mdl-rl-sac-update")


def _plan_rollout(y0, u):
    """The schematic model of the planning figure: one coordinate, the
    distance from upright, pushed down by the action, so the fan and its
    scores are computed rather than drawn freehand.  Returns the predicted
    states and the predicted return (state cost plus a small action cost)."""
    y = [y0]
    for a in u:
        y.append(max(y[-1] - 0.05 * a * np.sqrt(y[-1]), 0.0))
    y = np.array(y)
    ret = -20.0 * float(5.0 * (y[1:] ** 2).mean() + 0.02 * (u ** 2).sum())
    return y, ret

def fig_model_planning():     # F10 (ch. 15) -> mdl-rl-model-planning
    """Random shooting with a warm start, in the learned model.  (a) At time
    $t$: candidate action sequences are drawn around the previous plan shifted
    by one step (dashed), rolled out through the model and scored by the sum
    of predicted rewards; only the first action of the best one (blue) is
    executed.  (b) One real step later the agent is at the real $s_{t+1}$,
    a little off the prediction, and plans again around the best plan of (a)
    shifted by one step.  (c) The model's open-loop error grows with the
    horizon (schematic), which is one reason to replan at every step.  No
    critic appears anywhere."""
    H = 25
    k = np.arange(H)
    decay = np.exp(-k / 12.0)
    wig = np.sin(np.pi * k / H)
    PERT_A = [(-0.85, 0.25), (-0.40, -0.35), (0.20, 0.30), (0.50, -0.25),
              (0.95, 0.30)]
    PERT_B = [(-0.80, 0.30), (-0.45, -0.30), (-0.15, 0.30), (0.15, -0.30),
              (0.40, 0.30)]

    fig = plt.figure(figsize=(12.2, 3.6))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.55], figure=fig,
                  wspace=0.34)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    xs = np.arange(H + 1)

    def panel(ax, y0, centre, perts, label_s):
        spine, _ = _plan_rollout(y0, centre)
        cands = [np.clip(centre + d * decay + w * wig, 0.0, None)
                 for d, w in perts]
        rolls = [_plan_rollout(y0, u) for u in cands]
        best = int(np.argmax([r for _, r in rolls]))
        assert 0 < best < len(perts) - 1          # an interior winner
        ax.plot(xs, spine, color=GRAY, lw=1.8, ls=(0, (5, 3)), zorder=3)
        for i, (y, _) in enumerate(rolls):
            ax.plot(xs, y, color=BLUE if i == best else LIGHT,
                    lw=2.6 if i == best else 1.6, zorder=4 if i == best else 2)
        # the scores at the ends, spread apart (and off the axis) where two
        # ends nearly coincide; a short leader joins a moved label to its end
        ends = sorted((y[-1], i) for i, (y, _) in enumerate(rolls))
        pos, GAPL = np.array([max(e, 0.07) for e, _ in ends]), 0.075
        for _ in range(200):                      # relax overlapping pairs
            gap = np.diff(pos)
            j = int(np.argmin(gap))
            if gap[j] >= GAPL - 1e-9:
                break
            push = (GAPL - gap[j]) / 2
            pos[j] -= push
            pos[j + 1] += push
            pos = np.maximum(pos, 0.07 + GAPL * np.arange(len(pos)))
        for (yend, i), yl in zip(ends, pos):
            if abs(yl - yend) > 0.01:
                ax.plot([H + 0.15, H + 0.85], [yend, yl], color=GRAY,
                        lw=0.9, zorder=1)
            ax.text(H + 1.0, yl, f"${rolls[i][1]:.0f}$", ha="left",
                    va="center", fontsize=12,
                    color=BLUE if i == best else "black")
        yb = rolls[best][0]
        ax.plot(xs[:2], yb[:2], color=BLUE, lw=5.5, solid_capstyle="round",
                zorder=6)
        ax.plot([0], [y0], "o", color="black", ms=8, zorder=7)
        ax.text(-0.9, y0, label_s, ha="right", va="center", fontsize=13.5,
                color="black")
        ax.set_xlim(-0.5, H + 4.2)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0, H])
        ax.set_yticks([])
        ax.set_xlabel("model steps ahead", fontsize=13)
        _black_axes(ax)
        ax.spines["left"].set_visible(False)
        return cands[best], yb, spine

    # --- (a) the decision at t ---------------------------------------------- #
    spine_a = 1.6 * decay
    best_a, yb, _ = panel(axa, 0.85, spine_a, PERT_A, "$s_t$")
    axa.set_ylabel("distance from upright", fontsize=13)
    axa.annotate("execute $a_t$ only", xy=(1.05, yb[1]),
                 xytext=(2.4, 0.95), ha="left", va="center", fontsize=12.5,
                 color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE,
                                             lw=1.1, shrinkA=2, shrinkB=3))
    # an inline key in the empty upper right
    for yk, kw, txt in ((0.80, dict(color=GRAY, lw=1.8, ls=(0, (5, 3))),
                         "previous plan, shifted"),
                        (0.71, dict(color=BLUE, lw=2.6), "best candidate")):
        axa.plot([10.6, 13.0], [yk, yk], **kw)
        axa.text(13.6, yk, txt, ha="left", va="center", fontsize=12,
                 color="black")
    _letter(axa, "(a)", x=-0.10, y=1.0)

    # --- (b) one real step later --------------------------------------------- #
    y1_real = yb[1] + 0.035                           # the model was a bit off
    shifted = np.r_[best_a[1:], best_a[-1]]
    panel(axb, y1_real, shifted, PERT_B, "$s_{t+1}$")
    axb.plot([0], [yb[1]], "o", mfc="white", mec=BLUE, mew=1.6, ms=7,
             zorder=8)
    axb.annotate("predicted $s_{t+1}$", xy=(0.25, yb[1] - 0.012),
                 xytext=(2.6, 0.93), ha="left", va="center", fontsize=12,
                 color="black", arrowprops=dict(arrowstyle="->",
                                                color="black", lw=1.0,
                                                shrinkA=2, shrinkB=5))
    _letter(axb, "(b)", x=-0.10, y=1.0)

    # --- (c) error against the horizon (schematic) --------------------------- #
    kk = np.linspace(1, H, 200)
    axc.plot(kk, np.exp(kk / 5.5), color=ORANGE, lw=2.4)
    axc.set_xlim(0, H + 1)
    axc.set_ylim(0, np.exp(H / 5.5) * 1.08)
    axc.set_xticks([1, 10, 25])
    axc.set_yticks([])
    axc.set_xlabel("horizon $k$", fontsize=13)
    axc.set_ylabel("$k$-step model error", fontsize=13)
    _black_axes(axc)
    _letter(axc, "(c)", x=-0.16, y=1.0)
    fl.save(fig, "mdl-rl-model-planning")


REWARD_HAZARD = [5, 6]          # the hazard lane of the regularized section

def _reward_gridworld_data(hazard=REWARD_HAZARD, gamma=0.95):
    """The KL-regularized section's gridworld, its reference policy and the
    1,000 reference trajectories, replayed with the notebook's own seed and
    call order (rng 0, then rng.choice per step), so the counts drawn are the
    counts the notebook's fit sees."""
    P = np.zeros((16, 4, 16))
    for s, a in np.ndindex(16, 4):
        x, y = s % 4, s // 4
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
        P[s, a, min(3, max(0, y + dy)) * 4 + min(3, max(0, x + dx))] = 1.0
    P[15], P[15, :, 15] = 0.0, 1.0
    bonus = np.full(16, -0.04)
    bonus[hazard], bonus[15] = -1.5, 1.0
    r_true = np.einsum('sat,t->sa', P, bonus)
    r_true[15] = 0.0
    V = np.zeros(16)
    for _ in range(200):
        V = (r_true + gamma * P @ V).max(axis=1)
    Q = r_true + gamma * P @ V
    pi_ref = np.exp(Q / 0.25)
    pi_ref /= pi_ref.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    X, visits = [], np.zeros(16)
    for _ in range(1000):
        st, x, seen = 0, np.zeros((16, 4)), {0}
        for t in range(40):
            a = rng.choice(4, p=pi_ref[st])
            x[st, a] += gamma ** t
            st = P[st, a].argmax()
            seen.add(int(st))
            if st == 15:
                break
        for v in seen:
            visits[v] += 1
        X.append(x.ravel())
    X = np.stack(X)
    support = (X > 0).sum(0).reshape(16, 4)      # episodes containing (s, a)
    return P, bonus, r_true, support, visits

def fig_reward_gridworld():   # F5 (ch. 15) -> mdl-rl-reward-gridworld
    """(a) The map: start, goal, the hazard lane, and how many of the 1,000
    reference episodes visit each cell.  (b) Support of every state-action
    pair: each cell is split into four triangles (left, down, right, up),
    shaded by the number of episodes containing that pair on a log scale; a
    cross marks a pair that never occurs, and the eight pairs that enter the
    hazard lane are outlined.  Both panels replay the notebook's seed."""
    P, bonus, r_true, support, visits = _reward_gridworld_data()
    live = np.arange(15)
    assert (support[live] == 0).sum() + 4 == 9           # notebook: 9 pairs
    assert (support >= 50).sum() == 46                   # notebook: 46 pairs
    enter = [(s, a) for s in live for a in range(4)
             if r_true[s, a] == -1.5]
    assert len(enter) == 8
    cmap = plt.get_cmap("Blues")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.6))

    # --- (a) the map with state visits ------------------------------------- #
    for s in range(16):
        r, c = divmod(s, 4)
        if s == 15:
            fc = (*to_rgb(GREEN), 0.35)
        else:
            fc = cmap(0.08 + 0.75 * visits[s] / 1000.0)
        axa.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec="black", lw=1.2,
                                zorder=1))
        if s in REWARD_HAZARD:
            axa.add_patch(Rectangle((c + 0.05, r + 0.05), 0.9, 0.9, fc="none",
                                    ec=RED, lw=2.6, zorder=3))
        if s != 15:
            dark = visits[s] / 1000.0 > 0.6
            yc = r + (0.42 if s in REWARD_HAZARD else 0.55)
            axa.text(c + 0.5, yc, f"{visits[s]:.0f}", ha="center",
                     va="center", fontsize=13,
                     color="white" if dark else "black", zorder=4)
        if s in REWARD_HAZARD:
            axa.text(c + 0.5, r + 0.74, "$-1.5$", ha="center", va="center",
                     fontsize=12.5, color=RED, zorder=4)
    axa.text(0.10, 0.10, "S", ha="left", va="top", fontsize=13,
             fontweight="bold", color="white", zorder=4)
    axa.text(3.5, 3.5, "G\n$+1$", ha="center", va="center", fontsize=13,
             color="black", zorder=4, linespacing=1.2)
    axa.set_xlim(-0.02, 4.02)
    axa.set_ylim(4.02, -0.02)
    axa.set_aspect("equal")
    axa.axis("off")
    _letter(axa, "(a)", x=0.0, y=1.01)

    # --- (b) the support of every pair -------------------------------------- #
    # triangle of action a in cell (r, c): centre and the two corners on the
    # side the action moves towards (left, down, right, up; y grows downward)
    side = {0: [(0, 0), (0, 1)], 1: [(0, 1), (1, 1)], 2: [(1, 1), (1, 0)],
            3: [(1, 0), (0, 0)]}
    lmax = np.log10(1001.0)
    for s in range(16):
        r, c = divmod(s, 4)
        if s == 15:
            axb.add_patch(Rectangle((c, r), 1, 1, fc=(*to_rgb(GREEN), 0.35),
                                    ec="black", lw=1.2, zorder=1))
            axb.text(c + 0.5, r + 0.5, "G", ha="center", va="center",
                     fontsize=13, color="black", zorder=4)
            continue
        for a in range(4):
            tri = [(c + 0.5, r + 0.5)] + [(c + dx, r + dy)
                                          for dx, dy in side[a]]
            n = support[s, a]
            fc = cmap(0.08 + 0.80 * np.log10(1.0 + n) / lmax) if n else "white"
            axb.add_patch(plt.Polygon(tri, closed=True, fc=fc, ec="white",
                                      lw=0.8, zorder=1))
            cx = np.mean([p[0] for p in tri])
            cy = np.mean([p[1] for p in tri])
            if n == 0:
                axb.text(cx, cy, "$\\times$", ha="center", va="center",
                         fontsize=14, color="black", zorder=4)
            if (s, a) in enter:          # inset, so it never sits on a border
                inset = [(cx + 0.78 * (px - cx), cy + 0.78 * (py - cy))
                         for px, py in tri]
                axb.add_patch(plt.Polygon(inset, closed=True, fc="none",
                                          ec=RED, lw=1.8, zorder=3))
        axb.add_patch(Rectangle((c, r), 1, 1, fc="none", ec="black", lw=1.2,
                                zorder=2))
    axb.set_xlim(-0.02, 4.02)
    axb.set_ylim(4.02, -0.02)
    axb.set_aspect("equal")
    axb.axis("off")
    _letter(axb, "(b)", x=0.0, y=1.01)

    # a slim key for the shading of (b)
    cax = fig.add_axes([0.915, 0.20, 0.012, 0.60])
    grad = np.linspace(0.0, lmax, 256)[:, None]
    cax.imshow(cmap(0.08 + 0.80 * grad / lmax), aspect="auto",
               origin="lower", extent=(0, 1, 0, lmax))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks(np.log10([2, 11, 101, 1001]))
    cax.set_yticklabels(["1", "10", "100", "1000"])
    cax.tick_params(colors="black", labelsize=11)
    cax.set_ylabel("episodes containing the pair", fontsize=12)
    cax.yaxis.set_label_position("right")
    fig.subplots_adjust(left=0.01, right=0.90, wspace=0.08)
    fl.save(fig, "mdl-rl-reward-gridworld")


FIGURES = [
    fig_agent_env,
        fig_roadmap,
        fig_gridworld,
        fig_return_discount,
        fig_termination,
        fig_backups,
        fig_contraction,
        fig_value_wavefront,
        fig_gpi,
        fig_compounding_error,
        fig_exploration,
        fig_ucb,
        fig_score_ascent,
        fig_trajectory_chain,
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
        fig_token_mdp,
        fig_squashed_density,
        fig_sac_update,
        fig_model_planning,
        fig_reward_gridworld,
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
