// Slide diagrams for chapter_reinforcement-learning (§14 Foundations of RL).
import { C, TOKENS, grid, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// ═══════════════════════════════════════════════════════════════════════════
// §14.0 index.md (3 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 1. ch14-rl-delayed-credit
function delayedCredit() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Delayed Credit Assignment Across Horizon H = 1 / (1 − γ)', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(20, 38, 520, 46, N.panel, C.gray, 7, 1.8);
  o += tx(280, 50, 'Supervised Learning: immediate label y_t at every step t', { fs: 13.5, fw: 700, fill: C.ink });
  o += tx(280, 72, 'x₀ → y₀ (loss L₀)     x₁ → y₁ (loss L₁)     x₂ → y₂ (loss L₂)', { mono: true, fs: 12, fill: C.muted });

  o += box(20, 98, 520, 150, C.lblue, C.blue, 9, 2);
  o += tx(280, 118, 'Reinforcement Learning: sparse reward arrives only at goal', { fs: 14, fw: 700, fill: D.blue.dark });

  const xs = [74, 198, 322, 462], ny = 158;
  const labs = ['s₀, a₀', 's₁, a₁', 's₂, a₂', 's_T Goal'];
  const rews = ['r₀ = 0', 'r₁ = 0', 'r₂ = 0', 'r_T = +1'];
  xs.forEach((cx, i) => {
    const isGoal = i === 3;
    const bw = isGoal ? 96 : 84;
    o += box(cx - bw / 2, ny - 22, bw, 44, isGoal ? C.lgreen : '#FFFFFF', isGoal ? C.green : C.blue, 6, 2);
    o += tx(cx, ny - 9, labs[i], { mono: true, fs: 12.5, fw: 700, fill: isGoal ? D.green.dark : C.ink });
    o += tx(cx, ny + 9, rews[i], { mono: true, fs: 12, fw: 600, fill: isGoal ? D.green.dark : C.muted });
  });
  o += arrow(118, ny, 154, ny, C.blue);
  o += arrow(242, ny, 278, ny, C.blue);
  o += arrow(366, ny, 412, ny, C.blue, true);

  const by = 212;
  o += arrow(412, by, 118, by, D.orange.base);
  o += tx(280, 232, 'Bellman backup propagates discounted credit γᵀ⁻ᵗ · r_T backward', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// 2. ch14-rl-verifier-benchmark
function verifierBenchmark() {
  const W = 560, H = 268;
  let o = '';
  o += tx(280, 20, '4-Prompt Arithmetic Verifier Benchmark (9 Candidates)', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(20, 42, 140, 156, C.lblue, C.blue, 8, 2);
  o += tx(90, 62, 'Prompts x', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(90, 88, '2+2 → "4"', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(90, 112, '3*4 → "12"', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(90, 140, '8*8 (unsolv)', { mono: true, fs: 12, fw: 600, fill: D.red.dark });
  o += tx(90, 164, '9*9 (unsolv)', { mono: true, fs: 12, fw: 600, fill: D.red.dark });
  o += tx(90, 184, 'Ceiling: 50%', { fs: 12.5, fw: 700, fill: C.ink });

  o += arrow(164, 120, 200, 120, C.blue);

  o += box(204, 42, 160, 156, D.purple.tint, C.purple, 8, 2);
  o += tx(284, 62, '9 Candidates y', { fs: 14.5, fw: 700, fill: D.purple.dark });
  o += tx(284, 88, '8 numeric answers:', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(284, 108, '"0".."12" (θ_ref=0)', { mono: true, fs: 11.5, fill: C.muted });
  o += box(216, 126, 136, 58, C.lamber, C.amber, 6, 1.8);
  o += tx(284, 144, '9th: Hedge reply', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(284, 166, '"4, 12, 64 or 81"', { mono: true, fs: 11.5, fw: 700, fill: C.ink });

  o += arrow(368, 86, 404, 86, C.green);
  o += arrow(368, 154, 404, 154, D.red.base);

  o += box(408, 42, 132, 72, C.lgreen, C.green, 8, 2);
  o += tx(474, 62, 'Exact Checker', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(474, 82, 'Hedge → r = 0', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(474, 100, 'Max score 0.50', { fs: 12, fill: D.green.dark });

  o += box(408, 126, 132, 72, D.red.tint, D.red.base, 8, 2);
  o += tx(474, 146, 'Sloppy Checker', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(474, 166, 'Hedge → r = 1', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(474, 184, 'Reward hacked!', { fs: 12, fw: 700, fill: D.red.dark });

  o += box(20, 212, 520, 38, N.panel, C.gray, 6, 1.5);
  o += tx(280, 231, 'Reference π_ref scores 6.2% (≈1/16); exact RL reaches 50.0% ceiling', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 3. ch14-rl-chapter-roadmap
function chapterRoadmap() {
  const W = 560, H = 280;
  let o = '';
  o += tx(280, 20, 'Chapter 14 Progression: Relaxing Assumptions Step by Step', { fs: 15.5, fw: 700, fill: C.ink });

  const items = [
    { x: 16,  y: 42,  t: '§14.1–14.2 MDPs & DP', s: 'Known P, r · contraction', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, y: 42,  t: '§14.3 Tabular TD', s: 'Q-learning & SARSA', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 380, y: 42,  t: '§14.4 Exploration', s: 'UCB · Thompson · bonus', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, y: 156, t: '§14.5 Imitation', s: 'BC · O(εT²) · DAgger', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 198, y: 156, t: '§14.6 Optimal Control', s: 'LQR · iLQR · MPC', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 16,  y: 156, t: '§14.7–14.9 Policy Grad', s: 'REINFORCE · GRPO · MLP', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  const bw = 164, bh = 74;
  items.forEach((it) => {
    o += box(it.x, it.y, bw, bh, it.fill, it.st, 8, 2);
    o += tx(it.x + bw / 2, it.y + 26, it.t, { fs: 13.2, fw: 700, fill: it.dk });
    o += tx(it.x + bw / 2, it.y + 50, it.s, { fs: 12, fw: 600, fill: C.ink });
  });

  o += arrow(182, 79, 196, 79, C.blue);
  o += arrow(364, 79, 378, 79, C.amber);
  o += arrow(462, 118, 462, 154, C.purple);
  o += arrow(378, 193, 364, 193, C.green);
  o += arrow(196, 193, 182, 193, D.red.base);

  o += box(16, 242, 528, 28, N.panel, C.gray, 6, 1.5);
  o += tx(280, 256, 'Benchmarks: Slippery FrozenLake · 10-Armed Bandit · Pendulum · Verifier Task', { fs: 12.2, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.1 mdp.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 4. ch14-rl-mdp-tuple
function mdpTuple() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'The MDP Tuple (S, A, P, r) at Cell s = 9, Command a = down', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(20, 86, 114, 78, C.lblue, C.blue, 8, 2);
  o += tx(77, 112, 'State s ∈ S', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(77, 138, 's = 9 (|S|=16)', { mono: true, fs: 12, fw: 600, fill: C.ink });

  o += arrow(136, 125, 172, 125, C.blue);

  o += box(176, 86, 116, 78, D.purple.tint, C.purple, 8, 2);
  o += tx(234, 112, 'Action a ∈ A', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(234, 138, 'a = down (1)', { mono: true, fs: 12, fw: 600, fill: C.ink });

  const ys = [62, 125, 188];
  const lands = ["s' = 8 (left slip)", "s' = 13 (intended)", "s' = 10 (right slip)"];
  ys.forEach((ty, i) => {
    o += arrow(294, 125, 352, ty, C.amber);
    o += box(356, ty - 22, 184, 44, C.lgreen, C.green, 7, 1.8);
    o += tx(448, ty - 7, lands[i], { mono: true, fs: 12, fw: 700, fill: D.green.dark });
    o += tx(448, ty + 11, 'P = 1/3 · ρ = 0', { mono: true, fs: 11.5, fill: C.ink });
  });

  o += box(20, 220, 520, 30, C.lamber, C.amber, 6, 1.5);
  o += tx(280, 235, "∑ P(s' | s, a) = 1   ·   Expected reward r(s, a) = ∑ P(s' | s, a) ρ(s, a, s')", { mono: true, fs: 11.8, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// 5. ch14-rl-tensor-kernel
function tensorKernel() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Vectorizing Gymnasium P[s][a] into Dense Tensors', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(20, 46, 166, 146, N.panel, C.gray, 8, 2);
  o += tx(103, 68, 'Gym Dict P[s][a]', { fs: 14, fw: 700, fill: C.ink });
  o += tx(103, 96, "(1/3, s'=8, r=0)", { mono: true, fs: 11.8, fill: C.muted });
  o += tx(103, 118, "(1/3, s'=13, r=0)", { mono: true, fs: 11.8, fill: C.muted });
  o += tx(103, 140, "(1/3, s'=10, r=0)", { mono: true, fs: 11.8, fill: C.muted });
  o += tx(103, 170, '+= p on wall bounce', { fs: 11.8, fw: 700, fill: D.purple.dark });

  o += arrow(190, 118, 228, 118, C.purple);

  o += box(242, 42, 142, 138, C.lblue, C.blue, 8, 1.8);
  o += box(232, 52, 142, 138, C.lblue, C.blue, 8, 2);
  o += tx(303, 82, 'Tensor P', { fs: 16, fw: 700, fill: D.blue.dark });
  o += tx(303, 108, 'shape (16, 4, 16)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(303, 132, "P[s, a, s'] ≥ 0", { mono: true, fs: 11.8, fill: C.muted });
  o += tx(303, 156, "∑_{s'} P = 1.0", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += box(400, 52, 140, 138, C.lgreen, C.green, 8, 2);
  o += tx(470, 82, 'Matrix r', { fs: 16, fw: 700, fill: D.green.dark });
  o += tx(470, 108, 'shape (16, 4)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(470, 132, 'r[14, D/R/U]=1/3', { mono: true, fs: 11.3, fill: C.ink });
  o += tx(470, 156, 'r[11→15] = 0', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });

  o += box(20, 208, 520, 34, D.purple.tint, C.purple, 6, 1.5);
  o += tx(280, 225, 'Single-call matrix backup: Q = r + γ · (P @ V) in O(|S|²|A|)', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// 6. ch14-rl-geometric-tail
function geometricTail() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'Discounted Return Split: Exact Prefix + Bounded Geometric Tail', { fs: 15, fw: 700, fill: C.ink });

  const x0 = 46, base = 166, bw = 34, gap = 8;
  const hs = [108, 95, 83, 73, 64, 56, 49, 43, 38, 33];
  hs.forEach((h, i) => {
    const x = x0 + i * (bw + gap);
    const isTail = i >= 5;
    o += box(x, base - h, bw, h, isTail ? C.lamber : C.lblue, isTail ? C.amber : C.blue, 4, 1.8);
    o += tx(x + bw / 2, base + 16, `t=${i}`, { mono: true, fs: 11.5, fill: C.ink });
  });

  o += box(36, 194, 210, 42, C.lblue, C.blue, 6, 1.8);
  o += tx(141, 215, 'Prefix t < k: ∑ γ^t r_t', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += box(256, 194, 268, 42, C.lamber, C.amber, 6, 1.8);
  o += tx(390, 215, 'Tail t ≥ k: ≤ γ^k r_max / (1 − γ)', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// 7. ch14-rl-state-augmentation
function stateAugmentation() {
  const W = 560, H = 264;
  let o = '';
  o += box(20, 16, 520, 112, D.red.tint, D.red.base, 8, 2);
  o += tx(280, 36, '(a) Position Only s_t = p_t Violates the Markov Property', { fs: 14.5, fw: 700, fill: D.red.dark });
  const xs = [104, 280, 456], ny1 = 80;
  ['p_{t-1}', 'p_t, a_t', 'p_{t+1}'].forEach((lab, i) => {
    o += box(xs[i] - 50, ny1 - 20, 100, 40, '#FFFFFF', D.red.base, 6, 1.8);
    o += tx(xs[i], ny1, lab, { mono: true, fs: 13, fw: 700, fill: C.ink });
  });
  o += arrow(156, ny1, 228, ny1, C.gray);
  o += arrow(332, ny1, 404, ny1, D.red.base);
  o += tx(280, 114, 'p_{t+1} depends on velocity v_t ∝ p_t − p_{t-1} (history leak!)', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(20, 138, 520, 110, C.lgreen, C.green, 8, 2);
  o += tx(280, 158, '(b) Augmented State s̃_t = (p_t, v_t)ᵀ Restores First-Order Markov', { fs: 14.5, fw: 700, fill: D.green.dark });
  const ny2 = 204;
  o += box(44, ny2 - 22, 180, 44, '#FFFFFF', C.green, 6, 2);
  o += tx(134, ny2, 's̃_t = (p_t, v_t)ᵀ', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += arrow(228, ny2, 304, ny2, C.green);
  o += tx(266, ny2 - 16, 'action a_t', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += box(308, ny2 - 22, 212, 44, '#FFFFFF', C.green, 6, 2);
  o += tx(414, ny2, 's̃_{t+1} = (p_{t+1}, v_{t+1})ᵀ', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 8. ch14-rl-pomdp-belief
function pomdpBelief() {
  const W = 560, H = 268;
  let o = '';
  o += tx(280, 20, 'Partial Observability (POMDP): Restoring a Markov State', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(36, 42, 190, 56, N.panel, C.gray, 8, 2, true);
  o += tx(131, 62, 'Hidden State s_t', { fs: 14, fw: 700, fill: C.ink });
  o += tx(131, 82, '(true physics / cards)', { fs: 12, fill: C.muted });

  o += arrow(230, 70, 326, 70, C.purple);
  o += tx(278, 52, 'O(o_t | s_t)', { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  o += box(330, 42, 194, 56, C.lblue, C.blue, 8, 2);
  o += tx(427, 62, 'Observation o_t', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(427, 82, '(1 frame / partial view)', { fs: 12, fill: C.ink });

  const cards = [
    { x: 16,  t: '1. Belief State β_t', s1: 'P(s_t | o_{≤t}, a_{<t})', s2: 'Exact Bayes update', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 198, t: '2. k-Frame Stack', s1: '(o_{t-3}, ..., o_t)', s2: 'Infers velocity Δo/Δt', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, t: '3. RNN / Attention', s1: 'h_t = f(h_{t-1}, o_t)', s2: 'Learned context prefix', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cards.forEach((c) => {
    o += arrow(427, 100, c.x + 82, 134, C.gray);
    o += box(c.x, 138, 164, 110, c.fill, c.st, 8, 2);
    o += tx(c.x + 82, 164, c.t, { fs: 13.5, fw: 700, fill: c.dk });
    o += tx(c.x + 82, 192, c.s1, { mono: true, fs: 11.2, fw: 600, fill: C.ink });
    o += tx(c.x + 82, 222, c.s2, { fs: 12, fw: 600, fill: c.dk });
  });
  return svg(W, H, o);
}

// 9. ch14-rl-potential-shaping
function potentialShaping() {
  const W = 560, H = 272;
  let o = '';
  o += box(20, 18, 248, 236, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 42, 'Naive One-Way Bonus', { fs: 15, fw: 700, fill: D.red.dark });
  o += tx(144, 64, "r̃ = r + 0.3 · 1(ℓ(s') < ℓ(s))", { mono: true, fs: 11.8, fw: 700, fill: C.ink });

  o += box(38, 108, 72, 48, '#FFFFFF', D.red.base, 6, 2);
  o += tx(74, 132, 'Cell 13', { fs: 13.5, fw: 700, fill: C.ink });
  o += box(178, 108, 72, 48, '#FFFFFF', D.red.base, 6, 2);
  o += tx(214, 132, 'Cell 14', { fs: 13.5, fw: 700, fill: C.ink });

  o += arrow(112, 120, 176, 120, D.red.base);
  o += tx(144, 100, '+0.30', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += arrow(176, 144, 112, 144, C.gray);
  o += tx(144, 166, '+0.00 (free!)', { mono: true, fs: 11.5, fw: 600, fill: C.muted });

  o += tx(144, 202, 'Loop return: 2.92 > Goal 1.30', { fs: 12.8, fw: 700, fill: D.red.dark });
  o += tx(144, 226, 'Agent shuttles 13 ↔ 14 forever!', { fs: 12.2, fw: 600, fill: C.ink });

  o += box(292, 18, 248, 236, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Potential-Based Shaping', { fs: 15, fw: 700, fill: D.green.dark });
  o += tx(416, 64, "r̃ = r + γΦ(s') − Φ(s)", { mono: true, fs: 11.8, fw: 700, fill: C.ink });

  o += box(310, 108, 72, 48, '#FFFFFF', C.green, 6, 2);
  o += tx(346, 132, 'Cell 13', { fs: 13.5, fw: 700, fill: C.ink });
  o += box(450, 108, 72, 48, '#FFFFFF', C.green, 6, 2);
  o += tx(486, 132, 'Cell 14', { fs: 13.5, fw: 700, fill: C.ink });

  o += arrow(384, 120, 448, 120, C.green);
  o += tx(416, 100, '+0.315', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += arrow(448, 144, 384, 144, D.orange.base);
  o += tx(416, 166, '−0.270 charge', { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });

  o += tx(416, 202, 'Telescopes: Ṽ^{π}(s) = V^π(s) − Φ(s)', { fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(416, 226, 'Preserves optimal policy π*!', { fs: 12.8, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.2 value-iter.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 10. ch14-rl-v-vs-q
function vVsQ() {
  const W = 560, H = 258;
  let o = '';
  o += box(20, 18, 248, 222, C.lblue, C.blue, 9, 2);
  o += tx(144, 40, 'State Value V^π(s)', { fs: 15, fw: 700, fill: D.blue.dark });
  o += box(104, 62, 80, 38, '#FFFFFF', C.blue, 6, 2);
  o += tx(144, 81, 'State s', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  [76, 144, 212].forEach((ax, i) => {
    o += arrow(144, 102, ax, 142, C.blue);
    o += box(ax - 24, 146, 48, 32, D.purple.tint, C.purple, 5, 1.8);
    o += tx(ax, 162, `a_${i}`, { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  });
  o += tx(144, 200, 'Averages over policy π(a | s):', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(144, 220, 'V^π(s) = ∑_{a} π(a|s) Q^π(s, a)', { mono: true, fs: 11.8, fw: 700, fill: D.blue.dark });

  o += box(292, 18, 248, 222, C.lgreen, C.green, 9, 2);
  o += tx(416, 40, 'Action Value Q^π(s, a)', { fs: 15, fw: 700, fill: D.green.dark });
  o += box(360, 62, 112, 38, '#FFFFFF', C.green, 6, 2);
  o += tx(416, 81, 'Pair (s, a)', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  [348, 416, 484].forEach((sx, i) => {
    o += arrow(416, 102, sx, 142, C.amber);
    o += box(sx - 24, 146, 48, 32, C.lamber, C.amber, 5, 1.8);
    o += tx(sx, 162, `s'_${i}`, { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  });
  o += tx(416, 200, 'Commits to a, averages over P:', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(416, 220, "Q^π(s,a) = r + γ ∑_{s'} P V^π(s')", { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 11. ch14-rl-bellman-optimality
function bellmanOptimality() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, C.lamber, C.amber, 9, 2);
  o += tx(144, 42, 'Extracting π* from V*', { fs: 15, fw: 700, fill: D.orange.dark });
  o += tx(144, 64, '(Requires Known Model P, r)', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += box(36, 86, 216, 46, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(144, 109, '1-step lookahead P @ V*', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += arrow(144, 134, 144, 162, D.orange.base);
  o += box(36, 166, 216, 52, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(144, 184, 'π*(s) = argmax_{a}', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(144, 204, "r(s,a) + γ ∑ P(s'|s,a) V*(s')", { mono: true, fs: 10.8, fill: C.ink });

  o += box(292, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Extracting π* from Q*', { fs: 15, fw: 700, fill: D.green.dark });
  o += tx(416, 64, '(Model-Free Table Lookup!)', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += box(308, 86, 216, 46, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 109, "Row Q*(s, ·) caches E_{s'}", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += arrow(416, 134, 416, 162, C.green);
  o += box(308, 166, 216, 52, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 184, 'π*(s) = argmax_{a} Q*(s, a)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 204, 'O(|A|) lookup, no P needed', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 12. ch14-rl-stopping-certificate
function stoppingCertificate() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'Observable Stopping Certificate vs. Premature Raw-Diff Halting', { fs: 15, fw: 700, fill: C.ink });

  const y = 104;
  o += line(44, y, 506, y, C.gray, 2.5);
  const pts = [
    { x: 80,  w: 76, lab: 'V_{k-1}',    col: C.blue,   dk: D.blue.dark },
    { x: 196, w: 76, lab: 'V_k',        col: C.purple, dk: D.purple.dark },
    { x: 468, w: 96, lab: 'V* (exact)', col: C.green,  dk: D.green.dark },
  ];
  pts.forEach((p) => {
    o += box(p.x - p.w / 2, y - 18, p.w, 36, '#FFFFFF', p.col, 6, 2.2);
    o += tx(p.x, y, p.lab, { mono: true, fs: 12, fw: 700, fill: p.dk });
  });

  o += arrow(80, 62, 196, 62, C.blue);
  o += tx(138, 44, 'Δ_k = ||V_k − V_{k-1}||_{∞}', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += arrow(196, 148, 468, 148, D.orange.base);
  o += tx(332, 168, '||V_k − V*||_{∞} ≤ [γ / (1 − γ)] · Δ_k = 19 · Δ_k', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });

  o += box(20, 194, 520, 44, C.lamber, C.amber, 7, 1.8);
  o += tx(280, 216, 'At γ = 0.95: Δ_k ≤ 10⁻⁶ leaves 1.1×10⁻⁵ error; wait for 19·Δ_k ≤ 10⁻⁶!', { fs: 12.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 13. ch14-rl-linear-solve
function linearSolve() {
  const W = 560, H = 250;
  let o = '';
  o += tx(280, 20, 'Policy Evaluation: Fixed-Point Sweeps vs. Direct Matrix Solve', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 42, 248, 190, C.lblue, C.blue, 9, 2);
  o += tx(144, 66, 'Iterative Bellman Sweeps', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += box(36, 86, 216, 44, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(144, 108, 'V_{k+1} = r_π + γ P_π V_k', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(144, 154, 'Cost: O(|S|²) per sweep', { fs: 13, fw: 600, fill: C.ink });
  o += tx(144, 178, 'Needs ∝ 1/(1−γ) sweeps', { fs: 13, fw: 600, fill: D.blue.dark });
  o += tx(144, 206, 'Best for large sparse MDPs', { fs: 12.2, fill: C.muted });

  o += box(292, 42, 248, 190, C.lgreen, C.green, 9, 2);
  o += tx(416, 66, 'Direct Linear Solve', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += box(308, 86, 216, 44, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 108, 'V^π = (I − γ P_π)⁻¹ r_π', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(416, 154, 'Cost: O(|S|³) one-shot LU', { fs: 13, fw: 600, fill: C.ink });
  o += tx(416, 178, 'Machine precision in 1 call', { fs: 13, fw: 600, fill: D.green.dark });
  o += tx(416, 206, '4×4 Lake: V(s₀) = 0.0078', { mono: true, fs: 12, fill: C.muted });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.3 qlearning.md (5 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 14. ch14-rl-max-expectation-order
function maxExpectationOrder() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 42, 'State Value V*(s)', { fs: 15, fw: 700, fill: D.red.dark });
  o += box(34, 66, 220, 48, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 90, "max_{a} E_{s'}[r + γ V*(s')]", { mono: true, fs: 11.3, fw: 700, fill: C.ink });
  o += tx(144, 140, "max_{a} sits OUTSIDE E_{s'}", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(144, 168, '1 sample reveals only 1 action', { fs: 12.2, fill: C.ink });
  o += tx(144, 192, 'and cannot compare actions', { fs: 12.2, fill: C.ink });
  o += tx(144, 216, 'without transition model P!', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Action Value Q*(s, a)', { fs: 15, fw: 700, fill: D.green.dark });
  o += box(300, 66, 232, 48, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 90, "E_{s'}[r + γ max_{a'} Q*(s',a')]", { mono: true, fs: 11.3, fw: 700, fill: C.ink });
  o += tx(416, 140, "E_{s'} sits OUTSIDE max_{a'}", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(416, 168, "Environment samples s' ~ P,", { fs: 12.2, fill: C.ink });
  o += tx(416, 192, "then agent takes max_{a'} over", { fs: 12.2, fill: C.ink });
  o += tx(416, 216, "its own Q-table row Q(s', ·)!", { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 15. ch14-rl-mc-vs-td
function mcVsTd() {
  const W = 560, H = 264;
  let o = '';
  o += box(16, 16, 528, 108, C.lamber, C.amber, 8, 2);
  o += tx(280, 36, 'Monte Carlo Target R_t: Wait for Episode End (Unbiased, High Var)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  const xs = [70, 176, 282, 388, 490], y1 = 74;
  ['s_t', 's_{t+1}', 's_{t+2}', '...', 's_T'].forEach((s, i) => {
    o += box(xs[i] - 32, y1 - 18, 64, 36, '#FFFFFF', C.amber, 5, 1.8);
    o += tx(xs[i], y1, s, { mono: true, fs: 12, fw: 700, fill: C.ink });
    if (i < 4) o += arrow(xs[i] + 34, y1, xs[i + 1] - 34, y1, C.amber);
  });
  o += tx(280, 110, 'R_t = r_t + γ r_{t+1} + γ² r_{t+2} + ··· + γ^{T−t−1} r_{T−1} (full tail sum)', { mono: true, fs: 11.5, fw: 600, fill: C.ink });

  o += box(16, 140, 528, 108, C.lgreen, C.green, 8, 2);
  o += tx(280, 160, 'One-Step TD Target y_t: Bootstrap After 1 Step (Online, Low Var)', { fs: 13.5, fw: 700, fill: D.green.dark });
  const y2 = 200;
  o += box(44, y2 - 18, 96, 36, '#FFFFFF', C.green, 5, 2);
  o += tx(92, y2, '(s_t, a_t)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += arrow(144, y2, 232, y2, C.green);
  o += tx(188, y2 - 14, '1 step r_t', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += box(236, y2 - 18, 284, 36, '#FFFFFF', C.green, 5, 2);
  o += tx(378, y2, "y_t = r_t + γ max_{a'} Q(s_{t+1}, a')", { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(280, 234, 'Updates Q(s_t, a_t) immediately; trades tail noise for current table bias', { fs: 12.2, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 16. ch14-rl-visit-propagation
function visitPropagation() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, 'Backward Value Flow Across Realized Visits (γ = 0.95, α = 0.5)', { fs: 15, fw: 700, fill: C.ink });

  const rows = [
    { y: 42,  v: 'Visit 1: 14 → 15 (Goal, r = 1)', c13: '0.000', c14: '0 → 0.500', c15: 'Goal (+1)', col: C.green, dk: D.green.dark },
    { y: 118, v: 'Visit 2: 13 → 14 (r = 0, boot 0.500)', c13: '0 → 0.238', c14: '0.500', c15: 'Goal (+1)', col: C.blue, dk: D.blue.dark },
    { y: 194, v: 'Visit 3: 14 → 10 (slip to unvisited 0)', c13: '0.238', c14: '0.500 → 0.250', c15: 'Cell 10 (0)', col: C.amber, dk: D.orange.dark },
  ];
  rows.forEach((r) => {
    o += box(20, r.y, 520, 62, N.panel, r.col, 8, 1.8);
    o += tx(34, r.y + 18, r.v, { fs: 13, fw: 700, fill: r.dk, anchor: 'start' });
    o += box(34, r.y + 30, 144, 24, '#FFFFFF', C.gray, 4, 1.5);
    o += tx(106, r.y + 42, `Q(13,R): ${r.c13}`, { mono: true, fs: 11.3, fw: 700, fill: C.ink });
    o += box(194, r.y + 30, 172, 24, '#FFFFFF', r.col, 4, 1.8);
    o += tx(280, r.y + 42, `Q(14,R): ${r.c14}`, { mono: true, fs: 11.3, fw: 700, fill: r.dk });
    o += box(382, r.y + 30, 140, 24, '#FFFFFF', C.gray, 4, 1.5);
    o += tx(452, r.y + 42, r.c15, { mono: true, fs: 11.3, fw: 700, fill: C.ink });
  });
  return svg(W, H, o);
}

// 17. ch14-rl-double-sampling
function doubleSampling() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'The Double-Sampling Problem: Why TD Holds the Target Fixed', { fs: 15, fw: 700, fill: C.ink });

  o += box(18, 42, 224, 200, C.lblue, C.blue, 8, 2);
  o += tx(130, 64, 'Stochastic Branch', { fs: 14, fw: 700, fill: D.blue.dark });
  o += box(34, 118, 58, 36, '#FFFFFF', C.blue, 6, 2);
  o += tx(63, 136, 'State s', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += arrow(94, 128, 134, 96, C.green);
  o += arrow(94, 144, 134, 176, D.red.base);
  o += box(138, 76, 92, 36, C.lgreen, C.green, 5, 1.8);
  o += tx(184, 94, 'u₁ (r = 1)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += box(138, 160, 92, 36, D.red.tint, D.red.base, 5, 1.8);
  o += tx(184, 178, 'u₂ (r = 0)', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(130, 222, 'Each branch w.p. 1/2', { fs: 12.2, fw: 600, fill: C.ink });

  o += box(258, 42, 284, 94, D.red.tint, D.red.base, 8, 2);
  o += tx(400, 62, 'Full Grad on δ² (Biased!)', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(400, 86, 'E[δ²] = (Q − TQ)² + Var(y | s,a)', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(400, 112, 'Shrinks Var(y) by pulling u₁, u₂ close!', { fs: 11.8, fw: 600, fill: D.red.dark });

  o += box(258, 148, 284, 94, C.lgreen, C.green, 8, 2);
  o += tx(400, 168, 'Semi-Gradient TD (Exact TQ)', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(400, 192, "Hold y = r + γ max Q(s') fixed", { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(400, 218, 'E[δ | s,a] = (TQ)(s,a) − Q(s,a) = 0', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 18. ch14-rl-q-vs-sarsa
function qVsSarsa() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, C.lblue, C.blue, 9, 2);
  o += tx(144, 42, 'Off-Policy Q-Learning', { fs: 15, fw: 700, fill: D.blue.dark });
  o += tx(144, 64, "Target: r + γ max_{a'} Q(s', a')", { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += box(36, 86, 216, 62, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(144, 106, "At s': evaluates GREEDY max", { fs: 12.2, fw: 700, fill: D.blue.dark });
  o += tx(144, 128, 'even when behavior π_e explores!', { fs: 11.8, fill: C.ink });
  o += tx(144, 174, 'Converges to optimal Q*', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(144, 198, 'Can reuse replay buffers & logs', { fs: 12.2, fill: C.ink });

  o += box(292, 18, 248, 220, C.lamber, C.amber, 9, 2);
  o += tx(416, 42, 'On-Policy SARSA', { fs: 15, fw: 700, fill: D.orange.dark });
  o += tx(416, 64, "Target: r + γ Q(s', a')", { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += box(308, 86, 216, 62, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(416, 106, "At s': uses ACTUAL a' ~ π_e", { fs: 12.2, fw: 700, fill: D.orange.dark });
  o += tx(416, 128, "from quintuple (s, a, r, s', a')", { fs: 11.8, fill: C.ink });
  o += tx(416, 174, 'Converges to behavior Q^{π_e}', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(416, 198, 'Prices in future ε-random slips', { fs: 12.2, fill: C.ink });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.4 exploration.md (5 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 19. ch14-rl-exploration-tax
function explorationTax() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'The Exploration Tax: Blind Random Steps vs. Directed Uncertainty', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 42, 248, 192, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 66, 'Blind ε-Greedy (ε = 0.05)', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(144, 92, 'Tries ALL actions with prob ε/|A|', { fs: 12.2, fw: 600, fill: C.ink });
  o += box(38, 112, 212, 56, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 130, 'Optimal π* on FrozenLake:', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(144, 152, '73.0% → 54.2% (−18.8 pts!)', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(144, 194, 'Keeps pulling known fatal actions', { fs: 12.2, fw: 600, fill: D.red.dark });
  o += tx(144, 214, 'forever (linear regret Θ(εt))', { fs: 12.2, fill: C.ink });

  o += box(292, 42, 248, 192, C.lgreen, C.green, 9, 2);
  o += tx(416, 66, 'Directed Exploration (UCB / TS)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(416, 92, 'Targets uncertain high-upside arms', { fs: 12.2, fw: 600, fill: C.ink });
  o += box(310, 112, 212, 56, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 130, 'Bonus ∝ √(log t / n(a)):', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 152, 'Stops pulling once UCB < μ*', { fs: 12.2, fw: 700, fill: C.ink });
  o += tx(416, 194, 'Concentrates pulls on optimal arm', { fs: 12.2, fw: 600, fill: D.green.dark });
  o += tx(416, 214, '(logarithmic regret Θ(ln t))', { fs: 12.2, fill: C.ink });
  return svg(W, H, o);
}

// 20. ch14-rl-bandit-regret
function banditRegret() {
  const W = 560, H = 262;
  let o = '';
  o += tx(280, 20, '10-Armed Bandit Benchmark: Arm Means μ_a and Regret Gaps Δ_a', { fs: 15, fw: 700, fill: C.ink });

  const means = [0.50, 0.42, 0.90, 0.25, 0.55, 0.38, 0.60, 0.32, 0.50, 0.45];
  const x0 = 46, base = 176, bw = 38, gap = 10, scale = 130;

  const yStar = base - 0.90 * scale;
  o += line(26, yStar, 534, yStar, D.green.base, 2, true);

  means.forEach((m, i) => {
    const x = x0 + i * (bw + gap);
    const h = Math.round(m * scale);
    const isBest = i === 2;
    o += box(x, base - h, bw, h, isBest ? C.lgreen : C.lblue, isBest ? C.green : C.blue, 4, 1.8);
    o += tx(x + bw / 2, base - h - 12, m.toFixed(2), { mono: true, fs: 11.5, fw: 700, fill: isBest ? D.green.dark : C.ink });
    o += tx(x + bw / 2, base + 16, `a=${i}`, { mono: true, fs: 11.5, fill: C.muted });
  });

  o += box(20, 204, 520, 44, C.lamber, C.amber, 7, 1.8);
  o += tx(280, 226, 'Best μ* = μ₂ = 0.90  ·  Min gap Δ_min = 0.30  ·  Regret(t) = ∑ Δ_a n_t(a)', { mono: true, fs: 10.8, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 21. ch14-rl-greedy-lottery
function greedyLottery() {
  const W = 560, H = 254;
  let o = '';
  o += tx(280, 20, 'Why Pure Greedy (ε = 0) Locks Onto the First Paying Arm', { fs: 15, fw: 700, fill: C.ink });

  const stages = [
    { x: 16,  t: '1. All Estimates 0', s1: 'μ̂₀ = (0, 0, ..., 0)', s2: 'Random tie-break: arm 0', s3: '(true mean μ₀ = 0.50)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, t: '2. First Win (r = 1)', s1: 'μ̂(0) jumps to 1.00', s2: 'Untried arms sit at 0', s3: '1.00 > 0 (greedy max!)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, t: '3. Permanent Lock-In', s1: 'Pulls arm 0 forever!', s2: 'Arm 2 (0.90) untried', s3: 'Loses Δ₀ = 0.40 / pull', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  stages.forEach((st, i) => {
    o += box(st.x, 44, 164, 142, st.fill, st.st, 8, 2);
    o += tx(st.x + 82, 68, st.t, { fs: 13.5, fw: 700, fill: st.dk });
    o += tx(st.x + 82, 98, st.s1, { mono: true, fs: 11.8, fw: 700, fill: C.ink });
    o += tx(st.x + 82, 126, st.s2, { fs: 12, fill: C.ink });
    o += tx(st.x + 82, 152, st.s3, { fs: 12, fw: 600, fill: st.dk });
    if (i < 2) o += arrow(st.x + 165, 115, st.x + 181, 115, C.gray);
  });

  o += box(16, 202, 528, 38, N.panel, C.gray, 6, 1.5);
  o += tx(280, 221, 'Best arm 2 wins initial lottery w.p. μ₂ / ∑μ_b = 0.90 / 4.87 = 18.5% (E[Regret] ≈ 707)', { fs: 12, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 22. ch14-rl-thompson-beta
function thompsonBeta() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Thompson Sampling: Conjugate Beta(1 + w_a, 1 + n_a − w_a) Posteriors', { fs: 15, fw: 700, fill: C.ink });

  o += box(24, 40, 512, 152, '#FFFFFF', C.gray, 6, 1.8);
  o += line(48, 168, 512, 168, C.ink, 2);
  o += tx(48, 182, '0.0', { mono: true, fs: 11.5, fill: C.muted });
  o += tx(280, 182, '0.5', { mono: true, fs: 11.5, fill: C.muted });
  o += tx(512, 182, '1.0', { mono: true, fs: 11.5, fill: C.muted });

  // True Beta densities on a shared scale: x in [0, 1] -> 48..512 px,
  // density 1 -> 18.7 px (Beta(19, 3) peaks at 6.0, Beta(4, 4) at 2.2).
  const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
  const pdfPath = (a, b) => {
    const norm = fact(a + b - 1) / (fact(a - 1) * fact(b - 1));
    const pts = [];
    for (let k = 0; k <= 116; k++) {
      const x = k / 116;
      const y = norm * x ** (a - 1) * (1 - x) ** (b - 1);
      pts.push(`${(48 + 464 * x).toFixed(1)} ${(168 - 18.7 * y).toFixed(1)}`);
    }
    return 'M ' + pts.join(' L ');
  };

  // Curve 1: Untried Beta(1,1), flat at density 1
  o += line(48, 149.3, 512, 149.3, C.purple, 2.5, true);
  o += tx(114, 132, 'Untried: Beta(1, 1)', { fs: 12, fw: 700, fill: D.purple.dark });

  // Curve 2: Mediocre arm Beta(4, 4), centred at 0.5
  o += `<path d="${pdfPath(4, 4)}" fill="none" stroke="${C.amber}" stroke-width="2.5"/>`;
  o += tx(280, 110, 'Arm 0: Beta(4, 4)', { fs: 12.2, fw: 700, fill: D.orange.dark });

  // Curve 3: Best arm Beta(19, 3), sharp peak at 0.9
  o += `<path d="${pdfPath(19, 3)}" fill="none" stroke="${C.green}" stroke-width="2.8"/>`;
  o += tx(368, 64, 'Arm 2: Beta(19, 3)', { fs: 12.2, fw: 700, fill: D.green.dark });

  o += box(24, 204, 512, 42, C.lgreen, C.green, 7, 1.8);
  o += tx(280, 225, 'Sample μ̃_a ~ Beta(1 + w_a, 1 + n_a − w_a) per arm → pull argmax_{a} μ̃_a', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 23. ch14-rl-chain-mdp
function chainMdp() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Deep Exploration on the N-State Chain: Random Walk vs. Optimism', { fs: 15, fw: 700, fill: C.ink });

  const xs = [68, 178, 288, 392, 492], cy = 78;
  const labs = ['s = 0', 's = 1', 's = 2', '···', 'N−1 (+1)'];
  xs.forEach((cx, i) => {
    const isG = i === 4;
    const bw = isG ? 80 : 68;
    o += box(cx - bw / 2, cy - 20, bw, 40, isG ? C.lgreen : C.lblue, isG ? C.green : C.blue, 6, 2);
    o += tx(cx, cy, labs[i], { mono: true, fs: 12, fw: 700, fill: isG ? D.green.dark : C.ink });
    if (i < 4) o += arrow(cx + 36, cy, xs[i + 1] - (i === 3 ? 42 : 36), cy, C.blue);
  });
  o += tx(280, 44, 'Action a = 1 advances right; action a = 0 resets back to s = 0 (r = 0)', { fs: 12.5, fw: 600, fill: C.muted });

  o += box(20, 122, 248, 126, D.red.tint, D.red.base, 8, 2);
  o += tx(144, 144, 'Zero Init Q₀ = 0 (Random Walk)', { fs: 13.8, fw: 700, fill: D.red.dark });
  o += tx(144, 172, 'Needs N−1 right moves in a row:', { fs: 12.2, fill: C.ink });
  o += tx(144, 198, 'P(goal) = 2^{−(N−1)}', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(144, 224, 'N = 20: ~46,000 episodes!', { fs: 12.5, fw: 700, fill: C.ink });

  o += box(292, 122, 248, 126, C.lgreen, C.green, 8, 2);
  o += tx(416, 144, 'Optimistic Init Q₀ = 1.0 (ε = 0)', { fs: 13.8, fw: 700, fill: D.green.dark });
  o += tx(416, 172, 'Tried r=0 pair drops to γ < 1.0,', { fs: 12.2, fill: C.ink });
  o += tx(416, 198, 'so greedy tries untried 1.0 pairs!', { fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(416, 224, 'N = 20: solved in 8 episodes!', { fs: 12.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.5 imitation.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 24. ch14-rl-alvinn-shift
function alvinnShift() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Policy-Induced Distribution Shift in Behavior Cloning', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(20, 42, 520, 152, N.panel, C.gray, 10, 2);
  o += box(36, 92, 488, 50, C.lblue, C.blue, 6, 2);
  o += tx(280, 117, 'Expert Demonstration Tube d^{π*}: 100% of training labels live here', { fs: 13, fw: 700, fill: D.blue.dark });

  o += arrow(84, 90, 154, 64, D.red.base, true);
  o += tx(336, 64, 'Clone drifts off-path (ε = 0.182): NO labels!', { fs: 12.5, fw: 700, fill: D.red.dark });

  o += arrow(120, 144, 190, 170, D.red.base, true);
  o += tx(360, 170, 'Unvisited states stay uniform (1/4 random) → crash!', { fs: 12.5, fw: 700, fill: D.red.dark });

  o += box(20, 206, 520, 36, C.lamber, C.amber, 6, 1.5);
  o += tx(280, 224, 'Calm Lake: a clone 81.8% faithful wins 66.0%, from 3 or 1,000 expert runs', { fs: 11.8, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 25. ch14-rl-visitation-markov
function visitationMarkov() {
  const W = 560, H = 248;
  let o = '';
  o += tx(280, 20, 'Exact State-Visitation Propagation via Markov Matrix P_π', { fs: 15.5, fw: 700, fill: C.ink });

  const xs = [72, 208, 344], cy = 112;
  const labs = [['d₀^{π}', '[1, 0, ..., 0]'], ['d₁ = d₀ P_π', 'step 1 mass'], ['d_T = d₀ P_π^T', 'T = 100 steps']];
  xs.forEach((cx, i) => {
    o += box(cx - 54, cy - 34, 108, 68, C.lblue, C.blue, 8, 2);
    o += tx(cx, cy - 10, labs[i][0], { mono: true, fs: 11.8, fw: 700, fill: D.blue.dark });
    o += tx(cx, cy + 14, labs[i][1], { fs: 12, fill: C.ink });
  });
  o += arrow(128, cy, 152, cy, C.blue);
  o += arrow(264, cy, 288, cy, C.blue, true);

  o += arrow(400, cy - 14, 424, 74, C.green);
  o += box(428, 48, 112, 52, C.lgreen, C.green, 7, 2);
  o += tx(484, 66, 'd_{100}(Goal)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(484, 86, 'Exact win %', { fs: 12, fw: 600, fill: C.ink });

  o += arrow(400, cy + 14, 424, 150, D.red.base);
  o += box(428, 124, 112, 52, D.red.tint, D.red.base, 7, 2);
  o += tx(484, 142, 'd_{100}(Holes)', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(484, 162, 'Absorbing loss', { fs: 12, fw: 600, fill: C.ink });

  o += box(20, 196, 520, 38, N.panel, C.gray, 6, 1.5);
  o += tx(280, 215, 'Zero sampling noise: 100 matrix-vector products compute exact deployment success', { fs: 12.2, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 26. ch14-rl-hybrid-telescoping
function hybridTelescoping() {
  const W = 560, H = 260;
  let o = '';
  o += box(20, 18, 248, 224, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 42, 'Behavior Cloning: O(ε T²)', { fs: 15, fw: 700, fill: D.red.dark });
  o += tx(144, 66, 'Error ≤ ε only on expert d_t^{π*}', { fs: 12.2, fw: 600, fill: C.ink });
  o += box(34, 86, 220, 72, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 108, 'Mistake at step t forfeits', { fs: 12.2, fill: C.ink });
  o += tx(144, 128, 'all (T − t + 1) remaining steps:', { fs: 12.2, fill: C.ink });
  o += tx(144, 148, '∑_{t=1}^{T} ε(T−t+1) = εT(T+1)/2', { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  o += tx(144, 188, '64-step chain (ε = 0.01):', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(144, 212, 'Loses 17.07 of 64 rewards!', { fs: 13.2, fw: 700, fill: D.red.dark });

  o += box(292, 18, 248, 224, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Learner-State Fit: O(u ε T)', { fs: 15, fw: 700, fill: D.green.dark });
  o += tx(416, 66, 'Error ≤ ε on learner d_t^{π̂}', { fs: 12.2, fw: 600, fill: C.ink });
  o += box(306, 86, 220, 72, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 108, 'Hybrid π_k follows π̂ for k steps', { fs: 12.2, fill: C.ink });
  o += tx(416, 128, 'then expert π* (cost u per slip):', { fs: 12.2, fill: C.ink });
  o += tx(416, 148, '∑ [J(π_{k−1}) − J(π_k)] ≤ u ε T', { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  o += tx(416, 188, '64-step chain (u = 1, ε = 0.01):', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(416, 212, 'Loses only 0.64 rewards (27× less)!', { fs: 13.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 27. ch14-rl-dagger-loop
function daggerLoop() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'DAgger (Dataset Aggregation): States from Learner, Labels from Expert', { fs: 14.8, fw: 700, fill: C.ink });

  const bw = 224, bh = 74;
  const boxes = [
    { x: 24,  y: 44,  t: '1. Roll Out Learner π̂_k', s: 'Visit states s_t ~ d^{π̂_k}', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 312, y: 44,  t: '2. Query Expert π*(s_t)', s: 'Relabel with a*_{t} = π*(s_t)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 312, y: 162, t: '3. Aggregate Dataset D_k', s: 'D_k = D_{k−1} ∪ D_new', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 24,  y: 162, t: '4. Refit Policy π̂_{k+1}', s: 'Cross-entropy on all of D_k', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  boxes.forEach((b) => {
    o += box(b.x, b.y, bw, bh, b.fill, b.st, 8, 2);
    o += tx(b.x + bw / 2, b.y + 26, b.t, { fs: 13.8, fw: 700, fill: b.dk });
    o += tx(b.x + bw / 2, b.y + 52, b.s, { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  });

  o += arrow(250, 81, 308, 81, C.blue);
  o += arrow(424, 122, 424, 158, C.amber);
  o += arrow(308, 199, 250, 199, C.purple);
  o += arrow(136, 158, 136, 122, C.green);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.6 optimal-control.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 28. ch14-rl-grid-vs-quadratic
function gridVsQuadratic() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 42, 'Discretized State Grid', { fs: 15, fw: 700, fill: D.red.dark });
  o += tx(144, 66, '100 bins per state dimension d', { fs: 12.2, fill: C.ink });
  o += box(36, 86, 216, 68, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 108, 'Table size = 100^d cells', { mono: true, fs: 12.2, fw: 700, fill: D.red.dark });
  o += tx(144, 134, '7-DOF arm (d=14): 10²⁸ cells', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(144, 182, 'Humanoid (d=36): 10⁷² cells', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(144, 208, "Bellman's Curse of Dimensionality", { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, "Kalman's LQR Value Matrix", { fs: 15, fw: 700, fill: D.green.dark });
  o += tx(416, 66, 'V_t(x) = xᵀ P_t x (quadratic)', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += box(308, 86, 216, 68, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 108, 'Size = d(d + 1) / 2 entries', { mono: true, fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(416, 134, '7-DOF arm (d=14): 105 entries', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(416, 182, 'Humanoid (d=36): 666 numbers', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(416, 208, 'O(d³ + m³) matrix backup / step!', { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 29. ch14-rl-lqr-loop
function lqrLoop() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Closed-Loop Linear-Quadratic Regulator (LQR) System', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(24, 60, 190, 82, C.lgreen, C.green, 8, 2);
  o += tx(119, 86, 'Linear Feedback Law', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(119, 114, 'u*_{t} = −K_t x_t', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  o += arrow(216, 101, 298, 101, C.blue);
  o += tx(257, 83, 'u_t ∈ R^m', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += box(302, 60, 234, 82, C.lblue, C.blue, 8, 2);
  o += tx(419, 86, 'Linear Dynamics Plant', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(419, 114, 'x_{t+1} = A x_t + B u_t', { mono: true, fs: 13, fw: 700, fill: C.ink });

  // state feedback: the plant's x_t returns to the controller
  o += line(419, 144, 419, 166, C.green);
  o += line(419, 166, 119, 166, C.green);
  o += arrow(119, 166, 119, 144, C.green);
  o += tx(269, 155, 'state x_t fed back', { fs: 12, fw: 700, fill: D.green.dark });
  o += box(136, 186, 316, 52, C.lamber, C.amber, 8, 2);
  o += tx(294, 204, 'Quadratic Stage Cost c(x_t, u_t)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(294, 224, 'x_tᵀ Q x_t (state) + u_tᵀ R u_t (effort)', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 30. ch14-rl-riccati-backward
function riccatiBackward() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Backward Discrete-Time Riccati Recursion (t = H → 0)', { fs: 15.5, fw: 700, fill: C.ink });

  const xs = [466, 292, 104], y = 84;
  o += box(xs[0] - 64, y - 30, 128, 60, C.lamber, C.amber, 8, 2);
  o += tx(xs[0], y - 8, 'Terminal Step H', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(xs[0], y + 14, 'P_H = Q_H', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += arrow(400, y, 372, y, C.purple);

  o += box(xs[1] - 78, y - 30, 156, 60, D.purple.tint, C.purple, 8, 2);
  o += tx(xs[1], y - 8, 'Cost-to-Go P_{t+1}', { fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(xs[1], y + 14, "x'ᵀ P_{t+1} x'", { mono: true, fs: 11.5, fill: C.ink });

  o += arrow(214, y, 186, y, C.purple);

  o += box(xs[2] - 80, y - 30, 160, 60, C.lgreen, C.green, 8, 2);
  o += tx(xs[2], y - 8, 'Solve ∇_u Q_t = 0', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(xs[2], y + 14, 'Gain K_t & P_t', { mono: true, fs: 11.8, fw: 700, fill: C.ink });

  o += box(20, 136, 520, 104, N.panel, C.gray, 8, 1.8);
  o += tx(280, 158, 'K_t = (R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(280, 184, 'P_t = Q + Aᵀ P_{t+1} A − Aᵀ P_{t+1} B K_t', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 216, 'Independent of x₀  ·  Unchanged by Gaussian noise w_t ~ N(0, Σ)', { fs: 12.2, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 31. ch14-rl-pendulum-physics
function pendulumPhysics() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Pendulum-v1 Underactuation: Motor Clip |u| ≤ 2 vs. Gravity 5 sin θ', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 42, 248, 198, C.lblue, C.blue, 9, 2);
  o += tx(144, 64, 'Torque Limit vs. Gravity', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += box(34, 82, 220, 44, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 104, 'Peak gravity (θ=90°): 5.0', { mono: true, fs: 11.8, fw: 700, fill: D.red.dark });
  o += box(34, 138, 220, 44, '#FFFFFF', C.green, 6, 1.8);
  o += tx(144, 160, 'Max motor torque: |u_t| ≤ 2.0', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(144, 204, 'Static hold limit: arcsin(2/5)', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(144, 224, '|θ| ≤ 0.412 rad (23.6° basin)', { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  o += box(292, 42, 248, 198, C.lamber, C.amber, 9, 2);
  o += tx(416, 64, 'Swing-Up & Cost Bar', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(416, 92, 'c_t = θ² + 0.1 θ̇² + 0.001 u²', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += box(308, 112, 216, 52, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(416, 130, 'Outside 23.6°: cannot lift direct!', { fs: 12.2, fw: 700, fill: D.red.dark });
  o += tx(416, 150, 'Must pump energy back & forth', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(416, 192, 'Random torques: ≈ −1,200 return', { fs: 12.2, fill: C.muted });
  o += tx(416, 216, 'Successful swing-up: ≥ −200 return', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 32. ch14-rl-ilqr-cycle
function ilqrCycle() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Iterative LQR (iLQR): Gauss-Newton Trajectory Optimization Loop', { fs: 15, fw: 700, fill: C.ink });

  const bw = 224, bh = 74;
  const stages = [
    { x: 24,  y: 44,  t: '1. Nominal Rollout', s: 'x̄_{t+1} = f(x̄_t, 2 tanh v̄_t)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 312, y: 44,  t: '2. Local Taylor Jacobians', s: 'A_t = ∂f/∂x, B_t = ∂f/∂v', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 312, y: 162, t: '3. Backward Riccati Pass', s: 'Feedforward k_t & gains K_t', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 24,  y: 162, t: '4. Forward Line Search α', s: 'v̄_t + α k_t + K_t(x_t − x̄_t)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  stages.forEach((b) => {
    o += box(b.x, b.y, bw, bh, b.fill, b.st, 8, 2);
    o += tx(b.x + bw / 2, b.y + 26, b.t, { fs: 13.8, fw: 700, fill: b.dk });
    o += tx(b.x + bw / 2, b.y + 52, b.s, { mono: true, fs: 11.3, fw: 600, fill: C.ink });
  });

  o += arrow(250, 81, 308, 81, C.blue);
  o += arrow(424, 122, 424, 158, C.amber);
  o += arrow(308, 199, 250, 199, C.purple);
  o += arrow(136, 158, 136, 122, C.green);
  return svg(W, H, o);
}

// 33. ch14-rl-control-hierarchy
function controlHierarchy() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Three Execution Modes Under Impulse Disturbances (t = 45 Knock)', { fs: 15, fw: 700, fill: C.ink });

  const cols = [
    { x: 16,  t: '1. Open-Loop Plan', f: 'u_t = u̅_t', s1: 'Blind to state x_t', s2: '+0.5 knock: FALLS', s3: '(θ₁₂₀ = −2.94 rad)', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 198, t: '2. TV-LQR Tube', f: 'v̄_t + K_t Δx_t', s1: 'Local linear feedback', s2: '+0.5 knock: HOLDS', s3: '+2.2 knock: FALLS', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, t: '3. Receding MPC', f: 'Replan H_mpc at x_t', s1: 'Apply u*_{t|t}, re-plan', s2: '+2.2 knock: RECOVERS', s3: 'Plans fresh swing-up!', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cols.forEach((c) => {
    o += box(c.x, 42, 164, 198, c.fill, c.st, 8, 2);
    o += tx(c.x + 82, 66, c.t, { fs: 13.8, fw: 700, fill: c.dk });
    o += box(c.x + 12, 82, 140, 34, '#FFFFFF', c.st, 5, 1.6);
    o += tx(c.x + 82, 99, c.f, { mono: true, fs: 11.2, fw: 700, fill: C.ink });
    o += tx(c.x + 82, 138, c.s1, { fs: 12.2, fill: C.ink });
    o += tx(c.x + 82, 168, c.s2, { fs: 12.2, fw: 700, fill: c.dk });
    o += tx(c.x + 82, 196, c.s3, { fs: 12, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

// 34. ch14-rl-control-vs-rl
function controlVsRl() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Unifying Dictionary: Classical Optimal Control vs. Reinforcement Learning', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 40, 240, 204, C.lblue, C.blue, 9, 2);
  o += tx(136, 62, 'Classical Optimal Control', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(136, 92, 'State x_t ∈ R^d, u_t ∈ R^m', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(136, 122, 'Minimize Stage Cost ∑ c(x_t, u_t)', { fs: 12.2, fw: 700, fill: D.blue.dark });
  o += tx(136, 152, 'Quadratic Cost-to-Go xᵀ P_t x', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(136, 182, 'Backward Riccati / iLQR Jacobians', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(136, 212, 'Receding-Horizon MPC', { fs: 12.8, fw: 700, fill: D.blue.dark });

  o += box(304, 40, 240, 204, C.lgreen, C.green, 9, 2);
  o += tx(424, 62, 'Reinforcement Learning', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(424, 92, 'State s_t ∈ S, Action a_t ∈ A', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(424, 122, 'Maximize Return E[∑ γ^t r(s_t, a_t)]', { fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(424, 152, 'Value Table / Critic V^π, Q^π', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(424, 182, 'Bellman Operator T / Policy Grad', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(424, 212, 'Learned World-Model MPC (§15.6)', { fs: 12.8, fw: 700, fill: D.green.dark });

  [92, 122, 152, 182, 212].forEach((ry) => {
    o += arrow(260, ry, 298, ry, C.purple);
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.7 policy-gradient.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 35. ch14-rl-cloning-to-reinforce
function cloningToReinforce() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'Shared Log-Likelihood Primitive: Cloning Weights vs. Return Weights', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(28, 38, 504, 42, D.purple.tint, C.purple, 7, 2);
  o += tx(280, 59, 'policy_step: ascend  (1/N) ∑_{t} w_t · ∇_θ log π_θ(a_t | s_t)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });

  o += box(20, 96, 248, 140, C.lblue, C.blue, 8, 2);
  o += tx(144, 120, 'Behavior Cloning (§14.5)', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(144, 148, 'Data: Expert demos (s_t, a*_{t})', { fs: 12.5, fill: C.ink });
  o += tx(144, 176, 'Weight: w_t = 1 (always)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(144, 206, 'Requires interactive/logged expert', { fs: 12, fill: C.muted });

  o += box(292, 96, 248, 140, C.lgreen, C.green, 8, 2);
  o += tx(416, 120, 'REINFORCE (§14.7)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(416, 148, 'Data: Self-rollouts τ ~ π_θ', { fs: 12.5, fill: C.ink });
  o += tx(416, 176, 'Weight: w_t = R(τ) (return)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(416, 206, 'Learns from trial & error alone!', { fs: 12.2, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 36. ch14-rl-rare-reward-variance
function rareRewardVariance() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Why Sparse Rewards Cause High Policy-Gradient Variance', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 42, 248, 146, N.panel, C.gray, 8, 2);
  o += tx(144, 66, 'Failed Rollout (1 − p = 0.986)', { fs: 13.5, fw: 700, fill: C.muted });
  o += tx(144, 98, 'Reward R(τ) = 0', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(144, 128, 'Gradient estimate û = 0', { mono: true, fs: 12.5, fill: C.muted });
  o += tx(144, 160, '80% of 16-ep batches are all zero!', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 42, 248, 146, C.lamber, C.amber, 8, 2);
  o += tx(416, 66, 'Goal Rollout (prob p = 0.014)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(416, 98, 'Reward R(τ) = 1', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(416, 128, 'Estimate û = 1 − p = 0.986', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(416, 160, '70× larger than true mean p(1−p)!', { fs: 12.2, fw: 700, fill: D.orange.dark });

  o += box(20, 202, 520, 40, D.purple.tint, C.purple, 7, 1.8);
  o += tx(280, 222, 'Unbiased E[û] = p(1−p), but relative std √((1−p)/p) ≈ 8.4× true gradient!', { mono: true, fs: 11.8, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// 37. ch14-rl-log-derivative
function logDerivative() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'The Log-Derivative Trick: Turning ∇_θ E[R] into a Sampleable Expectation', { fs: 14.5, fw: 700, fill: C.ink });

  const steps = [
    { y: 42,  t: '1. Differentiate expectation:', f: '∇_θ J(θ) = ∑_{τ} [∇_θ P(τ; θ)] · R(τ)   (cannot sample ∇_θ P)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 112, t: '2. Apply identity ∇_θ P = P · ∇_θ log P:', f: '∇_θ J(θ) = ∑_{τ} P(τ; θ) · [R(τ) ∇_θ log P(τ; θ)]', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 182, t: '3. Rewrite as expectation over τ ~ P(·; θ):', f: '∇_θ J(θ) = E_{τ ~ P}[ R(τ) ∑_{t=0}^{T−1} ∇_θ log π_θ(a_t | s_t) ]', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s, i) => {
    o += box(20, s.y, 520, 54, s.fill, s.st, 8, 2);
    o += tx(36, s.y + 18, s.t, { fs: 13, fw: 700, fill: s.dk, anchor: 'start' });
    o += tx(280, s.y + 39, s.f, { mono: true, fs: 11.8, fw: 700, fill: C.ink });
    if (i < 2) o += arrow(280, s.y + 56, 280, s.y + 68, C.gray);
  });
  return svg(W, H, o);
}

// 38. ch14-rl-causality-credit
function causalityCredit() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Policy Gradient Theorem: Action a_t Cannot Cause Past Rewards r_{<t}', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(16, 48, 174, 86, N.panel, C.gray, 8, 2, true);
  o += tx(103, 70, 'Past Rewards (t\' < t)', { fs: 13, fw: 700, fill: C.muted });
  o += tx(103, 94, 'r₀, r₁, ..., r_{t−1}', { mono: true, fs: 12, fill: C.muted });
  o += tx(103, 118, 'E[r_{<t} · ∇log π_t] = 0', { mono: true, fs: 11, fw: 700, fill: D.red.dark });

  o += box(204, 48, 134, 86, D.purple.tint, C.purple, 8, 2.2);
  o += tx(271, 70, 'Step t Action', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(271, 94, 'a_t ~ π_θ(·|s_t)', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(271, 118, 'score ∇_θ log π_θ', { mono: true, fs: 11.2, fill: D.purple.dark });

  o += arrow(340, 91, 360, 91, C.green);

  o += box(364, 48, 180, 86, C.lgreen, C.green, 8, 2);
  o += tx(454, 70, 'Future Return (t\' ≥ t)', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(454, 94, 'Ĝ_t = ∑ γ^{t\'−t} r_{t\'}', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(454, 118, 'E[Ĝ_t] = Q^π(s_t, a_t)', { mono: true, fs: 11.2, fw: 700, fill: D.green.dark });

  o += box(16, 154, 528, 84, C.lblue, C.blue, 8, 1.8);
  o += tx(280, 178, 'Policy Gradient Theorem (Sutton et al., 2000):', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 202, '∇_θ J(θ) = E_τ[ ∑_{t≥0} γ^t ∇_θ log π_θ(a_t | s_t) Q^{π_θ}(s_t, a_t) ]', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(280, 224, 'Dropping past rewards r_{<t} eliminates prefix noise with zero bias!', { fs: 12.2, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.8 baselines.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 39. ch14-rl-all-or-nothing
function allOrNothing() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 42, 'Raw Return Weight R(τ) ≥ 0', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += box(34, 66, 220, 66, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 88, 'Win (R=0.77): w = +0.77 ↑', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(144, 114, 'Hole (R=0): w = 0.00 (idle!)', { mono: true, fs: 11.8, fw: 700, fill: D.red.dark });
  o += tx(144, 158, 'Fatal moves into holes receive', { fs: 12.2, fill: C.ink });
  o += tx(144, 180, 'ZERO negative gradient, while', { fs: 12.2, fw: 700, fill: D.red.dark });
  o += tx(144, 202, 'loops inside wins get boosted!', { fs: 12.2, fill: C.ink });

  o += box(292, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Centered Weight R(τ) − b(s_t)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += box(306, 66, 220, 66, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 88, 'Above mean: w > 0 (push ↑)', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(416, 114, 'Below mean: w < 0 (push ↓)', { mono: true, fs: 11.8, fw: 700, fill: D.orange.dark });
  o += tx(416, 158, 'Actively suppresses hole falls', { fs: 12.2, fill: C.ink });
  o += tx(416, 180, 'while leaving expected gradient', { fs: 12.2, fill: C.ink });
  o += tx(416, 202, 'E[û] = ∇_θ J completely unchanged!', { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 40. ch14-rl-prefix-admissibility
function prefixAdmissibility() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Admissible vs. Inadmissible Baselines at Step t', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 194, C.lgreen, C.green, 9, 2);
  o += tx(144, 68, 'Fixed Before Sampling a_t', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(144, 92, 'c(s₀, a₀, ..., s_t) → UNBIASED', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += box(34, 110, 220, 84, '#FFFFFF', C.green, 6, 1.8);
  o += tx(144, 130, '✓ Constant b or step index t', { fs: 12, fw: 600, fill: C.ink });
  o += tx(144, 154, '✓ State value V^π(s_t) or V̂(s_t)', { fs: 12, fw: 600, fill: C.ink });
  o += tx(144, 178, '✓ RLOO mean b_i of j ≠ i runs', { fs: 12, fw: 600, fill: C.ink });
  o += tx(144, 216, 'E[c · ∇log π(a_t|s_t)] = 0', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  o += box(292, 44, 248, 194, D.red.tint, D.red.base, 9, 2);
  o += tx(416, 68, 'Depends on Sampled a_t', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(416, 92, 'b(s_t, a_t, s_{t+1}) → BIASED!', { mono: true, fs: 11.8, fw: 700, fill: D.red.dark });
  o += box(306, 110, 220, 84, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(416, 130, '✗ Action value Q̂(s_t, a_t)', { fs: 12, fw: 600, fill: C.ink });
  o += tx(416, 154, '✗ Remaining length T − t', { fs: 12, fw: 600, fill: C.ink });
  o += tx(416, 178, '✗ Self-included batch mean R̄', { fs: 12, fw: 600, fill: C.ink });
  o += tx(416, 216, 'Couples with a_t → E[û] ≠ ∇J!', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 41. ch14-rl-advantage-centering
function advantageCentering() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Why the Baseline Must Depend on State: at π = π*, A*(s, a) = Q* − V*', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 194, C.lblue, C.blue, 9, 2);
  o += tx(144, 68, 'Start Cell s = 0 (V* = 0.180)', { fs: 14, fw: 700, fill: D.blue.dark });
  o += box(34, 86, 220, 72, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(144, 108, 'Best action (left): Q = 0.180', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(144, 128, '→ Advantage A = 0.000', { mono: true, fs: 11.8, fill: C.ink });
  o += tx(144, 148, 'Up: Q = 0.165 → A = −0.015', { mono: true, fs: 11.5, fill: D.red.dark });
  o += tx(144, 184, 'A global constant b = 0.45 would', { fs: 12.2, fill: C.ink });
  o += tx(144, 206, 'mark EVERY action at s=0 negative!', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 44, 248, 194, C.lgreen, C.green, 9, 2);
  o += tx(416, 68, 'Goal Neighbor s = 14 (V* = 0.724)', { fs: 14, fw: 700, fill: D.green.dark });
  o += box(306, 86, 220, 72, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 108, 'Best action (down): Q = 0.724', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(416, 128, '→ Advantage A = 0.000', { mono: true, fs: 11.8, fill: C.ink });
  o += tx(416, 148, 'Left: Q = 0.395 → A = −0.329', { mono: true, fs: 11.5, fill: D.red.dark });
  o += tx(416, 184, 'Subtracting V^π(s_t) centers every', { fs: 12.2, fill: C.ink });
  o += tx(416, 206, 'state at E_{a~π}[A^π(s, a)] = 0!', { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 42. ch14-rl-variance-mechanisms
function varianceMechanisms() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Three Distinct Operations Inside Standardized Policy Gradients', { fs: 15, fw: 700, fill: C.ink });

  const cols = [
    { x: 16,  t: '1. Drop Past r_{<t}', f: 'R(τ) → Ĝ_t', s1: 'Causality reduction', s2: 'Step-cost lake noise:', s3: '101.6 → 34.1 (3× cut)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, t: '2. Subtract V(s_t)', f: 'Ĝ_t → Ĝ_t − V(s_t)', s1: 'State control variate', s2: 'Lake relative noise:', s3: '18.1 → 9.1 (2× cut)', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 380, t: '3. Divide by Batch σ', f: '(Ĝ_t − R̄) / σ', s1: 'Step-size multiplier!', s2: 'Relative noise: 10.8', s3: 'Step norm: 3× larger', fill: C.lamber, st: C.amber, dk: D.orange.dark },
  ];
  cols.forEach((c) => {
    o += box(c.x, 42, 164, 200, c.fill, c.st, 8, 2);
    o += tx(c.x + 82, 66, c.t, { fs: 13.8, fw: 700, fill: c.dk });
    o += box(c.x + 12, 82, 140, 34, '#FFFFFF', c.st, 5, 1.6);
    o += tx(c.x + 82, 99, c.f, { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    o += tx(c.x + 82, 138, c.s1, { fs: 12.2, fw: 700, fill: c.dk });
    o += tx(c.x + 82, 168, c.s2, { fs: 12, fill: C.ink });
    o += tx(c.x + 82, 194, c.s3, { mono: true, fs: 11.3, fw: 700, fill: C.ink });
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §14.9 deep-rl.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 43. ch14-rl-continuous-generalization
function continuousGeneralization() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 42, 'Lookup Table on s ∈ R⁴', { fs: 15, fw: 700, fill: D.red.dark });
  o += tx(144, 66, 'CartPole: (x, ẋ, θ, θ̇) float32', { mono: true, fs: 12, fill: C.ink });
  o += box(36, 86, 216, 68, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 108, '0 exact state revisits across', { fs: 12.2, fw: 700, fill: D.red.dark });
  o += tx(144, 132, '1,000 continuous episodes!', { fs: 12.2, fill: C.ink });
  o += tx(144, 178, 'Update at θ = 0.0500 teaches', { fs: 12.2, fill: C.ink });
  o += tx(144, 202, 'NOTHING at θ = 0.0501!', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Neural Net V̂_w(s), π_θ(a|s)', { fs: 15, fw: 700, fill: D.green.dark });
  o += tx(416, 66, 'Shared parameters w, θ across R⁴', { fs: 12.2, fill: C.ink });
  o += box(308, 86, 216, 68, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 108, 'Smooth 64-unit tanh MLP', { fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(416, 132, 'interpolates nearby states', { fs: 12.2, fill: C.ink });
  o += tx(416, 178, '10 nearest states in R⁴ receive', { fs: 12.2, fill: C.ink });
  o += tx(416, 202, ">80% of s₀'s value update!", { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 44. ch14-rl-ntk-coupling
function ntkCoupling() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Neural Tangent Kernel Coupling: ΔV̂_w(s) ≈ η_V (y − V̂_w(s₀)) · k(s, s₀)', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(18, 44, 240, 194, C.lblue, C.blue, 8, 2);
  o += tx(138, 66, 'Tabular One-Hot Features e_s', { fs: 13.8, fw: 700, fill: D.blue.dark });
  o += tx(138, 90, 'k(s, s₀) = 1(s = s₀)', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  const M = [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']];
  o += grid(M, 90, 108, 28, 4, {
    fs: 12.5,
    fill: (r, c) => r === c ? [C.lgreen, C.green, false, D.green.dark] : ['#FFFFFF', C.gray, false, C.muted],
  });
  o += tx(138, 218, 'Diagonal: zero interference', { fs: 12.2, fw: 700, fill: D.blue.dark });

  o += box(276, 44, 266, 194, C.lamber, C.amber, 8, 2);
  o += tx(409, 66, 'MLP Features ∇_w V̂_w(s)', { fs: 13.8, fw: 700, fill: D.orange.dark });
  o += tx(409, 90, 'k(s, s₀) = ∇_w V̂(s)ᵀ ∇_w V̂(s₀)', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += box(290, 108, 238, 42, C.lgreen, C.green, 6, 1.6);
  o += tx(409, 129, 'Nearby ||s−s₀||: +0.43 (generalizes)', { fs: 11.8, fw: 700, fill: D.green.dark });
  o += box(290, 158, 238, 42, D.red.tint, D.red.base, 6, 1.6);
  o += tx(409, 179, 'Far ||s−s₀||: −0.07 (interferes!)', { fs: 11.8, fw: 700, fill: D.red.dark });
  o += tx(409, 218, 'Dense kernel couples all states', { fs: 12.2, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// 45. ch14-rl-actor-critic-mlps
function actorCriticMlps() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'ActorCritic Container: Separate 64-Unit tanh Policy and Value MLPs', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(18, 96, 106, 72, C.lblue, C.blue, 8, 2);
  o += tx(71, 122, 'State s_t', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(71, 146, 'obs ∈ R^d', { mono: true, fs: 12, fill: C.ink });

  o += arrow(126, 114, 170, 82, C.purple);
  o += arrow(126, 150, 170, 184, C.green);

  o += box(174, 42, 368, 82, D.purple.tint, C.purple, 8, 2);
  o += tx(358, 64, 'Actor Network π_θ (opt_pi: 1 step / batch)', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(358, 88, 'dense d→64 → tanh → dense 64→64 → tanh → head', { mono: true, fs: 11, fill: C.ink });
  o += tx(358, 108, 'Outputs discrete logits or Gaussian (μ_θ(s), log σ)', { fs: 11.8, fw: 600, fill: D.purple.dark });

  o += box(174, 142, 368, 82, C.lgreen, C.green, 8, 2);
  o += tx(358, 164, 'Critic Baseline V̂_w (opt_v: 40 steps / batch)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(358, 188, 'dense d→64 → tanh → dense 64→64 → tanh → scalar', { mono: true, fs: 11, fill: C.ink });
  o += tx(358, 208, '40 MSE passes explain ~99% of return variance!', { fs: 11.8, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 46. ch14-rl-truncation-bootstrap
function truncationBootstrap() {
  const W = 560, H = 262;
  let o = '';
  o += tx(280, 20, 'Bootstrapping Time-Limit Truncations on CartPole (T = 500, γ = 0.99)', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 198, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 68, 'Without Truncation Bootstrap', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(144, 92, 'Treats t = 500 clock cut as dead!', { fs: 12.2, fill: C.ink });
  o += box(34, 110, 220, 72, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 130, 'At step t = 490 (10 steps left):', { fs: 12, fill: C.ink });
  o += tx(144, 150, 'Ĝ_{490} = ∑ 0.99^k = 9.56', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(144, 170, 'Adv = 9.56 − 100 = −90.44!', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(144, 212, 'Punishes balanced pole near t=500!', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 44, 248, 198, C.lgreen, C.green, 9, 2);
  o += tx(416, 68, 'With Truncation Bootstrap', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(416, 92, 'y_t = Ĝ_t + (1−term_T) γ^{T−t} V̂(s_T)', { mono: true, fs: 10.8, fw: 700, fill: C.ink });
  o += box(306, 110, 220, 72, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 130, 'Adds γ¹⁰ V̂_w(s₅₀₀) ≈ 90.44:', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(416, 150, 'y₄₉₀ = 9.56 + 90.44 = 100.0', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(416, 170, 'Adv = 100.0 − 100 = 0.00!', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(416, 212, 'Holds 500-step ceiling cleanly!', { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

export const diagrams = {
  'ch14-rl-delayed-credit':            delayedCredit,
  'ch14-rl-verifier-benchmark':        verifierBenchmark,
  'ch14-rl-chapter-roadmap':           chapterRoadmap,
  'ch14-rl-mdp-tuple':                 mdpTuple,
  'ch14-rl-tensor-kernel':             tensorKernel,
  'ch14-rl-geometric-tail':            geometricTail,
  'ch14-rl-state-augmentation':        stateAugmentation,
  'ch14-rl-pomdp-belief':              pomdpBelief,
  'ch14-rl-potential-shaping':         potentialShaping,
  'ch14-rl-v-vs-q':                    vVsQ,
  'ch14-rl-bellman-optimality':        bellmanOptimality,
  'ch14-rl-stopping-certificate':      stoppingCertificate,
  'ch14-rl-linear-solve':              linearSolve,
  'ch14-rl-max-expectation-order':     maxExpectationOrder,
  'ch14-rl-mc-vs-td':                  mcVsTd,
  'ch14-rl-visit-propagation':         visitPropagation,
  'ch14-rl-double-sampling':           doubleSampling,
  'ch14-rl-q-vs-sarsa':                qVsSarsa,
  'ch14-rl-exploration-tax':           explorationTax,
  'ch14-rl-bandit-regret':             banditRegret,
  'ch14-rl-greedy-lottery':            greedyLottery,
  'ch14-rl-thompson-beta':             thompsonBeta,
  'ch14-rl-chain-mdp':                 chainMdp,
  'ch14-rl-alvinn-shift':              alvinnShift,
  'ch14-rl-visitation-markov':         visitationMarkov,
  'ch14-rl-hybrid-telescoping':        hybridTelescoping,
  'ch14-rl-dagger-loop':               daggerLoop,
  'ch14-rl-grid-vs-quadratic':         gridVsQuadratic,
  'ch14-rl-lqr-loop':                  lqrLoop,
  'ch14-rl-riccati-backward':          riccatiBackward,
  'ch14-rl-pendulum-physics':          pendulumPhysics,
  'ch14-rl-ilqr-cycle':                ilqrCycle,
  'ch14-rl-control-hierarchy':         controlHierarchy,
  'ch14-rl-control-vs-rl':             controlVsRl,
  'ch14-rl-cloning-to-reinforce':      cloningToReinforce,
  'ch14-rl-rare-reward-variance':      rareRewardVariance,
  'ch14-rl-log-derivative':            logDerivative,
  'ch14-rl-causality-credit':          causalityCredit,
  'ch14-rl-all-or-nothing':            allOrNothing,
  'ch14-rl-prefix-admissibility':      prefixAdmissibility,
  'ch14-rl-advantage-centering':       advantageCentering,
  'ch14-rl-variance-mechanisms':       varianceMechanisms,
  'ch14-rl-continuous-generalization': continuousGeneralization,
  'ch14-rl-ntk-coupling':              ntkCoupling,
  'ch14-rl-actor-critic-mlps':         actorCriticMlps,
  'ch14-rl-truncation-bootstrap':      truncationBootstrap,
};
