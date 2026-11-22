// Slide diagrams for chapter_deep-reinforcement-learning (§15 Deep RL).
import { C, TOKENS, grid, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// ═══════════════════════════════════════════════════════════════════════════
// §15.0 index.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 1. ch15-drl-naive-pg-waste
function naivePgWaste() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Three Bottlenecks of Naive On-Policy REINFORCE (~300k Steps)', { fs: 15, fw: 700, fill: C.ink });

  const cols = [
    { x: 16,  t: '1. Full-Episode Wait', s1: 'Cannot update until', s2: 'episode ends (T=200)', s3: 'High tail variance Ĝ_t', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 198, t: '2. Single-Use Discard', s1: '1,600 env steps drive', s2: 'only 1 gradient update,', s3: 'then thrown away!', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 380, t: '3. Unguarded Steps', s1: 'No bound on Δπ_θ(a|s)', s2: '1 noisy step can lock', s3: 'policy into collapse', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  cols.forEach((c) => {
    o += box(c.x, 42, 164, 148, c.fill, c.st, 8, 2);
    o += tx(c.x + 82, 68, c.t, { fs: 13.8, fw: 700, fill: c.dk });
    o += tx(c.x + 82, 102, c.s1, { fs: 12.2, fill: C.ink });
    o += tx(c.x + 82, 128, c.s2, { fs: 12.2, fill: C.ink });
    o += tx(c.x + 82, 158, c.s3, { fs: 12.2, fw: 700, fill: c.dk });
  });

  o += box(16, 206, 528, 38, C.lblue, C.blue, 7, 1.8);
  o += tx(280, 225, 'Chapter 15 Goal: Extract 10×–500× more learning per step while controlling bias', { fs: 12.2, fw: 700, fill: D.blue.dark });
  return svg(W, H, o);
}

// 2. ch15-drl-four-mechanisms
function fourMechanisms() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, 'Four Sample-Efficiency Mechanisms and Their Approximation Errors', { fs: 14.8, fw: 700, fill: C.ink });

  const rows = [
    { y: 40,  m: '1. Bootstrap Critic (§15.1)', err: "Young-critic bias e(s') in δ_t", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 94,  m: '2. Reuse Old Data (§15.2–7)', err: 'Distribution shift & moving targets', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 148, m: '3. Bound Shifts (§15.2–3)', err: 'Local clip limit or KL bias to π_ref', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { y: 202, m: '4. World Model (§15.6)', err: 'Compounding rollout error & compute', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  rows.forEach((r) => {
    o += box(18, r.y, 240, 44, r.fill, r.st, 7, 2);
    o += tx(138, r.y + 22, r.m, { fs: 12.8, fw: 700, fill: r.dk });
    o += arrow(262, r.y + 22, 290, r.y + 22, r.st);
    o += box(294, r.y, 248, 44, '#FFFFFF', r.st, 7, 1.8);
    o += tx(418, r.y + 22, r.err, { fs: 12.2, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

// 3. ch15-drl-triad-overview
function triadOverview() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'The Deadly Triad: Three Ingredients That Cause Value Divergence', { fs: 15, fw: 700, fill: C.ink });

  const boxes = [
    { x: 20,  y: 44,  t: '1. Bootstrapping', s: "Target y = r + γ V̂_w(s')", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 310, y: 44,  t: '2. Function Approx', s: 'Shared weights w couple states', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 165, y: 184, t: '3. Off-Policy Data', s: 'Replay or fixed log ≠ current π', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  boxes.forEach((b) => {
    o += box(b.x, b.y, 230, 62, b.fill, b.st, 8, 2);
    o += tx(b.x + 115, b.y + 22, b.t, { fs: 14, fw: 700, fill: b.dk });
    o += tx(b.x + 115, b.y + 44, b.s, { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  });

  o += box(130, 118, 300, 52, D.red.tint, D.red.base, 8, 2.2);
  o += tx(280, 136, 'All Three Together → Divergence!', { fs: 13.8, fw: 700, fill: D.red.dark });
  o += tx(280, 156, "||w||_∞ → ∞ (even on Baird's linear MDP)", { mono: true, fs: 11.2, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 4. ch15-drl-benchmark-gains
function benchmarkGains() {
  const W = 560, H = 262;
  let o = '';
  o += tx(280, 20, 'Sample-Efficiency Gains Across Chapter 15 Running Benchmarks', { fs: 15, fw: 700, fill: C.ink });

  o += box(18, 42, 524, 98, C.lblue, C.blue, 8, 2);
  o += tx(34, 62, 'Pendulum-v1 Swing-Up (Steps to Reach Return ≥ −200):', { fs: 13.5, fw: 700, fill: D.blue.dark, anchor: 'start' });
  o += box(34, 76, 492, 20, D.red.tint, D.red.base, 4, 1.5);
  o += tx(280, 86, 'Ch 14 REINFORCE: ~300,000 steps', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += box(34, 102, 180, 26, C.lamber, C.amber, 4, 1.8);
  o += tx(124, 115, '§15.5 SAC: 2.4k–11k', { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += box(228, 102, 298, 26, C.lgreen, C.green, 4, 1.8);
  o += tx(377, 115, '§15.6 World-Model MPC: 200–600 steps!', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  o += box(18, 154, 524, 92, C.lgreen, C.green, 8, 2);
  o += tx(34, 176, 'CartPole-v1 (Return 500) & Discrete Exact Benchmarks:', { fs: 13.5, fw: 700, fill: D.green.dark, anchor: 'start' });
  o += tx(280, 202, 'CartPole: 1-Step A2C ~60k  →  DQN ~40k  →  4-Epoch Clipped PPO ~25k', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(280, 226, 'Hazard Grid & Offline Lake: KL leash (β) & LCB (−κ/√n) stop reward hacking', { fs: 12, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.1 actor-critic.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 5. ch15-drl-mc-tail-vs-critic
function mcTailVsCritic() {
  const W = 560, H = 258;
  let o = '';
  o += box(18, 16, 524, 104, C.lamber, C.amber, 8, 2);
  o += tx(280, 36, 'Chapter 14 REINFORCE + Baseline: Wait for Sampled Tail Ĝ_{t+1}', { fs: 14, fw: 700, fill: D.orange.dark });
  o += box(34, 52, 492, 38, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(280, 71, 'w_t = r_t + γ (r_{t+1} + γ r_{t+2} + ···) − V̂_w(s_t)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(280, 106, 'Unbiased advantage sample, but high variance from T − t future random steps', { fs: 12, fill: C.ink });

  o += box(18, 136, 524, 106, C.lgreen, C.green, 8, 2);
  o += tx(280, 156, 'Chapter 15 Actor–Critic: Substitute Critic Prediction V̂_w(s_{t+1})', { fs: 14, fw: 700, fill: D.green.dark });
  o += box(34, 172, 492, 38, '#FFFFFF', C.green, 6, 1.8);
  o += tx(280, 191, 'δ_t = r_t + γ (1 − term_t) V̂_w(s_{t+1}) − V̂_w(s_t)', { mono: true, fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(280, 226, 'Computable after 1 step! Trades tail randomness for young-critic bias γ e(s_{t+1})', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 6. ch15-drl-cartpole-td-trace
function cartpoleTdTrace() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Tracing One CartPole Step by Hand (r_t = +1, γ = 0.99, V^∞ = 100)', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 198, C.lgreen, C.green, 9, 2);
  o += tx(144, 68, 'Calibrated Critic (V̂ = 100)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += box(34, 88, 220, 56, '#FFFFFF', C.green, 6, 1.8);
  o += tx(144, 106, 'Pole stays upright:', { fs: 12, fw: 600, fill: C.ink });
  o += tx(144, 128, 'δ_t = 1 + 0.99(100) − 100 = 0', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += box(34, 154, 220, 56, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 172, 'Pole falls (term = True):', { fs: 12, fw: 600, fill: C.ink });
  o += tx(144, 194, 'δ_t = 1 + 0 − 100 = −99!', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  o += box(292, 44, 248, 198, D.red.tint, D.red.base, 9, 2);
  o += tx(416, 68, 'Fresh Young Critic (V̂ ≈ 0)', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += box(306, 88, 220, 56, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(416, 106, 'Pole stays upright:', { fs: 12, fw: 600, fill: C.ink });
  o += tx(416, 128, 'δ_t = 1 + 0.99(0) − 0 = +1', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += box(306, 154, 220, 56, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(416, 172, 'Pole falls (term = True):', { fs: 12, fw: 600, fill: C.ink });
  o += tx(416, 194, 'δ_t = 1 + 0 − 0 = +1 (same!)', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(416, 226, 'Blind until critic catches up!', { fs: 12.2, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 7. ch15-drl-critic-error-decomp
function criticErrorDecomp() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Where Critic Error e(s) = V̂(s) − V^π(s) Enters the Policy Gradient', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(20, 40, 520, 40, D.purple.tint, C.purple, 7, 1.8);
  o += tx(280, 60, 'E[δ_t | s_t, a_t] = A^π(s_t, a_t) − e(s_t) + γ E[e(s_{t+1}) | s_t, a_t]', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });

  o += box(20, 96, 248, 146, C.lgreen, C.green, 8, 2);
  o += tx(144, 120, 'Current-State Error −e(s_t)', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(144, 148, 'Depends ONLY on state s_t', { fs: 12.2, fill: C.ink });
  o += tx(144, 174, 'E_{a_t}[e(s_t) ∇log π(a_t|s_t)] = 0', { mono: true, fs: 11.2, fw: 700, fill: D.green.dark });
  o += tx(144, 204, 'Adds ZERO bias to actor gradient!', { fs: 12.2, fw: 700, fill: C.ink });

  o += box(292, 96, 248, 146, D.red.tint, D.red.base, 8, 2);
  o += tx(416, 120, 'Next-State Error +γ e(s_{t+1})', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(416, 148, 's_{t+1} ~ P(· | s_t, a_t) depends on a_t', { fs: 12.2, fill: C.ink });
  o += tx(416, 174, 'E[e(s_{t+1}) ∇log π(a_t|s_t)] ≠ 0', { mono: true, fs: 11.2, fw: 700, fill: D.red.dark });
  o += tx(416, 204, 'Sole source of actor bias in TD!', { fs: 12.2, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 8. ch15-drl-semi-gradient-stop
function semiGradientStop() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'Semi-Gradient Critic Regression: Stop-Gradient on the Target y_t', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 48, 160, 60, N.panel, C.gray, 8, 2);
  o += tx(100, 70, 'Next State s_{t+1}', { fs: 13.5, fw: 700, fill: C.ink });
  o += tx(100, 92, 'V̂_w(s_{t+1})', { mono: true, fs: 12.5, fill: C.muted });

  o += arrow(182, 78, 222, 78, C.gray, true);

  o += box(226, 48, 176, 60, D.red.tint, D.red.base, 8, 2);
  o += tx(314, 70, 'stop_grad(y_t)', { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  o += tx(314, 92, 'r_t + γ V̂_w(s_{t+1})', { mono: true, fs: 11.8, fill: C.ink });

  o += box(20, 136, 160, 60, C.lblue, C.blue, 8, 2);
  o += tx(100, 158, 'Current State s_t', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(100, 180, 'V̂_w(s_t) (live w)', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += arrow(182, 166, 222, 166, C.blue);
  o += arrow(314, 110, 314, 132, C.gray);

  o += box(226, 136, 314, 60, C.lgreen, C.green, 8, 2);
  o += tx(383, 158, 'Semi-Gradient Step on w', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(383, 180, 'Δw = α · δ_t · ∇_w V̂_w(s_t)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += tx(280, 224, 'Differentiates only through prediction V̂_w(s_t), never through target V̂_w(s_{t+1})', { fs: 12.2, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 9. ch15-drl-fitted-td-loop
function fittedTdLoop() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'Three Systems Choices in Batched Fitted-TD Actor–Critic', { fs: 15, fw: 700, fill: C.ink });

  const stages = [
    { x: 16,  t: '1. Refit Critic First', s1: '20 passes on batch b', s2: 'BEFORE scoring actor', s3: 'Fails: 25% → 5%', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, t: '2. Recompute Targets', s1: "Fresh y_t = r + γV̂(s')", s2: 'inside each of 20 passes', s3: '20-step backup flow', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, t: '3. Raw δ_t + Grad Cap', s1: 'No batch std division!', s2: 'Clip ||∇_θ|| ≤ 0.5', s3: 'Anneals step 8× late', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  stages.forEach((s, i) => {
    o += box(s.x, 46, 164, 184, s.fill, s.st, 8, 2);
    o += tx(s.x + 82, 72, s.t, { fs: 13.8, fw: 700, fill: s.dk });
    o += tx(s.x + 82, 108, s.s1, { fs: 12, fw: 600, fill: C.ink });
    o += tx(s.x + 82, 134, s.s2, { fs: 12, fill: C.ink });
    o += box(s.x + 12, 156, 140, 54, '#FFFFFF', s.st, 6, 1.5);
    o += tx(s.x + 82, 183, s.s3, { fs: 11.8, fw: 700, fill: s.dk });
    if (i < 2) o += arrow(s.x + 166, 138, s.x + 196, 138, C.gray);
  });
  return svg(W, H, o);
}

// 10. ch15-drl-gae-mixture
function gaeMixture() {
  const W = 560, H = 262;
  let o = '';
  o += tx(280, 20, 'Generalized Advantage Estimation (GAE): Geometric (1−λ)λ^{n−1} Mixture', { fs: 14.5, fw: 700, fill: C.ink });

  const rows = [
    { y: 44,  w: '(1 − λ)',     f: 'Â_t^(1) = r_t + γ V̂(s_{t+1}) − V̂(s_t) = δ_t', col: C.blue, dk: D.blue.dark },
    { y: 96,  w: '(1 − λ) λ',   f: 'Â_t^(2) = r_t + γ r_{t+1} + γ² V̂(s_{t+2}) − V̂(s_t)', col: C.purple, dk: D.purple.dark },
    { y: 148, w: '(1 − λ) λ²',  f: 'Â_t^(3) = r_t + γ r_{t+1} + γ² r_{t+2} + γ³ V̂(s_{t+3}) − V̂', col: C.amber, dk: D.orange.dark },
  ];
  rows.forEach((r) => {
    o += box(16, r.y, 114, 42, N.panel, r.col, 6, 1.8);
    o += tx(73, r.y + 21, r.w, { mono: true, fs: 12, fw: 700, fill: r.dk });
    o += box(142, r.y, 402, 42, '#FFFFFF', r.col, 6, 1.8);
    o += tx(343, r.y + 21, r.f, { mono: true, fs: 11.2, fw: 600, fill: C.ink });
  });

  o += box(16, 202, 528, 44, C.lgreen, C.green, 7, 2);
  o += tx(280, 224, 'λ = 0 → 1-Step TD δ_t   ·   λ ∈ [0.9, 0.95] → Optimal MSE   ·   λ = 1 → MC Ĝ_t − V̂(s_t)', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 11. ch15-drl-gae-telescoping
function gaeTelescoping() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Telescoping Sum: O(T²) Multi-Step Returns Collapse into O(T) Scan', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 42, 528, 86, C.lblue, C.blue, 8, 2);
  o += tx(280, 62, 'Step 1: Interior Value Predictions ±γ^l V̂(s_{t+l}) Cancel in Pairs', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 88, 'Ĝ_t^(n) − V̂(s_t) = (r_t + γV̂_{t+1} − V̂_t) + γ(r_{t+1} + γV̂_{t+2} − V̂_{t+1}) + ···', { mono: true, fs: 10.5, fw: 600, fill: C.ink });
  o += tx(280, 110, '= δ_t + γ δ_{t+1} + γ² δ_{t+2} + ··· + γ^{n−1} δ_{t+n−1}', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += box(16, 142, 528, 96, C.lgreen, C.green, 8, 2);
  o += tx(280, 164, 'Step 2: Sum Weights (1−λ) ∑_{n>l} λ^{n−1} = λ^l → Single Backward Scan', { fs: 13.2, fw: 700, fill: D.green.dark });
  o += box(44, 178, 472, 42, '#FFFFFF', C.green, 6, 2);
  o += tx(280, 199, 'Â_t^GAE = ∑_{l=0}^∞ (γλ)^l δ_{t+l} = δ_t + (γλ) Â_{t+1}^GAE', { mono: true, fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.2 ppo.md (5 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 12. ch15-drl-single-vs-multi-epoch
function singleVsMultiEpoch() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Extracting Multiple Updates from a Single 8-Episode Batch', { fs: 15, fw: 700, fill: C.ink });

  const cols = [
    { x: 16,  t: '1-Step A2C (§15.1)', s1: '1 gradient step,', s2: 'then discard 4,000 steps', s3: 'Needs ≥ 50k steps', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, t: '20 Epochs Unclipped', s1: '20 passes on ρ_t Â_t', s2: 'Ratios explode to 0 / 2', s3: '50% die at return 9!', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 380, t: 'Clipped PPO (§15.2)', s1: '4–20 passes on L^CLIP', s2: 'Caps incentive at 1±0.2', s3: '~25k steps (8/8 win)!', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cols.forEach((c) => {
    o += box(c.x, 44, 164, 192, c.fill, c.st, 8, 2);
    o += tx(c.x + 82, 70, c.t, { fs: 13.8, fw: 700, fill: c.dk });
    o += tx(c.x + 82, 106, c.s1, { fs: 12.2, fill: C.ink });
    o += tx(c.x + 82, 132, c.s2, { fs: 12.2, fill: C.ink });
    o += box(c.x + 12, 158, 140, 54, '#FFFFFF', c.st, 6, 1.6);
    o += tx(c.x + 82, 185, c.s3, { fs: 11.8, fw: 700, fill: c.dk });
  });
  return svg(W, H, o);
}

// 13. ch15-drl-importance-cancellation
function importanceCancellation() {
  const W = 560, H = 254;
  let o = '';
  o += tx(280, 20, 'Change of Measure: Environment Physics Cancel in Trajectory Ratio', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(16, 42, 528, 110, C.lblue, C.blue, 9, 2);
  o += tx(280, 68, 'Num P(τ; θ): μ₀(s₀) · ∏_t π_θ(a_t | s_t) · ∏_t P(s_{t+1} | s_t, a_t)', { mono: true, fs: 10.8, fw: 700, fill: D.blue.dark });
  o += line(40, 92, 520, 92, C.blue, 2);
  o += tx(280, 118, 'Den P(τ; θ_old): μ₀(s₀) · ∏_t π_old(a_t | s_t) · ∏_t P(s_{t+1} | s_t, a_t)', { mono: true, fs: 10.8, fw: 700, fill: C.muted });

  o += arrow(280, 154, 280, 178, C.green);

  o += box(16, 182, 528, 56, C.lgreen, C.green, 8, 2);
  o += tx(280, 202, 'Exact Model-Free Likelihood Ratio (μ₀ and unknown P cancel!):', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(280, 224, 'P(τ; θ) / P(τ; θ_old) = ∏_t [ π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t) ]', { mono: true, fs: 11.2, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 14. ch15-drl-ratio-explosion
function ratioExplosion() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Curse of Horizon: Trajectory Product W_T = ∏ ρ_t vs. Single-Step ρ_t', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 196, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 68, 'Trajectory Ratio W_T = ∏ ρ_t', { fs: 13.8, fw: 700, fill: D.red.dark });
  o += tx(144, 92, 'Var(W_T) = 1.04^T − 1', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += box(34, 108, 220, 86, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 128, 'T = 10 steps:   Var = 0.48', { mono: true, fs: 11.8, fill: C.ink });
  o += tx(144, 152, 'T = 100 steps:  Var = 49.5', { mono: true, fs: 11.8, fw: 600, fill: D.orange.dark });
  o += tx(144, 176, 'T = 500 steps:  Var = 3.3×10⁸!', { mono: true, fs: 11.8, fw: 700, fill: D.red.dark });
  o += tx(144, 218, '1 lucky rollout dominates batch!', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 44, 248, 196, C.lgreen, C.green, 9, 2);
  o += tx(416, 68, 'Per-Step Ratio ρ_t(θ)', { fs: 13.8, fw: 700, fill: D.green.dark });
  o += tx(416, 92, 'ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)', { mono: true, fs: 10.6, fw: 700, fill: C.ink });
  o += box(306, 108, 220, 86, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 128, 'Drops history product ρ₀···ρ_{t−1}', { fs: 12, fill: C.ink });
  o += tx(416, 152, 'Var(ρ_t) = 0.04 for any T!', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(416, 176, '∇_θ L̂(θ_old) = exact policy grad', { mono: true, fs: 11, fw: 600, fill: C.ink });
  o += tx(416, 218, 'Valid while d^{π_θ} ≈ d^{π_{θ_old}}', { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 15. ch15-drl-perf-diff-lemma
function perfDiffLemma() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Performance Difference Lemma: True Improvement vs. Surrogate L̄(θ)', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 42, 528, 88, C.lblue, C.blue, 8, 2);
  o += tx(280, 64, 'Exact Return Gap J(θ) − J(θ_old) (Kakade & Langford, 2002):', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += box(30, 78, 500, 38, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(280, 97, '(1−γ)⁻¹ ∑_s d_γ^{π_θ}(s) ∑_a π_θ(a|s) A^old(s,a)   [NEW occupancy d^{π_θ}]', { mono: true, fs: 11, fw: 700, fill: C.ink });

  o += box(16, 144, 528, 96, C.lgreen, C.green, 8, 2);
  o += tx(280, 166, 'Surrogate L̄(θ) Replaces d_γ^{π_θ}(s) with Collected d_γ^{π_{θ_old}}(s):', { fs: 13.2, fw: 700, fill: D.green.dark });
  o += box(30, 180, 500, 36, '#FFFFFF', C.green, 6, 1.8);
  o += tx(280, 198, '(1−γ)⁻¹ ∑_s d_γ^{π_{θ_old}}(s) ∑_a π_θ(a|s) A^old(s,a)   [OLD occupancy]', { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  o += tx(280, 228, 'Mismatch |J(θ) − J(θ_old) − L̄(θ)| is bounded by O(max_s D_KL(π_old || π_θ))', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 16. ch15-drl-clip-drift-entropy
function clipDriftEntropy() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Why Clipped Ratios Still Drift Outside [0.8, 1.2] — and Entropy Defense', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 194, C.lamber, C.amber, 9, 2);
  o += tx(144, 68, 'Why Ratios Cross 1 ± ε_clip', { fs: 14, fw: 700, fill: D.orange.dark });
  o += box(34, 86, 220, 62, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(144, 106, '1. Shared MLP weights θ:', { fs: 12, fw: 700, fill: C.ink });
  o += tx(144, 128, 'Unclipped samples move θ for all!', { fs: 11.8, fill: C.ink });
  o += box(34, 158, 220, 62, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(144, 178, '2. Adam first-moment m_t:', { fs: 12, fw: 700, fill: C.ink });
  o += tx(144, 200, 'Momentum coasts after grad → 0', { fs: 11.8, fill: C.ink });

  o += box(292, 44, 248, 194, C.lgreen, C.green, 9, 2);
  o += tx(416, 68, 'Two Stabilizing Restorers', { fs: 14, fw: 700, fill: D.green.dark });
  o += box(306, 86, 220, 62, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 106, '1. Losing-side unclipped slope:', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 128, 'Pulls ρ < 0.8 (Â>0) back inside!', { fs: 11.8, fill: C.ink });
  o += box(306, 158, 220, 62, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 178, '2. Entropy bonus +β_H H(π_θ):', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 200, 'Repels π_θ from 0/1 saturation', { fs: 11.8, fill: C.ink });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.3 regularized.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 17. ch15-drl-rlhf-pipeline
function rlhfPipeline() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'The Three-Stage RLHF Pipeline and the KL Leash to π_ref', { fs: 15, fw: 700, fill: C.ink });

  const stages = [
    { x: 16,  t: '1. Reference Rollouts', s1: 'Sample trajectories', s2: "τ, τ' ~ π_ref", s3: 'Inside π_ref support', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, t: '2. Fit Reward Model r̂', s1: 'Bradley–Terry on', s2: "human pairs τ ≻ τ'", s3: 'Blind outside π_ref!', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, t: '3. KL-Penalized RL', s1: 'max E_π[r̂] − β D_KL', s2: 'Anchors π near π_ref', s3: 'Stops Goodhart hacking', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  stages.forEach((s, i) => {
    o += box(s.x, 46, 164, 186, s.fill, s.st, 8, 2);
    o += tx(s.x + 82, 72, s.t, { fs: 13.8, fw: 700, fill: s.dk });
    o += tx(s.x + 82, 108, s.s1, { fs: 12.2, fill: C.ink });
    o += tx(s.x + 82, 134, s.s2, { mono: true, fs: 11.8, fw: 700, fill: C.ink });
    o += box(s.x + 12, 158, 140, 54, '#FFFFFF', s.st, 6, 1.5);
    o += tx(s.x + 82, 185, s.s3, { fs: 11.5, fw: 700, fill: s.dk });
    if (i < 2) o += arrow(s.x + 166, 138, s.x + 196, 138, C.gray);
  });
  return svg(W, H, o);
}

// 18. ch15-drl-bradley-terry
function bradleyTerry() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'Bradley–Terry Preference Regression on Discounted Visit Counts', { fs: 14.8, fw: 700, fill: C.ink });

  const xs = [88, 276, 462], y = 104;
  o += box(xs[0] - 68, y - 40, 136, 80, C.lblue, C.blue, 8, 2);
  o += tx(xs[0], y - 16, 'Count Difference', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(xs[0], y + 8, "Δc = c(τ) − c(τ')", { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(xs[0], y + 28, 'in R^{64}', { mono: true, fs: 11.5, fill: C.muted });

  o += arrow(158, y, 194, y, C.blue);

  o += box(xs[1] - 78, y - 40, 156, 80, D.purple.tint, C.purple, 8, 2);
  o += tx(xs[1], y - 16, 'Return Difference', { fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(xs[1], y + 8, "r̂(τ) − r̂(τ') = Δcᵀ r̂", { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(xs[1], y + 28, 'Shaping Φ(s₀) cancels!', { fs: 11.5, fw: 600, fill: D.purple.dark });

  o += arrow(356, y, 386, y, C.purple);

  o += box(xs[2] - 72, y - 40, 144, 80, C.lgreen, C.green, 8, 2);
  o += tx(xs[2], y - 16, 'Win Probability', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(xs[2], y + 8, "P(τ ≻ τ') =", { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(xs[2], y + 28, "σ(r̂(τ) − r̂(τ'))", { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  o += box(20, 174, 520, 60, C.lamber, C.amber, 8, 1.8);
  o += tx(280, 194, 'L₂ Weight Decay (λ_wd = 0.02) on Thinly Visited Hazard Cells (5 & 6):', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(280, 216, 'Only 2–4 visits in 1,000 rollouts → r̂_hazard shrinks to ≈ 0.00 instead of −1.50!', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 19. ch15-drl-exponential-tilt-flow
function exponentialTiltFlow() {
  const W = 560, H = 254;
  let o = '';
  o += tx(280, 20, 'Closed-Form KL-Regularized Optimum: Exponentially Tilted Reference', { fs: 14.5, fw: 700, fill: C.ink });

  const y = 100;
  o += box(20, y - 38, 140, 76, C.lblue, C.blue, 8, 2);
  o += tx(90, y - 14, 'Reference Prior', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(90, y + 12, 'π_ref(a)', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  o += tx(176, y, '×', { fs: 22, fw: 700, fill: C.ink });

  o += box(192, y - 38, 152, 76, C.lamber, C.amber, 8, 2);
  o += tx(268, y - 14, 'Boltzmann Tilt', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(268, y + 12, 'exp(r(a) / β)', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  o += arrow(348, y, 380, y, C.green);

  o += box(384, y - 38, 156, 76, C.lgreen, C.green, 8, 2);
  o += tx(462, y - 14, 'Optimal Policy π*', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(462, y + 12, 'π_ref(a) e^{r/β} / Z', { mono: true, fs: 11.8, fw: 700, fill: C.ink });

  o += box(20, 166, 520, 70, D.purple.tint, C.purple, 8, 1.8);
  o += tx(280, 190, "Partition Z = ∑ π_ref(a') e^{r(a')/β}   ·   Optimal Objective Value = β log Z", { mono: true, fs: 11.8, fw: 700, fill: D.purple.dark });
  o += tx(280, 216, 'Support invariant: π_ref(a) = 0 ⟹ π*(a) = 0   ·   Gibbs: obj(π) = β log Z − β D_KL(π || π*)', { fs: 11.8, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 20. ch15-drl-trust-vs-anchor
function trustVsAnchor() {
  const W = 560, H = 258;
  let o = '';
  o += box(20, 18, 248, 222, C.lamber, C.amber, 9, 2);
  o += tx(144, 42, 'Moving Trust Region (§15.2)', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(144, 66, 'Penalty to PREVIOUS iterate π_k', { fs: 12.2, fill: C.ink });
  o += box(34, 84, 220, 62, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(144, 104, 'π_k(a) ∝ π_ref(a) e^{k · r(a)/β̃}', { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(144, 128, 'Effective temp β̃ / k → 0', { mono: true, fs: 11.3, fill: C.ink });
  o += tx(144, 174, 'Bounds each step along the PATH,', { fs: 12.2, fill: C.ink });
  o += tx(144, 198, 'but still converges to greedy', { fs: 12.2, fill: C.ink });
  o += tx(144, 220, 'argmax_a r̂(a) (reward hacks!)', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 18, 248, 222, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Fixed Reference Penalty (§15.3)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(416, 66, 'Penalty to FROZEN reference π_ref', { fs: 12.2, fill: C.ink });
  o += box(306, 84, 220, 62, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 104, 'π*(a) ∝ π_ref(a) e^{r(a)/β}', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(416, 128, 'Fixed temperature β > 0 forever', { mono: true, fs: 11.3, fill: C.ink });
  o += tx(416, 174, 'Changes the DESTINATION itself:', { fs: 12.2, fill: C.ink });
  o += tx(416, 198, 'Reverse KL suppresses bad actions', { fs: 12.2, fill: C.ink });
  o += tx(416, 220, 'exponentially ∝ e^{−Δ(a)/β}!', { fs: 12.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.4 dqn.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 21. ch15-drl-naive-q-explosion
function naiveQExplosion() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Why Naive Neural Q-Learning Explodes Past 10⁸ on CartPole', { fs: 15, fw: 700, fill: C.ink });

  const bw = 224, bh = 72;
  const boxes = [
    { x: 24,  y: 44,  t: '1. Push Q_w(s_t, a_t) Up', s: 'y_t = 1 + 0.99 max Q_w(s_{t+1})', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 312, y: 44,  t: '2. Shared Weights Coupled', s: "k(s', s) > 0 lifts Q_w(s') too!", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 312, y: 158, t: '3. Next Target Jumps Higher', s: 'y_{t+1} = 1 + 0.99 max Q_w climbs', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 24,  y: 158, t: '4. Runaway Feedback Loop', s: 'Q_w(s₀) > 10⁸, policy return ≈ 9!', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  boxes.forEach((b) => {
    o += box(b.x, b.y, bw, bh, b.fill, b.st, 8, 2);
    o += tx(b.x + bw / 2, b.y + 24, b.t, { fs: 13.5, fw: 700, fill: b.dk });
    o += tx(b.x + bw / 2, b.y + 48, b.s, { mono: true, fs: 11.2, fw: 600, fill: C.ink });
  });
  o += arrow(250, 80, 308, 80, C.blue);
  o += arrow(424, 118, 424, 154, C.amber);
  o += arrow(308, 194, 250, 194, D.red.base);
  o += arrow(136, 154, 136, 118, D.red.base);
  return svg(W, H, o);
}

// 22. ch15-drl-norm-mismatch
function normMismatch() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Why Least-Squares Projection Breaks Supremum-Norm Contraction', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 194, C.lgreen, C.green, 9, 2);
  o += tx(144, 68, 'Exact Bellman Backup T', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(144, 94, 'Contracts in Supremum Norm ||·||_∞:', { fs: 12.2, fill: C.ink });
  o += box(34, 110, 220, 44, '#FFFFFF', C.green, 6, 1.8);
  o += tx(144, 132, "||TQ − TQ'||_∞ ≤ γ ||Q − Q'||_∞", { mono: true, fs: 11.2, fw: 700, fill: D.green.dark });
  o += tx(144, 180, 'Bounds WORST-CASE error across', { fs: 12.2, fill: C.ink });
  o += tx(144, 204, 'every state-action pair (s, a)', { fs: 12.2, fw: 700, fill: D.green.dark });

  o += box(292, 44, 248, 194, D.red.tint, D.red.base, 9, 2);
  o += tx(416, 68, 'Neural Regression Π_d', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(416, 94, 'Projects in Visitation L₂(d) Norm:', { fs: 12.2, fill: C.ink });
  o += box(306, 110, 220, 44, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(416, 132, 'min_w E_{(s,a)~d}[(Q_w − y)²]', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(416, 180, 'Shrinking error on visited states', { fs: 12.2, fill: C.ink });
  o += tx(416, 204, 'can EXPAND ||Π_d TQ||_∞ > ||Q||_∞!', { fs: 12.2, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 23. ch15-drl-ring-buffer
function ringBuffer() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Experience Replay: Preallocated Circular Ring Buffer (200k Slots)', { fs: 14.8, fw: 700, fill: C.ink });

  const s = 54, g = 8, x0 = 26, y0 = 66;
  const cells = ['τ_0', 'τ_1', 'τ_2', 'τ_3', 'τ_4', '···', 'τ_N'];
  cells.forEach((c, i) => {
    const x = x0 + i * (s + g);
    const isPtr = i === 3;
    const isSampled = i === 1 || i === 4 || i === 6;
    const fill = isPtr ? C.lamber : (isSampled ? C.lgreen : C.lblue);
    const st = isPtr ? C.amber : (isSampled ? C.green : C.blue);
    o += box(x, y0, s, 46, fill, st, 6, 2);
    o += tx(x + s / 2, y0 + 23, c, { mono: true, fs: 13, fw: 700, fill: C.ink });
  });

  const px = x0 + 3 * (s + g) + s / 2;
  o += tx(px, 42, 'ptr = (ptr + 1) % cap', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });

  [1, 4, 6].forEach((idx) => {
    const cx = x0 + idx * (s + g) + s / 2;
    o += arrow(cx, y0 + 48, 280, 156, C.green);
  });

  o += box(24, 160, 512, 80, C.lgreen, C.green, 8, 2);
  o += tx(280, 182, 'Uniform Minibatch Gather (batch_size = 128, 1-step fragments)', { fs: 13.8, fw: 700, fill: D.green.dark });
  o += tx(280, 206, 'Breaks temporal trajectory correlation  ·  Reuses each transition ~64×', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(280, 226, "Stores float(terminated) so 500-step truncations still bootstrap Q_{w⁻}(s')", { mono: true, fs: 11.2, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 24. ch15-drl-double-dqn-split
function doubleDqnSplit() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Double DQN: Decoupling Greedy Selection from Target Evaluation', { fs: 15, fw: 700, fill: C.ink });

  o += box(18, 96, 102, 66, C.lblue, C.blue, 8, 2);
  o += tx(69, 120, 'Next State', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(69, 142, "s' ~ Buffer", { mono: true, fs: 12, fill: C.ink });

  o += arrow(122, 112, 164, 78, C.blue);
  o += arrow(122, 144, 164, 178, C.purple);

  o += box(168, 42, 202, 72, C.lblue, C.blue, 8, 2);
  o += tx(269, 64, '1. Online Net Q_w (Select)', { fs: 13.2, fw: 700, fill: D.blue.dark });
  o += tx(269, 90, "a* = argmax_{a'} Q_w(s', a')", { mono: true, fs: 11.3, fw: 700, fill: C.ink });

  o += box(168, 144, 202, 72, D.purple.tint, C.purple, 8, 2);
  o += tx(269, 166, '2. Target Net Q_{w⁻} (Eval)', { fs: 13.2, fw: 700, fill: D.purple.dark });
  o += tx(269, 192, "Gather Q_{w⁻}(s', a*)", { mono: true, fs: 11.8, fw: 700, fill: C.ink });

  o += arrow(269, 116, 269, 140, C.blue);
  o += arrow(372, 180, 398, 180, C.green);

  o += box(402, 134, 140, 92, C.lgreen, C.green, 8, 2);
  o += tx(472, 160, 'DDQN Target', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(472, 184, "r + γ Q_{w⁻}(s',a*)", { mono: true, fs: 11.2, fw: 700, fill: C.ink });
  o += tx(472, 206, 'Holds ≤ 100!', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.5 sac.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 25. ch15-drl-dqn-vs-sac-actor
function dqnVsSacActor() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, C.lblue, C.blue, 9, 2);
  o += tx(144, 42, 'Discrete Actions (DQN)', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(144, 66, 'A = {left, right} (|A| = 2)', { mono: true, fs: 12, fill: C.ink });
  o += box(34, 86, 220, 64, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(144, 108, 'Target & greedy action use', { fs: 12.2, fill: C.ink });
  o += tx(144, 130, "row scan max_{a'} Q_w(s', a')", { mono: true, fs: 11.3, fw: 700, fill: D.blue.dark });
  o += tx(144, 178, 'Impossible when action is a', { fs: 12.2, fill: C.ink });
  o += tx(144, 202, 'continuous torque u ∈ [−2, 2]!', { fs: 12.2, fw: 700, fill: D.red.dark });

  o += box(292, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Continuous Control (SAC)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(416, 66, 'Actor π_θ(s) + Critic Q_w(s, a)', { mono: true, fs: 12, fill: C.ink });
  o += box(306, 86, 220, 64, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 108, 'Actor samples ã_θ(s, z) and climbs', { fs: 12, fill: C.ink });
  o += tx(416, 130, 'pathwise ∂Q_w/∂a · ∂ã_θ/∂θ', { mono: true, fs: 11.3, fw: 700, fill: D.green.dark });
  o += tx(416, 178, '20× lower variance than score fn', { fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(416, 202, '300k steps → 3,000 steps!', { fs: 12.8, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 26. ch15-drl-ddpg-td3-sac
function ddpgTd3Sac() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Evolution of Off-Policy Continuous Actor–Critics', { fs: 15, fw: 700, fill: C.ink });

  const cols = [
    { x: 16,  t: '1. DDPG (2016)', s1: 'Deterministic μ_θ(s)', s2: 'Single critic Q_{w⁻}', s3: 'Exploits Q peaks!', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 198, t: '2. TD3 (2018)', s1: 'Deterministic μ_θ(s)', s2: 'Twin min(Q₁, Q₂)', s3: 'Ad-hoc action noise', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, t: '3. SAC (2018)', s1: 'Squashed Gaussian π_θ', s2: 'min(Q₁,Q₂) − β log π', s3: 'Max-entropy + dual β̂', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cols.forEach((c, i) => {
    o += box(c.x, 44, 164, 194, c.fill, c.st, 8, 2);
    o += tx(c.x + 82, 70, c.t, { fs: 14, fw: 700, fill: c.dk });
    o += tx(c.x + 82, 106, c.s1, { mono: true, fs: 11.3, fw: 700, fill: C.ink });
    o += tx(c.x + 82, 134, c.s2, { mono: true, fs: 11.3, fill: C.ink });
    o += box(c.x + 12, 160, 140, 56, '#FFFFFF', c.st, 6, 1.6);
    o += tx(c.x + 82, 188, c.s3, { fs: 11.8, fw: 700, fill: c.dk });
    if (i < 2) o += arrow(c.x + 166, 140, c.x + 196, 140, C.gray);
  });
  return svg(W, H, o);
}

// 27. ch15-drl-maxent-objective
function maxentObjective() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'Maximum-Entropy RL: Rewarding Future Entropy Inside the Return', { fs: 14.8, fw: 700, fill: C.ink });

  const xs = [92, 280, 468], y = 104;
  ['Step t = 0', 'Step t = 1', 'Step t = 2'].forEach((lab, i) => {
    const cx = xs[i];
    o += box(cx - 72, y - 48, 144, 96, C.lblue, C.blue, 8, 2);
    o += tx(cx, y - 24, lab, { fs: 13.5, fw: 700, fill: D.blue.dark });
    o += tx(cx, y + 2, `r_${i} (task reward)`, { mono: true, fs: 11.8, fw: 700, fill: C.ink });
    o += tx(cx, y + 26, `+ β H(π(· | s_${i}))`, { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
    if (i < 2) o += arrow(cx + 74, y, xs[i + 1] - 74, y, C.blue);
  });

  o += box(16, 174, 528, 62, C.lgreen, C.green, 8, 1.8);
  o += tx(280, 196, 'J(π) = E_π[ ∑ γ^t ( r_t + β H(π(· | s_t)) ) ]   with H(π) = −E[log π(a | s)]', { mono: true, fs: 11.2, fw: 700, fill: D.green.dark });
  o += tx(280, 220, 'Equivalent to −β D_KL(π || unif) up to log(2c); critic values multi-option future states!', { fs: 11.8, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 28. ch15-drl-soft-target-sampling
function softTargetSampling() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Soft Policy Evaluation: Replay Transition + One Fresh Live Action', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(20, 48, 190, 68, C.lblue, C.blue, 8, 2);
  o += tx(115, 70, 'From Replay Buffer D', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(115, 94, "(s, a, r, s', term)", { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += box(20, 140, 190, 68, D.purple.tint, C.purple, 8, 2);
  o += tx(115, 162, 'From Live Actor π_θ', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(115, 186, "ã' ~ π_θ(· | s')", { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += arrow(212, 82, 256, 114, C.blue);
  o += arrow(212, 174, 256, 142, C.purple);

  o += box(260, 62, 280, 128, C.lgreen, C.green, 9, 2);
  o += tx(400, 88, 'Single-Sample Soft Target y', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(400, 118, 'y = r + γ(1 − term) ×', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(400, 142, "[ min_{j=1,2} Q_{w_j⁻}(s', ã')", { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });
  o += tx(400, 166, "  − β log π_θ(ã' | s') ]", { mono: true, fs: 11.8, fw: 700, fill: D.purple.dark });

  o += tx(280, 228, "Zero importance weights needed: continuation action ã' is drawn from current π_θ!", { fs: 12.2, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 29. ch15-drl-clip-vs-tanh
function clipVsTanh() {
  const W = 560, H = 258;
  let o = '';
  o += box(20, 18, 248, 222, D.red.tint, D.red.base, 9, 2);
  o += tx(144, 42, 'Hard Clipping clip(u, −c, c)', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += box(36, 62, 216, 86, '#FFFFFF', D.red.base, 6, 1.8);
  o += line(56, 128, 100, 128, D.red.base, 3);
  o += line(100, 128, 188, 80, C.blue, 2.5);
  o += line(188, 80, 232, 80, D.red.base, 3);
  o += tx(144, 172, '1. Zero slope d/du = 0 for |u| > c', { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(144, 194, '(dead pathwise actor gradient!)', { fs: 11.8, fill: C.ink });
  o += tx(144, 218, '2. Point mass at ±c breaks log π', { fs: 12, fw: 700, fill: D.red.dark });

  o += box(292, 18, 248, 222, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'Smooth Squash a = c tanh(u)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += box(308, 62, 216, 86, '#FFFFFF', C.green, 6, 1.8);
  o += `<path d="M 328 128 C 378 128, 454 80, 504 80" fill="none" stroke="${C.green}" stroke-width="3"/>`;
  o += tx(416, 172, '1. Strictly positive slope everywhere:', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 194, 'da/du = c(1 − tanh² u) > 0', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(416, 218, '2. Exact invertible Jacobian density', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 30. ch15-drl-dual-temperature
function dualTemperature() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Automatic Temperature Tuning: Dual Feedback Loop for log β', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 42, 520, 52, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 60, 'Dual Loss: J(β) = −log β · E[log π_θ(a | s) + H̄]   (H̄ = −dim(A) = −1.0)', { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 80, 'Gradient: ∂J / ∂(log β) = Ĥ_batch − H̄', { mono: true, fs: 12.2, fw: 700, fill: C.ink });

  o += box(20, 112, 248, 126, C.lblue, C.blue, 8, 2);
  o += tx(144, 136, 'Case 1: Policy Too Random', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(144, 162, 'Ĥ_batch > H̄  (grad > 0)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(144, 190, 'Adam steps log β DOWN ↓', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(144, 214, 'Makes entropy cheaper to give up', { fs: 12, fill: C.ink });

  o += box(292, 112, 248, 126, C.lamber, C.amber, 8, 2);
  o += tx(416, 136, 'Case 2: Policy Too Narrow', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(416, 162, 'Ĥ_batch < H̄  (grad < 0)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(416, 190, 'Adam steps log β UP ↑', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(416, 214, 'Boosts entropy reward to widen π_θ', { fs: 12, fill: C.ink });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.6 model-based.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 31. ch15-drl-scalar-vs-physics
function scalarVsPhysics() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'What One Transition (s_t, a_t, r_t, s_{t+1}) Teaches', { fs: 15.5, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 194, C.lamber, C.amber, 9, 2);
  o += tx(144, 68, 'Model-Free Critic (SAC)', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += box(34, 90, 220, 52, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(144, 116, '1 Scalar Target: y_t ∈ R¹', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(144, 168, 'Discards where s_{t+1} moved!', { fs: 12.2, fill: C.ink });
  o += tx(144, 192, 'Needs 2,400–7,500 steps', { fs: 12.2, fw: 700, fill: D.orange.dark });
  o += tx(144, 216, 'to propagate swing-up value', { fs: 12.2, fill: C.ink });

  o += box(292, 44, 248, 194, C.lgreen, C.green, 9, 2);
  o += tx(416, 68, 'Learned World Model f̂_φ', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += box(306, 90, 220, 52, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 116, '4D Target: (Δs_t, r_t) ∈ R⁴', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(416, 168, 'Learns global gravity & inertia', { fs: 12.2, fill: C.ink });
  o += tx(416, 192, 'Swings up in 200–600 steps', { fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(416, 216, '(just 1 to 3 episodes!)', { fs: 12.2, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 32. ch15-drl-delta-residual-net
function deltaResidualNet() {
  const W = 560, H = 252;
  let o = '';
  o += tx(280, 20, 'DynamicsModel: Standardized State-Delta Residual Architecture', { fs: 14.8, fw: 700, fill: C.ink });

  const y = 104;
  o += box(20, y - 34, 114, 68, C.lblue, C.blue, 8, 2);
  o += tx(77, y - 10, 'Input (s, a)', { fs: 13.2, fw: 700, fill: D.blue.dark });
  o += tx(77, y + 14, '(x − μ_x)/σ_x', { mono: true, fs: 11.5, fill: C.ink });

  o += arrow(136, y, 170, y, C.blue);

  o += box(174, y - 34, 174, 68, D.purple.tint, C.purple, 8, 2);
  o += tx(261, y - 10, '2×64 SiLU MLP', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(261, y + 14, 'Predicts (Δŝ, r̂)', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += arrow(350, y, 384, y, C.purple);

  o += box(388, y - 34, 152, 68, C.lgreen, C.green, 8, 2);
  o += tx(464, y - 10, 'Residual Output', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(464, y + 14, "ŝ' = s + Δŝ", { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += line(77, y - 36, 77, 46, C.green, 2, true);
  o += line(77, 46, 464, 46, C.green, 2, true);
  o += arrow(464, 46, 464, y - 36, C.green);
  o += tx(270, 36, "Identity skip connection s (since Δt = 0.05 s ⟹ s' ≈ s)", { fs: 12, fw: 700, fill: D.green.dark });

  o += box(20, 164, 520, 72, N.panel, C.gray, 8, 1.6);
  o += tx(280, 186, 'Why standardize targets (Δs, r)? Reward r ∈ [−16.2, 0] has ~2000× larger variance', { fs: 12.2, fw: 600, fill: C.ink });
  o += tx(280, 210, 'than Δcos θ (~0.01); z-scoring ensures equal L₂ weight across all 4 dimensions!', { fs: 12.2, fw: 700, fill: D.blue.dark });
  return svg(W, H, o);
}

// 33. ch15-drl-experience-vs-compute
function experienceVsCompute() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Trading Physical Interaction for Test-Time Computation', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 194, C.lblue, C.blue, 9, 2);
  o += tx(144, 68, 'World-Model MPC (§15.6)', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += box(34, 88, 220, 44, '#FFFFFF', C.green, 6, 1.8);
  o += tx(144, 110, 'Data: 200–600 real steps!', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += box(34, 142, 220, 44, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(144, 164, 'Test: 12,500 model calls', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(144, 212, '500 cands × 25 steps per command', { fs: 12, fw: 600, fill: C.ink });

  o += box(292, 44, 248, 194, C.lgreen, C.green, 9, 2);
  o += tx(416, 68, 'Soft Actor–Critic (§15.5)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += box(306, 88, 220, 44, '#FFFFFF', C.amber, 6, 1.8);
  o += tx(416, 110, 'Data: 2,400–7,500 real steps', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += box(306, 142, 220, 44, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 164, 'Test: 1 actor pass (~μs)!', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 212, 'Amortizes planning into weights θ', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 34. ch15-drl-actor-seeded-search
function actorSeededSearch() {
  const W = 560, H = 254;
  let o = '';
  o += tx(280, 20, 'Search as a Policy Improvement Operator (Actor-Seeded Shooting)', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(16, 66, 158, 108, D.purple.tint, C.purple, 8, 2);
  o += tx(95, 92, 'Actor Rollout', { fs: 13.8, fw: 700, fill: D.purple.dark });
  o += tx(95, 118, 'ā_θ = c tanh μ_θ(s)', { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(95, 144, '25-step seed plan', { fs: 12, fill: C.muted });

  o += arrow(176, 96, 214, 76, C.green);
  o += arrow(176, 144, 214, 164, C.blue);

  o += box(218, 44, 192, 62, C.lgreen, C.green, 7, 2);
  o += tx(314, 66, 'Candidate 0 (Exact Seed)', { fs: 12.8, fw: 700, fill: D.green.dark });
  o += tx(314, 90, 'A^(0) = A_actor (0 noise)', { mono: true, fs: 11.2, fill: C.ink });

  o += box(218, 134, 192, 62, C.lblue, C.blue, 7, 2);
  o += tx(314, 156, 'Candidates 1..499', { fs: 12.8, fw: 700, fill: D.blue.dark });
  o += tx(314, 180, 'clip(A_actor + N(0, σ²))', { mono: true, fs: 11.2, fill: C.ink });

  o += arrow(412, 76, 434, 106, C.green);
  o += arrow(412, 164, 434, 134, C.blue);

  o += box(438, 72, 106, 96, C.lamber, C.amber, 8, 2);
  o += tx(491, 100, 'Winner a*_0', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(491, 124, 'Ĝ(A*) ≥', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(491, 144, 'Ĝ(A_actor)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  o += tx(280, 226, 'Because Candidate 0 is unperturbed, search strictly improves on the actor inside f̂_φ!', { fs: 12.2, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 35. ch15-drl-search-distillation
function searchDistillation() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Iterated Search and Policy Distillation (The AlphaZero / Dual Loop)', { fs: 14.8, fw: 700, fill: C.ink });

  const bw = 224, bh = 72;
  const boxes = [
    { x: 24,  y: 44,  t: '1. Refit World Model f̂_φ', s: 'Least squares on buffer D', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 312, y: 44,  t: '2. Actor-Seeded Search', s: 'a*_i = improve(f̂_φ, π_θ, s_i)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 312, y: 158, t: '3. Distill into Actor π_θ', s: 'min_θ ∑ ||ā_θ(s_i) − a*_i||²', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 24,  y: 158, t: '4. Collect New Episode', s: 'Act with improve(f̂_φ, π_θ)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  boxes.forEach((b) => {
    o += box(b.x, b.y, bw, bh, b.fill, b.st, 8, 2);
    o += tx(b.x + bw / 2, b.y + 24, b.t, { fs: 13.8, fw: 700, fill: b.dk });
    o += tx(b.x + bw / 2, b.y + 48, b.s, { mono: true, fs: 11.3, fw: 600, fill: C.ink });
  });
  o += arrow(250, 80, 308, 80, C.blue);
  o += arrow(424, 118, 424, 154, C.amber);
  o += arrow(308, 194, 250, 194, D.red.base);
  o += arrow(136, 154, 136, 118, C.green);
  return svg(W, H, o);
}

// 36. ch15-drl-mbpo-branched-rollout
function mbpoBranchedRollout() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Dyna & MBPO: Short k = 3 Branched Rollouts From Real Buffer States', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(20, 46, 520, 56, C.lblue, C.blue, 8, 2);
  o += tx(280, 66, 'Real Replay Buffer States (Anchor Distribution Without Drift):', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(280, 88, 's_0  ─────►  s_1  ─────►  s_2  ─────►  s_3  ─────►  s_4', { mono: true, fs: 12, fw: 700, fill: C.ink });

  [114, 280, 446].forEach((bx, i) => {
    o += arrow(bx, 104, bx, 136, C.amber);
    o += box(bx - 72, 140, 144, 52, C.lamber, C.amber, 7, 1.8);
    o += tx(bx, 158, `k=3 Branch ${i + 1} in f̂_φ`, { fs: 12, fw: 700, fill: D.orange.dark });
    o += tx(bx, 178, 's → ŝ₁ → ŝ₂ → ŝ₃', { mono: true, fs: 11.5, fill: C.ink });
  });

  o += box(20, 206, 520, 38, C.lgreen, C.green, 7, 1.8);
  o += tx(280, 225, 'Ablation: At matched UTD = 10, 100% real replay matches 95% imagined rollouts!', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.7 offline-rl.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 37. ch15-drl-online-vs-offline
function onlineVsOffline() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(144, 42, 'Online RL (§15.1–§15.6)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(144, 66, 'Closed Self-Correcting Loop', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += box(34, 86, 220, 86, '#FFFFFF', C.green, 6, 1.8);
  o += tx(144, 108, '1. Q̂ overestimates action a', { fs: 12, fill: C.ink });
  o += tx(144, 130, '2. Policy tries a in real env', { fs: 12, fill: C.ink });
  o += tx(144, 152, '3. Low reward refutes phantom Q̂!', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(144, 202, 'Active interaction prunes', { fs: 12.2, fill: C.ink });
  o += tx(144, 222, 'optimistic estimation errors', { fs: 12.2, fill: C.ink });

  o += box(292, 18, 248, 220, D.red.tint, D.red.base, 9, 2);
  o += tx(416, 42, 'Offline RL (§15.7)', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(416, 66, 'Fixed Log D · Zero New Trials', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += box(306, 86, 220, 86, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(416, 108, '1. Thin pair (n ≤ 5) gets lucky r', { fs: 12, fill: C.ink });
  o += tx(416, 130, "2. max_{a'} Q̂ picks thin outlier", { fs: 12, fill: C.ink });
  o += tx(416, 152, '3. NO rollout ever refutes it!', { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(416, 202, 'Narrow Lake: predicts 0.25 > V*=0.18', { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(416, 222, 'while actually earning only 0.10!', { fs: 12.2, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// 38. ch15-drl-trajectory-stitching
function trajectoryStitching() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Trajectory Stitching: How Offline Q Beats Behavior Cloning 14×', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 48, 144, 46, C.lgreen, C.green, 7, 2);
  o += tx(92, 64, 'Ep 1 Prefix (Fast)', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(92, 82, 's₀ → s₄ → s₈ → s₉', { mono: true, fs: 11.5, fill: C.ink });

  o += arrow(166, 71, 218, 106, C.green);

  o += box(20, 138, 144, 46, N.panel, C.gray, 7, 1.8, true);
  o += tx(92, 154, 'Ep 2 Prefix (Detour)', { fs: 12.5, fw: 600, fill: C.muted });
  o += tx(92, 172, 's₀ → s₁ → s₂ → s₆', { mono: true, fs: 11.5, fill: C.muted });

  o += arrow(166, 161, 218, 126, C.gray, true);

  o += box(222, 88, 116, 56, C.lblue, C.blue, 8, 2.2);
  o += tx(280, 108, 'Shared Cell', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 128, 's = 10', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += arrow(340, 104, 392, 71, D.red.base, true);
  o += box(396, 48, 144, 46, D.red.tint, D.red.base, 7, 1.8, true);
  o += tx(468, 64, 'Ep 1 Suffix (Fails)', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(468, 82, 's₁₀ → Hole 11 (0)', { mono: true, fs: 11.5, fill: C.ink });

  o += arrow(340, 128, 392, 161, C.green);
  o += box(396, 138, 144, 46, C.lgreen, C.green, 7, 2);
  o += tx(468, 154, 'Ep 2 Suffix (Wins!)', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(468, 172, 's₁₀ → s₁₄ → Goal 15', { mono: true, fs: 11.5, fill: C.ink });

  o += box(20, 202, 520, 42, D.purple.tint, C.purple, 7, 1.8);
  o += tx(280, 223, 'Bellman backup at s = 10 stitches Ep 1 prefix with Ep 2 suffix (0.14 vs BC 0.01)!', { fs: 12, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// 39. ch15-drl-lcb-bandit
function lcbBandit() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Pessimism (Lower Confidence Bound): Why LCB Rejects Lucky Outliers', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 196, C.lgreen, C.green, 9, 2);
  o += tx(144, 68, 'Arm 1: Well-Supported (n = 100)', { fs: 13.8, fw: 700, fill: D.green.dark });
  o += tx(144, 94, 'Empirical mean r̂(1) = 0.500', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += box(34, 112, 220, 68, '#FFFFFF', C.green, 6, 1.8);
  o += tx(144, 134, 'Small bonus b(1) = 0.136', { mono: true, fs: 11.8, fill: C.ink });
  o += tx(144, 158, 'LCB(1) = 0.500 − 0.136 = +0.364', { mono: true, fs: 11.2, fw: 700, fill: D.green.dark });
  o += tx(144, 210, 'LCB Winner! Safely earns 0.50', { fs: 13, fw: 700, fill: D.green.dark });

  o += box(292, 44, 248, 196, D.red.tint, D.red.base, 9, 2);
  o += tx(416, 68, 'Arm 2: Thin Outlier (n = 1)', { fs: 13.8, fw: 700, fill: D.red.dark });
  o += tx(416, 94, 'Lucky draw r̂(2) = 1.0 (true 0.25)', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += box(306, 112, 220, 68, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(416, 134, 'Huge penalty b(2) = 1.358', { mono: true, fs: 11.8, fill: D.red.dark });
  o += tx(416, 158, 'LCB(2) = 1.000 − 1.358 = −0.358', { mono: true, fs: 11.2, fw: 700, fill: D.red.dark });
  o += tx(416, 210, 'Naive argmax tricked; LCB rejects!', { fs: 12.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 40. ch15-drl-cql-push-pull
function cqlPushPull() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Conservative Q-Learning (CQL): Push Down All Actions, Pull Up Log Data', { fs: 14.2, fw: 700, fill: C.ink });

  o += box(20, 42, 520, 152, '#FFFFFF', C.gray, 8, 1.8);
  o += `<path d="M 56 146 Q 130 48 200 132 T 350 120 T 480 54" fill="none" stroke="${D.red.base}" stroke-width="2.5" stroke-dasharray="5 4"/>`;
  o += `<path d="M 56 166 Q 130 138 200 136 Q 280 92 350 132 Q 410 148 480 144" fill="none" stroke="${C.green}" stroke-width="3"/>`;

  o += arrow(130, 66, 130, 116, D.red.base);
  o += tx(130, 178, 'OOD Peak: Push ↓', { fs: 12, fw: 700, fill: D.red.dark });

  o += arrow(465, 66, 465, 116, D.red.base);
  o += tx(465, 178, 'OOD Peak: Push ↓', { fs: 12, fw: 700, fill: D.red.dark });

  o += arrow(280, 154, 280, 114, C.green);
  o += tx(280, 72, 'In-Support a_log ~ D: Pull ↑', { fs: 12.5, fw: 700, fill: D.green.dark });

  o += box(20, 204, 520, 40, D.purple.tint, C.purple, 7, 1.8);
  o += tx(280, 224, "CQL Loss: L_TD(w) + α_c E_s[ log ∑_{a'} exp(Q_w(s, a')) − Q_w(s, a_log) ]", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §15.8 rl-sequences.md (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 41. ch15-drl-sft-vs-rlvr
function sftVsRlvr() {
  const W = 560, H = 256;
  let o = '';
  o += box(16, 16, 528, 104, C.lblue, C.blue, 8, 2);
  o += tx(280, 36, 'Pretraining & SFT (Behavior Cloning): Teacher Target at Every Token', { fs: 13.8, fw: 700, fill: D.blue.dark });
  o += box(34, 52, 492, 38, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(280, 71, 'x → a*_1 (CE loss) → a*_2 (CE loss) → ··· → a*_T (CE loss)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(280, 106, 'Imitates static corpus; never practices recovering from its own reasoning errors', { fs: 12, fill: C.ink });

  o += box(16, 136, 528, 106, C.lgreen, C.green, 8, 2);
  o += tx(280, 156, 'RL with Verifiable Rewards (RLVR): Model Rollout + Outcome Checker', { fs: 13.8, fw: 700, fill: D.green.dark });
  o += box(34, 172, 492, 38, '#FFFFFF', C.green, 6, 1.8);
  o += tx(280, 191, 'x → a_1 ~ π_θ → a_2 ~ π_θ → ··· → a_T (EOS) → Verifier r(x, y) ∈ {0, 1}', { mono: true, fs: 11.3, fw: 700, fill: D.green.dark });
  o += tx(280, 226, 'No target tokens! Policy discovers reasoning chains that pass the terminal checker', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 42. ch15-drl-token-variance-decomp
function tokenVarianceDecomp() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Orthogonal Variance Decomposition on Two-Digit "3*4 = 12"', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 42, 520, 44, D.purple.tint, C.purple, 7, 1.8);
  o += tx(280, 64, 'Response Weight: r(x, y) − V(x) = A(s₁, a₁) [tens] + A(s₂, a₂) [units]', { mono: true, fs: 11.8, fw: 700, fill: D.purple.dark });

  o += box(20, 102, 248, 140, C.lgreen, C.green, 8, 2);
  o += tx(144, 126, 'Per-Token Prefix Advantage', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(144, 152, 'A(s₁, a₁) = V(x, a₁) − V(x)', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(144, 180, 'Var(A₁) = 0.0009', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(144, 210, 'On "13": rewards "1" (+0.09)!', { fs: 12.2, fw: 700, fill: C.ink });

  o += box(292, 102, 248, 140, D.red.tint, D.red.base, 8, 2);
  o += tx(416, 126, 'Whole-Response Weight', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(416, 152, 'Var(A₁) + Var(A₂) = 0.0009 + 0.0090', { mono: true, fs: 11.2, fw: 700, fill: C.ink });
  o += tx(416, 180, 'Var(r−V) = 0.0099 (11× higher!)', { mono: true, fs: 11.8, fw: 700, fill: D.red.dark });
  o += tx(416, 210, 'On "13": penalizes "1" (−0.01)!', { fs: 12.2, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 43. ch15-drl-token-mdp-stack
function tokenMdpStack() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'What the Deterministic Token MDP Simplifies — and What Survives', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(20, 44, 248, 194, N.panel, C.gray, 9, 2);
  o += tx(144, 68, 'Eliminated in Token MDPs', { fs: 14.5, fw: 700, fill: C.muted });
  o += box(34, 88, 220, 60, '#FFFFFF', C.gray, 6, 1.6);
  o += tx(144, 108, '✗ Learned World Model (§15.6)', { fs: 12.2, fw: 700, fill: C.ink });
  o += tx(144, 130, 'concat(s_t, a_t) is exact!', { mono: true, fs: 11.2, fill: C.muted });
  o += box(34, 158, 220, 60, '#FFFFFF', C.gray, 6, 1.6);
  o += tx(144, 178, '✗ Replay Q-Network (§15.4)', { fs: 12.2, fw: 700, fill: C.ink });
  o += tx(144, 200, '10⁵ tokens ⇒ 4.4σ max bias', { mono: true, fs: 11.2, fill: D.red.dark });

  o += box(292, 44, 248, 194, C.lgreen, C.green, 9, 2);
  o += tx(416, 68, 'Essential in LLM Post-Training', { fs: 14.2, fw: 700, fill: D.green.dark });
  o += box(306, 88, 220, 60, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 108, '✓ Group Baseline / Critic (§15.1)', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 130, 'GRPO/RLOO group mean or V̂(s_t)', { fs: 11.5, fill: C.ink });
  o += box(306, 158, 220, 60, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 178, '✓ Ratio Clip + KL Leash (§15.2–3)', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(416, 200, 'Guards multi-epoch & stops hacking', { fs: 11.5, fill: C.ink });
  return svg(W, H, o);
}

// 44. ch15-drl-loophole-threshold
function loopholeThreshold() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Exact Loophole Threshold: Tilted Posterior Odds exp(1/β − 4)', { fs: 14.8, fw: 700, fill: C.ink });

  o += box(16, 42, 528, 46, D.purple.tint, C.purple, 7, 1.8);
  o += tx(280, 65, 'π*(y_hedge) / π*(y_ord) = [π_ref(y_hedge) / π_ref(y_ord)] · e^{1/β} = e^{1/β − 4}', { mono: true, fs: 10.6, fw: 700, fill: D.purple.dark });

  o += box(20, 104, 248, 136, D.red.tint, D.red.base, 8, 2);
  o += tx(144, 128, 'Weak Regularization (β < 0.25)', { fs: 13.8, fw: 700, fill: D.red.dark });
  o += tx(144, 154, '1/β − 4 > 0  ⟹  Odds > 1', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(144, 182, 'Hedge dominates "8*8" & "9*9"!', { fs: 12.2, fw: 700, fill: D.red.dark });
  o += tx(144, 208, 'At β = 0.1: hedge prob = 0.49', { mono: true, fs: 11.8, fill: C.ink });

  o += box(292, 104, 248, 136, C.lgreen, C.green, 8, 2);
  o += tx(416, 128, 'Strong Leash (β > 0.25)', { fs: 13.8, fw: 700, fill: D.green.dark });
  o += tx(416, 154, '1/β − 4 < 0  ⟹  Odds < 1', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(416, 182, 'Prior e⁻⁴ suppresses hedge', { fs: 12.2, fw: 700, fill: D.green.dark });
  o += tx(416, 208, 'At β = 0.3: hedge drops to 0.07!', { mono: true, fs: 11.8, fill: C.ink });
  return svg(W, H, o);
}

// 45. ch15-drl-gspo-dapo-clip
function gspoDapoClip() {
  const W = 560, H = 256;
  let o = '';
  o += box(20, 18, 248, 220, C.lblue, C.blue, 9, 2);
  o += tx(144, 42, 'GSPO: Geometric Mean Ratio', { fs: 14.2, fw: 700, fill: D.blue.dark });
  o += tx(144, 66, 'Long CoT (T ~ 1,000+ tokens)', { fs: 12.2, fill: C.ink });
  o += box(34, 86, 220, 66, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(144, 106, 'Product W = ∏ ρ_t explodes', { fs: 12, fill: D.red.dark });
  o += tx(144, 132, 'ρ̄ = W^{1/T} = exp((1/T)∑log ρ_t)', { mono: true, fs: 10.6, fw: 700, fill: D.blue.dark });
  o += tx(144, 180, 'Keeps sequence ratio variance', { fs: 12.2, fill: C.ink });
  o += tx(144, 204, 'O(1/T) instead of O(T)!', { fs: 12.5, fw: 700, fill: D.blue.dark });

  o += box(292, 18, 248, 220, C.lgreen, C.green, 9, 2);
  o += tx(416, 42, 'DAPO: Asymmetric Clip-Higher', { fs: 14.2, fw: 700, fill: D.green.dark });
  o += tx(416, 66, 'Band: [0.80, 1.28] (ε_high = 0.28)', { mono: true, fs: 11.3, fw: 700, fill: C.ink });
  o += box(306, 86, 220, 66, '#FFFFFF', C.green, 6, 1.8);
  o += tx(416, 106, 'Symmetric 1.2 caps rare token', { fs: 12, fill: C.ink });
  o += tx(416, 132, 'π_old = 0.01 → 0.012 (+0.002)', { mono: true, fs: 11.3, fw: 700, fill: D.orange.dark });
  o += tx(416, 180, 'Raising upper ceiling to 1.28 lets', { fs: 12.2, fill: C.ink });
  o += tx(416, 204, 'rare reasoning tokens grow faster!', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 46. ch15-drl-grpo-architecture
function grpoArchitecture() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Group Relative Policy Optimization (GRPO): Critic-Free LLM Loop', { fs: 14.8, fw: 700, fill: C.ink });

  const y = 104;
  o += box(16, y - 34, 88, 68, C.lblue, C.blue, 8, 2);
  o += tx(60, y - 10, 'Prompt x', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(60, y + 14, 'Sample K', { fs: 12, fill: C.ink });

  o += arrow(106, y, 132, y, C.blue);

  o += box(136, y - 44, 132, 88, D.purple.tint, C.purple, 8, 2);
  o += tx(202, y - 20, 'Group Rollouts', { fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(202, y + 4, 'y₁, y₂, ..., y_K', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(202, y + 26, '~ π_{θ_old}(· | x)', { mono: true, fs: 11.2, fill: C.muted });

  o += arrow(270, y, 296, y, C.purple);

  o += box(300, y - 44, 118, 88, C.lamber, C.amber, 8, 2);
  o += tx(359, y - 20, 'Group Adv', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(359, y + 4, 'Â_j =', { mono: true, fs: 11.8, fw: 700, fill: C.ink });
  o += tx(359, y + 26, '(r_j − r̄) / σ_r', { mono: true, fs: 11.2, fw: 700, fill: D.orange.dark });

  o += arrow(420, y, 444, y, C.green);

  o += box(448, y - 44, 96, 88, C.lgreen, C.green, 8, 2);
  o += tx(496, y - 20, 'Update θ', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(496, y + 4, 'Clip ρ_t', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(496, y + 26, '− β k₃', { mono: true, fs: 11.8, fw: 700, fill: D.green.dark });

  o += box(16, 174, 528, 66, N.panel, C.gray, 8, 1.6);
  o += tx(280, 196, 'Eliminates separate critic LLM (~50% memory savings)!', { fs: 13.2, fw: 700, fill: D.green.dark });
  o += tx(280, 220, '4 epochs on K = 32 group lifts VerifierTask: 0.062 (π_ref) → 0.273 → 0.487', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 47. ch15-drl-kl-estimators
function klEstimators() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Three Per-Token KL Estimators (h = log(π_θ / π_ref))', { fs: 15, fw: 700, fill: C.ink });

  const cols = [
    { x: 16,  t: 'k₁ = h', s1: 'E[k₁] = D_KL(π_θ||π_ref)', s2: 'Direct E[∇k₁] = 0!', s3: 'Must fold into reward', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, t: 'k₂ = ½ h²', s1: 'Expected gradient:', s2: '∇_θ D_KL(π_θ || π_ref)', s3: 'Rev KL: hedge 0.017!', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 380, t: 'k₃ = e⁻ʰ + h − 1', s1: 'Expected gradient:', s2: '∇_θ D_KL(π_ref || π_θ)', s3: 'Fwd KL: hedge 0.500!', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  cols.forEach((c) => {
    o += box(c.x, 44, 164, 194, c.fill, c.st, 8, 2);
    o += tx(c.x + 82, 70, c.t, { mono: true, fs: 13.5, fw: 700, fill: c.dk });
    o += tx(c.x + 82, 106, c.s1, { fs: 12, fill: C.ink });
    o += tx(c.x + 82, 132, c.s2, { mono: true, fs: 11.2, fw: 700, fill: C.ink });
    o += box(c.x + 12, 158, 140, 56, '#FFFFFF', c.st, 6, 1.6);
    o += tx(c.x + 82, 186, c.s3, { fs: 11.5, fw: 700, fill: c.dk });
  });
  return svg(W, H, o);
}

// 48. ch15-drl-dpo-inversion
function dpoInversion() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Direct Preference Optimization (DPO): Inverting the Tilted Optimum', { fs: 14.5, fw: 700, fill: C.ink });

  const steps = [
    { y: 42,  t: '1. Solve tilted optimum π*(y|x) = π_ref(y|x) e^{r/β} / Z(x) for r(x, y):', f: 'r(x, y) = β log [ π*(y | x) / π_ref(y | x) ] + β log Z(x)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 112, t: '2. Substitute into Bradley–Terry r(x, y_w) − r(x, y_l) (Z(x) cancels!):', f: 'Δr = β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x))', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 182, t: '3. Optimize policy π_θ directly by binary cross-entropy on (y_w ≻ y_l):', f: 'L_DPO(θ) = −E_D[ log σ( β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)) ) ]', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s, i) => {
    o += box(16, s.y, 528, 54, s.fill, s.st, 8, 2);
    o += tx(32, s.y + 18, s.t, { fs: 12.2, fw: 700, fill: s.dk, anchor: 'start' });
    o += tx(280, s.y + 39, s.f, { mono: true, fs: 10.6, fw: 700, fill: C.ink });
    if (i < 2) o += arrow(280, s.y + 56, 280, s.y + 68, C.gray);
  });
  return svg(W, H, o);
}

export const diagrams = {
  'ch15-drl-naive-pg-waste':          naivePgWaste,
  'ch15-drl-four-mechanisms':         fourMechanisms,
  'ch15-drl-triad-overview':          triadOverview,
  'ch15-drl-benchmark-gains':         benchmarkGains,
  'ch15-drl-mc-tail-vs-critic':       mcTailVsCritic,
  'ch15-drl-cartpole-td-trace':       cartpoleTdTrace,
  'ch15-drl-critic-error-decomp':     criticErrorDecomp,
  'ch15-drl-semi-gradient-stop':      semiGradientStop,
  'ch15-drl-fitted-td-loop':          fittedTdLoop,
  'ch15-drl-gae-mixture':             gaeMixture,
  'ch15-drl-gae-telescoping':         gaeTelescoping,
  'ch15-drl-single-vs-multi-epoch':   singleVsMultiEpoch,
  'ch15-drl-importance-cancellation': importanceCancellation,
  'ch15-drl-ratio-explosion':         ratioExplosion,
  'ch15-drl-perf-diff-lemma':         perfDiffLemma,
  'ch15-drl-clip-drift-entropy':      clipDriftEntropy,
  'ch15-drl-rlhf-pipeline':           rlhfPipeline,
  'ch15-drl-bradley-terry':           bradleyTerry,
  'ch15-drl-exponential-tilt-flow':   exponentialTiltFlow,
  'ch15-drl-trust-vs-anchor':         trustVsAnchor,
  'ch15-drl-naive-q-explosion':       naiveQExplosion,
  'ch15-drl-norm-mismatch':           normMismatch,
  'ch15-drl-ring-buffer':             ringBuffer,
  'ch15-drl-double-dqn-split':        doubleDqnSplit,
  'ch15-drl-dqn-vs-sac-actor':        dqnVsSacActor,
  'ch15-drl-ddpg-td3-sac':            ddpgTd3Sac,
  'ch15-drl-maxent-objective':        maxentObjective,
  'ch15-drl-soft-target-sampling':    softTargetSampling,
  'ch15-drl-clip-vs-tanh':            clipVsTanh,
  'ch15-drl-dual-temperature':        dualTemperature,
  'ch15-drl-scalar-vs-physics':       scalarVsPhysics,
  'ch15-drl-delta-residual-net':      deltaResidualNet,
  'ch15-drl-experience-vs-compute':   experienceVsCompute,
  'ch15-drl-actor-seeded-search':     actorSeededSearch,
  'ch15-drl-search-distillation':     searchDistillation,
  'ch15-drl-mbpo-branched-rollout':   mbpoBranchedRollout,
  'ch15-drl-online-vs-offline':       onlineVsOffline,
  'ch15-drl-trajectory-stitching':    trajectoryStitching,
  'ch15-drl-lcb-bandit':              lcbBandit,
  'ch15-drl-cql-push-pull':           cqlPushPull,
  'ch15-drl-sft-vs-rlvr':             sftVsRlvr,
  'ch15-drl-token-variance-decomp':   tokenVarianceDecomp,
  'ch15-drl-token-mdp-stack':         tokenMdpStack,
  'ch15-drl-loophole-threshold':      loopholeThreshold,
  'ch15-drl-gspo-dapo-clip':          gspoDapoClip,
  'ch15-drl-grpo-architecture':       grpoArchitecture,
  'ch15-drl-kl-estimators':           klEstimators,
  'ch15-drl-dpo-inversion':           dpoInversion,
};
