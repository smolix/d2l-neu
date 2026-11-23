// Slide diagrams for Chapter 38 (chapter_mdl-dynamics).
// Rendered and audited by tools/build_module_diagrams.mjs with 0 visual defects.

import { C, TOKENS, tx, arrow, svg } from './engine.mjs';
import { sec, num } from './sections.mjs';
import { textWidth } from './mdl-textwidth.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const rect = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, sw = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const circle = (cx, cy, r, fill, stroke, sw = 2, dash = false) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="4 3"' : ''}/>`;

const path = (d, stroke, fill = 'none', sw = 2.5, dash = false) =>
  `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Text that does not fit is shrunk, but never below MIN_FS. A mono string
// still too wide at MIN_FS is set in the (narrower) sans face instead; a
// string still too wide after that is wrapped onto two lines at the space
// nearest its middle. Widths come from the real font metrics
// (mdl-textwidth.mjs).
const MIN_FS = 11;
export const _fitLog = [];
function split2(s) {
  const mid = s.length / 2;
  let best = -1;
  for (let i = 1; i < s.length - 1; i++) {
    if (s[i] === ' ' && (best < 0 || Math.abs(i - mid) < Math.abs(best - mid))) best = i;
  }
  return best < 0 ? null : [s.slice(0, best).trimEnd(), s.slice(best + 1).trimStart()];
}

// { lines: [one or two strings], fs, mono } for text t in a box of width maxW.
function fitLines(t, maxW, o = {}) {
  const str = String(t);
  const avail = maxW - 22;
  const base = o.fs || 14.5;
  const bold = (o.fw || 500) >= 600;
  const fit1 = (mono) => {
    const w = textWidth(str, base, mono, bold);
    return w <= avail ? base : Math.floor((base * avail / w) * 10) / 10;
  };
  let mono = !!o.mono;
  let fs = fit1(mono);
  if (fs < MIN_FS && mono) { mono = false; fs = fit1(false); }
  if (fs >= MIN_FS) return { lines: [str], fs, mono };
  const parts = split2(str);
  if (!parts) {
    _fitLog.push(['over', str, MIN_FS]);
    return { lines: [str], fs: MIN_FS, mono };
  }
  const w2 = Math.max(...parts.map(p => textWidth(p, base, mono, bold)));
  fs = Math.max(MIN_FS, Math.min(base, Math.floor((base * avail / w2) * 10) / 10));
  _fitLog.push(['wrap', str, fs, w2 * fs / base > avail]);
  return { lines: parts, fs, mono };
}

// Centred text fitted to width maxW (one line, or two around y).
function fitTx(x, y, t, maxW, o = {}) {
  const { lines, fs, mono } = fitLines(t, maxW, o);
  if (lines.length === 1) return tx(x, y, lines[0], { ...o, fs, mono });
  const dy = 0.6 * fs;
  return tx(x, y - dy, lines[0], { ...o, fs, mono }) + tx(x, y + dy, lines[1], { ...o, fs, mono });
}

// Stack fitted items: a wrapped item adds 1.2 fs of height between its two
// lines; sp separates items. Returns the markup.
function stackLines(cx, cy, items, maxW, sp) {
  const fitted = items.map(it => ({ it, ...fitLines(it.t, maxW, it) }));
  const extra = fitted.map(f => (f.lines.length > 1 ? 1.2 * f.fs : 0));
  const total = (fitted.length - 1) * sp + extra.reduce((a, b) => a + b, 0);
  let y = cy - total / 2;
  let o = '';
  fitted.forEach((f, i) => {
    f.lines.forEach((ln, j) => {
      o += tx(cx, y + j * 1.2 * f.fs, ln, { ...f.it, fs: f.fs, mono: f.mono });
    });
    y += extra[i] + sp;
  });
  return o;
}

function wrapExtra(items, maxW) {
  return items.reduce((a, it) => {
    const f = fitLines(it.t, maxW, it);
    return a + (f.lines.length > 1 ? 1.2 * f.fs : 0);
  }, 0);
}

// High-level clean card with lines of text guaranteed inside borders
function card(x, y, w, h, fill, stroke, lines, opts = {}) {
  let o = rect(x, y, w, h, fill, stroke, opts.rx || 8, opts.sw || 2, opts.dash || false);
  const n = lines.length;
  const items = lines.map((ln, i) => {
    const item = typeof ln === 'string' ? { t: ln } : ln;
    return {
      t: item.t,
      fs: item.fs || (i === 0 ? 15 : 13.5),
      fw: item.fw || (i === 0 ? 700 : 500),
      mono: item.mono || false,
      fill: item.fill || (i === 0 ? C.ink : C.muted),
    };
  });
  const ex = wrapExtra(items, w);
  const spacing = opts.spacing || (n <= 1 ? 0 : Math.min(22, (h - 24 - ex) / (n - 1)));
  return o + stackLines(x + w / 2, y + h / 2, items, w, spacing);
}

// Header bar + body card
function titledPanel(x, y, w, h, title, accent, bodyLines, opts = {}) {
  let o = rect(x, y, w, h, opts.fill || '#FFFFFF', accent.base, 10, 2);
  o += rect(x + 6, y + 6, w - 12, 30, accent.tint, 'none', 6, 0);
  o += fitTx(x + w / 2, y + 21, title, w - 12, { fs: 14.5, fw: 700, fill: accent.dark });
  const bodyTop = y + 40;
  const bodyH = h - 44;
  const n = bodyLines.length;
  const items = bodyLines.map(ln => {
    const item = typeof ln === 'string' ? { t: ln } : ln;
    return {
      t: item.t,
      fs: item.fs || 13.5,
      fw: item.fw || 600,
      mono: item.mono || false,
      fill: item.fill || C.ink,
    };
  });
  const ex = wrapExtra(items, w);
  const sp = opts.spacing || (n <= 1 ? 0 : Math.min(22, (bodyH - 20 - ex) / (n - 1)));
  return o + stackLines(x + w / 2, bodyTop + bodyH / 2, items, w, sp);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.0 index.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function unifiedPrimitives() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 12, 500, 40, D.purple.tint, C.purple, [
    { t: 'One Primitive Operation: Moving a State Along a Continuous Velocity Field', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 66, 156, 146, '1. ResNet Block', D.blue, [
    { t: 'x_{ℓ+1} = x_ℓ + f(x_ℓ)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Forward Euler step', fs: 12.5, fill: C.ink },
    { t: 'with h = 1 on ODE', fs: 12.5, fill: C.ink },
    { t: 'ẋ(t) = f(x(t), t)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);

  o += titledPanel(192, 66, 156, 146, '2. Gradient Step', D.orange, [
    { t: 'w_{k+1} = w_k − η∇L', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Euler step h = η on', fs: 12.5, fill: C.ink },
    { t: 'gradient flow −∇L', fs: 12.5, fill: C.ink },
    { t: 'Stable iff η < 2/λ_max', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(364, 66, 156, 146, '3. Diffusion / Flow', D.green, [
    { t: 'dX = f dt + g dW', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '(dW)² = dt adds ½g²Δ', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Reversed by score', fs: 12.5, fill: C.ink },
    { t: '∇_x log p_t(x)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 224, 500, 48, C.lblue, C.blue, [
    { t: 'Unifying Bridge:  ½ g² Δp_t = ∇ · (p_t · ½ g² ∇log p_t)  converts noise into score transport!', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function threeStageRoadmap() {
  const W = 540, H = 292;
  let o = '';
  o += card(20, 12, 500, 74, C.lblue, C.blue, [
    { t: `Stage 1 · Deterministic ODEs, Solvers & Flows (§§${num('sec_mdl-odes-solvers')}–${num('sec_mdl-neural-odes-cnf')})`, fs: 15, fw: 700, fill: D.blue.dark },
    { t: 'Picard–Lindelöf uniqueness · Matrix exp & ZOH · Euler/RK4 stability R(hλ) · Adjoint & Liouville −tr(J)', fs: 12.5, fill: C.ink },
  ]);
  o += arrow(270, 88, 270, 108, C.blue);

  o += card(20, 112, 500, 74, C.lamber, C.amber, [
    { t: `Stage 2 · Stochastic Calculus & Density Evolution (§§${num('sec_mdl-sdes')}–${num('sec_mdl-fokker-planck-probability-flow')})`, fs: 15, fw: 700, fill: D.orange.dark },
    { t: 'Wiener process & (dW)² = dt · Itô’s lemma & OU kernel · Fokker–Planck PDE · Anderson Reverse SDE & PF-ODE', fs: 12.5, fill: C.ink },
  ]);
  o += arrow(270, 188, 270, 208, D.orange.base);

  o += card(20, 212, 500, 68, C.lgreen, C.green, [
    { t: `Stage 3 · Learning Generative Vector Fields (§§${num('sec_mdl-score-matching-diffusion-flow')}–${num('sec_mdl-flow-matching-transport')})`, fs: 15, fw: 700, fill: D.green.dark },
    { t: 'Vincent DSM & Tweedie · DDPM/DDIM & CFG · Conditional Flow Matching · Rectified Flow Reflow & W₂ OT', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function fiveBenchmarks() {
  const W = 540, H = 290;
  let o = '';
  o += fitTx(270, 18, `Five Closed-Form Running Benchmarks Anchoring Chapter ${num('chap_mdl-dynamics')}`, 500, { fs: 15, fw: 700, fill: C.ink });

  const items = [
    { name: `1. Planar Spiral Sink (§§${num('sec_mdl-odes-solvers')}–${num('sec_mdl-neural-odes-cnf')})`, eq: 'ẋ = (−½I + K)x,  λ = −0.5 ± i', test: 'e^{At}, RK4 order 4, area e⁻ᵗ', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { name: `2. Stiff Two-Mode Decay (${sec('sec_mdl-euler-runge-kutta')})`, eq: 'ẋ = diag(−50, −1)x', test: 'Explicit h < 2/|λ|, A-stability', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { name: `3. Ornstein–Uhlenbeck (${sec('sec_mdl-sdes')})`, eq: 'dX = −θX dt + √(2θ) dW', test: 'Kernel N(α_t x₀, 1−α_t²), VP-SDE', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { name: `4. Bimodal Mixture (§§${num('sec_mdl-fokker-planck-probability-flow')}–${num('sec_mdl-score-matching-diffusion-flow')})`, eq: 'w₁ N(−2, 0.25) + w₂ N(+2, 0.25)', test: 'Score tanh, PF-ODE, DDIM, CFG', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { name: `5. Two-Moons Transport (${sec('sec_mdl-flow-matching-transport')})`, eq: 'x_t = (1 − t)x₀ + tε', test: 'Crossing floor C=1.37, Reflow 1-step', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  items.forEach((it, i) => {
    const y = 34 + i * 50;
    o += card(16, y, 174, 42, it.fill, it.st, [
      { t: it.name, fs: 12.5, fw: 700, fill: it.dk },
    ], { rx: 6, sw: 2 });
    o += card(198, y, 166, 42, '#FFFFFF', it.st, [
      { t: it.eq, mono: true, fs: 12, fw: 700, fill: C.ink },
    ], { rx: 6, sw: 1.5 });
    o += card(372, y, 152, 42, '#FFFFFF', it.st, [
      { t: it.test, fs: 12, fw: 600, fill: C.muted },
    ], { rx: 6, sw: 1.5 });
  });
  return svg(W, H, o);
}

function fourEquationsBridge() {
  const W = 540, H = 286;
  let o = '';
  o += titledPanel(20, 12, 240, 122, '1. Forward Noising SDE (0 → T)', D.blue, [
    { t: 'dX = f(X, t) dt + g(t) dW', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Smooths p₀ into prior p_T ≈ N(0, I)', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 12, 240, 122, '2. Anderson Reverse SDE (T → 0)', D.green, [
    { t: 'dX = [f − g(t)² ∇log p_t(X)] dt + g dW̄', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Full g² undoes forward + reverse noise', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(20, 148, 240, 124, '3. Fokker–Planck Density PDE', D.orange, [
    { t: '∂_{t} p_t = −∇·(f p_t) + ½ g(t)² Δp_t', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: '= −∇ · (p_t u_t)   (Continuity Eq.)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);

  o += titledPanel(280, 148, 240, 124, '4. Probability-Flow ODE (0 ↔ T)', D.purple, [
    { t: 'ẋ = u_t(x) = f(x,t) − ½ g(t)² ∇log p_t', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
    { t: 'Deterministic twin sharing all p_t!', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.1 mdl-odes-solvers.md (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function wellPosedTriad() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 38, D.purple.tint, C.purple, [
    { t: 'When Does an ODE ẋ = f(x, t) Define a Global Invertible Flow Map Φ_T?', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 62, 156, 154, '1. Globally Lipschitz', D.green, [
    { t: 'ẋ = −x,  x(0) = 1', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Unique: x(t) = e⁻ᵗ', mono: true, fs: 13, fill: C.ink },
    { t: 'Global bijection', fs: 12.5, fill: D.green.dark },
    { t: 'Φ_t(x₀) = e⁻ᵗ x₀', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(192, 62, 156, 154, '2. Non-Lipschitz', D.red, [
    { t: 'ẋ = √|x|,  x(0) = 0', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Slope 1/(2√x) → ∞', mono: true, fs: 12, fill: C.ink },
    { t: 'Infinitely many paths', fs: 12.5, fill: D.red.dark },
    { t: 'x_c(t) = (t − c)² / 4', mono: true, fs: 12, fill: C.ink },
  ]);

  o += titledPanel(364, 62, 156, 154, '3. Local Lipschitz', D.orange, [
    { t: 'ẋ = x²,  x(0) = x₀ > 0', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'x(t) = x₀ / (1 − x₀t)', mono: true, fs: 12, fill: C.ink },
    { t: 'Blows up to +∞ at', fs: 12.5, fill: D.red.dark },
    { t: 'finite t* = 1 / x₀!', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 228, 500, 44, C.lgreen, C.green, [
    { t: 'Picard–Lindelöf Requirement: Global Lipschitz ‖f(x,t)−f(y,t)‖ ≤ L‖x−y‖ guarantees existence + uniqueness', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function decayPlusRotation() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, '1. Radial Decay: ẋ = −x', D.blue, [
    { t: 'Velocity points straight inward', fs: 12.5, fill: C.ink },
    { t: 'd/dt ‖x‖² = 2 xᵀ(−x) = −2‖x‖²', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    { t: '‖x(t)‖ = e^{−t} ‖x₀‖', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Pure real eigenvalue λ = −1', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 188, '2. Orthogonal Turn: ẋ = K x', D.green, [
    { t: 'K = [0, −1; 1, 0]  (90° rotation)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'xᵀ K x = −x₁x₂ + x₂x₁ = 0  (⊥ x!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'd/dt ‖x(t)‖² = 2 xᵀ K x = 0', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Imaginary eigenvalues λ = ±i', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 216, 500, 52, D.purple.tint, C.purple, [
    { t: 'Combined Planar Spiral:  A = −½ I + K  ⇒  λ = −0.5 ± i', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: 'decays as e^{−t/2} while rotating at 1 rad/s', fs: 12.5, fill: C.ink },
  ], { spacing: 18 });
  return svg(W, H, o);
}

function picardFixedPoint() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Picard Integral Operator:   (Pφ)(t) = x₀ + ∫₀ᵗ f(φ(s), s) ds   (Fixed Point φ = Pφ ⇔ ODE Solution)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  const stages = [
    { k: 'φ₀(t)', val: '1', sub: 'Initial guess', fill: N.panel, st: C.gray, dk: C.ink },
    { k: 'φ₁(t)', val: '1 + t', sub: '1st integral', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { k: 'φ₂(t)', val: '1 + t + t²/2', sub: '2nd integral', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { k: 'φ_{∞}(t)', val: '∑ tᵏ/k! = eᵗ', sub: 'Unique limit', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  stages.forEach((s, i) => {
    const x = 20 + i * 128;
    o += card(x, 72, 114, 116, s.fill, s.st, [
      { t: s.k, mono: true, fs: 14, fw: 700, fill: s.dk },
      { t: s.val, mono: true, fs: 13.5, fw: 700, fill: C.ink },
      { t: s.sub, fs: 12, fill: C.muted },
    ]);
    if (i < 3) {
      o += arrow(x + 116, 130, x + 126, 130, C.purple);
    }
  });

  o += card(20, 206, 500, 60, D.purple.tint, C.purple, [
    { t: 'Banach Contraction Proof:  d(Pφ, Pψ) ≤ Lδ · d(φ, ψ) = ½ d(φ, ψ)  on δ = 1 / (2L)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Because global L is state-independent, ⌈2LT⌉ uniform restarts cover any finite horizon [0, T]!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function finiteTimeBlowup() {
  const W = 540, H = 282;
  let o = '';
  // Left plot of x(t) = x0 / (1 - x0 t) blowing up at t* = 1/x0
  o += rect(20, 16, 240, 250, '#FFFFFF', C.gray, 8, 1.5);
  o += line(40, 224, 240, 224, C.gray, 1.5, false);
  o += line(40, 36, 40, 224, C.gray, 1.5, false);
  // Vertical asymptote at t* = 1 / x0 (x = 196)
  o += line(196, 48, 196, 224, D.red.base, 2, true);
  // Curve rising steeply to asymptote
  o += path('M 40 190 Q 154 182 186 48', C.blue, 'none', 3.5, false);
  o += circle(40, 190, 4.5, C.blue, '#FFFFFF', 1.5);
  o += fitTx(140, 32, 'Asymptote t* = 1 / x₀', 210, { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += fitTx(140, 246, 'ẋ = x²  ⇒  x(t) = x₀ / (1 − x₀t)', 220, { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });

  o += card(276, 16, 244, 76, C.lblue, C.blue, [
    { t: 'Local Lipschitz Only', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '|x² − y²| = |x + y| |x − y|', mono: true, fs: 12.5, fill: C.ink },
    { t: 'On |x| ≤ ρ:  L_ρ = 2ρ → ∞', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += card(276, 103, 244, 76, C.lamber, C.amber, [
    { t: 'Shrinking Picard Steps', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Safe step δ_ρ = 1/(2L_{ρ}) = 1/(4ρ)', mono: true, fs: 12, fill: C.ink },
    { t: 'As x(t) ↑, steps shrink to 0', fs: 12.5, fill: C.muted },
  ]);

  o += card(276, 190, 244, 76, D.red.tint, D.red.base, [
    { t: 'Finite-Horizon Explosion', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Restarts sum to a finite time', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Bounded activations (tanh) fix this', fs: 12, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function flowGroupBijection() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 186, 174, C.lblue, C.blue, [
    { t: 'Initial Space ℝᵈ', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 't = 0', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Distinct starts', fs: 13, fill: C.ink },
    { t: 'x₀ ≠ y₀', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Φ₀ = id', mono: true, fs: 13, fill: C.muted },
  ]);

  o += arrow(212, 82, 328, 82, C.blue);
  o += fitTx(270, 62, 'Forward Φ_T (+f)', 110, { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += arrow(328, 124, 212, 124, C.green);
  o += fitTx(270, 146, 'Inverse Φ_T⁻¹ (−f)', 110, { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  o += card(334, 14, 186, 174, C.lgreen, C.green, [
    { t: 'Terminal Space ℝᵈ', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 't = T', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Never cross:', fs: 13, fill: C.ink },
    { t: 'Φ_T(x₀) ≠ Φ_T(y₀)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Φ_{t+s} = Φ_t ∘ Φ_s', mono: true, fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 204, 500, 60, D.purple.tint, C.purple, [
    { t: 'Non-Crossing Theorem: Two trajectories of a Lipschitz ODE can NEVER meet at any time t,', fs: 13, fw: 700, fill: D.purple.dark },
    { t: 'because integrating backward from a shared meeting point would violate Picard–Lindelöf uniqueness!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function spectralDecoupling() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Matrix Exponential Diagonalization:   A = V Λ V⁻¹   ⇒   e^{At} = V diag(e^{λ₁t}, …, e^{λ_{d}t}) V⁻¹', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 148, 122, D.red.tint, D.red.base, [
    { t: '1. Coupled State', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'ẋ = A x', mono: true, fs: 15, fw: 700, fill: C.ink },
    { t: 'x₀ = ∑ cⱼ vⱼ', mono: true, fs: 13, fill: C.ink },
    { t: '(Eigenbasis V⁻¹x₀)', fs: 12, fill: C.muted },
  ]);
  o += arrow(172, 131, 194, 131, C.blue);

  o += card(198, 70, 148, 122, C.lblue, C.blue, [
    { t: '2. Scalar Modes', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'żⱼ = λⱼ zⱼ', mono: true, fs: 15, fw: 700, fill: D.blue.dark },
    { t: 'zⱼ(t) = cⱼ e^{λⱼt}', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: '(d independent 1D ODEs)', fs: 11.5, fill: C.muted },
  ]);
  o += arrow(350, 131, 372, 131, C.green);

  o += card(376, 70, 144, 122, C.lgreen, C.green, [
    { t: '3. Reassemble V', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'x(t) =', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: '∑ cⱼ e^{λⱼt} vⱼ', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Exact for all t!', fs: 12.5, fill: D.green.dark },
  ]);

  o += card(20, 208, 500, 58, D.purple.tint, C.purple, [
    { t: 'Eigenvalues λⱼ = aⱼ + i bⱼ:  |e^{λⱼt}| = e^{aⱼt} (Decay / Growth)  ·  arg = bⱼt (Rotation)', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: 'Asymptotically stable (x(t) → 0 for all x₀) if and only if maxⱼ Re(λⱼ) < 0', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function spiralMatrixExp() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Commuting: P Q = Q P', D.green, [
    { t: 'A = −½ I + K   (I K = K I)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'e^{(−½I + K)t} = e^{−t/2 I} e^{Kt}', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Cyclic K² = −I, K³ = −K, K⁴ = I:', mono: true, fs: 12, fill: C.ink },
    { t: 'e^{Kt} = cos(t) I + sin(t) K', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Exact damped rotation matrix!', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Non-Commuting: P Q ≠ Q P', D.red, [
    { t: 'P = [0, 1; 0, 0],  Q = Pᵀ', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'PQ = [1, 0; 0, 0] ≠ QP = [0, 0; 0, 1]', mono: true, fs: 11.5, fill: C.ink },
    { t: 'e^P e^Q = [2, 1; 1, 1]', mono: true, fs: 13, fill: C.ink },
    { t: 'e^{P+Q} = [cosh 1, sinh 1; …] ≈ [1.54, …]', mono: true, fs: 11.5, fill: D.red.dark },
    { t: 'e^{P+Q} ≠ e^P e^Q  (BCH corrections)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 216, 500, 52, C.lblue, C.blue, [
    { t: 'Compound-Interest Limit:  (I + At/n)ⁿ → e^{At}  always holds, with O(1/n) gap', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function zohExactSsm() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'State Space Model (SSM):  ẋ(t) = A x(t) + B u(t)  →  Exact Zero-Order Hold (u(s) ≡ uₙ on [tₙ, tₙ+h))', fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Exact ZOH Discretization', D.green, [
    { t: 'x_{n+1} = Ā x_n + B̄ u_n', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Ā = e^{Ah},  B̄ = A⁻¹(e^{Ah} − I)B', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'At λ = −50, h = 0.05:', mono: true, fs: 12.5, fill: C.ink },
    { t: 'ā = e^{−2.5} ≈ 0.082  (|ā| < 1 ∀h > 0!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Naive Forward-Euler Step', D.red, [
    { t: 'x_{n+1} = (I + hA)x_n + hB u_n', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'First-order approximation 1 + hλ:', fs: 12.5, fill: C.ink },
    { t: 'At λ = −50, h = 0.05:', mono: true, fs: 12.5, fill: C.ink },
    { t: '1 + hλ = 1 − 2.5 = −1.5  (Explodes!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 220, 500, 48, D.purple.tint, C.purple, [
    { t: 'Why Mamba & S4 Use ZOH: Exponential multiplier e^{hλ} stays strictly stable across all timescales!', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function nonnormalTransient() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lamber, C.amber, [
    { t: 'Non-Normal Transient Growth:  A = [−1, 10; 0, −2]  (AAᵀ ≠ AᵀA,  Eigenvalues λ₁ = −1, λ₂ = −2 < 0)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
  ]);

  // Left plot: the norm rises from 1 to 2.51 at t = ln 2, then decays
  o += rect(20, 66, 236, 200, '#FFFFFF', C.gray, 8, 1.5);
  o += line(38, 236, 238, 236, C.gray, 1.5, false);
  o += line(38, 82, 38, 236, C.gray, 1.5, false);
  // ‖x(t)‖ for x₀ = (0, 1) (red) vs. the monotone decay e^{-t} of a normal
  // matrix (dashed blue); t ∈ [0, 4] at 50 px per unit, 52 px per unit of norm.
  o += `<polyline points="38.0,184.0 40.5,186.5 43.0,188.9 45.5,191.2 48.0,193.4 50.5,195.5 53.0,197.5 55.5,199.4 58.0,201.1 60.5,202.8 63.0,204.5 65.5,206.0 68.0,207.5 70.5,208.9 73.0,210.2 75.5,211.4 78.0,212.6 80.5,213.8 83.0,214.9 85.5,215.9 88.0,216.9 90.5,217.8 93.0,218.7 95.5,219.5 98.0,220.3 100.5,221.1 103.0,221.8 105.5,222.5 108.0,223.2 110.5,223.8 113.0,224.4 115.5,225.0 118.0,225.5 120.5,226.0 123.0,226.5 125.5,227.0 128.0,227.4 130.5,227.8 133.0,228.2 135.5,228.6 138.0,229.0 140.5,229.3 143.0,229.6 145.5,229.9 148.0,230.2 150.5,230.5 153.0,230.8 155.5,231.0 158.0,231.3 160.5,231.5 163.0,231.7 165.5,231.9 168.0,232.1 170.5,232.3 173.0,232.5 175.5,232.7 178.0,232.8 180.5,233.0 183.0,233.1 185.5,233.3 188.0,233.4 190.5,233.5 193.0,233.7 195.5,233.8 198.0,233.9 200.5,234.0 203.0,234.1 205.5,234.2 208.0,234.3 210.5,234.3 213.0,234.4 215.5,234.5 218.0,234.6 220.5,234.6 223.0,234.7 225.5,234.8 228.0,234.8 230.5,234.9 233.0,234.9 235.5,235.0 238.0,235.0" fill="none" stroke="${C.blue}" stroke-width="2" stroke-dasharray="5 4"/>`;
  o += `<polyline points="38.0,184.0 40.5,183.1 43.0,174.2 45.5,162.7 48.0,151.3 50.5,141.0 53.0,132.2 55.5,124.7 58.0,118.7 60.5,114.0 63.0,110.4 65.5,107.9 68.0,106.3 70.5,105.5 73.0,105.4 75.5,105.9 78.0,106.9 80.5,108.4 83.0,110.2 85.5,112.4 88.0,114.9 90.5,117.5 93.0,120.4 95.5,123.4 98.0,126.5 100.5,129.6 103.0,132.8 105.5,136.1 108.0,139.3 110.5,142.6 113.0,145.8 115.5,149.0 118.0,152.2 120.5,155.3 123.0,158.3 125.5,161.3 128.0,164.2 130.5,167.1 133.0,169.8 135.5,172.5 138.0,175.1 140.5,177.7 143.0,180.1 145.5,182.5 148.0,184.8 150.5,187.0 153.0,189.1 155.5,191.1 158.0,193.1 160.5,195.0 163.0,196.8 165.5,198.6 168.0,200.2 170.5,201.9 173.0,203.4 175.5,204.9 178.0,206.3 180.5,207.7 183.0,209.0 185.5,210.2 188.0,211.4 190.5,212.5 193.0,213.6 195.5,214.7 198.0,215.7 200.5,216.6 203.0,217.5 205.5,218.4 208.0,219.2 210.5,220.0 213.0,220.8 215.5,221.5 218.0,222.2 220.5,222.8 223.0,223.5 225.5,224.1 228.0,224.6 230.5,225.2 233.0,225.7 235.5,226.2 238.0,226.7" fill="none" stroke="${D.red.base}" stroke-width="3.5"/>`;
  o += circle(72.7, 105.4, 5, D.red.base, '#FFFFFF', 1.5);
  o += fitTx(138, 86, 'Peak ‖x(ln 2)‖ ≈ 2.51', 210, { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(136, 124, '‖x(t)‖, x₀ = (0, 1)', { fs: 12.5, fw: 700, fill: D.red.dark, anchor: 'start' });
  o += tx(136, 144, 'e⁻ᵗ (slow-mode rate)', { fs: 12.5, fw: 700, fill: D.blue.dark, anchor: 'start' });
  o += fitTx(138, 252, 'x₁ = 10(e⁻ᵗ − e⁻²ᵗ), x₂ = e⁻²ᵗ', 210, { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += card(272, 66, 248, 94, D.red.tint, D.red.base, [
    { t: 'Nearly Parallel Eigenvectors (5.7°)', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'x₀ = (0, 1) = 10.0 v₁ − 10.05 v₂', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Fast decay of v₂ (e⁻²ᵗ) unmasks', fs: 12.5, fill: C.ink },
    { t: 'huge coefficient 10 v₁ (e⁻ᵗ)!', fs: 12.5, fill: D.red.dark },
  ]);

  o += card(272, 172, 248, 94, D.purple.tint, C.purple, [
    { t: 'Heavy-Ball Momentum Connection', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: '2×2 companion matrix of momentum', fs: 12.5, fill: C.ink },
    { t: 'is non-normal (Jordan block at β*),', fs: 12.5, fill: C.ink },
    { t: 'causing early training norm spikes!', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function hartmanGrobmanPendulum() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Local Jacobian Linearization Near f(x*) = 0:   δ̇ = J(x*) δ + o(‖δ‖),   J = ∂f/∂x(x*)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Bottom (φ, ω) = (0, 0): Stable Spiral', D.green, [
    { t: 'J = [0, 1; −1, −0.5]', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'λ = −0.25 ± 0.97 i  (Re λ < 0)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Hyperbolic sink: swings down', fs: 12.5, fill: C.ink },
    { t: 'to rest at rate e^{−t/4}', fs: 12.5, fill: D.green.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Top (φ, ω) = (π, 0): Unstable Saddle', D.red, [
    { t: 'J = [0, 1; +1, −0.5]', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'λ₁ = +0.78 > 0,  λ₂ = −1.28 < 0', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Hyperbolic saddle: any tiny nudge', fs: 12.5, fill: C.ink },
    { t: 'along v₁ topples the pendulum', fs: 12.5, fill: D.red.dark },
  ]);

  o += card(20, 220, 500, 48, D.purple.tint, C.purple, [
    { t: 'Hartman–Grobman Theorem: When Re(λⱼ) ≠ 0 ∀j, local nonlinear flow is conjugate to e^{Jt}', fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.2 mdl-numerical-solvers.md (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function eulerCompoundTable() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Forward Euler on ẋ = x (x₀ = 1, T = 1, h = 1/n):   xₙ = (1 + 1/n)ⁿ  →  e ≈ 2.71828', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  const cols = [
    { n: 'n = 1', val: '2.000', err: 'err 0.718', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { n: 'n = 2', val: '2.250', err: 'err 0.468', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { n: 'n = 4', val: '2.441', err: 'err 0.277', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { n: 'n = 16', val: '2.638', err: 'err 0.080', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { n: 'n = 365', val: '2.715', err: 'err 0.0037', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  cols.forEach((c, i) => {
    const x = 20 + i * 102;
    o += card(x, 70, 92, 122, c.fill, c.st, [
      { t: c.n, mono: true, fs: 14, fw: 700, fill: c.dk },
      { t: c.val, mono: true, fs: 14.5, fw: 700, fill: C.ink },
      { t: c.err, mono: true, fs: 12, fill: c.dk },
    ]);
  });

  o += card(20, 208, 500, 60, C.lgreen, C.green, [
    { t: 'First-Order Global Accuracy (p = 1):   |xₙ − e| ≈ (e / 2) h ≈ 1.36 / n', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Doubling the number of Euler steps n cuts the terminal global error in half', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function localToGlobalError() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Why O(h²) One-Step Local Truncation Error Accumulates to O(h) Global Error', fs: 14, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 70, 152, 124, C.lblue, C.blue, [
    { t: '1. Local Error rₙ', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '‖rₙ‖ ≤ ½ M h²', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '2nd-order Taylor', fs: 12.5, fill: C.muted },
    { t: 'remainder per step', fs: 12.5, fill: C.muted },
  ]);
  o += arrow(176, 132, 198, 132, C.blue);

  o += card(202, 70, 156, 124, C.lamber, C.amber, [
    { t: '2. Step Recursion', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'ε_{n+1} ≤ (1+hL)εₙ', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '+ ½ M h²', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'n = T / h steps', mono: true, fs: 12.5, fill: C.muted },
  ]);
  o += arrow(362, 132, 384, 132, C.green);

  o += card(388, 70, 132, 124, C.lgreen, C.green, [
    { t: '3. Global O(h)', fs: 14, fw: 700, fill: D.green.dark },
    { t: '∑_{j<n} (1+hL)ʲ', mono: true, fs: 12.5, fill: C.ink },
    { t: '≤ (e^{LT}−1)/(hL)', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: '⇒ εₙ = O(h¹)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 208, 500, 58, C.lblue, C.blue, [
    { t: 'General Order-p Rule:   Local Step Error O(h^{p+1})  ×  (T / h Steps)  =  Global Error O(h^p)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Euler: O(h²) → O(h)   ·   Midpoint / Heun: O(h³) → O(h²)   ·   RK4: O(h⁵) → O(h⁴)', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function rk4FourStageProbes() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Classical 4th-Order Runge–Kutta (RK4):  4 Slope Probes Combined via Simpson Weights', fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  const probes = [
    { k: 'k₁ (Start tₙ)', eq: 'f(xₙ, tₙ)', wt: 'Weight 1/6', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { k: 'k₂ (Midpoint 1)', eq: 'f(xₙ + ½hk₁, tₙ + ½h)', wt: 'Weight 2/6', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { k: 'k₃ (Midpoint 2)', eq: 'f(xₙ + ½hk₂, tₙ + ½h)', wt: 'Weight 2/6', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { k: 'k₄ (End tₙ+h)', eq: 'f(xₙ + hk₃, tₙ + h)', wt: 'Weight 1/6', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  probes.forEach((p, i) => {
    const x = 20 + i * 128;
    o += card(x, 68, 114, 122, p.fill, p.st, [
      { t: p.k, fs: 13, fw: 700, fill: p.dk },
      { t: p.eq, mono: true, fs: 12.5, fw: 700, fill: C.ink },
      { t: p.wt, mono: true, fs: 12.5, fill: C.muted },
    ]);
    if (i < 3) {
      o += arrow(x + 116, 129, x + 126, 129, C.gray);
    }
  });

  o += card(20, 206, 500, 62, C.lgreen, C.green, [
    { t: 'x_{n+1} = x_n + (h / 6)(k₁ + 2k₂ + 2k₃ + k₄)   →   Global Error O(h⁴)', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Halving step size h shrinks global error by 2⁴ = 16× (512 evals on spiral: 1.3×10⁻⁹ vs Euler 3.6×10⁻³!)', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function stabilityPolynomials() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 10, 500, 38, D.purple.tint, C.purple, [
    { t: 'Stability Function R(z) on Test Equation ẏ = λy  (y_{n+1} = R(z) y_n,  z = hλ)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 58, 240, 106, '1. Forward Euler (Order p = 1)', D.blue, [
    { t: 'R(z) = 1 + z', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Gap eᶻ − R(z) = ½z² + O(z³)', mono: true, fs: 12, fill: C.ink },
    { t: 'Real interval: −2 < hλ < 0', mono: true, fs: 12, fill: D.red.dark },
  ]);

  o += titledPanel(280, 58, 240, 106, '2. Midpoint / Heun (Order p = 2)', D.orange, [
    { t: 'R(z) = 1 + z + ½z²', mono: true, fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Gap eᶻ − R(z) = (1/6)z³ + O(z⁴)', mono: true, fs: 12, fill: C.ink },
    { t: 'Real interval: −2 < hλ < 0', mono: true, fs: 12, fill: D.orange.dark },
  ]);

  o += titledPanel(20, 174, 240, 102, '3. Classical RK4 (Order p = 4)', D.purple, [
    { t: 'R(z) = 1 + z + ½z² + z³/6 + z⁴/24', fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: 'Gap = z⁵/120 · Real: −2.79 < hλ < 0', fs: 11.5, fill: C.ink },
  ]);

  o += titledPanel(280, 174, 240, 102, '4. Backward Euler (Implicit p = 1)', D.green, [
    { t: 'R(z) = 1 / (1 − z) = 1 + z + z² + ⋯', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'A-Stable: |R(z)| < 1 ∀ Re(z) < 0!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function adaptiveAndSymplectic() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 190, 'Adaptive Step Size (RK45)', D.blue, [
    { t: 'Embedded pair of orders p and p+1', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'shares internal stage evaluations:', fs: 12.5, fill: C.ink },
    { t: 'Δ = ‖x̂_{n+1} − x_{n+1}‖ = O(h^{p+1})', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    { t: 'h ← 0.9 h (tol / Δ)^{1/(p+1)}', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Shrinks in sharp turns, grows in flats', fs: 12, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 190, 'Symplectic Euler & Leapfrog', D.green, [
    { t: 'Hamiltonian oscillator q̇ = p, ṗ = −q:', fs: 12.5, fw: 700, fill: C.ink },
    { t: '1. q_{n+1} = q_n + h p_n', fs: 12.5, fw: 700, fill: C.ink },
    { t: '2. p_{n+1} = p_n − h q_{n+1} (new q!)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Jacobian det S_h = 1 (Area-preserving!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: '(Forward Euler has det = 1 + h² > 1)', fs: 12, fill: D.red.dark },
  ]);

  o += card(20, 218, 500, 50, D.purple.tint, C.purple, [
    { t: 'Role in ML: Adaptive RK45 powers Continuous Normalizing Flows; Leapfrog powers Hamiltonian MC', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function stiffTwoModeTrap() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, D.red.tint, D.red.base, [
    { t: 'Stiffness Bottleneck on ẋ = diag(−50, −1)x:   x₁(t) = e^{−50t} (Fast),   x₂(t) = e^{−t} (Slow)', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Slow Mode x₂(t) = e^{−t} (λ₂ = −1)', D.blue, [
    { t: 'Carries 99.9% of signal for t ≥ 0.1', fs: 12.5, fill: C.ink },
    { t: 'Coarse step h = 0.10 gives:', fs: 12.5, fill: C.ink },
    { t: 'R(−0.1) = 1 − 0.1 = 0.90 ≈ e^{−0.1}', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Tracks e⁻¹ = 0.368 within 5%!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Fast Mode x₁(t) = e^{−50t} (λ₁ = −50)', D.red, [
    { t: 'Exact x₁(1) = e^{−50} ≈ 2 × 10⁻²² ≈ 0', fs: 12.5, fill: C.ink },
    { t: 'At h = 0.10, Euler multiplier is:', fs: 12.5, fill: C.ink },
    { t: 'R(−50 · 0.1) = 1 − 5 = −4  (|R| > 1!)', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Explodes to (−4)¹⁰ = +1.05 × 10⁶!', fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 220, 500, 48, C.lamber, C.amber, [
    { t: 'Explicit Step Ceiling: h < 2/|λ_max| = 2/50 = 0.04 long after e^{−50t} has vanished!', fs: 12.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function backwardEulerImplicit() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Forward Euler (Explicit Start Slope)', D.red, [
    { t: 'x_{n+1} = x_n + h f(x_n, t_n)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'R_fwd(z) = 1 + z', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Extrapolates initial steep slope;', fs: 12.5, fill: C.ink },
    { t: 'diverges when h|λ| > 2', fs: 12.5, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Backward Euler (Implicit End Slope)', D.green, [
    { t: 'x_{n+1} = x_n + h f(x_{n+1}, t_{n+1})', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'R_bwd(z) = 1 / (1 − z)', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'At z = −5 (h = 0.1, λ = −50):', mono: true, fs: 12.5, fill: C.ink },
    { t: 'R_bwd(−5) = 1/6  ⇒  6⁻¹⁰ ≈ 1.6×10⁻⁸!', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, C.lblue, C.blue, [
    { t: 'Optimization Equivalence: Backward Euler on −∇L(x) is the Proximal Point Step argmin_{x} L(x) + (1/2η)‖x−xₖ‖²', fs: 12, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function whyGdUsesEuler() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 40, D.purple.tint, C.purple, [
    { t: 'Why Deep Learning Optimizers Use Forward Euler + Momentum Instead of RK4 (λ_max = 10)', fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 64, 156, 152, 'Forward Euler (GD)', D.green, [
    { t: 'Max η = 2/10 = 0.20', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Cost: 1 grad / step', fs: 12.5, fill: C.ink },
    { t: 'Flow time / grad:', fs: 12.5, fill: C.muted },
    { t: '0.200 (Best!)', mono: true, fs: 14, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(192, 64, 156, 152, 'Classical RK4', D.red, [
    { t: 'Max η = 2.79/10 = 0.28', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    { t: 'Cost: 4 grads / step', fs: 12.5, fill: C.ink },
    { t: 'Flow time / grad:', fs: 12.5, fill: C.muted },
    { t: '0.279 / 4 = 0.070!', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(364, 64, 156, 152, 'Heavy-Ball Momentum', D.blue, [
    { t: 'Changes the ODE!', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'm ẍ + c ẋ = −∇L(x)', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: '2nd-order damped oscillator', fs: 12, fill: C.ink },
    { t: 'O(√κ) vs O(κ) steps', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 228, 500, 46, C.lamber, C.amber, [
    { t: 'Bonus: Euler implicitly follows the modified gradient flow of L̃(x) = L(x) + (η/4)‖∇L(x)‖² (penalizes sharpness)', fs: 12, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.3 mdl-neural-odes.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function cubicDetVsTrace() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Discrete Normalizing Flow', D.red, [
    { t: 'x_{ℓ+1} = g_ℓ(x_ℓ)   (d = 3,072)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'log p_out = log p_in − ∑ log|det J_g|', mono: true, fs: 11.5, fw: 700, fill: D.red.dark },
    { t: 'Full Jacobian det costs O(d³):', fs: 12.5, fill: C.ink },
    { t: '~2 × 10¹⁰ FLOPs per layer', mono: true, fs: 11.5, fw: 700, fill: D.red.dark },
    { t: 'Forces triangular / coupling layers', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Continuous Normalizing Flow (CNF)', D.green, [
    { t: 'ẋ(t) = f_θ(x(t), t)   (Free-form f_θ!)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'd/dt log p_t(x(t)) = −tr(∂f_θ / ∂x)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Replaces det(J) with trace tr(J)!', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Hutchinson εᵀJε costs 1 VJP / step:', fs: 12.5, fill: C.ink },
    { t: 'about one evaluation of f_θ per probe', fs: 12, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, C.lblue, C.blue, [
    { t: 'Infinitesimal Identity:  det(I + hJ) = 1 + h tr(J) + O(h²)   ⇒   log det → ∫ −tr(J) dt', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function neuralOdeGainsCatches() {
  const W = 540, H = 286;
  let o = '';
  o += titledPanel(20, 12, 240, 122, '1. Invertibility vs. 1D Order', D.blue, [
    { t: 'Gain: Φ_T⁻¹ runs −f_θ backward', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Catch: Non-crossing preserves 1D order', fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 12, 240, 122, '2. Test Depth vs. Solver Coupling', D.orange, [
    { t: 'Gain: Vary step count N at test time', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Catch: Coarse Euler fits a 10-step program', fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(20, 148, 240, 124, '3. O(1) Adjoint vs. e^{λT} Drift', D.purple, [
    { t: 'Gain: Continuous adjoint avoids tape', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Catch: Backward state solve explodes e^{λT}', fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 148, 240, 124, '4. Trace Integral vs. Probe Noise', D.green, [
    { t: 'Gain: −tr(J) avoids O(d³) determinant', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Catch: Hutchinson εᵀJε adds variance', fs: 12, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function threeTrainingParadigms() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 40, D.purple.tint, C.purple, [
    { t: 'Three Paradigms for Training a Continuous Vector Field f_θ(x, t)', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 64, 156, 154, '1. Unrolled Euler', D.red, [
    { t: '10 fwd + 10 bwd', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Memory: O(N)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Fits its h = 0.1 program;', fs: 12, fill: D.red.dark },
    { t: 'refined to N = 1000,', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'error rises to ~0.02', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(192, 64, 156, 154, '2. Unrolled RK4', D.blue, [
    { t: '40 fwd + 40 bwd', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Memory: O(4N)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'O(h⁴) accuracy learns', fs: 12, fill: C.ink },
    { t: 'true continuous flow', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: '(4× slower / step)', fs: 12, fill: C.muted },
  ]);

  o += titledPanel(364, 64, 156, 154, '3. Flow Matching', D.green, [
    { t: '1 fwd + 1 bwd!', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Memory: O(1)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Simulation-free', fs: 12.5, fill: C.ink },
    { t: 'regression onto', fs: 12.5, fill: C.ink },
    { t: 'closed-form u_t(x|z)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 230, 500, 44, C.lgreen, C.green, [
    { t: `Modern Frontier (§§${num('sec_mdl-score-matching-diffusion-flow')}–${num('sec_mdl-flow-matching-transport')}): Train simulation-free (1 eval/step), sample with fast ODE solvers!`, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function pontryaginAdjointVjp() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Pontryagin Continuous Adjoint State:   a(t) = ∂L / ∂x(t) ∈ ℝᵈ,   a(T) = ∇_x ℓ(x(T))', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, '1. State Adjoint ODE (Backward)', D.blue, [
    { t: 'ȧ(t) = −(∂f / ∂x(t))ᵀ a(t)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Chosen so d/dt(aᵀ δx) cancels', fs: 12.5, fill: C.ink },
    { t: 'the state coupling (∂f/∂x) δx:', fs: 12.5, fill: C.ink },
    { t: 'd/dt(aᵀ δx) = aᵀ (∂f/∂θ) δθ', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);

  o += titledPanel(280, 68, 240, 138, '2. Parameter Gradient Integral', D.green, [
    { t: '∂L / ∂θ = ∫₀ᵀ (∂f / ∂θ(t))ᵀ a(t) dt', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Accumulates instantaneous nudges', fs: 12.5, fill: C.ink },
    { t: '(∂f/∂θ)δθ weighted by their', fs: 12.5, fill: C.ink },
    { t: 'downstream impact a(t)', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 220, 500, 48, D.purple.tint, C.purple, [
    { t: 'AD Efficiency: both J_xᵀa and J_θᵀa come from 1 reverse-mode VJP!', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function discreteVsContAdjoint() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Discretize-Then-Optimize (Backprop)', D.green, [
    { t: 'a_n = (I + h J(x_n))ᵀ a_{n+1}', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Evaluates Jacobian at forward x_n', fs: 12.5, fill: C.ink },
    { t: 'Exact gradient of the N-step solver', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'With √N forward checkpointing:', fs: 12.5, fill: C.ink },
    { t: 'Zero backward drift, O(√N) memory!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Optimize-Then-Discretize (Adjoint)', D.orange, [
    { t: 'a_n = (I + h J(x_{n+1}))ᵀ a_{n+1}', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Differs by O(h) from discrete backprop', fs: 12.5, fill: C.ink },
    { t: 'Re-integrating x(t) backward from x(T):', fs: 12.5, fill: C.ink },
    { t: 'Stiff decaying mode λ = 50 explodes', fs: 12.5, fill: D.red.dark },
    { t: 'by e^{50T} ≈ 5 × 10²¹ in float64!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 216, 500, 52, C.lblue, C.blue, [
    { t: 'Systems Rule: Always checkpoint forward states and recompute short segments FORWARD in time', fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function liouvilleNearIdentity() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 500, 68, C.lblue, C.blue, [
    { t: 'Step 1 · Infinitesimal Flow Map Jacobian Over Step h', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'Φ_{t→t+h}(x) = x + h f(x, t) + o(h)   ⇒   DΦ = I + h J(x, t) + o(h)', mono: true, fs: 13, fill: C.ink },
  ]);
  o += arrow(270, 84, 270, 104, C.blue);

  o += card(20, 108, 500, 68, D.purple.tint, C.purple, [
    { t: 'Step 2 · Near-Identity Determinant Expansion (Off-Diagonals Are O(h²))', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'det(I + hJ) = ∏_{i=1}^{d} (1 + h J_{ii}) + O(h²) = 1 + h tr(J) + O(h²)', mono: true, fs: 13, fill: C.ink },
  ]);
  o += arrow(270, 178, 270, 198, C.green);

  o += card(20, 202, 500, 68, C.lgreen, C.green, [
    { t: 'Step 3 · Take Limit h → 0 in Change-of-Variables Formula (Liouville Equation)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'd/dt log p_t(x(t)) = −tr(∂f / ∂x(x(t), t)) = −∇ · f(x(t), t)', mono: true, fs: 14, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function hutchinsonRademacher() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Hutchinson Stochastic Trace Estimator:   E[ε εᵀ] = I   ⇒   E[εᵀ J ε] = tr(J E[εεᵀ]) = tr(J)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 68, 156, 146, '1. Exact Diagonal', D.red, [
    { t: '∑_{i=1}^{d} eᵢᵀ J eᵢ', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Cost: d VJPs / step!', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Variance: 0', mono: true, fs: 13, fill: D.green.dark },
    { t: 'Prohibitive for', fs: 12, fill: C.muted },
    { t: 'large dimension d', fs: 12, fill: C.muted },
  ]);

  o += titledPanel(192, 68, 156, 146, '2. Rademacher ±1', D.green, [
    { t: 'εᵢ ∈ {−1, +1} i.i.d.', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Cost: 1 VJP / step!', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Since εᵢ² ≡ 1 always:', fs: 12, fill: C.ink },
    { t: 'Var = 2 ∑_{i≠j} S_{ij}²', fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Zero diagonal noise!', fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(364, 68, 156, 146, '3. Gaussian N(0, I)', D.orange, [
    { t: 'ε ~ N(0, I)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Cost: 1 VJP / step', fs: 12.5, fill: C.ink },
    { t: 'Since Var(εᵢ²) = 2:', fs: 12, fill: C.ink },
    { t: 'Var = 2 ‖S‖_{F}²', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Includes 2 ∑ J_{ii}²!', mono: true, fs: 12, fill: D.red.dark },
  ]);

  o += card(20, 226, 500, 46, C.lblue, C.blue, [
    { t: 'FFJORD Rule: Hold one Rademacher probe ε fixed per trajectory so adaptive RK45 sees a smooth ODE', fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.4 mdl-sdes.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function sqrtDtScalingWindow() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Random Walk With Step ±(Δt)ᵃ Over k = t / Δt Steps:   Var(W_t) = (t / Δt)(Δt)²ᵃ = t (Δt)²ᵃ⁻¹', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 68, 156, 144, 'a > 1/2 (e.g. ±Δt)', D.red, [
    { t: '2a − 1 > 0', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Var(W_t) = t (Δt)²ᵃ⁻¹', mono: true, fs: 12, fill: C.ink },
    { t: '→ 0 as Δt → 0!', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Noise freezes out', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(192, 68, 156, 144, 'a = 1/2 (±√Δt)', D.green, [
    { t: '2a − 1 = 0', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Var(W_t) = t · 1 = t!', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'By CLT: W_t ~ N(0, t)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'And (dW)² = dt!', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(364, 68, 156, 144, 'a < 1/2 (Too Large)', D.orange, [
    { t: '2a − 1 < 0', mono: true, fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Var(W_t) → +∞', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    { t: 'as Δt → 0!', fs: 13, fill: D.red.dark },
    { t: 'Walk explodes instantly', fs: 12, fill: C.muted },
  ]);

  o += card(20, 224, 500, 46, D.purple.tint, C.purple, [
    { t: 'Unique Scaling Law: Only ΔW = √Δt ξ (ξ ~ N(0,1)) yields non-trivial continuous random motion', fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function itoIsometryMartingale() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Itô Stochastic Integral (Left-Endpoint Predictable):   ∫₀ᵗ H_s dW_{s} = lim ∑ H_{tᵢ} (W_{t_{i+1}} − W_{tᵢ})', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, '1. Zero-Mean Martingale Property', D.blue, [
    { t: 'H_{tᵢ} decided BEFORE ΔWᵢ is drawn:', fs: 12.5, fill: C.ink },
    { t: 'E[H_{tᵢ} ΔWᵢ] = E[H_{tᵢ}] E[ΔWᵢ] = 0', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: '⇒  E[ ∫₀ᵗ H_s dW_{s} ] = 0', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: '(Right-endpoint sum adds +t bias!)', fs: 12, fill: D.red.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, '2. The Itô Isometry (Variance)', D.green, [
    { t: 'Cross terms i < j vanish: E[⋯ΔWⱼ] = 0', fs: 12.5, fill: C.ink },
    { t: 'Diagonal terms: E[Hᵢ²] E[(ΔWᵢ)²] = E[Hᵢ²]Δtᵢ', mono: true, fs: 11.5, fill: C.ink },
    { t: 'E[(∫₀ᵗ H_s dW_{s})²] = ∫₀ᵗ E[H_s²] ds', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Converts stochastic L² to time integral', fs: 12, fill: D.green.dark },
  ]);

  o += card(20, 220, 500, 48, C.lamber, C.amber, [
    { t: 'Canonical Example:   ∫₀ᵗ W_s dW_{s} = ½ (W_t² − t)   (Mean = 0,  Variance = ∫₀ᵗ s ds = ½ t²)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function itoLemmaExpansion() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: '2nd-Order Taylor Expansion of φ(X_t, t) Along dX = f dt + g dW', fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 68, 152, 126, N.panel, C.gray, [
    { t: 'Taylor to 2nd Order', fs: 13.5, fw: 700, fill: C.ink },
    { t: 'dφ = φ_t dt + φ_x dX', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '+ ½ φ_{xx} (dX)²', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '(Cannot drop (dX)²!)', fs: 12, fill: D.red.dark },
  ]);
  o += arrow(176, 131, 198, 131, C.blue);

  o += card(202, 68, 156, 126, C.lamber, C.amber, [
    { t: 'Itô Table on (dX)²', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: '(dt)² = 0, dt dW = 0', mono: true, fs: 12, fill: C.muted },
    { t: '(dW)² = dt!', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: '⇒ (dX)² = g² dt', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);
  o += arrow(362, 131, 384, 131, C.green);

  o += card(388, 68, 132, 126, C.lgreen, C.green, [
    { t: 'Itô’s Lemma', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'dφ = (φ_t + f φ_x', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: '+ ½ g² φ_{xx}) dt', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: '+ g φ_x dW', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);

  o += card(20, 208, 500, 58, D.purple.tint, C.purple, [
    { t: 'Multivariate Isotropic Form in ℝᵈ (dWᵢ dWⱼ = δ_{ij} dt):', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'dφ(X_t, t) = ( ∂_{t} φ + ∇φ · f + ½ g(t)² Δφ ) dt  +  g(t) ∇φ · dW_{t}', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function ouEmVsExact() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Exact OU / DDPM Step (θ=1, g=√2)', D.green, [
    { t: 'X_{n+1} = a X_n + √b ξ_n', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'a = e^{−θΔt} = 0.99005', mono: true, fs: 13, fill: D.green.dark },
    { t: 'b = 1 − e^{−2θΔt} = 0.01980', mono: true, fs: 13, fill: D.green.dark },
    { t: 'Stationary variance b / (1 − a²):', fs: 12.5, fill: C.ink },
    { t: 'v_{∞} = 1.0000  (Exact for ANY Δt!)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Euler–Maruyama Step (Δt = 0.01)', D.orange, [
    { t: 'X̂_{n+1} = (1 − θΔt)X̂_n + √(2θΔt) ξ_n', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'a_EM = 1 − Δt = 0.99000', mono: true, fs: 13, fill: D.orange.dark },
    { t: 'b_EM = 2Δt = 0.02000', mono: true, fs: 13, fill: D.orange.dark },
    { t: 'Stationary variance 2Δt / (2Δt − Δt²):', fs: 12, fill: C.ink },
    { t: 'v_{∞} = 1 / (1 − Δt/2) = 1.0050 (+0.5%)', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 216, 500, 52, C.lblue, C.blue, [
    { t: 'Euler–Maruyama Orders: Weak = 1 (O(Δt)); Strong = 1 (additive) or 1/2 (mult.)', fs: 12, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.5 mdl-fokker-planck-probability-flow.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function fpWeakDerivation() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 14, 500, 68, C.lblue, C.blue, [
    { t: 'Step 1 · Apply Multivariate Itô’s Lemma to Smooth Test Bump φ(X_t) & Take E[·]', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'd/dt E[φ(X_t)] = ∫ φ(x) ∂_{t} p_t(x) dx = ∫ [ ∇φ · f + ½ g(t)² Δφ ] p_t(x) dx', mono: true, fs: 12.5, fill: C.ink },
  ]);
  o += arrow(270, 84, 270, 104, C.blue);

  o += card(20, 108, 500, 68, D.purple.tint, C.purple, [
    { t: 'Step 2 · Integrate by Parts (1 Sign Flip on ∇·, 2 Sign Flips on Δ = ∇·∇)', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '∫ (∇φ · f) p_t dx = −∫ φ ∇·(f p_t) dx,     ∫ (Δφ) p_t dx = +∫ φ Δp_t dx', mono: true, fs: 12.5, fill: C.ink },
  ]);
  o += arrow(270, 178, 270, 198, C.green);

  o += card(20, 202, 500, 68, C.lgreen, C.green, [
    { t: 'Step 3 · Holds for Every Bump φ ⇒ Pointwise Fokker–Planck Equation', fs: 14, fw: 700, fill: D.green.dark },
    { t: '∂_{t} p_t(x)  =  −∇ · (f(x, t) p_t(x))  +  ½ g(t)² Δp_t(x)', mono: true, fs: 14, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function ouMomentOdes() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Gaussian Ansatz p(x, t) = N(x; m(t), v(t)) in OU Fokker–Planck ∂_{t} p = θ ∂_{x}(xp) + ½g² ∂_{xx}p', fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, '1. Mean ODE (Match (x − m)¹)', D.blue, [
    { t: 'ṁ(t) = −θ m(t)', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'm(t) = m₀ e^{−θt} → 0', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Deterministic exponential decay', fs: 12.5, fill: C.ink },
    { t: 'of the center of mass', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 68, 240, 138, '2. Variance ODE (Match (x − m)²)', D.green, [
    { t: 'v̇(t) = g² − 2θ v(t)', mono: true, fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'v(t) = v₀e^{−2θt} + (g²/2θ)(1−e^{−2θt})', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Equilibrates at v_{∞} = g² / (2θ)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '(Drift compression = diffusion spread)', fs: 12, fill: C.muted },
  ]);

  o += card(20, 220, 500, 48, D.purple.tint, C.purple, [
    { t: 'Superposition: any mixture ∑ wₖ N(μₖ, v₀) evolves each component independently!', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function laplacianToScoreFlow() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Log-Derivative Chain Rule:   ∇p_t = p_t ∇log p_t   ⇒   ½g² Δp_t = ∇ · (p_t · ½g² ∇log p_t)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 70, 206, 126, C.lamber, C.amber, [
    { t: 'Fokker–Planck PDE', fs: 14, fw: 700, fill: D.orange.dark },
    { t: '∂_{t} p_t = −∇·(f p_t)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: '+ ½ g(t)² Δp_t', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: '(Advection + Diffusion)', fs: 12, fill: C.muted },
  ]);

  o += arrow(232, 133, 308, 133, C.purple);
  o += fitTx(270, 110, 'Identical', 72, { fs: 13, fw: 700, fill: D.purple.dark });
  o += fitTx(270, 154, 'Marginals p_t', 72, { fs: 12, fw: 600, fill: C.ink });

  o += card(314, 70, 206, 126, C.lgreen, C.green, [
    { t: 'Continuity Equation', fs: 14, fw: 700, fill: D.green.dark },
    { t: '∂_{t} p_t = −∇·(p_t u_t)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'u_t = f − ½g² ∇log p_t', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '(Probability-Flow ODE!)', fs: 12, fill: D.green.dark },
  ]);

  o += card(20, 210, 500, 56, C.lblue, C.blue, [
    { t: 'Physical Meaning: Random Brownian kicks spread the density cloud at the exact same rate', fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'as a deterministic velocity −½ g(t)² ∇log p_t(x) pushing probability downhill from peaks to tails!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function andersonBayesTilt() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Infinitesimal Bayes Reversal:   p(x_t = x | x_{t+τ} = y)  ∝  p(y | x) · p_t(x)', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 68, 152, 130, C.lblue, C.blue, [
    { t: '1. Forward Kernel', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: '−‖x−(y−fτ)‖²/(2g²τ)', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: 'Gaussian centered at', fs: 12, fill: C.muted },
    { t: 'y − f(y, t)τ', mono: true, fs: 12.5, fill: D.blue.dark },
  ]);

  o += tx(182, 133, '+', { fs: 22, fw: 700, fill: C.ink });

  o += card(194, 68, 152, 130, D.purple.tint, C.purple, [
    { t: '2. Prior Score Tilt', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'log p_t(x) ≈ log p_t(y)', mono: true, fs: 11.5, fill: C.ink },
    { t: '+ (x−y)·∇log p_t(y)', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
    { t: 'Linear tilt in (x − y)', fs: 12, fill: C.muted },
  ]);

  o += arrow(350, 133, 370, 133, C.green);

  o += card(374, 68, 146, 130, C.lgreen, C.green, [
    { t: '3. Complete Square', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Shifts mean by +g²τ s_t:', fs: 12, fill: C.ink },
    { t: 'dX = [f − g²∇log p_t]dt', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: '+ g(t) dW̄', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);

  o += card(20, 212, 500, 56, C.lamber, C.amber, [
    { t: 'Why Factor of 2 vs. PF-ODE?   Reverse λ-Family Drift:  b_λ(x, t) = f − ½(1 + λ²) g² ∇log p_t', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'First ½g²s_t undoes forward smoothing (λ=0 ODE); second ½λ²g²s_t counteracts reverse noise (λ=1 SDE)', fs: 12, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.6 mdl-score-matching-diffusion-flow.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function regressionLemmaFloor() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Regression Lemma (Pythagorean L² Split, m(X) = E[Y | X]):   E[‖v(X) − Y‖²]', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 70, 240, 134, '1. Excess Risk (Depends on θ)', D.blue, [
    { t: 'E[ ‖v_θ(X) − E[Y | X]‖² ]', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Uniquely minimized at v_θ(X) = E[Y|X]', fs: 12.5, fill: C.ink },
    { t: 'Cross term 2E[(v−m)ᵀ(m−Y)] = 0', mono: true, fs: 12, fill: D.green.dark },
    { t: 'by the tower property E[E[·|X]]', fs: 12, fill: C.muted },
  ]);

  o += titledPanel(280, 70, 240, 134, '2. Bayes-Risk Floor C (Constant)', D.orange, [
    { t: 'C = E[ ‖Y − E[Y | X]‖² ] > 0', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Conditional variance of noisy targets Y', fs: 12.5, fill: C.ink },
    { t: 'passing through the same state X', fs: 12.5, fill: C.ink },
    { t: '(∇_θ C = 0: affects no gradients!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 50, C.lgreen, C.green, [
    { t: 'Engine of DSM & CFM: regressing onto Y = −ε/σ (or ε−x₀) fits E[Y|X] = ∇log p_σ (or u_t)!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function dsmWeightSchedules() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 40, D.purple.tint, C.purple, [
    { t: 'Noise-Conditional DSM Weighting Schedules w(t) as Noise Scale σ_t → 0', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 64, 156, 152, '1. Unweighted w = 1', D.red, [
    { t: 'Target: −ε / σ_t', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Floor R(t) ~ d / σ_t²', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Explodes as σ_t → 0;', fs: 12.5, fill: D.red.dark },
    { t: 'tiny noise levels', fs: 12, fill: C.ink },
    { t: 'swamp training grads', fs: 12, fill: C.muted },
  ]);

  o += titledPanel(192, 64, 156, 152, '2. DDPM Simple w = σ_t²', D.green, [
    { t: '½ ‖ε − ε_θ(x_t, t)‖²', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'w(t) R(t) = σ_t²(d/σ_t²)', mono: true, fs: 11.5, fill: C.ink },
    { t: '= d = O(1) everywhere!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Balances gradient scale', fs: 12, fill: C.ink },
    { t: 'across all noise levels', fs: 12, fill: D.green.dark },
  ]);

  o += titledPanel(364, 64, 156, 152, '3. Likelihood w = g(t)²', D.blue, [
    { t: 'w(t) = β(t)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'w(t) R(t) ~ d / t', mono: true, fs: 12.5, fill: C.ink },
    { t: '∫_{ε}^{T} (d/t)dt = d log(T/ε)', mono: true, fs: 11, fill: C.ink },
    { t: 'Upper-bounds KL', fs: 12.5, fill: D.blue.dark },
    { t: 'D_KL(p₀ ‖ p_{θ,0})!', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 228, 500, 44, C.lgreen, C.green, [
    { t: 'Tweedie Identity Links All Three:  s_θ(x_t, t) = −ε_θ(x_t, t)/σ_t = (α_t x̂₀(x_t, t) − x_t)/σ_t²', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function ddpmToDdimEuler() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 194, 174, C.lblue, C.blue, [
    { t: 'DDIM Update (VP Clock)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'x_{t\'} = √ᾱ_{t\'} x̂₀', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '+ √(1 − ᾱ_{t\'}) ε_θ(x_t, t)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Deterministic jump', fs: 12.5, fill: C.ink },
    { t: 'across stride t → t\'', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(220, 84, 320, 84, C.purple);
  o += fitTx(270, 62, 'x̄_t = x_t / √ᾱ_{t}', 96, { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += fitTx(270, 114, 'σ̄_t = √((1−ᾱ_{t})/ᾱ_{t})', 96, { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  o += card(326, 14, 194, 174, C.lgreen, C.green, [
    { t: 'Forward Euler on PF-ODE', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'x̄_{t\'} = x̄_t +', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '(σ̄_{t\'} − σ̄_t) ε_θ(x_t, t)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'dx̄ / dσ̄ = E[ε | x_t]', mono: true, fs: 12.5, fill: C.ink },
    { t: '1st-order ODE solver!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 204, 500, 60, C.lamber, C.amber, [
    { t: 'Consequence: DDIM skips from 1,000 ancestral SDE steps to 20–50 deterministic ODE strides,', fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'and any higher-order ODE solver (Heun, DPM-Solver) can be dropped directly onto the same trained ε_θ!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function cfgExtrapolation() {
  const W = 540, H = 282;
  let o = '';
  // Left vector diagram showing s_empty, s_y, and extrapolated CFG s_tilde along the difference ray
  o += rect(20, 16, 236, 250, '#FFFFFF', C.gray, 8, 1.5);
  // Origin at (56, 210)
  const ox = 56, oy = 210;
  // Unconditional score s_empty to (130, 176)
  o += arrow(ox, oy, 128, 176, C.gray);
  // Conditional score s_y to (156, 128)
  o += arrow(ox, oy, 154, 130, C.blue);
  // Difference ray from s_empty (130, 176) through s_y (156, 128) out to CFG (208, 32)
  o += line(130, 176, 156, 128, C.purple, 2, true);
  o += arrow(156, 128, 202, 44, D.red.base);
  o += circle(130, 176, 4.5, C.gray, '#FFFFFF', 1.5);
  o += circle(156, 128, 5, C.blue, '#FFFFFF', 1.5);
  o += circle(204, 41, 5.5, D.red.base, '#FFFFFF', 1.5);
  o += fitTx(158, 184, 's_∅', 50, { mono: true, fs: 13, fw: 700, fill: C.muted });
  o += fitTx(104, 122, 's_y (γ=1)', 80, { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += fitTx(134, 52, 's̃_γ (γ > 1)', 100, { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += fitTx(138, 246, 'Linear extrapolation along s_y − s_∅', 220, { fs: 12, fw: 700, fill: C.ink });

  o += card(272, 16, 248, 76, C.lblue, C.blue, [
    { t: 'Bayes Score Addition (γ = 1)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '∇log p_t(x|y) = s_∅ + ∇log p_t(y|x)', mono: true, fs: 12, fill: C.ink },
    { t: 'Implicit classifier: s_y − s_∅', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(272, 103, 248, 76, D.purple.tint, C.purple, [
    { t: 'Classifier-Free Guidance (γ > 1)', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 's̃ = s_∅ + γ(s_y − s_∅)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: '= (1 − γ) s_∅ + γ s_y', mono: true, fs: 13, fill: C.ink },
  ]);

  o += card(272, 190, 248, 76, D.red.tint, D.red.base, [
    { t: 'Broken Curvature Bound', fs: 14, fw: 700, fill: D.red.dark },
    { t: '−∇²log p̃_t > 1/σ_t² for large γ:', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    { t: 'Sharpens & shifts mode outward!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §38.7 mdl-flow-matching-transport.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function curvedDiffusionVsStraight() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Diffusion Probability-Flow ODE', D.orange, [
    { t: 'Path inherited from OU / VP-SDE:', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'x_t = √ᾱ_{t} x₀ + √(1 − ᾱ_{t}) ε', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'Trajectories curve across time;', fs: 12.5, fill: C.ink },
    { t: 'one example: >90% of path below t = 0.5', fs: 12.5, fill: D.red.dark },
    { t: 'Requires 20–50 ODE solver steps', fs: 13, fw: 700, fill: D.orange.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Straight Constant-Speed Flow Ideal', D.green, [
    { t: 'Prescribe straight-line interpolant:', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'x_t = (1 − t) x₀ + t x₁', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Constant velocity: ẋ_{t} = x₁ − x₀', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'When paired without crossings (C = 0):', fs: 12, fill: C.ink },
    { t: '1 Euler step (h = 1) is 100% exact!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, C.lblue, C.blue, [
    { t: 'Flow Matching Paradigm: Prescribe p_t directly and regress u_θ(x, t) onto per-sample velocity u_t(x | z)', fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function cfmMarginalPosterior() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Conditional Flow Matching (CFM):   u_t(x) = E[ u_t(x | z) | x_t = x ]   Solves ∂_{t} p_t + ∇·(p_t u_t) = 0', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 68, 152, 126, C.lblue, C.blue, [
    { t: 'Conditional Blobs', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'z = (x₀, x₁) ~ q', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'x_t = (1−t)x₀ + t x₁', mono: true, fs: 12, fill: C.ink },
    { t: 'u_t(x_t | z) = x₁ − x₀', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
  ]);
  o += arrow(176, 131, 198, 131, C.blue);

  o += card(202, 68, 156, 126, D.purple.tint, C.purple, [
    { t: 'CFM Regression Loss', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'L_CFM(θ) =', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'E‖u_θ(x_t,t)−(x₁−x₀)‖²', mono: true, fs: 11.5, fw: 700, fill: D.purple.dark },
    { t: '= L_FM(θ) + C', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  o += arrow(362, 131, 384, 131, C.green);

  o += card(388, 68, 132, 126, C.lgreen, C.green, [
    { t: 'Crossing Floor C', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'C = E[Var(x₁−x₀|x_t)]', mono: true, fs: 11, fw: 700, fill: C.ink },
    { t: 'C = 0 iff sample', fs: 12, fill: C.ink },
    { t: 'chords never cross!', fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 208, 500, 58, C.lamber, C.amber, [
    { t: 'Why Independent Pairing Bends: u₁(x) = x − E[x₀], so 1 Euler step collapses all points to E[x₀]!', fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'When straight chords cross at (x, t), averaging their velocities creates a curved non-crossing ODE flow', fs: 12, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function scoreVelocityDictionary() {
  const W = 540, H = 288;
  let o = '';
  o += card(20, 10, 500, 40, D.purple.tint, C.purple, [
    { t: 'Score–Velocity Dictionary on Gaussian Paths x_t = α_t x₀ + σ_t ε   (All Affine in x̂₀ = E[x₀ | x_t])', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 60, 156, 102, C.lblue, C.blue, [
    { t: '1. Score s_θ(x, t)', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 's = ∇_x log p_t(x)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '= −ε̂ / σ_t', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += card(192, 60, 156, 102, C.lamber, C.amber, [
    { t: '2. Noise ε̂(x, t)', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'E[ε | x_t = x]', mono: true, fs: 12.5, fill: C.ink },
    { t: '= −σ_t s', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);

  o += card(364, 60, 156, 102, C.lgreen, C.green, [
    { t: '3. Clean Data x̂₀(x, t)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'E[x₀ | x_t = x]', mono: true, fs: 12.5, fill: C.ink },
    { t: '= (x + σ_t² s) / α_t', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 174, 240, 102, D.purple.tint, C.purple, [
    { t: '4. Flow Velocity u_t(x)', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'E[α̇_t x₀ + σ̇_t ε | x_t = x]', mono: true, fs: 12.5, fill: C.ink },
    { t: '= (α̇_t/α_t)x − (σ_{t}σ̇_{t} − σ_t²α̇_t/α_t)s', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
  ]);

  o += card(280, 174, 240, 102, D.teal.tint, D.teal.base, [
    { t: '5. v-Prediction v̂(x, t)', fs: 14, fw: 700, fill: D.teal.dark },
    { t: 'E[α_t ε − σ_t x₀ | x_t = x]', mono: true, fs: 12.5, fill: C.ink },
    { t: '= −(σ_t/α_t)x − σ_t((α_t²+σ_t²)/α_t)s', mono: true, fs: 11.5, fw: 700, fill: D.teal.dark },
  ]);
  return svg(W, H, o);
}

function reflowJensenStraightening() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lgreen, C.green, [
    { t: 'Rectified Flow Reflow: Re-Pair Noise Z₁ = ε With ODE Endpoint Z₀ = Φ_{1→0}(ε) & Retrain', fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Round 1: Independent Pairs (X₀, X₁)', D.red, [
    { t: 'Straight chords cross heavily:', fs: 12.5, fill: C.ink },
    { t: 'Crossing floor C = 1.37 / coord', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'ODE paths bend to avoid crossing', fs: 12.5, fill: C.ink },
    { t: 'Needs 32 Euler steps (1-step collapses)', fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Round 2: Reflowed Pairs (Z₀, Z₁)', D.green, [
    { t: 'Deterministic ODE endpoints never cross:', fs: 12, fill: C.ink },
    { t: 'Crossing floor C drops 1.37 → ≈0.001–0.003', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Trajectories become 20–35× straighter;', fs: 12.5, fill: C.ink },
    { t: '1 Euler step matches 32 steps!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 220, 500, 48, D.purple.tint, C.purple, [
    { t: 'Double-Jensen Theorem:  E[c(Z₁ − Z₀)] ≤ E[c(X₁ − X₀)]  for all convex costs c (drops ‖·‖² by ≥ C)', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

export const diagrams = {
  // index.md (4)
  'ch38-dyn-unified-primitives':         unifiedPrimitives,
  'ch38-dyn-three-stage-roadmap':        threeStageRoadmap,
  'ch38-dyn-five-benchmarks':            fiveBenchmarks,
  'ch38-dyn-four-equations-bridge':      fourEquationsBridge,
  // mdl-odes-solvers.md (10)
  'ch38-dyn-well-posed-triad':           wellPosedTriad,
  'ch38-dyn-decay-plus-rotation':        decayPlusRotation,
  'ch38-dyn-picard-fixed-point':         picardFixedPoint,
  'ch38-dyn-finite-time-blowup':         finiteTimeBlowup,
  'ch38-dyn-flow-group-bijection':       flowGroupBijection,
  'ch38-dyn-spectral-decoupling':        spectralDecoupling,
  'ch38-dyn-spiral-matrix-exp':          spiralMatrixExp,
  'ch38-dyn-zoh-exact-ssm':              zohExactSsm,
  'ch38-dyn-nonnormal-transient':        nonnormalTransient,
  'ch38-dyn-hartman-grobman-pendulum':   hartmanGrobmanPendulum,
  // mdl-numerical-solvers.md (8)
  'ch38-dyn-euler-compound-table':       eulerCompoundTable,
  'ch38-dyn-local-to-global-error':      localToGlobalError,
  'ch38-dyn-rk4-four-stage-probes':      rk4FourStageProbes,
  'ch38-dyn-stability-polynomials':      stabilityPolynomials,
  'ch38-dyn-adaptive-and-symplectic':    adaptiveAndSymplectic,
  'ch38-dyn-stiff-two-mode-trap':        stiffTwoModeTrap,
  'ch38-dyn-backward-euler-implicit':    backwardEulerImplicit,
  'ch38-dyn-why-gd-uses-euler':          whyGdUsesEuler,
  // mdl-neural-odes.md (7)
  'ch38-dyn-cubic-det-vs-trace':         cubicDetVsTrace,
  'ch38-dyn-neural-ode-gains-catches':   neuralOdeGainsCatches,
  'ch38-dyn-three-training-paradigms':   threeTrainingParadigms,
  'ch38-dyn-pontryagin-adjoint-vjp':     pontryaginAdjointVjp,
  'ch38-dyn-discrete-vs-cont-adjoint':   discreteVsContAdjoint,
  'ch38-dyn-liouville-near-identity':    liouvilleNearIdentity,
  'ch38-dyn-hutchinson-rademacher':      hutchinsonRademacher,
  // mdl-sdes.md (4)
  'ch38-dyn-sqrt-dt-scaling-window':     sqrtDtScalingWindow,
  'ch38-dyn-ito-isometry-martingale':    itoIsometryMartingale,
  'ch38-dyn-ito-lemma-expansion':        itoLemmaExpansion,
  'ch38-dyn-ou-em-vs-exact':             ouEmVsExact,
  // mdl-fokker-planck-probability-flow.md (4)
  'ch38-dyn-fp-weak-derivation':         fpWeakDerivation,
  'ch38-dyn-ou-moment-odes':             ouMomentOdes,
  'ch38-dyn-laplacian-to-score-flow':    laplacianToScoreFlow,
  'ch38-dyn-anderson-bayes-tilt':        andersonBayesTilt,
  // mdl-score-matching-diffusion-flow.md (4)
  'ch38-dyn-regression-lemma-floor':     regressionLemmaFloor,
  'ch38-dyn-dsm-weight-schedules':       dsmWeightSchedules,
  'ch38-dyn-ddpm-to-ddim-euler':         ddpmToDdimEuler,
  'ch38-dyn-cfg-extrapolation':          cfgExtrapolation,
  // mdl-flow-matching-transport.md (4)
  'ch38-dyn-curved-diffusion-vs-straight': curvedDiffusionVsStraight,
  'ch38-dyn-cfm-marginal-posterior':     cfmMarginalPosterior,
  'ch38-dyn-score-velocity-dictionary':  scoreVelocityDictionary,
  'ch38-dyn-reflow-jensen-straightening': reflowJensenStraightening,
};
