// Slide diagrams for ch09-optimization.mjs (Chapter 9: Optimization Algorithms)
import { C, TOKENS, FS, FM, grid, tx, baselineY, arrow, block, chip, rc, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const circ = (cx, cy, r, fill, stroke, sw = 2, dash = false) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="4 3"' : ''}/>`;

const ellip = (cx, cy, rx, ry, fill, stroke, sw = 2, dash = false) =>
  `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const card = (x, y, w, h, accent, tint, title, l1, l2, l3) => {
  let o = box(x, y, w, h, tint, accent, 8, 2);
  const cx = x + w / 2;
  if (l3) {
    o += tx(cx, y + 20, title, { fs: 13.5, fw: 700, fill: accent });
    o += tx(cx, y + 44, l1, { fs: 12.5, fw: 600, fill: C.ink });
    o += tx(cx, y + 66, l2, { mono: true, fs: 11.5, fw: 600, fill: C.ink });
    o += tx(cx, y + 88, l3, { fs: 12, fw: 600, fill: C.muted });
  } else if (l2 && h < 90) {
    // short card: pull the three lines up so the mono l2 line clears the border
    o += tx(cx, y + 20, title, { fs: 14, fw: 700, fill: accent });
    if (l1) o += tx(cx, y + 42, l1, { fs: 12.5, fw: 600, fill: C.ink });
    o += tx(cx, y + 62, l2, { mono: true, fs: 12, fw: 600, fill: C.muted });
  } else {
    o += tx(cx, y + 22, title, { fs: 14, fw: 700, fill: accent });
    if (l1) o += tx(cx, y + 48, l1, { fs: 12.5, fw: 600, fill: C.ink });
    if (l2) o += tx(cx, y + 72, l2, { mono: true, fs: 12, fw: 600, fill: C.muted });
  }
  return o;
};

// ═══════════════════════════════════════════════════════════════════════
// §9.0 INDEX (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function twoObstacles() {
  const W = 560, H = 280;
  let o = '';
  // Left panel: Ill-conditioning
  o += box(16, 16, 254, 248, C.lblue, C.blue, 10, 2);
  o += tx(143, 38, '1. Ill-Conditioning (κ = 20)', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(143, 58, 'f(x) = 0.1 x₁² + 2 x₂²', { mono: true, fs: 12, fw: 600, fill: C.muted });
  o += ellip(143, 128, 96, 34, '#FFFFFF', C.blue, 1.8);
  o += ellip(143, 128, 56, 18, C.lblue, C.blue, 1.5, true);
  o += `<polyline points="62,106 86,150 110,110 134,144 158,118 182,132" fill="none" stroke="${D.red.base}" stroke-width="2.2"/>`;
  o += circ(194, 128, 4.5, C.green, D.green.dark, 1.5);
  o += box(28, 184, 230, 66, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(143, 205, 'Steep λ₂ = 4 caps η < 0.5', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(143, 230, 'Flat λ₁ = 0.2 creeps slowly', { fs: 12.5, fw: 600, fill: C.ink });

  // Right panel: Minibatch Gradient Noise
  o += box(290, 16, 254, 248, C.lamber, C.amber, 10, 2);
  o += tx(417, 38, '2. Minibatch Gradient Noise', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(417, 58, 'g_t = ∇f(x_t) + ξ_t', { mono: true, fs: 12, fw: 600, fill: C.muted });
  o += ellip(417, 128, 76, 34, '#FFFFFF', C.amber, 2, true);
  const pts = [[371,122],[388,140],[404,114],[426,138],[444,118],[412,130],[434,126],[394,124]];
  pts.forEach(([px, py]) => { o += circ(px, py, 3.5, D.orange.base, D.orange.dark, 1); });
  o += circ(417, 128, 5, C.green, D.green.dark, 1.5);
  o += box(302, 184, 230, 66, '#FFFFFF', C.amber, 6, 1.5);
  o += tx(417, 205, 'Stationary Noise Ball at x*', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(417, 230, 'E[x_∞²] ≈ η σ² / (2 λ b)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function threeLevers() {
  const W = 560, H = 290;
  let o = '';
  o += card(16, 12, 528, 80, D.blue.dark, C.lblue,
    'Lever 1: Step Direction & Preconditioning (§§9.4–9.6)',
    'Momentum velocity · AdamW coordinate RMS · Muon spectral UVᵀ',
    'Targets ill-conditioning κ ≫ 1 and 40× layer scale gaps');
  o += card(16, 104, 528, 80, D.purple.dark, D.purple.tint,
    'Lever 2: Step-Size Schedule η_t (§9.7)',
    'Linear warmup (0 → η₀) · Travel budget B_T = Σ η_t · Cooldown',
    'Steers below 2/η₀ early and shrinks terminal noise ball');
  o += card(16, 196, 528, 80, D.green.dark, C.lgreen,
    'Lever 3: Batch Size, Width Scaling & Guards (§§9.8–9.10)',
    'Critical batch b_crit ≈ b_noise · μP width LR η/m · Clip & EMA',
    'Balances step–data Pareto curve and guards against spikes');
  return svg(W, H, o);
}

function fourTestbeds() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 114, D.blue.dark, C.lblue,
    '1. Quadratic Valley (§9.1)',
    '0.1 x₁² + 2 x₂² (d = 2, κ = 20)',
    'Exact 2/λ, √κ damping, noise',
    'Closed-form analytical model');
  o += card(290, 14, 254, 114, D.orange.dark, C.lamber,
    '2. NASA Airfoil Data (§9.2)',
    'n = 1,500 examples, d = 6',
    'Exact L = 2.10, tr(Σ) = 3.24',
    'Verifies η tr(Σ) / (4b) floor');
  o += card(16, 144, 254, 114, D.green.dark, C.lgreen,
    '3. FashionCNN (§9.3+)',
    '421,642 params (2 Conv + 2 FC)',
    '8.5 epochs in 2,000 steps',
    'Test accuracy & weight EMA');
  o += card(290, 144, 254, 114, D.purple.dark, D.purple.tint,
    '4. TinyLM Transformer (§9.3+)',
    '470,784 params (2 pre-LN blocks)',
    '0.03 epochs on WikiText bytes',
    '40× grad spread · AdamW & Muon');
  return svg(W, H, o);
}

function roadmap() {
  const W = 560, H = 280;
  let o = '';
  const items = [
    { x: 16,  y: 14,  sec: '§§9.1–9.2 Basics', title: 'Landscapes & SGD', sub: 'η < 2/λ · ησ²/(2λb)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 198, y: 14,  sec: '§9.3 Testbeds',     title: 'TinyLM & CNN',     sub: 'Landmarks · 3× seed', fill: N.panel, st: C.gray, dk: C.ink },
    { x: 380, y: 14,  sec: '§9.4 Momentum',     title: 'Heavy-Ball & NAG', sub: 'η/(1−β) · O(√κ)',     fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 16,  y: 152, sec: '§§9.5–9.6 Adaptive', title: 'AdamW & Muon',    sub: 'τ=1/(ηλ) · −ηUVᵀ',    fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 198, y: 152, sec: '§§9.7–9.8 Scaling',  title: 'Schedules & Batch', sub: 'B_T · b_crit ≈ b_noise', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 380, y: 152, sec: '§§9.9–9.10 Recipe',  title: 'μP, Clip & EMA',   sub: 'LR η/m · θ=1.0 · EMA', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  items.forEach(it => {
    o += box(it.x, it.y, 164, 110, it.fill, it.st, 8, 2);
    o += tx(it.x + 82, it.y + 24, it.sec, { mono: true, fs: 12, fw: 700, fill: it.dk });
    o += tx(it.x + 82, it.y + 54, it.title, { fs: 14, fw: 700, fill: C.ink });
    o += tx(it.x + 82, it.y + 84, it.sub, { mono: true, fs: 11.5, fw: 600, fill: C.muted });
  });
  o += arrow(182, 69, 196, 69, C.blue);
  o += arrow(364, 69, 378, 69, C.blue);
  o += arrow(182, 207, 196, 207, C.amber);
  o += arrow(364, 207, 378, 207, C.green);
  o += line(462, 125, 462, 138, C.blue) + line(462, 138, 98, 138, C.blue);
  o += arrow(98, 137, 98, 151, C.blue);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.1 GRADIENT DESCENT (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function gdObstacles() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 114, D.blue.dark, C.lblue,
    '1. Anisotropic Curvature',
    'Condition number κ = λ_max / λ_min',
    'Sharp λ_max caps η < 2/λ_max',
    'Flat λ_min creeps at ~2/κ');
  o += card(290, 14, 254, 114, D.purple.dark, D.purple.tint,
    '2. High-d Saddle Points',
    'Minima prob ~ 2⁻ᵈ at high loss',
    'Unstable mode (1 + ημ) > 1',
    'Repels iterates exponentially');
  o += card(16, 144, 528, 114, D.orange.dark, C.lamber,
    '3. Progressive Sharpening → Edge of Stability (λ_max → 2/η)',
    'Hessian sharpness λ_max(∇²f(x_t)) rises steadily during training',
    'Arrests and plateaus at stability ceiling 2 / η',
    'Self-stabilizes via non-monotonic steps along sharpest eigenvector');
  return svg(W, H, o);
}

function gdEigenmodes() {
  const W = 560, H = 276;
  let o = '';
  o += box(16, 14, 528, 56, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 33, 'Eigenbasis Expansion:  x_t = Σ (1 − η λ_i)ᵗ c_i v_i', { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 54, 'Condition Number κ = λ_max / λ_min governs convergence rate', { fs: 12.5, fw: 600, fill: C.ink });

  o += card(16, 86, 254, 90, D.red.dark, D.red.tint,
    'Sharpest Mode (λ_max)',
    'Requires |1 − η λ_max| < 1',
    'Global ceiling: η < 2 / λ_max');
  o += card(290, 86, 254, 90, D.blue.dark, C.lblue,
    'Flattest Mode (λ_min)',
    'Factor 1 − η λ_min > 1 − 2/κ',
    'Shrinks by < 2/κ per step');

  o += box(16, 192, 528, 70, C.lgreen, C.green, 8, 2);
  o += tx(280, 215, 'Optimal Constant Rate:  η* = 2 / (λ_min + λ_max)', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(280, 241, 'Factor ρ* = (κ − 1)/(κ + 1)   ⇒   Steps t ≈ (κ / 2) ln(1 / ε)', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function gdSaddleEscape() {
  const W = 560, H = 266;
  let o = '';
  o += box(16, 16, 246, 234, N.panel, C.gray, 10, 2);
  o += tx(139, 38, 'Strict Saddle Geometry', { fs: 14.5, fw: 700, fill: C.ink });
  const cx = 139, cy = 134;
  o += line(cx - 84, cy, cx + 84, cy, D.red.base, 2);
  o += line(cx, cy - 64, cx, cy + 64, C.blue, 2);
  o += arrow(cx, cy - 60, cx, cy - 18, C.blue);
  o += arrow(cx, cy + 60, cx, cy + 18, C.blue);
  o += arrow(cx + 16, cy, cx + 78, cy, D.red.base);
  o += arrow(cx - 16, cy, cx - 78, cy, D.red.base);
  o += circ(cx, cy, 5, '#FFFFFF', C.ink, 2);
  o += tx(139, 224, 'Blue: λ > 0 · Red: −μ < 0', { fs: 12.5, fw: 600, fill: C.ink });

  o += card(278, 16, 266, 108, D.red.dark, D.red.tint,
    'Unstable Mode (λ = −μ < 0)',
    'c_{t+1} = (1 + η μ) c_t',
    'Exponentially repels δ > 0');
  o += card(278, 142, 266, 108, D.green.dark, C.lgreen,
    'Logarithmic Escape Time',
    't_esc ≈ ln(1 / δ) / (η μ)',
    'Escapes saddles w.p. 1');
  return svg(W, H, o);
}

function gdDescentLemma() {
  const W = 560, H = 268;
  let o = '';
  o += box(16, 14, 528, 64, C.lblue, C.blue, 8, 2);
  o += tx(280, 35, 'L-Smooth Upper Bound (‖∇²f(x)‖₂ ≤ L everywhere):', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(280, 58, 'f(x − η∇f) ≤ f(x) − η(1 − ηL/2) ‖∇f(x)‖²', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += card(16, 94, 164, 158, D.green.dark, C.lgreen,
    'Optimal Step',
    'η = 1 / L',
    'Drop ‖∇f‖²/(2L)',
    'Max guarantee');
  o += card(198, 94, 164, 158, D.orange.dark, C.lamber,
    'Safe Window',
    '0 < η < 2 / L',
    '1 − ηL/2 > 0',
    'Strict descent');
  o += card(380, 94, 164, 158, D.red.dark, D.red.tint,
    'Overshoot',
    'η ≥ 2 / L',
    '1 − ηL/2 ≤ 0',
    'No descent guarantee');
  return svg(W, H, o);
}

function gdSharpnessFilter() {
  const W = 560, H = 268;
  let o = '';
  o += box(16, 14, 528, 52, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 40, 'Linear Stability of a Minimum x*:  |1 − η f″(x*)| < 1  ⇔  f″(x*) < 2/η', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });

  o += card(16, 82, 254, 170, D.blue.dark, C.lblue,
    'Small LR: η = 0.5 (2/η = 4)',
    'Deep sharp basin (f″ = 1.81 < 4)',
    '1 − 0.5(1.81) = +0.09',
    'STABLE: converges to deep basin');
  o += card(290, 82, 254, 170, D.orange.dark, C.lamber,
    'Large LR: η = 2.0 (2/η = 1)',
    'Deep sharp basin (f″ = 1.81 > 1)',
    '1 − 2.0(1.81) = −2.62',
    'EJECTED → flat basin (f″ = 0.98)');
  return svg(W, H, o);
}

function gdNewtonHazards() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 52, C.lblue, C.blue, 8, 2);
  o += tx(280, 40, 'Newton Step:  x ← x − H⁻¹ ∇f(x)   (rescales mode i by 1/λ_i → κ = 1)', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });

  o += card(16, 82, 164, 174, D.red.dark, D.red.tint,
    '1. O(d²) Cost',
    '10¹⁸ entries',
    'at d = 10⁹',
    'Infeasible RAM');
  o += card(198, 82, 164, 174, D.orange.dark, C.lamber,
    '2. Saddles',
    'λ_i < 0 flips',
    'step uphill',
    'Attracts saddles!');
  o += card(380, 82, 164, 174, D.purple.dark, D.purple.tint,
    '3. Noise',
    'Inverting tiny',
    'noisy λ_i',
    'Explodes var');
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.2 MINIBATCH SGD (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function sgdCostVsNoise() {
  const W = 560, H = 268;
  let o = '';
  o += card(16, 14, 254, 144, D.red.dark, D.red.tint,
    'Full-Batch GD (b = 1.28M)',
    '1.28M images per update',
    '450K updates = 450K epochs',
    '5,000× excess compute cost!');
  o += card(290, 14, 254, 144, D.green.dark, C.lgreen,
    'Minibatch SGD (b = 256)',
    '256 images per update',
    '450K updates = 90 epochs',
    '5,000× cheaper per step');
  o += box(16, 174, 528, 78, C.lamber, C.amber, 8, 2);
  o += tx(280, 198, 'The Fundamental Stochastic Trade-Off', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(280, 226, 'Cheaper updates inject sampling variance tr(Σ)/b → Stationary Noise Ball', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function sgdSignalPlusNoise() {
  const W = 560, H = 264;
  let o = '';
  o += box(16, 18, 154, 92, C.lblue, C.blue, 8, 2);
  o += tx(93, 46, 'True Signal', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(93, 76, '∇f(x_t)', { mono: true, fs: 14.5, fw: 700, fill: C.ink });

  o += tx(186, 64, '+', { fs: 22, fw: 700, fill: C.ink });

  o += box(202, 18, 156, 92, C.lamber, C.amber, 8, 2);
  o += tx(280, 46, 'Sampling Noise', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 76, 'ξ_t (E[ξ_t]=0)', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  o += tx(374, 64, '=', { fs: 22, fw: 700, fill: C.ink });

  o += box(390, 18, 154, 92, C.lgreen, C.green, 8, 2);
  o += tx(467, 46, 'Sampled Grad', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(467, 76, '∇f_i(x_t)', { mono: true, fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 130, 528, 116, N.panel, C.gray, 8, 2);
  o += tx(280, 156, 'At Global Minimum x* where ∇f(x*) = 0:', { fs: 14, fw: 700, fill: C.ink });
  o += tx(280, 186, 'Per-example gradients ∇f_i(x*) ≠ 0 whenever examples disagree', { fs: 12.5, fw: 600, fill: D.red.dark });
  o += tx(280, 216, '⇒ Constant-rate iterates keep receiving random kicks −η ξ_t', { mono: true, fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function sgdNoiseEquilibrium() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 56, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 33, 'Second-Moment Recursion Along Eigendirection λ:', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 54, 'E[x_{t+1}²] = (1 − ηλ)² E[x_t²] + η² σ²', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  o += card(16, 86, 240, 88, D.blue.dark, C.lblue,
    'Curvature Contraction',
    'Multiplies by (1 − ηλ)²',
    'Removes 2ηλ(1−ηλ/2) E[x²]');
  o += tx(280, 130, '⇌', { fs: 22, fw: 700, fill: C.ink });
  o += card(304, 86, 240, 88, D.orange.dark, C.lamber,
    'Step Noise Injection',
    'Adds variance +η² σ²',
    'Fresh kick every step');

  o += box(16, 190, 528, 66, C.lgreen, C.green, 8, 2);
  o += tx(280, 213, 'Stationary Noise-Ball Balance (E[x_{t+1}²] = E[x_t²]):', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(280, 238, 'E[x_∞²] = η σ² / [λ(2 − ηλ)]  ≈  η σ² / (2 λ)', { mono: true, fs: 13, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function sgdThreeLaws() {
  const W = 560, H = 282;
  let o = '';
  o += card(16, 12, 528, 78, D.red.dark, D.red.tint,
    'Law 1: Stability Ceiling Preserved (η < 2 / λ_max)',
    'Denominator (2 − ηλ) → 0 as η → 2/λ; variance blows up at 2/λ',
    'Steep valley wall λ₂ = 4 still caps η < 0.5');
  o += card(16, 102, 528, 78, D.blue.dark, C.lblue,
    'Law 2: Inverse Curvature Scaling (E[x_∞²] ∝ 1 / λ)',
    'Flat λ₁ = 0.2 has var 0.253 vs. steep λ₂ = 4 var 0.016',
    'Noise cloud has 16× higher variance (4× wider std) along flat axis');
  o += card(16, 192, 528, 78, D.green.dark, C.lgreen,
    'Law 3: Linear η / b Scaling (E[x_∞²] ∝ η / b)',
    'Halving η or doubling batch size b halves stationary variance',
    'Ratio η / b controls the steady-state noise floor');
  return svg(W, H, o);
}

function sgdRobbinsMonro() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 56, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 33, 'Robbins–Monro Exact Convergence Conditions (1951)', { fs: 14.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 54, '1) Infinite Travel: Σ η_t = ∞      2) Finite Noise: Σ η_t² < ∞', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += card(16, 86, 164, 170, D.orange.dark, C.lamber,
    'Constant η',
    'Σ η_t = ∞ (OK)',
    'Σ η_t² = ∞',
    'Stalls in ball');
  o += card(198, 86, 164, 170, D.red.dark, D.red.tint,
    'Geometric η₀γᵗ',
    'Σ η_t² < ∞ (OK)',
    'Σ η_t < ∞',
    'Freezes early!');
  o += card(380, 86, 164, 170, D.green.dark, C.lgreen,
    'Poly c/(t+t₀)',
    'Σ η_t = ∞',
    'Σ η_t² < ∞',
    'MSE = O(1/t)');
  return svg(W, H, o);
}

function sgdCurvatureCancel() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 16, 240, 92, C.lblue, C.blue, 8, 2);
  o += tx(136, 40, 'Mode k Iterate Variance', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(136, 66, 'v_k ≈ η σ_k² / (2 λ_k b)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(136, 90, 'Scales as 1 / λ_k', { fs: 12, fw: 600, fill: C.muted });

  o += tx(280, 62, '×', { fs: 22, fw: 700, fill: C.ink });

  o += box(304, 16, 240, 92, C.lamber, C.amber, 8, 2);
  o += tx(424, 40, 'Mode k Loss Curvature', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(424, 66, 'Δf_k = ½ λ_k v_k', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(424, 90, 'Scales as λ_k', { fs: 12, fw: 600, fill: C.muted });

  o += arrow(280, 114, 280, 144, C.purple);

  o += box(16, 150, 528, 106, C.lgreen, C.green, 8, 2);
  o += tx(280, 176, 'Curvature λ_k Cancels Out in Every Eigendirection!', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(280, 204, '½ λ_k · [η σ_k² / (2 λ_k b)] = η σ_k² / (4 b)', { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  o += tx(280, 234, 'Total Excess Loss:  E[f(x_∞)] − f(x*) ≈ η tr(Σ(x*)) / (4 b)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.3 TWO TEST MODELS (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function testQuadVsNet() {
  const W = 560, H = 268;
  let o = '';
  o += card(16, 14, 254, 140, D.blue.dark, C.lblue,
    'Analytical Quadratic Valley',
    'd = 2, static H = diag(0.2, 4)',
    'Exact η_max = 0.5',
    'Zero seed-to-seed variance');
  o += card(290, 14, 254, 140, D.purple.dark, D.purple.tint,
    'Neural Testbeds (TinyLM/CNN)',
    'd ≈ 4.5×10⁵, dynamic λ_max(x_t)',
    '40× cross-layer scale spread',
    'Stochastic init & batch noise');
  o += box(16, 170, 528, 82, C.lgreen, C.green, 8, 2);
  o += tx(280, 194, 'Three Requirements for Fair Optimizer Benchmarking', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(280, 218, '1. Complementary architectures (Transformer + CNN)   2. Loss landmarks', { fs: 12, fw: 600, fill: C.ink });
  o += tx(280, 238, '3. Independent LR sweep per arm + 3× multi-seed significance threshold', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function testTinylmArch() {
  const W = 560, H = 276;
  let o = '';
  o += box(16, 14, 528, 52, C.lblue, C.blue, 8, 2);
  o += tx(280, 32, 'Input Bytes X: (B=64, T=64) ∈ {0..255}', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(280, 52, 'H₀ = E_tok(X) + E_pos(0..63) ∈ ℝ^{64×64×128}  (40,960 emb params)', { mono: true, fs: 11.5, fw: 600, fill: C.ink });

  o += arrow(280, 68, 280, 92, C.blue);

  o += box(16, 96, 528, 92, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 118, '2× Pre-LN Causal Transformer Blocks (d=128, 2 heads)', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 144, 'H ← H + CausalMHA(LN₁(H))  [qkv 128→384, proj 128→128]', { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(280, 168, 'H ← H + MLP_GELU(LN₂(H))   [mlp1 128→512, mlp2 512→128]', { mono: true, fs: 12, fw: 600, fill: C.ink });

  o += arrow(280, 190, 280, 214, C.purple);

  o += box(16, 218, 528, 46, C.lgreen, C.green, 8, 2);
  o += tx(280, 241, 'Readout: Linear_{128→256}(LN(H)) → Next-Byte Logits (470,784 params)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function testRegimeComparison() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 254, 244, D.purple.tint, C.purple, 10, 2);
  o += tx(143, 38, 'TinyLM (WikiText-103)', { fs: 15, fw: 700, fill: D.purple.dark });
  o += tx(143, 60, '2-Block Transformer (471K)', { mono: true, fs: 11.5, fw: 600, fill: C.muted });
  o += box(28, 78, 230, 52, '#FFFFFF', C.purple, 6, 1.5);
  o += tx(143, 96, 'Single-Pass Pretraining', { fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(143, 116, '0.03 epochs (3% of 270M B)', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += box(28, 142, 230, 52, '#FFFFFF', C.purple, 6, 1.5);
  o += tx(143, 160, 'Zero Memorization Gap', { fs: 13, fw: 700, fill: C.ink });
  o += tx(143, 180, 'Fresh unseen text every step', { fs: 11.5, fw: 600, fill: C.muted });
  o += tx(143, 226, 'Metric: Nats/byte & PPL', { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  o += box(290, 14, 254, 244, C.lblue, C.blue, 10, 2);
  o += tx(417, 38, 'FashionCNN (Images)', { fs: 15, fw: 700, fill: D.blue.dark });
  o += tx(417, 60, '2-Conv + 2-FC CNN (422K)', { mono: true, fs: 11.5, fw: 600, fill: C.muted });
  o += box(302, 78, 230, 52, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(417, 96, 'Multi-Epoch Classifier', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(417, 116, '8.5 epochs (512K / 60K img)', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += box(302, 142, 230, 52, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(417, 160, 'Generalization Gap', { fs: 13, fw: 700, fill: C.ink });
  o += tx(417, 180, '95% in fc1; can overfit', { fs: 11.5, fw: 600, fill: C.muted });
  o += tx(417, 226, 'Metric: Loss & Test Acc', { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  return svg(W, H, o);
}

function testHarnessFlow() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 16, 254, 110, D.purple.dark, D.purple.tint,
    '1. lr_sweep(model, opt)',
    'Clones shared seed-0 weights',
    'Sweeps log η grid per arm');
  o += card(290, 16, 254, 110, D.blue.dark, C.lblue,
    '2. train_steps(T=2000)',
    'Fixed 2,000-step budget',
    'Same loop for every optimizer');
  o += card(290, 144, 254, 110, D.orange.dark, C.lamber,
    '3. final_loss & steps_to',
    'Mean of last k=100 steps',
    '25-step block threshold');
  o += card(16, 144, 254, 110, D.green.dark, C.lgreen,
    '4. Held-Out Evaluation',
    '1.1M WikiText val bytes',
    '10K Fashion-MNIST test');
  o += arrow(272, 71, 288, 71, C.gray);
  o += arrow(417, 128, 417, 142, C.gray);
  o += arrow(288, 199, 272, 199, C.gray);
  return svg(W, H, o);
}

function testLossLandmarks() {
  // Bar length is proportional to the loss in nats (36 px per nat).
  const W = 560, H = 280, X0 = 206, K = 36;
  let o = '';
  const bars = [
    { y: 16,  v: 5.71, val: '5.71 nats (303)', title: 'Untrained TinyLM (step 1)', sub: 'ln 256 + s²/2 (s² ≈ 1/3)', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { y: 70,  v: 5.55, val: '5.55 nats (256)', title: 'Uniform Random Guess', sub: 'p = 1/256 per byte', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 124, v: 3.18, val: '3.18 nats (24.0)', title: 'Unigram Byte Counts', sub: 'Marginal p(b) frequency', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 178, v: 2.50, val: '2.50 nats (12.2)', title: 'Bigram Byte-Pair Table', sub: 'p(b_t | b_{t−1}) 1-byte ctx', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { y: 232, v: 2.13, v2: 1.50, val: '2.13 → 1.50 nats', title: 'Trained TinyLM (64-B ctx)', sub: '', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  bars.forEach(b => {
    const cy = b.y + 20;
    o += tx(16, b.sub ? b.y + 11 : cy, b.title, { fs: 13, fw: 700, fill: b.dk, anchor: 'start' });
    if (b.sub) o += tx(16, b.y + 30, b.sub, { fs: 11.5, fw: 500, fill: C.ink, anchor: 'start' });
    if (b.v2) {
      o += box(X0, cy - 13, b.v * K, 26, '#FFFFFF', b.st, 5, 1.8, true);
      o += box(X0, cy - 13, b.v2 * K, 26, b.fill, b.st, 5, 2);
    } else {
      o += box(X0, cy - 13, b.v * K, 26, b.fill, b.st, 5, 2);
    }
    o += tx(X0 + b.v * K + 8, cy, b.val, { mono: true, fs: 12.5, fw: 700, fill: b.dk, anchor: 'start' });
  });
  return svg(W, H, o);
}

function testLrDiagnosis() {
  const W = 560, H = 282;
  let o = '';
  o += card(16, 12, 528, 78, D.blue.dark, C.lblue,
    'η = 0.1 (Too Low — Under-Budgeted Travel B_T = 200)',
    'Smooth monotonic drop; still descending at step 2,000 (2.48 nats)',
    'Action: Raise η (short of travel budget, not trapped in noise ball)');
  o += card(16, 102, 528, 78, D.green.dark, C.lgreen,
    'η = 0.5 (Workable — Near Initial Sharpness Ceiling 2/L)',
    'Early >1-nat spike in steps 1–30, then steady descent to 2.13 nats',
    'Action: Best constant SGD rate; add warmup (§9.7) or clipping (§9.10)');
  o += card(16, 192, 528, 78, D.red.dark, D.red.tint,
    'η = 2.5 (Too High — Exceeds Stability Limit η > 2/λ_max)',
    'Step 1 overshoots into steeper region; overflows to inf in 10 steps',
    'Action: Lower learning rate η immediately');
  return svg(W, H, o);
}

function testNuisanceProtocol() {
  const W = 560, H = 274;
  let o = '';
  o += card(16, 14, 164, 144, D.purple.dark, D.purple.tint,
    'Scientific Arm',
    'Update rule',
    'SGD vs AdamW',
    'vs Muon');
  o += card(198, 14, 164, 144, D.orange.dark, C.lamber,
    'Nuisance η',
    'Tune on log grid',
    'independently',
    'for each arm!');
  o += card(380, 14, 164, 144, D.blue.dark, C.lblue,
    'Fixed Scope',
    'Shared seed-0',
    'Same batch b',
    'Same T = 2,000');

  o += box(16, 174, 528, 84, C.lgreen, C.green, 8, 2);
  o += tx(280, 198, 'Two Metrics Reported from Every Learning-Rate Sweep', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(280, 222, '1. Peak Score at η*: Lowest final_loss / validation loss across grid', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(280, 242, '2. Usable LR Window: Span of η finishing within +0.15 nats of η*', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function testSeedThreshold() {
  const W = 560, H = 266;
  let o = '';
  o += box(16, 14, 528, 52, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 40, 'Decision Rule: Single-Run Gap Must Exceed 3× Five-Seed Range', { fs: 14, fw: 700, fill: D.purple.dark });

  o += card(16, 82, 254, 168, D.blue.dark, C.lblue,
    'TinyLM Loss (nats/byte)',
    '5-seed range: 0.010–0.015',
    '3× Threshold: ≥ 0.05 nats',
    'Smaller gap = statistical tie');
  o += card(290, 82, 254, 168, D.green.dark, C.lgreen,
    'FashionCNN Loss & Acc',
    '5-seed range: 0.02 / 1.0%',
    'Threshold: ≥ 0.06 / ≥ 3.0%',
    'Replicate across seeds if closer');
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.4 MOMENTUM (5 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function momPersistentVsAlt() {
  const W = 560, H = 268;
  let o = '';
  o += box(16, 16, 528, 112, C.lgreen, C.green, 8, 2);
  o += tx(280, 38, 'Flat Valley Floor (x₁, λ₁ = 0.2): Persistent Gradients', { fs: 14, fw: 700, fill: D.green.dark });
  for (let i = 0; i < 4; i++) {
    o += arrow(44 + i * 74, 72, 104 + i * 74, 72, D.green.dark);
  }
  o += tx(360, 72, '⇒', { fs: 22, fw: 700, fill: D.green.dark });
  o += arrow(390, 72, 515, 72, D.green.dark);
  o += tx(280, 106, 'Constructive sum in v_t = Σ βᵏ g_{t−k}  →  Amplified by 1/(1 − β)', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += box(16, 142, 528, 112, D.red.tint, D.red.base, 8, 2);
  o += tx(280, 164, 'Steep Valley Walls (x₂, λ₂ = 4.0): Alternating Gradients', { fs: 14, fw: 700, fill: D.red.dark });
  o += arrow(104, 192, 44, 192, D.red.base);
  o += arrow(124, 192, 184, 192, D.red.base);
  o += arrow(264, 192, 204, 192, D.red.base);
  o += arrow(284, 192, 344, 192, D.red.base);
  o += tx(375, 192, '⇒', { fs: 22, fw: 700, fill: D.red.dark });
  o += arrow(420, 192, 470, 192, C.blue);
  o += tx(280, 232, 'Destructive cancellation in v_t = Σ (−β)ᵏ g  →  Damped by 1/(1 + β)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function momVelocityUnroll() {
  const W = 560, H = 266;
  let o = '';
  o += box(16, 14, 528, 54, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 32, 'Heavy-Ball Update:  v_t = β v_{t−1} + g_t,   x_{t+1} = x_t − η v_t', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(280, 53, 'Unrolled History:  v_t = g_t + β g_{t−1} + β² g_{t−2} + ···', { mono: true, fs: 12, fw: 600, fill: C.ink });

  const weights = [
    { x: 28,  h: 96, lab: 'g_t',     w: '1' },
    { x: 118, h: 78, lab: 'g_{t-1}', w: 'β' },
    { x: 208, h: 62, lab: 'g_{t-2}', w: 'β²' },
    { x: 298, h: 48, lab: 'g_{t-3}', w: 'β³' },
    { x: 388, h: 36, lab: 'g_{t-4}', w: 'β⁴' },
    { x: 472, h: 24, lab: '···',     w: 'βᵏ' },
  ];
  const base = 212;
  weights.forEach(b => {
    o += box(b.x, base - b.h, 60, b.h, C.lblue, C.blue, 6, 2);
    o += tx(b.x + 30, base - b.h / 2, b.w, { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
    o += tx(b.x + 30, base + 20, b.lab, { mono: true, fs: 12, fw: 600, fill: C.ink });
  });
  o += tx(280, 252, 'Memory Window: τ = 1 / (1 − β) steps  (Half-life ≈ 0.693 τ)', { fs: 12.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function momDualFilter() {
  const W = 560, H = 268;
  let o = '';
  o += card(16, 16, 254, 148, D.green.dark, C.lgreen,
    'Persistent Floor (g_{t−k} ≈ g)',
    '1 + β + β² + ··· = 1/(1 − β)',
    'η_eff = η / (1 − β)',
    '1.2 at η = 0.6, β = 0.5 (2×)');
  o += card(290, 16, 254, 148, D.blue.dark, C.lblue,
    'Alternating Wall ((−1)ᵏ g)',
    '1 − β + β² − ··· = 1/(1 + β)',
    'η_alt = η / (1 + β)',
    '0.4 < 0.5 at η = 0.6 (stable!)');

  o += box(16, 180, 528, 72, C.lamber, C.amber, 8, 2);
  o += tx(280, 204, 'Expanded Stability Ceiling:  η λ_max < 2 (1 + β)', { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 230, 'Unlocks (1 + β)/(1 − β) times larger effective floor steps than plain GD', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function momNoiseImpulse() {
  const W = 560, H = 268;
  let o = '';
  o += box(16, 16, 528, 76, C.lamber, C.amber, 8, 2);
  o += tx(280, 38, 'Cumulative Impulse of One Noise Kick ξ_t in Velocity v_t:', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(280, 66, '−η (1 + β + β² + ···) ξ_t = −[η / (1 − β)] ξ_t = −η_eff ξ_t', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += card(16, 108, 254, 144, D.blue.dark, C.lblue,
    'Plain SGD at Rate η_eff',
    'Full kick −η_eff ξ_t in 1 step',
    'Noise ball ∝ η_eff σ²/(2λb)',
    'Instantaneous step jitter');
  o += card(290, 108, 254, 144, D.purple.dark, D.purple.tint,
    'Momentum at η = (1−β)η_eff',
    'Spreads −η_eff ξ_t over τ steps',
    'Same stationary noise ball!',
    'Scale η ∝ (1 − β) under noise');
  return svg(W, H, o);
}

function momSqrtKappa() {
  const W = 560, H = 274;
  let o = '';
  o += card(16, 14, 254, 144, D.red.dark, D.red.tint,
    'Optimal Plain GD (β = 0)',
    'Contraction ρ* = (κ−1)/(κ+1)',
    '1 − 2 / κ per step',
    'Steps t ≈ (κ / 2) ln(1 / ε)');
  o += card(290, 14, 254, 144, D.green.dark, C.lgreen,
    'Optimal Heavy-Ball (β*)',
    'Contraction ρ* = (√κ−1)/(√κ+1)',
    '1 − 2 / √κ per step',
    'Steps t ≈ (√κ / 2) ln(1 / ε)');

  o += box(16, 174, 528, 86, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 198, 'Why All Underdamped Modes Contract at Exact Rate √β:', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 222, 'z² − (1 + β − ηλ)z + β = 0  has root product  z₁ z₂ = β', { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(280, 244, 'For β ≥ β_c(λ) = (1 − √(ηλ))², complex roots satisfy |z₁| = |z₂| = √β!', { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.5 ADAM & ADAMW (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function adamLayerScales() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 142, D.red.dark, D.red.tint,
    'Raw Gradient RMS (Step 0)',
    'token_emb: 1.7×10⁻⁴',
    'biases: 6.8×10⁻³ (40× gap!)',
    '15× disparity at step 500');
  o += card(290, 14, 254, 142, D.green.dark, C.lgreen,
    'Adam Update RMS ‖Δx‖/η',
    'Divides by coordinate √v̂_{t,i}',
    'All 8 groups in 0.14–0.28 η',
    'Compressed into a 2× band!');

  o += box(16, 172, 528, 84, C.lblue, C.blue, 8, 2);
  o += tx(280, 198, 'Why Nominal η Shifts 160× (SGD 0.5 vs. AdamW 3×10⁻³):', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 224, 'In AdamW, η is a direct coordinate step length in parameter space', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(280, 244, 'Invariant to diagonal rescaling of loss or coordinates', { fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function adamColdStartJump() {
  const W = 560, H = 266;
  let o = '';
  o += card(16, 16, 254, 148, D.red.dark, D.red.tint,
    '1. Cold-Start Jump (No Bias Fix)',
    'v₁ = (1 − β₂) g₁²',
    '|Δx₁| = η / √(1 − β₂)',
    '3.16 η (β₂=0.9) · 31.6 η (0.999)!');
  o += card(290, 16, 254, 148, D.orange.dark, C.lamber,
    '2. Non-Vanishing Late Steps',
    'Steady |g_t| ⇒ √v_t ≈ |g_t|',
    'Δx_t ≈ −η sign(g_t)',
    'Hops in ±η/2 band until η→0');

  o += box(16, 180, 528, 72, C.lgreen, C.green, 8, 2);
  o += tx(280, 204, 'Bias Correction Divides by (1 − β₁ᵗ) and (1 − β₂ᵗ)', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(280, 230, 'Restores exact step-1 magnitude |Δx_{1,i}| = η (45° diagonal step)', { mono: true, fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function adamDualMoments() {
  const W = 560, H = 278;
  let o = '';
  o += box(16, 22, 108, 118, N.panel, C.gray, 8, 2);
  o += tx(70, 68, 'Gradient', { fs: 14, fw: 700, fill: C.ink });
  o += tx(70, 96, 'g_t', { mono: true, fs: 15, fw: 700, fill: D.red.dark });

  o += arrow(126, 44, 158, 44, C.blue);
  o += arrow(126, 122, 158, 122, C.amber);

  o += card(160, 10, 212, 68, D.blue.dark, C.lblue,
    '1st Moment m_t (β₁=0.9)',
    'm̂_t = m_t / (1−β₁ᵗ) [τ₁=10]', '');
  o += card(160, 88, 212, 68, D.orange.dark, C.lamber,
    '2nd Moment v_t (β₂=0.999)',
    'v̂_t = v_t / (1−β₂ᵗ) [τ₂=1000]', '');

  o += arrow(374, 44, 404, 64, C.blue);
  o += arrow(374, 122, 404, 92, C.amber);

  o += box(406, 36, 138, 92, C.lgreen, C.green, 8, 2);
  o += tx(475, 62, 'Adam Update', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(475, 92, '−η m̂_t/(√v̂_t+ε)', { mono: true, fs: 11.5, fw: 700, fill: C.ink });

  o += box(16, 174, 528, 88, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 198, 'Scale Invariance & Step-Length Bounds', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 222, 'Rescaling g_{t,i} → c·g_{t,i} cancels c between m̂_{t,i} and √v̂_{t,i}', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(280, 244, 'Persistent gradient steps by ≈ η  ·  Zero-mean noise steps by ≈ 0.23 η', { mono: true, fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function adamSignAndEps() {
  const W = 560, H = 282;
  let o = '';
  o += card(16, 12, 528, 78, D.purple.dark, D.purple.tint,
    '1. Adam as Smoothed Sign Descent (β₁ = β₂ = 0 ⇒ −η sign(g_t))',
    'Sign descent with momentum matches Adam’s lead over SGD on Transformers',
    'Equalizes step lengths across blocks and rare token embeddings');
  o += card(16, 102, 528, 78, D.blue.dark, C.lblue,
    '2. Role of ε as Maximum Effective LR Cap (η / (√v̂ + ε) ≤ η / ε)',
    'For ε ≫ √v̂_{t,i}, Adam reduces to SGD with momentum at rate (1−β₁)η/ε',
    'Default ε = 10⁻⁸ stays well below typical gradient RMS (10⁻⁴–10⁻²)');
  o += card(16, 192, 528, 78, D.red.dark, D.red.tint,
    '3. Small-β₂ Pathology (Reddi et al., 2019: Periodic +4, −1, −1)',
    'Short-memory β₂ shrinks √v_t on −1 steps and inflates √v_t on +4 step',
    'Keeping β₂ ∈ [0.95, 0.999] stabilizes √v̂_t over 20–1,000 steps');
  return svg(W, H, o);
}

function adamCoupledVsDecoupled() {
  const W = 560, H = 270;
  let o = '';
  o += card(16, 14, 254, 148, D.red.dark, D.red.tint,
    'Coupled ℓ₂ Penalty in Adam',
    'g_t ← g_t + λ x_t inside m_t, v_t',
    'Decay: [ηλ / (√v̂_{t,i}+ε)] x_{t,i}',
    'Collapses matrix RMS 40×!');
  o += card(290, 14, 254, 148, D.green.dark, C.lgreen,
    'Decoupled Decay (AdamW)',
    'Moments use data grad g_t only',
    'x_{t+1} = (1 − ηλ) x_t − η u_t',
    'Uniform shrinkage across all W');

  o += box(16, 178, 528, 76, C.lamber, C.amber, 8, 2);
  o += tx(280, 202, 'Why Coupled ℓ₂ Fails on Adaptive Optimizers:', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(280, 230, 'Weights with small data gradients are divided by tiny √v̂_{t,i} and over-decayed!', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function adamDecayTimescale() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 60, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 34, 'Unrolled AdamW: Weights as an EMA of Past Normalized Updates u_s', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 56, 'x_t = (1 − ηλ)ᵗ x₀ − η Σ_{s=0}^{t−1} (1 − ηλ)^{t−1−s} u_s', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += card(16, 90, 254, 166, D.blue.dark, C.lblue,
    'Decay Timescale τ',
    '(1 − ηλ)ᵏ ≈ exp(−k / τ)',
    'τ = 1 / (η λ) steps',
    '3,333 steps at η=3e-3, λ=0.1');
  o += card(290, 90, 254, 166, D.green.dark, C.lgreen,
    'Decoupled LR Sweeps',
    'Fix shrinkage s = ηλ = 1/τ',
    'Set λ = s / η when sweeping η',
    'Keeps memory horizon τ fixed!');
  return svg(W, H, o);
}

function adamRotationalEq() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 146, D.red.dark, D.red.tint,
    'Without Weight Decay (λ = 0)',
    'Random-walk norm growth:',
    'RMS(x_t) ∝ η √t',
    'Angular step η/RMS decays 1/√t');
  o += card(290, 14, 254, 146, D.green.dark, C.lgreen,
    'With AdamW Decay (t ≳ 2τ)',
    'Equilibrium weight norm:',
    'RMS(x_∞) ≈ √[η / (2λ)]',
    'Locks angular step √(2ηλ)!');

  o += box(16, 176, 528, 80, C.lblue, C.blue, 8, 2);
  o += tx(280, 200, 'Rotational Equilibrium Across Scale-Invariant LayerNorm Sublayers', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 226, 'Every decayed 2D matrix rotates at constant angular rate √(2ηλ) per step', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(280, 244, 'Independent of initialization scale and gradient magnitude (Kosson et al., 2024)', { fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.6 MATRIX PRECONDITIONING & MUON (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function muonOperatorView() {
  const W = 560, H = 266;
  let o = '';
  o += card(16, 14, 254, 144, D.orange.dark, C.lamber,
    'Flat-Vector View (AdamW)',
    'Treats W ∈ ℝ^{m×n} as mn',
    'independent scalar entries',
    'Permutation-blind to matrix!');
  o += card(290, 14, 254, 144, D.purple.dark, D.purple.tint,
    'Operator View (Muon)',
    'W maps x ∈ ℝⁿ ↦ y = Wx ∈ ℝᵐ',
    '83.5% of TinyLM sits in',
    '8 hidden 2D weight matrices');

  o += box(16, 174, 528, 76, C.lblue, C.blue, 8, 2);
  o += tx(280, 198, 'Bounding the Induced Activation Shift Δy = ΔW x:', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(280, 226, '‖ΔW x‖₂ / ‖x‖₂ ≤ ‖ΔW‖₂ = σ₁(ΔW)   (Spectral Operator Norm)', { mono: true, fs: 13, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function muonSpectralVsFrob() {
  const W = 560, H = 268;
  let o = '';
  o += box(16, 14, 528, 52, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 40, 'SVD of Matrix Update:  ΔW = U Σ Vᵀ = Σ_{i=1}^r σ_i u_i v_iᵀ', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });

  o += card(16, 82, 254, 170, D.blue.dark, C.lblue,
    'Spectral Norm ‖ΔW‖₂',
    'max_{‖x‖₂=1} ‖ΔW x‖₂ = σ₁',
    'Exact max directional stretch',
    'of any unit input vector x');
  o += card(290, 82, 254, 170, D.orange.dark, C.lamber,
    'Frobenius Norm ‖ΔW‖_F',
    '√(σ₁² + σ₂² + ··· + σ_r²)',
    'Sums all r modes in quadrature',
    'Overstates r=128 by √128≈11.3×');
  return svg(W, H, o);
}

function muonSvdEqualize() {
  const W = 560, H = 266;
  let o = '';
  o += box(16, 16, 230, 234, D.red.tint, D.red.base, 8, 2);
  o += tx(131, 38, 'Raw Gradient G = U Σ Vᵀ', { fs: 14, fw: 700, fill: D.red.dark });
  const sBars = [110, 42, 18, 8];
  sBars.forEach((bh, i) => {
    const bx = 44 + i * 46;
    o += box(bx, 186 - bh, 32, bh, '#FFFFFF', D.red.base, 4, 2);
    o += tx(bx + 16, 206, `σ_${i+1}`, { mono: true, fs: 12, fw: 700, fill: C.ink });
  });
  o += tx(131, 232, 'Top mode σ₁ dominates step', { fs: 12, fw: 600, fill: D.red.dark });

  o += arrow(252, 133, 308, 133, C.purple);
  o += tx(280, 112, 'Σ → I_r', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });

  o += box(314, 16, 230, 234, C.lgreen, C.green, 8, 2);
  o += tx(429, 38, 'Spectral Step −η U Vᵀ', { fs: 14, fw: 700, fill: D.green.dark });
  for (let i = 0; i < 4; i++) {
    const bx = 342 + i * 46;
    o += box(bx, 186 - 84, 32, 84, '#FFFFFF', C.green, 4, 2);
    o += tx(bx + 16, 144, '1', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
    o += tx(bx + 16, 206, `σ_${i+1}`, { mono: true, fs: 12, fw: 700, fill: C.ink });
  }
  o += tx(429, 232, 'All r modes step by η!', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function muonRank1Collapse() {
  const W = 560, H = 268;
  let o = '';
  o += box(16, 14, 528, 46, N.panel, C.gray, 8, 2);
  o += tx(280, 37, '2×2 Gradient G = [[3, 1], [1, 1]]  (σ₁ = 3.41, σ₂ = 0.59)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += card(16, 74, 164, 178, D.blue.dark, C.lblue,
    'Euclidean ‖·‖_F',
    '−G / ‖G‖_F',
    'σ = (0.99, 0.17)',
    'Starves 2nd mode');
  o += card(198, 74, 164, 178, D.red.dark, D.red.tint,
    'Entry Box ‖·‖_∞',
    '−sign(G) = −11ᵀ',
    'σ = (2.00, 0.00)',
    'RANK-1 COLLAPSE!');
  o += card(380, 74, 164, 178, D.green.dark, C.lgreen,
    'Spectral ‖·‖₂',
    '−U Vᵀ = −I₂',
    'σ = (1.00, 1.00)',
    'Full step on both!');
  return svg(W, H, o);
}

function muonNsLoop() {
  const W = 560, H = 270;
  let o = '';
  o += box(16, 16, 528, 56, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 36, 'Odd Matrix Polynomial Preserves Singular Vectors U and Vᵀ:', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 56, 'p(X) = a X + b (X Xᵀ) X + c (X Xᵀ)² X  =  U p(Σ) Vᵀ', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += card(16, 88, 160, 166, D.blue.dark, C.lblue,
    '1. Normalize',
    'X₀ = G / ‖G‖_F',
    'σ_i ∈ [0, 1]',
    'No SVD needed');
  o += arrow(178, 171, 198, 171, C.purple);
  o += card(200, 88, 160, 166, D.purple.dark, D.purple.tint,
    '2. 5× Quintic',
    'X_{k+1} = p(X_k)',
    'p′(0) = 3.4445',
    '485× small-σ lift');
  o += arrow(362, 171, 382, 171, C.green);
  o += card(384, 88, 160, 166, D.green.dark, C.lgreen,
    '3. Orthogonal',
    'NS₅(G) ≈ U Vᵀ',
    'σ_i ∈ [0.68, 1.13]',
    '<1% FLOP cost');
  return svg(W, H, o);
}

function muonRmsAlignment() {
  const W = 560, H = 274;
  let o = '';
  o += box(16, 14, 528, 60, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 34, 'Complete Muon Update on m × n Hidden Matrix W:', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 56, 'W_{t+1} = (1 − ηλ) W_t − η · 0.2 √max(m, n) · NS₅(G_t + β M_t)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += card(16, 90, 254, 168, D.blue.dark, C.lblue,
    '1. Nesterov Average (β=0.95)',
    'Average G_t + β M_t first',
    '20-step memory window',
    'Extracts persistent modes');
  o += card(290, 90, 254, 168, D.green.dark, C.lgreen,
    '2. RMS Alignment (0.2√max)',
    'RMS(UVᵀ) = 1 / √max(m, n)',
    'Scales entry RMS to 0.2 η',
    'Shares η & λ with AdamW!');
  return svg(W, H, o);
}

function muonQkClip() {
  const W = 560, H = 266;
  let o = '';
  o += card(16, 16, 254, 234, D.orange.dark, C.lamber,
    '1. Weight Decay (λ > 0)',
    'NS₅ sets singular values ~1',
    'Without (1 − ηλ) shrinkage,',
    '‖W‖_F grows & η/‖W‖_F drops');
  o += card(290, 16, 254, 234, D.purple.dark, D.purple.tint,
    '2. QK-Clip (Kimi K2 1T)',
    'qᵀk = xᵀ W_Qᵀ W_K x is bilinear',
    'Product of two matrices drifts',
    'Rescale W_Q, W_K if logit > 100');
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.7 LEARNING-RATE SCHEDULES (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function schedEarlyLate() {
  const W = 560, H = 268;
  let o = '';
  o += card(16, 14, 254, 146, D.red.dark, D.red.tint,
    'Opening Steps (t = 1..W)',
    'Cold AdamW steps by ±η₀',
    '67% jump at η₀ = 0.03',
    'Before m_t, v_t stabilize!');
  o += card(290, 14, 254, 146, D.orange.dark, C.lamber,
    'Terminal Steps (t → T)',
    'Constant η₀ traps SGD in ball',
    'η σ² / (2 λ b)',
    'AdamW/Muon hop by O(η₀)');

  o += box(16, 176, 528, 76, C.lgreen, C.green, 8, 2);
  o += tx(280, 200, 'Schedule Remedy:  Warmup (0 → η₀) + High Plateau + Decay (η_T → 0)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(280, 228, 'Steers below 2/η_t early, maximizes travel budget B_T, and cools noise to 0', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function schedBudgetVsFloor() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 56, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 34, 'Bias–Variance Decomposition Under a Schedule η_t:', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 55, 'E[x_t²] ≈ exp(−2 λ B_t) x₀²  +  η_t σ² / (2 λ),    B_t = Σ_{s<t} η_s', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += card(16, 86, 254, 170, D.blue.dark, C.lblue,
    'Cumulative Budget B_T',
    'Area under LR curve:',
    'B_T = Σ_{t=0}^{T−1} η_t',
    'Erases start as exp(−2λ B_T)');
  o += card(290, 86, 254, 170, D.green.dark, C.lgreen,
    'Terminal Rate η_T',
    'Final step height at t = T',
    'Noise floor ≈ η_T σ² / (2λ)',
    'Requires slow tail R_s ≳ 1/λ');
  return svg(W, H, o);
}

function schedClassicalFamilies() {
  const W = 560, H = 276;
  let o = '';
  o += card(16, 14, 254, 118, D.blue.dark, C.lblue,
    'Constant (η_t = 0.1)',
    'B_{3000} = 300.0 · η_T = 0.100',
    'Erases x₀ fast',
    'Stalls at noise ball 0.268');
  o += card(290, 14, 254, 118, D.red.dark, D.red.tint,
    'Exponential (0.4 × 0.98ᵗ)',
    'B_∞ = 20.0 · η_T = 1.9×10⁻²⁷',
    'Tail budget starves early',
    'FREEZES at MSE 0.125!');
  o += card(16, 146, 254, 118, D.orange.dark, C.lamber,
    'Polynomial α = 1/2',
    'B_{3000} = 200.2 · η_T = 0.036',
    'η_0 (1 + t/25)⁻⁰·⁵',
    'Tracks noise ball to ≈0.09–0.10');
  o += card(290, 146, 254, 118, D.green.dark, C.lgreen,
    'Polynomial α = 1',
    'B_{3000} = 48.2 · η_T = 0.0033',
    'η_0 (1 + t/25)⁻¹',
    'Tracks noise ball to ≈0.011–0.012');
  return svg(W, H, o);
}

function schedRemainingBudget() {
  const W = 560, H = 274;
  let o = '';
  o += box(16, 14, 528, 60, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 34, 'Why Exponential Decay Freezes: Noise at Step s Needs Future Budget R_s', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 56, 'E[x_T²] ≈ e^{−2λB_T} x₀² + σ² Σ η_s² e^{−2λR_s},  R_s = Σ_{u>s} η_u', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += card(16, 90, 254, 168, D.red.dark, D.red.tint,
    'Geometric Tail Starvation',
    'Remaining R_s ≈ η_s / (1 − γ)',
    'Once η_s < (1−γ)/(2λ) = 0.05',
    '2λ R_s < 1 → contraction stops!');
  o += card(290, 90, 254, 168, D.orange.dark, C.lamber,
    'Exact Frozen Noise Floor',
    'E[x_∞²] ≈ (1 − γ) σ² / (4 λ²)',
    'At γ = 0.98, λ₁ = 0.2:',
    '0.02 / (4 × 0.04) = 0.125!');
  return svg(W, H, o);
}

function schedLateDecayWin() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 164, 148, D.blue.dark, C.lblue,
    'Constant',
    'Budget 100%',
    'B_T = 6.0',
    'Ends ≈1.56 nats');
  o += card(198, 14, 164, 148, D.purple.dark, D.purple.tint,
    'Cosine Decay',
    'Budget 50%',
    'B_T = 3.0',
    'Ends ≈1.56 nats');
  o += card(380, 14, 164, 148, D.green.dark, C.lgreen,
    'Stable–Decay',
    'Budget 90%',
    'B_T = 5.4',
    'Wins: ≈1.51 nats!');

  o += box(16, 178, 528, 78, C.lamber, C.amber, 8, 2);
  o += tx(280, 202, 'Never Rank Schedules Mid-Run Before Cooldown (Step 1,500 vs. 2,000):', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 226, 'At step 1,500 cosine leads (≈1.58 vs ≈1.61); during steps 1,600–2,000', { fs: 12, fw: 600, fill: C.ink });
  o += tx(280, 245, 'late decay sheds 0.10 nats of noise and wins by 0.05 nats!', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function schedWarmupMechanics() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 148, D.red.dark, D.red.tint,
    'Cold Start (η₀ = 0.03)',
    'Step 1 moves weights by ±0.03',
    '0.03 / 0.045 = 67% jump!',
    'Scrambles init → stalls at 2.7');
  o += card(290, 14, 254, 148, D.green.dark, C.lgreen,
    '20-Step Linear Warmup',
    'Step 1 starts at η₀/20 = 0.0015',
    '3.3% relative weight step',
    'Moments fill over τ₁ = 10 steps');

  o += box(16, 178, 528, 78, C.lblue, C.blue, 8, 2);
  o += tx(280, 202, 'Warmup Duration Rule:  W = 1%–5% of T  (always W ≥ 2/(1−β₁) = 20)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 228, 'Triangle budget loss ΔB = ½ W η₀ is < 2.5% of B_T, while steering below 2/η_t', { fs: 12, fw: 600, fill: C.ink });
  o += tx(280, 246, 'Tune peak η₀ with warmup active (optimal η₀* is 3–10× below stall ceiling)', { fs: 11.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function schedCosineVsWsd() {
  const W = 560, H = 270;
  let o = '';
  o += card(16, 14, 254, 240, D.purple.dark, D.purple.tint,
    'Cosine Decay (Fixed T)',
    'Must fix horizon T at step 0',
    '50% travel budget (0.5 η₀ T)',
    'Cannot extend without re-warm');
  o += card(290, 14, 254, 240, D.green.dark, C.lgreen,
    'Warmup–Stable–Decay (WSD)',
    'Horizon-free plateau at η₀',
    '85–90% travel budget',
    'Fork 10–20% cooldown anytime!');
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.8 BATCH SIZE & CRITICAL BATCH SIZE (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function bsTwoRegimes() {
  const W = 560, H = 270;
  let o = '';
  o += card(16, 14, 254, 146, D.green.dark, C.lgreen,
    'Small Batch (b: 4 → 16)',
    'Noise-dominated (b ≪ b_crit)',
    '>3.3× fewer steps',
    '~1.2× data cost (near-linear!)');
  o += card(290, 14, 254, 146, D.orange.dark, C.lamber,
    'Large Batch (b: 64 → 256)',
    'Curvature-dominated (b ≳ b_crit)',
    'Only 1.8× fewer steps',
    '2.2× data & FLOP bill!');

  o += box(16, 176, 528, 78, C.lblue, C.blue, 8, 2);
  o += tx(280, 200, 'The Scaling Knee: Critical Batch Size b_crit ≈ b_noise', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(280, 228, 'Gradient-noise scale:  b_noise = tr(Σ) / ‖∇f‖²  (McCandlish et al.)', { mono: true, fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function bsSnrCrossover() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 56, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 34, 'Minibatch Expected Squared Norm Decomposition:', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 55, 'E[‖ĝ_b‖²] = ‖∇f‖² (Signal Power)  +  tr(Σ) / b (Noise Power)', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += card(16, 86, 164, 170, D.orange.dark, C.lamber,
    'b ≪ b_noise',
    'Noise ≫ Signal',
    'E‖ĝ_b‖² ≈ tr(Σ)/b',
    '1/b variance drop');
  o += card(198, 86, 164, 170, D.green.dark, C.lgreen,
    'b = b_noise',
    'Noise = Signal',
    'tr(Σ) / ‖∇f‖²',
    '50% max progress');
  o += card(380, 86, 164, 170, D.blue.dark, C.lblue,
    'b ≫ b_noise',
    'Signal ≫ Noise',
    'E‖ĝ_b‖² ≈ ‖∇f‖²',
    'Diminishing gain');
  return svg(W, H, o);
}

function bsNoisyDescent() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 56, C.lblue, C.blue, 8, 2);
  o += tx(280, 34, 'Expected L-Smooth Descent Lemma Under Minibatch Noise:', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(280, 55, 'E[f(x − η ĝ_b)] − f(x) ≤ −η ‖∇f‖² + ½ L η² ‖∇f‖² (1 + b_noise / b)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += card(16, 86, 254, 170, D.purple.dark, D.purple.tint,
    'Optimal Step Size η_opt(b)',
    'η_opt(b) = (1/L)/(1 + b_noise/b)',
    'Linear ∝ b for b ≪ b_noise',
    'Saturates at 1/L for b ≫ b_noise');
  o += card(290, 86, 254, 170, D.green.dark, C.lgreen,
    'Guaranteed Drop Δf_opt(b)',
    '[‖∇f‖²/(2L)] / (1 + b_noise/b)',
    'At b = b_noise: exactly 50%',
    'of full-batch single-step drop!');
  return svg(W, H, o);
}

function bsParetoRegimes() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 56, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 34, 'Universal Step–Data Pareto Hyperbola (E_min = b_crit S_min):', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 55, 'S/S_min = 1 + b_crit/b,  E/E_min = 1 + b/b_crit  ⇒  (E/E_min−1)(S/S_min−1) = 1', { mono: true, fs: 11, fw: 700, fill: C.ink });

  o += card(16, 86, 164, 170, D.blue.dark, C.lblue,
    'b = 0.1 b_crit',
    'S = 11.0 S_min',
    'E = 1.10 E_min',
    'Compute optimal');
  o += card(198, 86, 164, 170, D.green.dark, C.lgreen,
    'b = b_crit',
    'S = 2.0 S_min',
    'E = 2.00 E_min',
    'Balanced knee!');
  o += card(380, 86, 164, 170, D.red.dark, D.red.tint,
    'b = 10 b_crit',
    'S = 1.1 S_min',
    'E = 11.0 E_min',
    '11× data waste!');
  return svg(W, H, o);
}

function bsLinearVsSqrt() {
  const W = 560, H = 274;
  let o = '';
  o += card(16, 14, 254, 152, D.blue.dark, C.lblue,
    'SGD / Momentum: Linear (η ∝ b)',
    'k steps at (η₀, b₀) equal',
    '1 step at (k η₀, k b₀)',
    'Preserves η/b noise floor');
  o += card(290, 14, 254, 152, D.purple.dark, D.purple.tint,
    'Adam / AdamW: Sqrt (η ∝ √b)',
    '√v̂_t ≈ σ/√b already scales √b!',
    'η(b) = √(b / b₀) η₀',
    'Matches drift & noise variance');

  o += box(16, 182, 528, 76, C.lamber, C.amber, 8, 2);
  o += tx(280, 206, 'Why Both Rules Saturate Above b_crit and Require Warmup:', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 234, 'For b ≫ b_crit, η_opt saturates at 1/L; at t = 0, b_noise(0) ≈ 2 ≪ b!', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function bsRampPlaybook() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 114, D.blue.dark, C.lblue,
    '1. Hardware Saturation',
    'Raise b until step time',
    'scales linearly with b',
    'No idle GPU tensor cores');
  o += card(290, 14, 254, 114, D.purple.dark, D.purple.tint,
    '2. Post-Warmup b_noise',
    'Measure two-batch estimator',
    'at step 300–500',
    'b_noise grows >30× from t=0');
  o += card(16, 144, 254, 114, D.green.dark, C.lgreen,
    '3. Position b vs. b_crit',
    'Data-bound: b ≲ 0.3 b_crit',
    'Wall-clock: b ≈ b_crit',
    'Or ramp b dynamically');
  o += card(290, 144, 254, 114, D.orange.dark, C.lamber,
    '4. Scale η & Warmup',
    'η ∝ b (SGD) or √b (AdamW)',
    'up to b_crit knee',
    'Warmup bridges b > b_noise(0)');
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.9 WIDTH PARAMETERIZATION & μP (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function mupProxyTransfer() {
  const W = 560, H = 270;
  let o = '';
  o += card(16, 14, 254, 146, D.red.dark, D.red.tint,
    'Standard Parametrization (SP)',
    'Width 128 optimum η* = 3.9e-3',
    'drifts 16× lower at n = 8,192',
    'Transfer loses 0.12 nats!');
  o += card(290, 14, 254, 146, D.green.dark, C.lgreen,
    'Maximal Update (μP)',
    'Scales layer LRs & readout',
    'by width factor m = n / n_base',
    'Zero-shot transfer (≤0.003 nat)!');

  o += box(16, 176, 528, 78, C.lblue, C.blue, 8, 2);
  o += tx(280, 200, 'Tune on a Narrow Proxy, Train Once at Full Scale (Yang et al., 2022)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 228, 'Aligns optimal learning rate η* across a 64× width range (135K → 141M params)', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function mupCoherentSum() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 148, D.blue.dark, C.lblue,
    'Initialization (Incoherent)',
    'W_{ij} ~ N(0, 1/n) ⊥ h_j',
    'Var(z_i) = Σ (1/n) E[h_j²]',
    'Sums in quadrature → Θ(1)');
  o += card(290, 14, 254, 148, D.red.dark, D.red.tint,
    'Adam Step 1 (Aligned Signs)',
    'ΔW = −η sign(δ) sign(h)ᵀ',
    '(ΔW h)_i = −η sign(δ_i) Σ|h_j|',
    'Coherent sum blows up → Θ(η n)!');

  o += box(16, 178, 528, 78, C.lamber, C.amber, 8, 2);
  o += tx(280, 202, 'The Fan-In Blow-Up Law Under Standard Parametrization:', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 226, 'At η = 0.0039, step 1 shifts features by 0.5× at n=128 vs. 32× at n=8,192!', { fs: 12, fw: 600, fill: C.ink });
  o += tx(280, 245, 'Keeping Δz = ΔW h = Θ(1) needs hidden Adam LR η_hidden ∝ 1/n', { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

function mupThreeRules() {
  const W = 560, H = 278;
  let o = '';
  o += box(16, 12, 528, 42, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 33, 'μP Scaling Rules for AdamW Relative to Base Width n_base (m = n / n_base)', { fs: 13.5, fw: 700, fill: D.purple.dark });

  o += card(16, 66, 164, 196, D.blue.dark, C.lblue,
    '1. Input & 1D',
    'Fixed fan-in d_in',
    'Init var: 1 / d_in',
    'LR: η  ·  Mult: 1');
  o += card(198, 66, 164, 196, D.purple.dark, D.purple.tint,
    '2. Hidden (n×n)',
    'Fan-in & out ∝ n',
    'Init var: 1 / n',
    'LR: η / m  ·  Mult: 1');
  o += card(380, 66, 164, 196, D.green.dark, C.lgreen,
    '3. Readout',
    'Fan-in ∝ n',
    'Init: 1 / n_base',
    'LR: η  ·  Mult: 1/m');
  return svg(W, H, o);
}

function mupOutputLayer() {
  const W = 560, H = 270;
  let o = '';
  o += card(16, 14, 254, 148, D.blue.dark, C.lblue,
    '1. Weight Update: (1/m) ΔW h',
    'ΔW_out h sums n aligned terms',
    'Forward logit multiplier 1/m',
    'keeps logit update Θ(1) at LR η');
  o += card(290, 14, 254, 148, D.green.dark, C.lgreen,
    '2. Feature Update: (1/m) W Δh',
    'Aligned Δh requires O(1/n) W',
    'Fixed init var 1/n_base × 1/m',
    'scales effective weights O(1/n)!');

  o += box(16, 178, 528, 76, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 202, 'Base Width Anchor n_base:  At n = n_base (m = 1), μP Equals SP', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 230, 'Fixes constant ratios between input, hidden, and output layer updates', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function mupTransformerRules() {
  const W = 560, H = 276;
  let o = '';
  o += card(16, 14, 254, 116, D.blue.dark, C.lblue,
    'Embeddings & 1D Norms/Biases',
    'One-hot lookup (fan-in = 1)',
    'Init var 1 · Full AdamW LR η');
  o += card(290, 14, 254, 116, D.purple.dark, D.purple.tint,
    'Internal 2D Matrices',
    'qkv, proj, mlp1, mlp2 (∝ d)',
    'Init 1/fan-in · Scaled LR η/m');
  o += card(16, 144, 254, 116, D.green.dark, C.lgreen,
    'Readout Head (d → V)',
    'Fan-in ∝ d, fixed vocab V',
    'Init 1/d_base · Logits ×1/m');
  o += card(290, 144, 254, 116, D.orange.dark, C.lamber,
    'Attention Scores (qᵀk)',
    'Aligned Δq, Δk sum d_k terms',
    'Scale by 1/d_k (not 1/√d_k)!');
  return svg(W, H, o);
}

function mupSpectralCondition() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 60, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 34, 'Universal Spectral Scaling Condition (Yang, Simon & Bernstein, 2023):', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 56, '‖W‖₂ = Θ(√(d_out / d_in))     and     ‖ΔW‖₂ = Θ(√(d_out / d_in))', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  o += card(16, 90, 254, 166, D.red.dark, D.red.tint,
    'Adam Step-1 Outer Product',
    'ΔW = −η sign(δ) sign(h)ᵀ',
    '‖ΔW‖₂ = η ‖s‖₂ ‖t‖₂',
    '= Θ(η √(d_out d_in))');
  o += card(290, 90, 254, 166, D.green.dark, C.lgreen,
    'Matching Θ(√(d_out / d_in))',
    'η √(d_out d_in) = √(d_out/d_in)',
    'Canceling √d_out gives:',
    'η ∝ 1 / d_in  (fan-in law!)');
  return svg(W, H, o);
}

function mupMuonVsAdam() {
  const W = 560, H = 272;
  let o = '';
  o += box(16, 14, 528, 48, C.lblue, C.blue, 8, 2);
  o += tx(280, 38, 'Width Scaling on an n × n Hidden Matrix:  Target ‖ΔW‖₂ = Θ(1)', { fs: 14, fw: 700, fill: D.blue.dark });

  o += card(16, 78, 164, 178, D.purple.dark, D.purple.tint,
    'AdamW Step',
    'Rank-1 sign step',
    '‖ΔW‖₂ = Θ(η n)',
    'Scale LR: η ∝ 1/n');
  o += card(198, 78, 164, 178, D.orange.dark, C.lamber,
    'Muon (0.2√n)',
    'Entry-RMS scale',
    '‖ΔW‖₂ = Θ(η √n)',
    'Scale LR: η ∝ 1/√n');
  o += card(380, 78, 164, 178, D.green.dark, C.lgreen,
    'Muon (Spectral)',
    '√(d_out/d_in) NS₅',
    '‖ΔW‖₂ = Θ(η)',
    'Scale LR: η = Θ(1)!');
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════
// §9.10 PRACTICAL DIAGNOSTICS & RECIPES (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════

function pracThreePillars() {
  const W = 560, H = 274;
  let o = '';
  o += card(16, 14, 164, 152, D.red.dark, D.red.tint,
    '1. Spike Guards',
    'Global clip θ=1.0',
    'β₂ = 0.95 memory',
    'Checkpoint rewind');
  o += card(198, 14, 164, 152, D.blue.dark, C.lblue,
    '2. Weight EMA',
    'x̄_t = αx̄ + (1−α)x',
    'Window 1/(1−α)',
    'Smooths noise ball');
  o += card(380, 14, 164, 152, D.green.dark, C.lgreen,
    '3. Short Proxy',
    '10%-horizon sweep',
    'Calibrate peak η₀',
    'Check b_noise knee');

  o += box(16, 182, 528, 76, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 206, 'Why Operational Guards Matter at Scale (PaLM 540B on 6,144 TPUs):', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 234, 'Suffered ~20 mid-run spikes despite clipping — requiring fast recovery', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function pracHypotheses() {
  const W = 560, H = 268;
  let o = '';
  o += card(16, 14, 254, 238, D.green.dark, C.lgreen,
    'Hypothesis A: Opening Sharpness',
    'η = 1.0 exceeds 2/L only in',
    'opening transient (steps 1–50)',
    'Clips <2% of steps → RESCUED!');
  o += card(290, 14, 254, 238, D.red.dark, D.red.tint,
    'Hypothesis B: Globally Too High',
    'η = 3.0 exceeds 2/L across',
    'entire training trajectory',
    'Clips ~100% of steps → STALLS!');
  return svg(W, H, o);
}

function pracGuardVsBrake() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 254, 148, D.green.dark, C.lgreen,
    'Healthy Guard (η=1.0, θ=1.0)',
    'Clipped steps: < 2%–5%',
    'Median ‖g_t‖₂ ≈ 0.25 ≈ θ/4',
    'Action: Keep θ = 1.0');
  o += card(290, 14, 254, 148, D.red.dark, D.red.tint,
    'Constant Brake (η=3.0, θ=1.0)',
    'Clipped steps: > 50%–100%',
    'Saturates step at length ηθ',
    'Action: Lower η first!');

  o += box(16, 178, 528, 78, C.lamber, C.amber, 8, 2);
  o += tx(280, 202, 'Golden Rule for Global-Norm Gradient Clipping Threshold θ:', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 230, 'When clipping fires on >20%–50% of steps, LOWER η first, then tune θ', { fs: 12.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function pracAdamwVSpike() {
  const W = 560, H = 274;
  let o = '';
  o += box(16, 14, 528, 58, C.lblue, C.blue, 8, 2);
  o += tx(280, 34, 'In AdamW, a Single Spike g_t ≫ √v_{t−1} Has Bounded Step |Δx_t| ≤ 3.16 η', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(280, 55, 'What Clipping Protects Is the Second Moment v_t = β₂ v_{t−1} + (1−β₂) g_t²!', { fs: 12.5, fw: 700, fill: D.red.dark });

  o += card(16, 88, 254, 170, D.red.dark, D.red.tint,
    'Unclipped 100× Spike (β₂=0.999)',
    'Injects 0.001 × 10⁴ = 10× v_t',
    'Effective LR drops to 30%',
    'Paralyzed for ~2,300 steps!');
  o += card(290, 88, 254, 170, D.green.dark, C.lgreen,
    'Clipped (θ=4×) + β₂ = 0.95',
    'Caps v_t injection below 2%',
    'β₂ = 0.95 window τ₂ = 20 steps',
    '20× faster recovery (~120 steps)');
  return svg(W, H, o);
}

function pracSpikeRewind() {
  const W = 560, H = 272;
  let o = '';
  o += card(16, 14, 164, 146, D.red.dark, D.red.tint,
    '1. Detect Spike',
    'Marginal weights',
    '+ outlier batch',
    'Sudden loss jump');
  o += arrow(182, 87, 196, 87, C.purple);
  o += card(198, 14, 164, 146, D.blue.dark, C.lblue,
    '2. Rewind ~100',
    'Reload weights W',
    '+ moments m_t, v_t',
    '+ step counter t');
  o += arrow(364, 87, 378, 87, C.green);
  o += card(380, 14, 164, 146, D.green.dark, C.lgreen,
    '3. Skip Batches',
    'Advance loader',
    '200–500 batches',
    'Past spike window');

  o += box(16, 176, 528, 80, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 200, 'Auxiliary Softmax Logit Stabilization:  z-Loss = 10⁻⁴ log²(Z)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(280, 224, 'Restoring gradient 2×10⁻⁴ (log Z) p pulls log-partition log Z → 0', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(280, 244, 'Prevents logit drift without altering softmax probabilities p', { fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function pracFrontierRecipes() {
  const W = 560, H = 276;
  let o = '';
  o += card(16, 14, 254, 116, D.blue.dark, C.lblue,
    'Llama 3 405B & OLMo 2 7B',
    'AdamW (0.9, 0.95) · Clip θ = 1.0',
    'λ = 0.1 (2D only) · Cosine decay');
  o += card(290, 14, 254, 116, D.purple.dark, D.purple.tint,
    'DeepSeek-V3 671B',
    'AdamW (0.9, 0.95) · Clip θ = 1.0',
    'λ = 0.1 · WSD schedule + ramp');
  o += card(16, 144, 528, 116, D.green.dark, C.lgreen,
    'Kimi K2 1T (MuonClip) & Shared Frontier Consensus',
    'Muon on 2D hidden matrices + AdamW on embeddings/head/1D · WSD cooldown',
    'Defaults: (β₁,β₂)=(0.9,0.95), 2D λ=0.1, clip θ=1.0 + warmup');
  return svg(W, H, o);
}

function pracDiagnosticTree() {
  const W = 560, H = 280;
  let o = '';
  o += card(16, 12, 254, 78, D.red.dark, D.red.tint,
    'Explodes in Steps 1–20',
    'Exceeds 2/L or cold ±η₀ jump',
    '→ Add 1–5% warmup + clip 1.0');
  o += card(290, 12, 254, 78, D.orange.dark, C.lamber,
    'Stalls High / Clips > 20%',
    'Peak rate η₀ globally too high',
    '→ Lower η₀ by 2–3× first');
  o += card(16, 102, 254, 78, D.blue.dark, C.lblue,
    'Plateaus with High Jitter',
    'Trapped in noise ball at η/b',
    '→ Decay η_t→0, EMA, or ↑ b');
  o += card(290, 102, 254, 78, D.purple.dark, D.purple.tint,
    'Constant Slope at Step T',
    'Under-budgeted travel B_T',
    '→ Raise η₀ or extend horizon T');
  o += box(16, 194, 528, 72, C.lgreen, C.green, 8, 2);
  o += tx(280, 218, 'Four Mandatory Telemetry Curves on Every Training Run:', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(280, 244, '1. Loss   2. Pre-clip ‖g_t‖₂   3. Clip (‖g_t‖₂ > θ)   4. Step ‖ΔW‖ / ‖W‖', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

export const diagrams = {
  // §9.0 index.md
  'ch09-opt-two-obstacles':          twoObstacles,
  'ch09-opt-three-levers':           threeLevers,
  'ch09-opt-four-testbeds':          fourTestbeds,
  'ch09-opt-roadmap':                roadmap,
  // §9.1 gd.md
  'ch09-opt-gd-obstacles':           gdObstacles,
  'ch09-opt-gd-eigenmodes':          gdEigenmodes,
  'ch09-opt-gd-saddle-escape':       gdSaddleEscape,
  'ch09-opt-gd-descent-lemma':       gdDescentLemma,
  'ch09-opt-gd-sharpness-filter':    gdSharpnessFilter,
  'ch09-opt-gd-newton-hazards':      gdNewtonHazards,
  // §9.2 sgd.md
  'ch09-opt-sgd-cost-vs-noise':      sgdCostVsNoise,
  'ch09-opt-sgd-signal-plus-noise':  sgdSignalPlusNoise,
  'ch09-opt-sgd-noise-equilibrium':  sgdNoiseEquilibrium,
  'ch09-opt-sgd-three-laws':         sgdThreeLaws,
  'ch09-opt-sgd-robbins-monro':      sgdRobbinsMonro,
  'ch09-opt-sgd-curvature-cancel':   sgdCurvatureCancel,
  // §9.3 test-models.md
  'ch09-opt-test-quad-vs-net':       testQuadVsNet,
  'ch09-opt-test-tinylm-arch':       testTinylmArch,
  'ch09-opt-test-regime-comparison': testRegimeComparison,
  'ch09-opt-test-harness-flow':      testHarnessFlow,
  'ch09-opt-test-loss-landmarks':    testLossLandmarks,
  'ch09-opt-test-lr-diagnosis':      testLrDiagnosis,
  'ch09-opt-test-nuisance-protocol': testNuisanceProtocol,
  'ch09-opt-test-seed-threshold':    testSeedThreshold,
  // §9.4 momentum.md
  'ch09-opt-mom-persistent-vs-alt':  momPersistentVsAlt,
  'ch09-opt-mom-velocity-unroll':    momVelocityUnroll,
  'ch09-opt-mom-dual-filter':        momDualFilter,
  'ch09-opt-mom-noise-impulse':      momNoiseImpulse,
  'ch09-opt-mom-sqrt-kappa':         momSqrtKappa,
  // §9.5 adam.md
  'ch09-opt-adam-layer-scales':         adamLayerScales,
  'ch09-opt-adam-cold-start-jump':      adamColdStartJump,
  'ch09-opt-adam-dual-moments':         adamDualMoments,
  'ch09-opt-adam-sign-and-eps':         adamSignAndEps,
  'ch09-opt-adam-coupled-vs-decoupled': adamCoupledVsDecoupled,
  'ch09-opt-adam-decay-timescale':      adamDecayTimescale,
  'ch09-opt-adam-rotational-eq':        adamRotationalEq,
  // §9.6 muon.md
  'ch09-opt-muon-operator-view':     muonOperatorView,
  'ch09-opt-muon-spectral-vs-frob':  muonSpectralVsFrob,
  'ch09-opt-muon-svd-equalize':      muonSvdEqualize,
  'ch09-opt-muon-rank1-collapse':    muonRank1Collapse,
  'ch09-opt-muon-ns-loop':           muonNsLoop,
  'ch09-opt-muon-rms-alignment':     muonRmsAlignment,
  'ch09-opt-muon-qk-clip':           muonQkClip,
  // §9.7 lr-scheduler.md
  'ch09-opt-sched-early-late':         schedEarlyLate,
  'ch09-opt-sched-budget-vs-floor':    schedBudgetVsFloor,
  'ch09-opt-sched-classical-families': schedClassicalFamilies,
  'ch09-opt-sched-remaining-budget':   schedRemainingBudget,
  'ch09-opt-sched-late-decay-win':     schedLateDecayWin,
  'ch09-opt-sched-warmup-mechanics':   schedWarmupMechanics,
  'ch09-opt-sched-cosine-vs-wsd':      schedCosineVsWsd,
  // §9.8 batch-size.md
  'ch09-opt-bs-two-regimes':         bsTwoRegimes,
  'ch09-opt-bs-snr-crossover':       bsSnrCrossover,
  'ch09-opt-bs-noisy-descent':       bsNoisyDescent,
  'ch09-opt-bs-pareto-regimes':      bsParetoRegimes,
  'ch09-opt-bs-linear-vs-sqrt':      bsLinearVsSqrt,
  'ch09-opt-bs-ramp-playbook':       bsRampPlaybook,
  // §9.9 scaling.md
  'ch09-opt-mup-proxy-transfer':     mupProxyTransfer,
  'ch09-opt-mup-coherent-sum':       mupCoherentSum,
  'ch09-opt-mup-three-rules':        mupThreeRules,
  'ch09-opt-mup-output-layer':       mupOutputLayer,
  'ch09-opt-mup-transformer-rules':  mupTransformerRules,
  'ch09-opt-mup-spectral-condition': mupSpectralCondition,
  'ch09-opt-mup-muon-vs-adam':       mupMuonVsAdam,
  // §9.10 practice.md
  'ch09-opt-prac-three-pillars':     pracThreePillars,
  'ch09-opt-prac-hypotheses':        pracHypotheses,
  'ch09-opt-prac-guard-vs-brake':    pracGuardVsBrake,
  'ch09-opt-prac-adamw-v-spike':     pracAdamwVSpike,
  'ch09-opt-prac-spike-rewind':      pracSpikeRewind,
  'ch09-opt-prac-frontier-recipes':  pracFrontierRecipes,
  'ch09-opt-prac-diagnostic-tree':   pracDiagnosticTree,
};
