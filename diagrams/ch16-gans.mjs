// Slide diagrams for Chapter 16 (Generative Adversarial Networks)
import { C, TOKENS, tx, arrow, svg } from './engine.mjs';
import { sec, num } from './sections.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Helper: clean 3-card vertical stack
function stack3(title, c1, c2, c3, footer) {
  const W = 560, H = 290;
  let o = '';
  o += tx(280, 18, title, { fs: 15, fw: 700, fill: C.ink });
  const cards = [c1, c2, c3];
  const ys = [34, 110, 186];
  cards.forEach((c, i) => {
    const y = ys[i];
    o += box(14, y, 532, 62, c.fill, c.stroke, 8, 2);
    o += box(22, y + 10, 126, 42, '#FFFFFF', c.stroke, 6, 1.5);
    o += tx(85, y + 31, c.badge, { mono: !!c.monoBadge, fs: 12.5, fw: 700, fill: c.dark });
    o += tx(160, y + 22, c.head, { fs: 13.5, fw: 700, fill: C.ink, anchor: 'start' });
    o += tx(160, y + 44, c.sub, { mono: !!c.monoSub, fs: 12, fw: 600, fill: C.muted, anchor: 'start' });
  });
  if (footer) {
    o += box(14, 254, 532, 26, N.panel, C.gray, 6, 1.5);
    o += tx(280, 267, footer, { fs: 12, fw: 700, fill: C.ink });
  }
  return svg(W, H, o);
}

// Helper: 2-column comparison layout with top title and bottom synthesis bar
function compare2(title, left, right, bottom) {
  const W = 560, H = 290;
  let o = '';
  o += tx(280, 18, title, { fs: 15, fw: 700, fill: C.ink });
  const cols = [
    { x: 14, ...left },
    { x: 288, ...right },
  ];
  cols.forEach((c) => {
    o += box(c.x, 36, 258, 198, c.fill, c.stroke, 10, 2);
    o += tx(c.x + 129, 58, c.head, { fs: 14, fw: 700, fill: c.dark });
    o += tx(c.x + 129, 80, c.sub, { fs: 12, fw: 600, fill: C.muted });
    c.rows.forEach((r, idx) => {
      const ry = 98 + idx * 42;
      o += box(c.x + 10, ry, 238, 34, '#FFFFFF', c.stroke, 6, 1.5);
      o += tx(c.x + 129, ry + 17, r, { fs: 12, fw: 700, fill: C.ink });
    });
  });
  if (bottom) {
    o += box(14, 244, 532, 34, bottom.fill || D.purple.tint, bottom.stroke || C.purple, 8, 2);
    o += tx(280, 261, bottom.text, { fs: 12.5, fw: 700, fill: bottom.dark || D.purple.dark });
  }
  return svg(W, H, o);
}

// Helper: 3-stage horizontal pipeline with bottom note bar
function pipeline3(title, s1, s2, s3, bottom) {
  const W = 560, H = 276;
  let o = '';
  o += tx(280, 18, title, { fs: 15, fw: 700, fill: C.ink });
  const xs = [14, 201, 388], bw = 158, by = 40, bh = 166;
  const stages = [s1, s2, s3];
  stages.forEach((s, i) => {
    const x = xs[i];
    o += box(x, by, bw, bh, s.fill, s.stroke, 9, 2);
    o += tx(x + bw / 2, by + 22, s.tag, { fs: 12, fw: 700, fill: s.dark });
    o += tx(x + bw / 2, by + 44, s.head, { fs: 13.5, fw: 700, fill: C.ink });
    o += box(x + 8, by + 62, bw - 16, 42, '#FFFFFF', s.stroke, 6, 1.5);
    o += tx(x + bw / 2, by + 83, s.eq, { mono: true, fs: 11.5, fw: 700, fill: s.dark });
    o += tx(x + bw / 2, by + 124, s.note1, { fs: 12, fw: 600, fill: C.ink });
    o += tx(x + bw / 2, by + 146, s.note2, { fs: 11.5, fw: 600, fill: C.muted });
  });
  o += arrow(xs[0] + bw + 3, by + bh / 2, xs[1] - 3, by + bh / 2, C.blue);
  o += arrow(xs[1] + bw + 3, by + bh / 2, xs[2] - 3, by + bh / 2, C.purple);
  if (bottom) {
    o += box(14, 222, 532, 40, bottom.fill || C.lgreen, bottom.stroke || C.green, 8, 2);
    o += tx(280, 242, bottom.text, { fs: 12.5, fw: 700, fill: bottom.dark || D.green.dark });
  }
  return svg(W, H, o);
}

// Helper: 2x2 matrix grid
function matrix2x2(title, tl, tr, bl, br, footer) {
  const W = 560, H = 290;
  let o = '';
  o += tx(280, 18, title, { fs: 15, fw: 700, fill: C.ink });
  const cells = [
    { x: 14, y: 36, ...tl },
    { x: 288, y: 36, ...tr },
    { x: 14, y: 144, ...bl },
    { x: 288, y: 144, ...br },
  ];
  cells.forEach((c) => {
    o += box(c.x, c.y, 258, 96, c.fill, c.stroke, 8, 2);
    o += tx(c.x + 129, c.y + 20, c.head, { fs: 13.5, fw: 700, fill: c.dark });
    o += box(c.x + 10, c.y + 34, 238, 30, '#FFFFFF', c.stroke, 6, 1.5);
    o += tx(c.x + 129, c.y + 49, c.eq, { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    o += tx(c.x + 129, c.y + 80, c.sub, { fs: 12, fw: 600, fill: C.muted });
  });
  if (footer) {
    o += box(14, 250, 532, 28, N.panel, C.gray, 6, 1.5);
    o += tx(280, 264, footer, { fs: 12, fw: 700, fill: C.ink });
  }
  return svg(W, H, o);
}

export const diagrams = {
  // ── index.md (3 diagrams) ───────────────────────────────────────────
  'ch16-gan-three-axes': () => stack3(
    'Three Design Axes of Adversarial Formulations',
    {
      badge: 'Axis 1: Target',
      head: 'What the Critic Estimates',
      sub: 'Log-ratio λ(x)=log(p/q) vs. link D*(λ) or witness T(x)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Axis 2: Arity',
      head: 'How Samples Are Scored',
      sub: 'Single-sample D(x) ≷ 0 vs. pairwise σ(D(x) − D(x\'))',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'Axis 3: Penalty',
      head: 'How Critic Smoothness Is Enforced',
      sub: 'Unconstrained vs. 1-Lipschitz (W₁) or R₁+R₂ = γE[‖∇D‖²]',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'R3GAN Recipe: Pairwise RpGAN + Zero-Centered R₁ + R₂ Penalties'
  ),

  'ch16-gan-roadmap': () => pipeline3(
    `Chapter ${num('chap_gans')} Progression: Divergences → Dynamics → Synthesis`,
    {
      tag: `Part I · §§${num('sec_basic_gan')}–${num('sec_gan_relativistic')}`,
      head: 'Objectives',
      eq: 'JS, f-GAN, W₁',
      note1: 'Log-ratio & links',
      note2: 'Pairwise ranking',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: `Part II · ${sec('sec_gan_convergence')}`,
      head: 'Game Dynamics',
      eq: 'R₁ + R₂ Damping',
      note1: 'Dirac-GAN orbits',
      note2: 'Re(ν) = −γ/2 < 0',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      tag: `Part III · §§${num('sec_dcgan')}–${num('sec_gan_beyond')}`,
      head: 'Images & Critics',
      eq: 'DCGAN, cGAN, ADD',
      note1: 'FID / KID / P&R',
      note2: 'Projection & Patch',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Unified Thread: Replace Intractable Likelihoods with a Learned Critic D' }
  ),

  'ch16-gan-notation-map': () => matrix2x2(
    `Core Mathematical Quantities Across Chapter ${num('chap_gans')}`,
    {
      head: 'Densities & Log-Ratio',
      eq: 'ρ = p/q,  λ(x) = log(p/q)',
      sub: 'Equilibrium q = p  ⇒  ρ = 1, λ = 0',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Bayes Posterior & Critic',
      eq: 'η(x) = p/(p+q) = σ(λ(x))',
      sub: 'Equilibrium η(x) = 1/2, D*(x) = 0',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: 'Generator Gradient Weights',
      eq: 'w_sat = σ(D),  w_ns = σ(−D)',
      sub: 'w_Rp = E_p[σ(D(x\') − D(x))] = 1/2 at q = p',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'Zero-Centered Penalties',
      eq: 'R₁+R₂ = ½γ(E_p‖∇D‖² + E_q‖∇D‖²)',
      sub: 'Vanishes at q = p where ∇_x D* = 0',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Every formulation modifies the payoff, pairing arity, or gradient penalty on D'
  ),

  // ── gan.md (7 diagrams) ─────────────────────────────────────────────
  'ch16-gan-bce-game': () => compare2(
    'Two-Player Minimax Value V(D) on Real and Fake Draws',
    {
      head: 'Discriminator (max_{D} V)',
      sub: 'Classifies source label y ∈ {1, 0}',
      rows: [
        'Real x ~ p: push D(x) > 0',
        'Fake x\' ~ q: push D(x\') < 0',
        'Baseline D=0: V(0) = −2 log 2',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Generator (min_{G} max_{D} V)',
      sub: 'Moves x\' = G(z) uphill on D',
      rows: [
        'Minimax: E_z[log σ(−D(G(z)))]',
        'Non-sat: −E_z[log σ(D(G(z)))]',
        'Pushes q(x) → p(x) everywhere',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Equilibrium (q = p): D*(x) = 0,  σ(D*) = 1/2,  −V(D*) = 2 log 2 ≈ 1.386' }
  ),

  'ch16-gan-pointwise-opt': () => pipeline3(
    'Pointwise Derivation of the Optimal Log-Loss Critic D*(x)',
    {
      tag: 'Step 1 · Integrand',
      head: 'Decouple at x',
      eq: 'p·ln u + q·ln(1−u)',
      note1: 'Let u = σ(D(x)) ∈ (0,1)',
      note2: 'Concave in u at each x',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Step 2 · Stationarity',
      head: 'Set ∂/∂u = 0',
      eq: 'p/u = q/(1 − u)',
      note1: 'u* = p(x) / (p(x) + q(x))',
      note2: 'Posterior P(y = 1 | x)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Step 3 · Logit',
      head: 'Invert Sigmoid',
      eq: 'D*(x) = log(p / q)',
      note1: 'Log density ratio λ(x)',
      note2: 'D*(x) = 0  ⇔  p = q',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Log loss pins the zero threshold: D*(x) > 0 where p(x) > q(x)' }
  ),

  'ch16-gan-gaussian-ratio': () => compare2(
    'Optimal Critic Complexity: Equal vs. Unequal Covariances',
    {
      head: 'Equal Variance (Σ_p = Σ_q = I)',
      sub: 'p = N(0, 1)  vs.  q = N(μ, 1)',
      rows: [
        '−x²/2 terms cancel exactly!',
        'D*(x) = μ(μ/2 − x)  (Affine)',
        'Sign flips at midpoint x = μ/2',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Unequal Covariance (Σ_p ≠ Σ_q)',
      sub: 'p = N(μ_p, Σ_p) vs. q = N(μ_q, Σ_q)',
      rows: [
        'Quadratic −½ xᵀ(Σ_p⁻¹ − Σ_q⁻¹)x',
        'Curved elliptical boundaries',
        'Requires nonlinear MLP critic',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'Critic architecture must be expressive enough to represent log(p(x)/q(x))' }
  ),

  'ch16-gan-js-mixture': () => pipeline3(
    'Substituting D*(x) Recovers Jensen–Shannon Divergence',
    {
      tag: 'Mixture m(x)',
      head: 'Pooled Density',
      eq: 'm = ½(p + q)',
      note1: 'σ(D*) = p / (2m)',
      note2: 'σ(−D*) = q / (2m)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Two KL Terms',
      head: 'Distance to m',
      eq: 'KL(p‖m) + KL(q‖m)',
      note1: 'Bounded: p/(2m) ≤ 1',
      note2: 'Symmetric in (p, q)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Minimax Value',
      head: 'Global Minimum',
      eq: '2 JS(p,q) − 2log2',
      note1: 'JS(p,q) ≥ 0',
      note2: 'JS = 0  ⇔  q = p',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Minimizing max_{D} V(D) uniquely matches q = p at value −2 log 2' }
  ),

  'ch16-gan-js-saturation': () => {
    const W = 560, H = 280;
    let o = '';
    o += tx(280, 18, 'JS(p, q) = I(x; y) Saturates at log 2 ≈ 0.6931 Nats', { fs: 15, fw: 700, fill: C.ink });
    const bx = 56, by = 48, bw = 448, bh = 152;
    o += box(bx, by, bw, bh, '#FFFFFF', C.gray, 6, 1.5);
    const ceilY = by + 24;
    o += line(bx + 8, ceilY, bx + bw - 8, ceilY, D.red.base, 2, true);
    o += tx(bx + 16, ceilY - 10, 'Ceiling: log 2 ≈ 0.6931 nats (zero slope!)', { mono: true, fs: 12, fw: 700, fill: D.red.dark, anchor: 'start' });

    const bars = [
      { mu: 'μ=1', val: '0.111', h: 20, col: C.lblue, st: C.blue },
      { mu: 'μ=2', val: '0.337', h: 61, col: C.lblue, st: C.blue },
      { mu: 'μ=4', val: '0.633', h: 114, col: C.lamber, st: C.amber },
      { mu: 'μ=8', val: '0.693', h: 125, col: D.red.tint, st: D.red.base },
      { mu: 'Disjoint', val: 'log 2', h: 125, col: D.red.tint, st: D.red.base },
    ];
    bars.forEach((b, i) => {
      const x = bx + 32 + i * 82;
      const y = by + bh - b.h;
      o += box(x, y, 56, b.h, b.col, b.st, 4, 2);
      o += tx(x + 28, y + b.h / 2, b.val, { mono: true, fs: 12, fw: 700, fill: C.ink });
      o += tx(x + 28, by + bh + 18, b.mu, { mono: true, fs: 12.5, fw: 700, fill: C.ink });
    });
    o += box(14, 230, 532, 38, D.red.tint, D.red.base, 8, 2);
    o += tx(280, 249, 'When supports separate, I(x; y) = log 2 is constant → ∂_{θ} JS(p, q_θ) = 0', { fs: 12.5, fw: 700, fill: D.red.dark });
    return svg(W, H, o);
  },

  'ch16-gan-soft-rev-kl': () => compare2(
    'Non-Saturating Generator Descends Softened Reverse KL 2 KL(m ‖ p)',
    {
      head: 'Spurious Fake Samples',
      sub: 'q(x) > 0  where  p(x) = 0',
      rows: [
        'Mixture m(x) = q(x)/2 > 0',
        'Integrand: m(x) log(m/0) = +∞',
        'Infinite penalty kills artifacts!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Dropped Real Data Modes',
      sub: 'p(x) > 0  where  q(x) = 0',
      rows: [
        'Mixture m(x) = p(x)/2',
        'Integrand: (p/2) log(1/2), finite',
        'Total 2 KL(m ‖ p) ≈ 0.26 nats',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: '2 KL(m ‖ p) heavily punishes bad samples, lightly punishes missing modes' }
  ),

  'ch16-gan-mode-drop': () => stack3(
    'Penalty for Dropping Half the Dataset (1 of 2 Equal Modes)',
    {
      badge: 'Forward KL',
      head: 'Maximum Likelihood: KL(p ‖ q) = ∞',
      sub: 'Zero density q(x) = 0 on a real mode incurs infinite penalty',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      badge: 'Reverse KL',
      head: 'Mode-Seeking: KL(q ‖ p) = log 2 ≈ 0.693 nats',
      sub: 'Doubling density on one mode (q = 2p) costs only log 2 nats',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      badge: 'GAN (NS / JS)',
      head: '2 KL(m ‖ p) ≈ 0.262 nats  ·  JS(p, q) ≈ 0.216 nats',
      sub: 'Costs ~1/4 nat; on covered mode D*(x) = −log 2 has ∇D* = 0',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    'Zero local restoring force: constant D*(x) = −log 2 gives no pull to missing mode'
  ),

  // ── objectives.md (7 diagrams) ──────────────────────────────────────
  'ch16-gan-three-stall': () => stack3(
    'Three Distinct Mechanisms That Can Stall a Distant Generator',
    {
      badge: '1. Divergence',
      head: 'Population Divergence Saturation',
      sub: 'JS(p, q) flattens at log 2 when supports separate (μ ≫ 1)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: '2. Critic Class',
      head: 'Critic Function-Class Geometry T',
      sub: 'Does D ∈ T jump abruptly or interpolate across the gap?',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: '3. G Weighting',
      head: 'Generator Loss Rescaling Weight w(x\')',
      sub: 'Does G keep unit weight (w_ns ≈ 1) or vanish as e^D → 0?',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'A distant generator moves iff ∇_x D(x\') ≠ 0 and the generator weight is nonzero'
  ),

  'ch16-gan-general-sup': () => matrix2x2(
    'Variational Template: d(p, q) = sup_{T ∈ T} { E_p[ℓ_p(T)] + E_q[ℓ_q(T)] }',
    {
      head: 'Single Sample · Unconstrained T',
      eq: 'Logistic, LSGAN, Hinge, f-GAN',
      sub: 'Recovers f-divergences D_f(p ‖ q)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Single Sample · Constrained T',
      eq: 'ℓ_p(t)=t, ℓ_q(t)=−t on F',
      sub: 'IPMs: RKHS ball (MMD), 1-Lip (W₁)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      head: 'Ordered Pair · Unconstrained T',
      eq: 'E_{p,q}[log σ(D(x) − D(x\'))]',
      sub: `RpGAN (${sec('sec_gan_relativistic')}) → JS(p⊗q, q⊗p)`,
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: 'Ordered Pair · Penalized T',
      eq: 'RpGAN − (γ/2)(R₁ + R₂)',
      sub: `R3GAN (${sec('sec_gan_convergence')}) → Dual Sobolev filter`,
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'Always convex in probability measure q (sup of linear expectations), nonconvex in θ'
  ),

  'ch16-gan-bayes-risk-gap': () => stack3(
    'Two-Outcome Bayes-Risk Gap Δ_ℓ(p, q) = L(1/2) − E_m[L(η(x))]',
    {
      badge: 'Logistic Loss',
      head: 'L(1/2) = 0.693  →  E_m[L(η)] = 0.562',
      sub: 'Gap Δ = 0.131 nats = Jensen–Shannon divergence JS(p, q)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Squared (LSGAN)',
      head: 'L(1/2) = 1/4 = 0.250  →  E_m[L(η)] = 3/16 = 0.1875',
      sub: 'Gap Δ = 1/16 = 0.0625 = ¼ Vincze–Le Cam χ²(p ‖ m)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'Hinge Loss',
      head: 'L(1/2) = 1.000  →  E_m[L(η)] = 1/2 = 0.500',
      sub: 'Gap Δ = 1/2 = 0.500 = Total Variation distance TV(p, q)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'All three losses score the same posterior η = p/(p+q) = (3/4, 1/4)'
  ),

  'ch16-gan-f-divergence': () => pipeline3(
    'Every Symmetric Margin Loss Defines an f-Divergence',
    {
      tag: '1. Pointwise Inf',
      head: 'Bayes Risk L(η)',
      eq: 'inf_{t} E[loss | η]',
      note1: 'Inf of affine funcs of η',
      note2: 'Concave on [0,1]',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Density Ratio',
      head: 'Change to u = p/q',
      eq: 'η = u / (u + 1)',
      note1: 'Optimal link D*(λ)',
      note2: 'Monotone in λ = log u',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Generator f(u)',
      head: 'f-Divergence Form',
      eq: 'L(½) − ½(u+1)L(η)',
      note1: 'Convex in u, f(1) = 0',
      note2: 'Δ_ℓ(p,q) = E_q[f(p/q)]',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Links: Logistic D*=λ  ·  LSGAN D*=σ(λ)  ·  Sq.Hinge D*=tanh(λ/2)  ·  Hinge D*=sign(λ)' }
  ),

  'ch16-gan-fenchel-dual': () => pipeline3(
    'Fenchel–Young Conjugate Dual for General f-Divergences (f-GAN)',
    {
      tag: 'Convex Conjugate',
      head: 'Fenchel Inequality',
      eq: 'f(u) ≥ u·t − f*(t)',
      note1: 'f*(t) = sup_{u} {u·t − f(u)}',
      note2: 'Tight at t* = f\'(u)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Substitute u = p/q',
      head: 'Expectation Under q',
      eq: 'E_p[T] − E_q[f*(T)]',
      note1: 'Uses E_q[(p/q)T] = E_p[T]',
      note2: 'Optimal critic T* = f\'(p/q)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Forward KL Case',
      head: 'f(u) = u log u',
      eq: 'E_p[T] − E_q[e^T/e]',
      note1: 'f*(t) = exp(t − 1)',
      note2: 'T*(x) = 1 + log(p/q)',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'Asymmetric payoffs ℓ_p ≠ ℓ_q reach unbounded divergences like KL(p ‖ q)' }
  ),

  'ch16-gan-w1-lipschitz': () => stack3(
    'Kantorovich–Rubinstein W₁ Duality and Lipschitz Critics',
    {
      badge: 'Weight Clipping',
      head: 'WGAN (2017): Clamp w ∈ [−c, c] After Each Update',
      sub: 'Enforces κ-Lipschitzness, but restricts weights to ±c',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      badge: 'WGAN-GP',
      head: 'Gradient Penalty (2017): γ E[(‖∇_x T(x̃)‖ − 1)²]',
      sub: 'Soft unit-slope constraint; 2-sided GP creates barrier at q = p',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      badge: 'Spectral Norm',
      head: 'SN-GAN (2018): Normalize W ← W / σ_max(W)',
      sub: 'Bounds each layer Lipschitz constant ≤ 1 via power iteration',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'W₁(p, q) = sup { E_p[T] − E_q[T] : ‖T‖_{Lip} ≤ 1 } keeps unit slope at all distances'
  ),

  'ch16-gan-sample-complexity': () => stack3(
    'Finite-Sample Minibatch Scaling in High Dimension d_x',
    {
      badge: 'Kernel MMD²',
      head: 'Parametric Rate O(n⁻¹/²) in Any Dimension d_x',
      sub: 'Unbiased U-statistic over pairwise kernel evaluations k(x, x\')',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      badge: 'Empirical W₁',
      head: 'Curse of Dimensionality O(n⁻¹/ᵈˣ) for d_x > 2',
      sub: 'Halving OT error takes 2ᵈˣ samples; WGAN uses smooth net',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      badge: 'Forward-KL f-GAN',
      head: 'Exponential Term exp(T*(x\') − 1) = p(x\')/q(x\') Explodes',
      sub: 'Variance under q is χ²(p ‖ q), which can be ∞ even when KL < ∞',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    'Smooth parametric critic architectures act as low-variance witnesses in high dimensions'
  ),

  // ── relativistic.md (7 diagrams) ────────────────────────────────────
  'ch16-gan-shift-invariance': () => compare2(
    'Single-Sample Zero Threshold vs. Shift-Invariant Pairwise Ranking',
    {
      head: 'Classic Log-Loss V(D)',
      sub: 'Compares D(x) against fixed 0',
      rows: [
        'V(D + b) ≠ V(D) for b ≠ 0',
        'Pins zero level: D*(x) = λ(x)',
        'Global drift b flips all signs!',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'Relativistic Pairing Φ(D)',
      sub: 'Scores difference D(x) − D(x\')',
      rows: [
        'Φ(D + b) = Φ(D) for all b ∈ ℝ',
        'Optimal ranker: D*(x) = λ(x) + c',
        'Diagnostic: P(D(x) > D(x\'))',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    { text: 'Functional derivative δΦ/δD(t) = 0 is solved uniquely by D*(x) = log(p(x)/q(x)) + c' }
  ),

  'ch16-gan-lifting-product': () => pipeline3(
    'The Lifting Lemma: Product-Space Game V_{P,Q}(D) = 2 Φ(D)',
    {
      tag: '1. Product Space',
      head: 'Ordered Pairs (a₁, a₂)',
      eq: 'P = p⊗q,  Q = q⊗p',
      note1: 'P: real sample first',
      note2: 'Q: real sample second',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Additive Ratio',
      head: 'Factors Separate',
      eq: 'log(dP/dQ) = λ₁ − λ₂',
      note1: 'λ(a₁) − λ(a₂) exact!',
      note2: 'Separates additively',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Zero Gap',
      head: 'Difference Critics',
      eq: 'D(a₁) − D(a₂)',
      note1: 'Contains D* = λ₁ − λ₂',
      note2: 'sup V_{P,Q} = 2 sup Φ',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Restricting pair critics to D(a₁) − D(a₂) loses zero statistical power' }
  ),

  'ch16-gan-drp-entropy': () => stack3(
    'Calibrated Pairing Divergence d_Rp(p, q) = sup_{D} Φ(D) + log 2',
    {
      badge: 'Product JS',
      head: 'd_Rp(p, q) = JS(p ⊗ q,  q ⊗ p)',
      sub: 'Equals H(½(p⊗q + q⊗p)) − H(p) − H(q) ≥ 0, vanishing iff q = p',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Explicit Form',
      head: 'log 2 − E_{x~p, x\'~q}[ log(1 + (q(x)p(x\')) / (p(x)q(x\'))) ]',
      sub: 'Evaluated at optimal difference D*(x) − D*(x\') = λ(x) − λ(x\')',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'Pair Info',
      head: 'd_Rp(p, q) = I((a₁, a₂); y) = log 2 − H(y | a₁, a₂)',
      sub: 'Mutual information (nats) that an ordered pair carries about y',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'At q = p: Φ(D*) = −log 2 ≈ −0.6931,  d_Rp(p, p) = 0,  P(D(x) > D(x\')) = 1/2'
  ),

  'ch16-gan-mi-sandwich': () => stack3(
    'Information Sandwich: d_Rp(p, q) = 2 JS(p, q) − I(a₁; a₂)',
    {
      badge: 'Near Equilibrium',
      head: 'q → p:  d_Rp(p, q) ≈ 2 JS(p, q)  (Ratio → 2)',
      sub: 'Marginal JS = Θ(‖p−q‖²) while redundancy I(a₁; a₂) = Θ(‖p−q‖⁴)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      badge: 'Overlap Regime',
      head: 'Sandwich: JS(p, q) ≤ d_Rp(p, q) ≤ min(2 JS(p, q), log 2)',
      sub: 'Second pair member adds I(a₂; y | a₁) = JS(p, q) − I(a₁; a₂) ≥ 0',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Disjoint Supports',
      head: 'p ⊥ q:  d_Rp(p, q) = JS(p, q) = log 2  (Ratio → 1)',
      sub: 'First sample a₁ determines y (I(a₁; a₂) = log 2); still saturates!',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'Chain rule of mutual information explains both the 2× local gain and log 2 ceiling'
  ),

  'ch16-gan-rank-weight': () => compare2(
    'Relativistic Rank Weight in Generator Updates',
    {
      head: 'Saturating Pairing (min_{G} Φ)',
      sub: 'Weight w_Rp(x\') = E_p[σ(D(x\')−D(x))]',
      rows: [
        'Win prob of x\' against real x~p',
        'When D(x\') ≪ D(x): w_Rp → 0',
        'Stalls when initialized far away!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Non-Saturating (rpgan_loss_G)',
      sub: 'Weight 1 − w_Rp = E_p[σ(D(x)−D(x\'))]',
      rows: [
        'Loss prob of x\' against real x~p',
        'When D(x\') ≪ D(x): 1 − w_Rp → 1',
        'Preserves full gradient far away!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Shift-invariant weighting: compares D(x\') against live real scores D(x), not fixed 0' }
  ),

  'ch16-gan-collision-graphs': () => compare2(
    'Directed Pairing Graphs on Mode Collisions (Sun et al., 2020)',
    {
      head: 'All-Pairs E_all & Classic E_GAN',
      sub: 'Collision x\'₁ = x\'₂ = x₁ (drops x₂)',
      rows: [
        'Every perturbation toward x₂',
        'strictly INCREASES best-response E',
        'n^n − n! strict collision minima!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Fixed 1-to-1 Coupling E_diag',
      sub: 'Permanent pairs x₁↔x\'₁, x₂↔x\'₂',
      rows: [
        'Cycle 1→1 tied at −½ log 2',
        'Acyclic edge 2→x\'₂ stays at 0',
        'Flat escape plateau: 0 local minima!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Cycle rule: differences cancel on cycles (−log 2) and diverge on acyclic edges (0)' }
  ),

  'ch16-gan-minibatch-repair': () => pipeline3(
    'Why Minibatch Training Evaluates All-Pairs E_all, Not E_diag',
    {
      tag: 'Step k · Draw',
      head: 'Random Batches',
      eq: 'X ~ p^B,  X\' ~ q^B',
      note1: 'No persistent ID i ↔ i',
      note2: 'Fresh random draws',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Step k · Pair',
      head: 'Random Index Match',
      eq: '(X_i, X\'_i) ~ p⊗q',
      note1: 'Re-paired every step',
      note2: 'All n² pairs over time',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Expected Loss',
      head: 'All-Pairs Landscape',
      eq: 'E_all = d_Rp − log2',
      note1: 'Retains n^n − n! minima',
      note2: 'Needs R₁ + R₂ penalties!',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: `In ${sec('sec_gan_convergence')}, our 2×2 experiment on 25 Gaussians proves R₁+R₂ drives mode coverage` }
  ),

  // ── convergence.md (9 diagrams) ─────────────────────────────────────
  'ch16-gan-equilibrium-spiral': () => compare2(
    'Static Population Equilibrium vs. Minimax Gradient Dynamics',
    {
      head: 'Single-Objective Minimization',
      sub: 'Potential flow θ̇ = −∇_θ L(θ)',
      rows: [
        'Symmetric Hessian ∇²L(θ*)',
        'Real eigenvalues ν_i < 0',
        'Descends monotonically to θ*',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      head: 'Two-Player Minimax Game',
      sub: 'θ̇ = −∂_{θ} V(θ,ψ),  ψ̇ = +∂_{ψ} V(θ,ψ)',
      rows: [
        'Antisymmetric off-diagonal blocks',
        'Imaginary eigenvalues ν = ±i ω',
        'Orbits or spirals outward!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    { text: 'On the 2-parameter Dirac-GAN, 300 Euler steps double the distance to (0, 0)!' }
  ),

  'ch16-gan-dirac-setup': () => pipeline3(
    'The Dirac-GAN: Minimal Two-Parameter Minimax System',
    {
      tag: 'Real Data p',
      head: 'Point Mass at 0',
      eq: 'p = δ₀  (x = 0)',
      note1: 'D_ψ(0) = ψ · 0 = 0',
      note2: 'Real payoff ℓ(0)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Generator q_θ',
      head: 'Point Mass at θ',
      eq: 'q_θ = δ_θ (x\' = θ)',
      note1: 'D_ψ(θ) = ψ · θ',
      note2: 'Fake payoff ℓ(−ψθ)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Value V(θ, ψ)',
      head: 'Unique Equilibrium',
      eq: 'V = ℓ(0) + ℓ(−ψθ)',
      note1: '(θ*, ψ*) = (0, 0)',
      note2: 'sup_{ψ} V = ℓ(0) for θ≠0',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Linear critic D_ψ(x) = ψx has spatial gradient ∇_x D_ψ(x) = ψ everywhere' }
  ),

  'ch16-gan-circular-flow': () => compare2(
    'Continuous-Time Simultaneous Gradient Flow Orbits on Circles',
    {
      head: 'Orthogonal Velocity Field',
      sub: 'u(θ, ψ) = ℓ\'(−ψθ) · (ψ, −θ)',
      rows: [
        'θ̇ = −∂_{θ} V = +ψ ℓ\'(−ψθ)',
        'ψ̇ = +∂_{ψ} V = −θ ℓ\'(−ψθ)',
        'd/dt(θ² + ψ²) = 2θψℓ\' − 2ψθℓ\' = 0',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Linear Center at (0, 0)',
      sub: 'Jacobian J at equilibrium',
      rows: [
        'J = [[0, ℓ\'(0)], [−ℓ\'(0), 0]]',
        'Eigenvalues ν₁,₂ = ±i ℓ\'(0)',
        'Re(ν) = 0 → zero damping!',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'Every trajectory orbits indefinitely at constant radius r(t) = r(0) for any payoff ℓ' }
  ),

  'ch16-gan-euler-vs-alt': () => compare2(
    'Discretization Hazards: Simultaneous Euler vs. Alternating Steps',
    {
      head: 'Simultaneous Euler Steps',
      sub: '(θ, ψ)_{k+1} = (θ, ψ)_{k} + h u_k',
      rows: [
        'Since u ⊥ (θ, ψ), by Pythagoras:',
        '‖z_{k+1}‖² = ‖z_k‖² + h²‖u_k‖²',
        '|1 ± i h ℓ\'(0)| > 1 (Spirals Out!)',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Alternating Half-Steps',
      sub: 'Update ψ first, then θ using ψ_{k+1}',
      rows: [
        'Composes two shears (a = h ℓ\'(0)):',
        'M = [[1 − a², a], [−a, 1]], det M = 1',
        'Area-preserving bounded annulus!',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'Neither simultaneous nor alternating unpenalized steps converge to (0, 0)' }
  ),

  'ch16-gan-r1-r2-flat': () => pipeline3(
    'Why Zero-Centered Input-Gradient Penalties Vanish at Equilibrium',
    {
      tag: 'At Equilibrium q = p',
      head: 'Flat Optimal Critic',
      eq: 'D*(x) = log(p/q) = 0',
      note1: 'Constant across supp(p)',
      note2: '∇_x D*(x) = 0 everywhere!',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Penalties R₁ and R₂',
      head: 'Input Gradient Norms',
      eq: 'R₁ = ½γ E_p[‖∇D‖²]',
      note1: 'R₂ = ½γ E_q[‖∇D‖²]',
      note2: 'R₁+R₂ = γ E_m[‖∇D‖²]',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Unshifted Optimum',
      head: 'Damped Critic Step',
      eq: 'V_ℓ(D) − R₁ − R₂',
      note1: 'Does not move q* = p',
      note2: 'Adds −γψ damping to ψ̇',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Penalizing ‖∇_x D‖² damps critic oscillations while leaving the equilibrium q = p intact' }
  ),

  'ch16-gan-damping-regimes': () => stack3(
    'Damped Oscillator Regimes: ν₁,₂ = −γ/2 ± √(γ²/4 − ℓ\'(0)²)',
    {
      badge: '1. Underdamped',
      head: '0 < γ < 2ℓ\'(0)  (0 < γ < 1 for logistic ℓ\'(0) = 1/2)',
      sub: 'Complex roots with Re(ν) = −γ/2 < 0; spirals inward (Euler h < 4γ)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: '2. Critical',
      head: 'γ = 2ℓ\'(0)  (γ = 1 for logistic loss)',
      sub: 'Double root ν = −ℓ\'(0) (−1/2 for logistic); fastest monotone decay',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      badge: '3. Overdamped',
      head: 'γ > 2ℓ\'(0)  (γ > 1 for logistic loss)',
      sub: 'Two negative real roots; slow mode ν_slow ≈ −ℓ\'(0)²/γ',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'For every γ > 0, Re(ν₁,₂) < 0 makes the equilibrium (0, 0) locally exponentially stable'
  ),

  'ch16-gan-zero-vs-one-gp': () => compare2(
    'Penalty Design: Why Zero-Centered R₁ + R₂ Beats WGAN-GP and R₁ Alone',
    {
      head: 'One-Centered WGAN-GP',
      sub: 'Penalizes γ(‖∇_x D‖ − 1)²',
      rows: [
        'Forces unit slope ‖∇D‖ = 1 at q = p',
        'Dirac-GAN: cycles (γ < 2) or parks at |θ| = γ',
        'Cannot rest flat at equilibrium',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Both R₁ + R₂ vs. R₁ Alone',
      sub: 'p = δ₀, q = δ₁, critic D(x) = −cx²',
      rows: [
        'At x=0: ∇D(0) = 0 ⇒ R₁ = 0!',
        'Under R₁ alone: c → ∞ explodes!',
        'Adding R₂ = 2γc² caps c* < 1/(8γ)',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Penalizing both real (R₁) and fake (R₂) gradients bounds critic steepness everywhere' }
  ),

  'ch16-gan-sobolev-filter': () => pipeline3(
    'Local Value Under R₁ + R₂: Dual Sobolev Spatial Low-Pass Filter',
    {
      tag: 'Quadratic Form',
      head: 'Laplacian Operator',
      eq: '∫m‖∇D‖² = ⟨D, L_m D⟩',
      note1: 'L_m D = −∇·(m ∇D)',
      note2: 'Weighted by mixture m',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Optimal Local Value',
      head: 'Inverse Operator',
      eq: '(|ℓ\'\'|m + γL_m)⁻¹',
      note1: 'γ → 0: 2 JS(p, q)',
      note2: 'γ → ∞: ‖p−q‖²_{H⁻¹} / γ',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Fourier Mode ω',
      head: 'Low-Pass Filter',
      eq: '(|ℓ\'\'| + γω²)⁻¹',
      note1: 'Cutoff scale 2√γ',
      note2: 'γ has units of length²',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'R₁ + R₂ smooths fine-scale density spikes so the critic provides coarse global pulls' }
  ),

  'ch16-gan-25g-factorial': () => matrix2x2(
    '25-Gaussian 5×5 Grid: 2×2 Factorial Design (10,000 Steps, β₁ = 0)',
    {
      head: 'Arm 1: NS (γ = 0)',
      eq: 'Classic Logistic, No Penalty',
      sub: 'Steep critic (‖∇D‖² ~ 10), drops modes',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Arm 2: Pairing (γ = 0)',
      eq: 'RpGAN Ranking, No Penalty',
      sub: 'Steep critic; re-pairing drops modes',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'Arm 3: NS + R₁ + R₂ (γ = 1)',
      eq: 'Classic Logistic + Penalties',
      sub: '‖∇D‖² ~ 10⁻³; covers all 25 modes',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Arm 4: Pairing + R₁ + R₂ (γ = 1)',
      eq: 'R3GAN (RpGAN + R₁ + R₂)',
      sub: '‖∇D‖² ~ 10⁻³; covers all 25 modes',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Plus γ = 10 overdamped control: confirms R₁+R₂ penalties drive mode coverage'
  ),

  // ── dcgan.md (5 diagrams) ───────────────────────────────────────────
  'ch16-gan-transposed-conv': () => pipeline3(
    'Spatial Upsampling Hierarchy vs. Checkerboard Overlap Pitfall',
    {
      tag: 'Stage 1 · Project',
      head: '1×1×100 → 4×4×512',
      eq: 'k=4, s=1, p=0',
      note1: 'n_out = s(n−1) + k − 2p',
      note2: '1(0) + 4 − 0 = 4×4 grid',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Stages 2–5 · Double',
      head: '4 → 8 → 16 → 32 → 64',
      eq: 'k=4, s=2, p=1 → 2n',
      note1: 'k=4 divisible by s=2',
      note2: 'Even 2×2 stamp overlap',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      tag: 'Pitfall · Uneven k/s',
      head: 'Checkerboard Grid',
      eq: 'k=3, s=2 → 2,1,2,1',
      note1: 'Uneven stamp counts',
      note2: 'Fix: Bilinear ↑ + 3×3',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    { text: 'R3GAN replaces 4×4 transposed convolutions with bilinear resampling + 3×3 convs' }
  ),

  'ch16-gan-normfree-gen': () => pipeline3(
    'Normalization-Free Generator: 4×4 Constant + 4 Bilinear Stages',
    {
      tag: '4×4 Stem',
      head: 'Const + Projected z',
      eq: '4×4×128 + Linear(z)',
      note1: 'Concat along channels',
      note2: 'Plus class embed if cGAN',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '4 Upsample Stages',
      head: '4 → 8 → 16 → 32 → 64',
      eq: 'Bilinear ↑2× + Conv3',
      note1: 'LeakyReLU(0.2), He init',
      note2: 'Zero BatchNorm layers!',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'RGB Head',
      head: '3×3 Conv → 64×64×3',
      eq: 'Linear RGB (no tanh)',
      note1: '2.8M total parameters',
      note2: 'Independent batch rows',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'He fan-in init Var(w) = 2 / ((1 + 0.2²) n_in) stabilizes variance without BatchNorm' }
  ),

  'ch16-gan-normfree-critic': () => pipeline3(
    'Normalization-Free Critic: Independent Per-Sample Rows for R₁ + R₂',
    {
      tag: 'Input Image',
      head: '64×64×3 RGB Batch',
      eq: 'x tracks gradients',
      note1: 'No BatchNorm coupling!',
      note2: 'D(x_i) depends on x_i',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '4 Downsample Stages',
      head: '64 → 32 → 16 → 8 → 4',
      eq: 'Conv3×3 + Pool ↓2×',
      note1: 'LeakyReLU(0.2) + He init',
      note2: '1.6M total parameters',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Pooled Head & R₁/R₂',
      head: '4×4 Features φ(x)',
      eq: '∇_x (∑ D(x_i))',
      note1: 'Input gradient kept in graph',
      note2: 'Exact per-sample ‖∇D‖²',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Omitting BatchNorm ensures ∇_{x_i} ∑_{j} D(x_j) = ∇_{x_i} D(x_i) in one backward pass' }
  ),

  'ch16-gan-nonleaking-aug': () => compare2(
    'Why Augmenting Only Real Images Leaks vs. Non-Leaking Augmentation',
    {
      head: 'Real-Only Augmentation (Leaks!)',
      sub: 'Apply random flip T_ρ only to x ~ p',
      rows: [
        'Critic compares T_ρ p against q',
        'Equilibrium forces q = T_ρ p!',
        'Generates flipped left-only sprites',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Non-Leaking (ADA / DiffAugment)',
      sub: 'Apply T_ρ to BOTH x and x\' inside D',
      rows: [
        'Critic compares T_ρ p against T_ρ q',
        'For invertible T_ρ (ρ ≠ 1/2):',
        'T_ρ q = T_ρ p  ⇒  q = p (Exact!)',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Differentiating through T_ρ(G(z)) in D prevents memorization without corrupting q' }
  ),

  'ch16-gan-eval-summary': () => stack3(
    'Complementary Feature-Space Evaluation Metrics (φ(x) ∈ ℝ^{d})',
    {
      badge: 'FID (Gaussian W₂²)',
      head: '‖μ_p − μ_q‖² + tr(Σ_p + Σ_q − 2(Σ_p¹/² Σ_q Σ_p¹/²)¹/²)',
      sub: 'Compares first 2 moments; has positive +O(1/n) covariance bias',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'KID (Cubic MMD²)',
      head: 'Unbiased U-Statistic with Kernel k(u, v) = (uᵀv/d + 1)³',
      sub: 'Compares up to 3rd moments; zero finite-sample bias at any n',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'k-NN Prec & Recall',
      head: 'Precision = q in real balls B_p  ·  Recall = p in fake balls B_q',
      sub: 'Separates fidelity (precision) from mode coverage (recall)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Always calibrate FID against a real-vs.-real reference floor at the same sample size n'
  ),

  // ── conditional.md (5 diagrams) ─────────────────────────────────────
  'ch16-gan-cond-game': () => pipeline3(
    'Conditional Minimax Game on Joint Pairs (x, c)',
    {
      tag: 'Real Joint p(x, c)',
      head: 'Data + True Label',
      eq: 'p(c) · p(x | c)',
      note1: 'Draw (x, c) from data',
      note2: 'Score D(x, c) > 0',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Fake Joint q(x, c)',
      head: 'Same Prior p(c)',
      eq: 'p(c) · q(x | c)',
      note1: 'Draw c~p(c), x\'=G(z,c)',
      note2: 'Score D(x\', c) < 0',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Optimal D*(x, c)',
      head: 'Prior p(c) Cancels!',
      eq: 'log(p(x|c)/q(x|c))',
      note1: 'p(c) in num & denom',
      note2: 'Slice-wise log ratio!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Sharing the condition prior p(c) cancels p(c) out of the optimal critic D*(x, c)' }
  ),

  'ch16-gan-slice-js': () => compare2(
    'Expected Slice-Wise JS Divergence vs. Rare-Class Imbalance',
    {
      head: 'Independent Game per Slice c',
      sub: 'max_{D} V = E_c[2 JS(p(·|c), q(·|c))] − 2log2',
      rows: [
        'Min −2 log 2 iff q(·|c) = p(·|c) ∀c',
        'Cyclic shift q(·|c) = p(·|c+1):',
        'Every slice has JS = log 2 (Worst!)',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Rare-Class Underweighting Hazard',
      sub: 'Each slice is weighted by prior p(c)',
      rows: [
        'Rare condition with p(c) = ε = 10⁻³',
        'Contributes ≤ 2ε log 2 ≈ 0.0014 nats!',
        'Remedy: class-balanced sampling',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'Unlike marginal JS(p(x), q(x)), conditional JS detects cyclic label shifts immediately' }
  ),

  'ch16-gan-projection-head': () => pipeline3(
    'Bayes\' Rule Derivation of the Miyato–Koyama Projection Head',
    {
      tag: '1. Bayes Factoring',
      head: 'Split p(x|c)/q(x|c)',
      eq: 'log(p(c|x)/q(c|x))',
      note1: '+ log(p(x) / q(x))',
      note2: 'Alignment + Realism',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Softmax Logits',
      head: 'Linear in Features φ',
      eq: '(u_c − w_c)ᵀ φ(x)',
      note1: 'p(c|x) ∝ exp(u_cᵀ φ)',
      note2: 'q(c|x) ∝ exp(w_cᵀ φ)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Projection Head',
      head: 'Inner Product + ψ₀',
      eq: 'e_cᵀ φ(x) + ψ₀(x)',
      note1: 'e_c = u_c − w_c (embed)',
      note2: 'ψ₀ absorbs log(p(x)/q(x))',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'One inner product e_cᵀ φ(x) scores class alignment; scalar head ψ₀(x) scores realism' }
  ),

  'ch16-gan-acgan-bias': () => compare2(
    'Why Projection Critics Beat Auxiliary Classifier GANs (AC-GAN)',
    {
      head: 'AC-GAN Classifier Loss (Biased)',
      sub: 'Minimizes −E[log C(c | x\')] = H(q(c|x\')) + KL',
      rows: [
        'Forces entropy H(q(c | x\')) → 0',
        'Shuns regions where classes overlap',
        'Collapses onto canonical prototypes!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Projection Critic Head (Unbiased)',
      sub: 'Estimates log(p(c|x)/q(c|x)) + log(p(x)/q(x))',
      rows: [
        'Vanishes when q(x | c) = p(x | c)',
        'Keeps true overlap entropy H(p(c|x))',
        'Maintains full within-class diversity!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Forcing low class entropy rewards prototype collapse; log-ratios preserve diversity' }
  ),

  'ch16-gan-three-eval-axes': () => stack3(
    'Three Orthogonal Failure Axes in Conditional Image Evaluation',
    {
      badge: 'Axis 1: Alignment',
      head: 'Wrong Class (q(·|c) = p(·|c\')) → Referee Classifier',
      sub: 'Marginal FD/MMD² are blind if marginal q(x) = p(x) matches!',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Axis 2: Fidelity',
      head: 'Blur / Artifacts → Marginal FD & Per-Class Precision',
      sub: 'Generated features fall outside 3-NN support balls of real images',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'Axis 3: Diversity',
      head: 'Intra-Class Prototype Collapse → Per-Class 3-NN Recall',
      sub: 'Referee accuracy and precision RISE on prototypes, recall drops!',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'Complete evaluation: Referee Accuracy + Marginal FD + Per-Class Precision/Recall'
  ),

  // ── adversarial-losses.md (10 diagrams) ─────────────────────────────
  'ch16-gan-multimodal-blur': () => compare2(
    'Pointwise Regression vs. Adversarial Critics on Multimodal Targets',
    {
      head: 'Pointwise Regression (L₂ / L₁)',
      sub: 'Predicts conditional mean E[x | c]',
      rows: [
        'Averages all plausible sharp textures',
        'Lands in empty void between modes',
        'Fails on super-res, inpainting & audio',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Learned Adversarial Critic D(x, c)',
      sub: 'Penalizes empty corridor (D* = log(p/q) ≪ 0)',
      rows: [
        '1. Diffusion distillation (ADD / DMD2)',
        '2. Latent tokenizers (VQGAN / SD-VAE)',
        '3. Speech vocoders (HiFi-GAN)',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Critics survive across modern foundation pipelines to prevent conditional-mean blur' }
  ),

  'ch16-gan-orth-decomp': () => pipeline3(
    'Orthogonal Decomposition of Expected Squared Error E[(x − g(c))²]',
    {
      tag: 'Add & Subtract μ(c)',
      head: 'Cross Term Vanishes',
      eq: 'E[(x−μ)(μ−g)] = 0',
      note1: 'μ(c) = E[x | c]',
      note2: 'Iterated expectation',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Irreducible Spread',
      head: 'Aleatoric Variance',
      eq: 'E[(x−μ)²] = Var(x|c)',
      note1: 'Independent of g(c)',
      note2: 'Large when multimodal',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Predictor Error',
      head: 'Unique L₂ Minimizer',
      eq: 'E[(μ(c) − g(c))²]',
      note1: 'Minimized at g* = μ(c)',
      note2: 'L₁ picks median (void!)',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'When p(x | c) is bimodal, both the mean μ(c) and median sit in the empty void' }
  ),

  'ch16-gan-two-bands': () => {
    const W = 560, H = 276;
    let o = '';
    o += tx(280, 18, 'Bimodal Two-Band Benchmark: x = c ± 1 + 0.05 ξ', { fs: 15, fw: 700, fill: C.ink });
    const px = 22, py = 38, pw = 236, ph = 186;
    o += box(px, py, pw, ph, '#FFFFFF', C.gray, 8, 2);
    o += line(px + 24, py + 76, px + 210, py + 28, C.blue, 6);
    o += tx(px + 86, py + 26, 'x = c + 1', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
    o += line(px + 24, py + 116, px + 210, py + 68, D.red.base, 2.5, true);
    o += line(px + 24, py + 156, px + 210, py + 108, C.blue, 6);
    o += tx(px + 132, py + 162, 'x = c − 1', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
    o += box(px + 54, py + 82, 128, 22, D.red.tint, D.red.base, 5, 1.5);
    o += tx(px + 118, py + 93, 'μ(c) = c (void)', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

    o += box(276, 38, 266, 86, D.red.tint, D.red.base, 8, 2);
    o += tx(409, 58, 'Conditional Mean μ(c) = c', { fs: 14, fw: 700, fill: D.red.dark });
    o += tx(409, 80, 'MSE = 1² + 0.05² = 1.0025', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    o += tx(409, 102, 'Minimum MSE, 20σ from data!', { fs: 12, fw: 600, fill: D.red.dark });

    o += box(276, 138, 266, 86, C.lgreen, C.green, 8, 2);
    o += tx(409, 158, 'On-Band Predictor g(c) = c + 1', { fs: 14, fw: 700, fill: D.green.dark });
    o += tx(409, 180, 'MSE = ½(0²) + ½(2²) + 0.05² = 2.0025', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    o += tx(409, 202, '100% on manifold, 2× higher MSE!', { fs: 12, fw: 600, fill: D.green.dark });

    o += box(14, 234, 532, 32, D.purple.tint, C.purple, 6, 1.5);
    o += tx(280, 250, 'Adversarial critic D*(x, c) = log(p/q) ≪ 0 in corridor pushes g(c) onto a valid band', { fs: 12.5, fw: 700, fill: D.purple.dark });
    return svg(W, H, o);
  },

  'ch16-gan-perception-distortion': () => stack3(
    'The Perception–Distortion Trade-Off (Blau & Michaeli, 2018)',
    {
      badge: 'Blurry Mean μ(c)',
      head: 'Minimum Distortion: MSE = Var(x | c) = 1.0025 (0 dB)',
      sub: 'Optimal under L₂ / PSNR, rejected with probability 1 by critic',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      badge: '1-Band Predictor',
      head: 'Mode-Dropped Manifold: MSE = 2.0025 (−3.00 dB PSNR)',
      sub: 'Every sample is realistic, but drops the lower band (recall = 50%)',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      badge: 'Ideal Sampler x̂~p',
      head: 'Exact Realism (q = p): MSE = 2 Var(x|c) = 2.0050',
      sub: 'E[(x − x̂)²] = E[(x − μ)²] + E[(μ − x̂)²] = 2 Var(x|c) (−3.01 dB)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Perfect perceptual distribution matching costs up to 3.01 dB in PSNR vs. the mean'
  ),

  'ch16-gan-latent-routing': () => compare2(
    'Why Multimodal Coverage Requires a Stochastic Latent Input z',
    {
      head: 'Deterministic Map x̂ = g(c)',
      sub: 'Single output per condition c',
      rows: [
        'c carries 0 bits about band b ∈ {±1}',
        'Function g(c) hits at most 1 value',
        'Infinite capacity still drops 1 band!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Stochastic Generator x̂ = g(c, z)',
      sub: 'Routes latent quantiles z ~ N(0, 1)',
      rows: [
        'z > 0 → upper band c + 1 + 0.05ξ',
        'z < 0 → lower band c − 1 + 0.05ξ',
        'Covers both modes (49% / 49%)!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Target ambiguity requires a latent z; capacity limits require teacher distillation' }
  ),

  'ch16-gan-high-noise-blur': () => compare2(
    'Diffusion Denoiser D*_{τ}(x_τ) = E[x₀ | x_τ] at Low vs. High Noise',
    {
      head: 'Low Noise (τ ≪ 1)',
      sub: 'Posterior p(x₀ | x_τ) is unimodal',
      rows: [
        'Noisy x_τ = x₀ + τξ pins down x₀',
        'Posterior mean E[x₀ | x_τ] is sharp',
        'Multi-step solvers finish at τ → 0',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      head: 'High Noise (τ ≫ 1)',
      sub: 'Posterior p(x₀ | x_τ) spans all modes',
      rows: [
        'Many clean images match x_τ',
        'E[x₀ | x_τ] is a blurry average!',
        '1-step regression onto teacher blurs',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    { text: 'Regressing a 1-step student onto a high-noise denoiser pulls outputs into the void' }
  ),

  'ch16-gan-tanh-contraction': () => pipeline3(
    'Fixed-Point Contraction of High-Noise Teacher Regression (τ = 2)',
    {
      tag: 'Exact Teacher',
      head: 'Posterior Log-Odds',
      eq: 'c + tanh((x̃−c)/4)',
      note1: 'At τ=2: τ²+0.05² ≈ 4',
      note2: 'Slope ≤ 1/4 = 0.25 < 1',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Teacher-Only Loss',
      head: 'Unique Fixed Point',
      eq: 'u=E[tanh((u+2ξ)/4)]',
      note1: 'Contraction |f\'(u)| < 1',
      note2: 'Collapses to u* = 0!',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      tag: 'Teacher + Critic D',
      head: '400× Stronger Pull',
      eq: 'Slope ~ δ / 0.05²',
      note1: '400δ beats 0.008 pull',
      note2: '99% on bands (49/49%)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Sharp critic restoring force (σ = 0.05) overcomes high-noise teacher contraction' }
  ),

  'ch16-gan-add-dmd2': () => compare2(
    'One-Step Foundation Model Distillation: ADD vs. DMD2',
    {
      head: 'ADD / SDXL-Turbo (Sauer et al.)',
      sub: 'Teacher Denoiser + DINOv2 Critic',
      rows: [
        '1. Regress noised x̂ onto teacher D_τ',
        '2. Frozen DINOv2 conditional critic',
        'Teacher anchors prompt; D sharpens',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'DMD & DMD2 (Yin et al., 2024)',
      sub: 'Two Score Nets + Real-Data Critic',
      rows: [
        'DMD: ∇log p_τ(x_τ) − ∇log q_τ(x_τ)',
        'Equals optimal critic slope ∇log(p/q)!',
        'DMD2 adds real-image GAN critic',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Both ADD and DMD2 pair a diffusion teacher with an adversarial discriminator' }
  ),

  'ch16-gan-vqgan-patchgan': () => stack3(
    'Three Complementary Decoder Losses in VQGAN / Latent Diffusion VAEs',
    {
      badge: '1. Pixel L₁ / L₂',
      head: 'Pointwise Reconstruction: ‖x̂ − x‖₁ or ‖x̂ − x‖₂²',
      sub: 'Anchors low-frequency color and layout, blurs fine textures',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: '2. Perceptual LPIPS',
      head: 'Pretrained Feature Matching: ∑_{ℓ} ‖φ_ℓ(x̂) − φ_ℓ(x)‖₂²',
      sub: 'Preserves mid-level structure, permits repetitive grid artifacts',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: '3. PatchGAN Critic',
      head: 'Local Convolutional Discriminator on Every N × N Patch',
      sub: 'Forces decoder to commit to crisp high-frequency textures',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Latent diffusion autoencoders (e.g., Stable Diffusion\'s) use PatchGAN-trained decoders'
  ),

  'ch16-gan-hifigan-phase': () => compare2(
    'HiFi-GAN Vocoders: Why L₂ Waveform Regression Collapses to Silence',
    {
      head: 'Phase Uncertainty in Audio',
      sub: 'x_n = sin(ωn + ϑ),  ϑ ~ U[−a, a]',
      rows: [
        'Mel-spectrogram drops phase ϑ',
        'E_{ϑ}[sin(ωn + ϑ)] = (sin a / a) sin(ωn)',
        'Full uncertainty a = π ⇒ sin π / π = 0!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'HiFi-GAN Discriminator Suite',
      sub: 'Mel L₁ + Time-Domain Critics',
      rows: [
        'Multi-Period Discriminator (MPD)',
        'Multi-Scale Discriminator (MSD)',
        'Enforces amplitude & phase continuity!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Without a time-domain waveform critic, phase ambiguity shrinks L₂ audio to silence' }
  ),
};
