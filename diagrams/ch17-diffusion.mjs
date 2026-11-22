// Slide diagrams for Chapter 17 (Diffusion and Flow-Based Generative Models)
import { C, TOKENS, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

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
    o += tx(85, y + 31, c.badge, { mono: !!c.monoBadge, fs: 12, fw: 700, fill: c.dark });
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
  // ── index.md (1 diagram) ────────────────────────────────────────────
  'ch17-diff-roadmap': () => stack3(
    'Chapter 17 Roadmap: Score Matching → Fast Solvers → Tokens',
    {
      badge: '§§17.1–17.3 Core',
      head: 'Denoising Scores, Gaussian Paths & ODE/SDE Transport',
      sub: 'Trained MLP matches exact 2D oracle weights within 0.004',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: '§§17.4–17.6 Images',
      head: 'U-Net + CFG (§17.4) · Fast Solvers (§17.5) · ELBO (§17.6)',
      sub: '20-NFE Heun matches 1,000-step FD; ODE NLL matches entropy',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: '§§17.7–17.9 Scale',
      head: 'Latent VAE & DiT (§17.7) · Flow Maps (§17.8) · Tokens',
      sub: 'Latent U-Net cuts FLOPs >10×; 2 jumps match 15 Euler steps',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Unifying Thread: Regress onto clean data across noise levels, integrate t: 1 → 0'
  ),

  // ── denoising.md (9 diagrams) ───────────────────────────────────────
  'ch17-diff-mse-posterior': () => pipeline3(
    'Least-Squares Denoising Learns the Posterior Mean E[x₀ | x̃]',
    {
      tag: '1. Corrupt Data',
      head: 'Gaussian Blur',
      eq: 'x̃ = x₀ + σ ε',
      note1: 'x₀ ~ p,  ε ~ N(0, I)',
      note2: 'Marginal x̃ ~ p_σ',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Orthogonal Split',
      head: 'Condition on x̃',
      eq: '‖D(x̃)−E[x₀|x̃]‖² + C',
      note1: 'Cross term vanishes',
      note2: 'C = tr Cov[x₀ | x̃]',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Bayes Optimum',
      head: 'Posterior Mean',
      eq: 'D*(x̃) = E[x₀ | x̃]',
      note1: 'Unique MSE minimizer',
      note2: 'Averages plausible x₀',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Minimizing E‖D(x̃) − x₀‖² over random pairs recovers E[x₀ | x̃] without knowing p(x₀)' }
  ),

  'ch17-diff-vincents-dsm': () => pipeline3(
    'Vincent\'s Theorem (2011): Conditional to Marginal Score Matching',
    {
      tag: 'Conditional Kernel',
      head: 'Known Noise Score',
      eq: '∇log p_σ(x̃|x₀)',
      note1: '= −ε/σ, points back to x₀',
      note2: 'Equals (x₀ − x̃) / σ²',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'DSM Regression',
      head: 'Per-Example Target',
      eq: '½ E‖s_θ(x̃) + ε/σ‖²',
      note1: 'O(1) backprop pass',
      note2: 'No Jacobian trace!',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Bayes Posterior Avg',
      head: 'Marginal Score!',
      eq: 's*(x̃) = ∇log p_σ(x̃)',
      note1: 'E[−ε/σ | x̃] = ∇p_σ/p_σ',
      note2: 'Exact blurred score',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Regressing onto the score of the noise we added fits the marginal score ∇log p_σ(x̃)' }
  ),

  'ch17-diff-gaussian-shrinkage': () => stack3(
    'Analytical Tweedie Shrinkage on a Gaussian Mode N(μ, v I)',
    {
      badge: 'Blurred Score',
      head: 'p_σ = N(μ, (v + σ²)I)  ⇒  ∇log p_σ(x̃) = −(x̃ − μ) / (v + σ²)',
      sub: 'Adding isotropic noise σ² inflates variance from v = s² to v + σ²',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Wiener Shrinkage',
      head: 'E[x₀ | x̃] = x̃ + σ² ∇log p_σ(x̃) = μ + [v / (v + σ²)](x̃ − μ)',
      sub: 'At v = 0.25, σ = 0.3: shrinkage factor v/(v + σ²) = 0.735',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'MSE Reduction',
      head: 'Per-Coordinate MSE Drops from σ² = 0.090 to 0.066',
      sub: 'In 2D (d = 2): point MSE drops from 0.180 to 0.132 (−27%)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Tweedie step x̃ + σ² s_θ(x̃) shrinks noisy points toward high-density mode centers'
  ),

  'ch17-diff-three-sm-losses': () => stack3(
    'Three Score-Matching Objectives Compared',
    {
      badge: 'Fisher Divergence',
      head: 'Exact Target: ½ E_p[‖s_θ(x) − ∇log p(x)‖²]',
      sub: 'Ideal objective, intractable since ground-truth ∇log p is unknown',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Explicit SM (2005)',
      head: 'Integration by Parts: E_p[½‖s_θ(x)‖² + ∇ · s_θ(x)] + const',
      sub: 'Eliminates ∇log p, but Jacobian trace ∇ · s_θ costs O(d) passes',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      badge: 'Denoising SM',
      head: 'Noise Regression: ½ E_{x₀,ε}[‖s_θ(x₀ + σε) + ε/σ‖²]',
      sub: 'Fast O(1) pass in any dimension d; fits blurred score ∇log p_σ(x̃)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Denoising Score Matching trades O(d) Jacobian traces for a controllable σ-blur bias'
  ),

  'ch17-diff-sigma-dilemma': () => compare2(
    'The Single-Noise-Level Dilemma: Target Variance vs. Smoothing Bias',
    {
      head: 'Small Noise (σ → 0)',
      sub: 'Target −ε/σ explodes in variance',
      rows: [
        'Target variance E‖ε/σ‖² = d / σ²',
        'Loss floor diverges as d / σ² → ∞',
        'High gradient noise & mode trapping',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Large Noise (σ ≫ s)',
      sub: 'Blurred score ∇log p_σ ≠ ∇log p',
      rows: [
        'Bias = d σ⁴ / [2s²(s² + σ²)²]',
        'σ=0.05: 0.0004  →  σ=1.0: 2.56',
        'Fast mixing, but 5× the data variance',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'No single σ can give both low-variance global mixing and sharp local density' }
  ),

  'ch17-diff-manifold-cliff': () => compare2(
    'Anisotropic Score Geometry Near a Low-Dimensional Manifold (x₂ = 0)',
    {
      head: 'Tangential Direction (x̃₁)',
      sub: 'Along the data manifold',
      rows: [
        'Marginal density q_σ(x̃₁) is smooth',
        'Tangential score ∂₁ log q_σ is O(1)',
        'Buried under O(1/σ) target noise!',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Normal Direction (x̃₂)',
      sub: 'Orthogonal to the manifold',
      rows: [
        'Gaussian thickness N(x̃₂; 0, σ²)',
        'Normal score ∂₂ log p_σ = −x̃₂ / σ²',
        'Curvature 1/σ² → ∞ (steep cliff!)',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    { text: 'At tiny σ, a network must fit a −1/σ² normal cliff while averaging O(1/σ) noise' }
  ),

  'ch17-diff-langevin-step': () => pipeline3(
    'Overdamped Langevin Dynamics: x_{k+1} = x_k + (h/2) s(x_k) + √h ξ_k',
    {
      tag: 'Drift + Diffusion',
      head: 'Stationary Balance',
      eq: '+(h/2)∇log p + √h ξ',
      note1: 'Uphill drift + Brownian',
      note2: 'Leaves p(x) invariant',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Finite Step Size h',
      head: 'Variance Inflation',
      eq: 'v_h = v / (1 − h/4v)',
      note1: 'At v=0.25, h=0.05: +5%',
      note2: 'Exact as step h → 0',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Kramers Escape',
      head: 'Valley Bottleneck',
      eq: 'τ_esc ∝ exp(ΔU)',
      note1: 'ΔU = 10–12 nat valley',
      note2: 'Requires ~10⁶ steps!',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    { text: 'Local score ascent cannot cross deep low-density barriers between separated modes' }
  ),

  'ch17-diff-single-level-table': () => stack3(
    'Diagnosing the Five Single-Level Langevin Runs (2,000 Steps)',
    {
      badge: 'Clean Score ∇log p',
      head: 'Weights [0.34, 0.33, 0.33]  ·  Strays 1.3%  ·  Spread 0.50',
      sub: 'Sharp spread, but 10–12 nat barriers freeze initial 1/3 weights!',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      badge: 'Wide Blur ∇log p₁.₀',
      head: 'Weights [0.49, 0.31, 0.20]  ·  Strays 41%  ·  Spread 1.01',
      sub: 'Mixes mode weights accurately, but with 5× the variance!',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      badge: 'MLP Score (σ=0.3)',
      head: 'Weights [0.39–0.46, ...]  ·  Strays 5–6.5%  ·  Spread 0.60',
      sub: '7–9 nat barriers block mixing; valley drift driven by MLP errors',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    'Target Ground Truth p: Weights [0.50, 0.30, 0.20] · Strays 1.1% · Spread 0.48'
  ),

  'ch17-diff-wiener-vs-cnn': () => compare2(
    '784-D Fashion-MNIST Denoising (σ = 0.5): Linear Wiener vs. Convnet',
    {
      head: 'Optimal Linear Wiener Filter',
      sub: 'μ + Σ(Σ + σ²I)⁻¹(x̃ − μ)',
      rows: [
        'Shrinks PCA axis i by λ_i/(λ_i + σ²)',
        'Projects onto a flat linear subspace',
        'Test MSE = 0.045 (blurry edges)',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: '4-Layer Nonlinear Convnet',
      sub: 'Learned Posterior Mean E[x₀ | x̃]',
      rows: [
        'Projects onto curved garment manifold',
        '33% lower test MSE = 0.030 (30 SE!)',
        'Still blurs fine texture at high σ',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'One-step denoising returns the posterior average E[x₀ | x̃]; generation needs a path' }
  ),

  // ── gaussian-paths.md (8 diagrams) ──────────────────────────────────
  'ch17-diff-continuum-scales': () => pipeline3(
    'Connecting Data (t = 0) to Pure Noise (t = 1) Along a Gaussian Path',
    {
      tag: 't = 1 · Coarse Scale',
      head: 'High Noise (σ ≫ s)',
      eq: 'x₁ = ε ~ N(0, I)',
      note1: 'Valleys completely filled',
      note2: 'Establishes 0.5/0.3/0.2',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 't ∈ (0, 1) · Path',
      head: 'Continuous Schedule',
      eq: 'x_t = α_t x₀ + σ_t ε',
      note1: 'One network u_θ(x_t, t)',
      note2: 'Smooth ODE/SDE transport',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 't → 0 · Fine Scale',
      head: 'Low Noise (σ ≪ s)',
      eq: 'x₀ ~ p_data',
      note1: '∇log p_σ → ∇log p',
      note2: 'Sharp local geometry!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Annealing from t = 1 down to t = 0 combines global mixing with local sharpness' }
  ),

  'ch17-diff-log-snr-clock': () => pipeline3(
    'Log-SNR λ_t = log(α_t² / σ_t²) Aligns Every Gaussian Schedule',
    {
      tag: 'Any Gaussian Path',
      head: 'Point at Time t',
      eq: 'x_t = α_t x₀ + σ_t ε',
      note1: 'Signal-to-noise α_t/σ_t',
      note2: 'λ_t = 2 log(α_t / σ_t)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Invertible Rescaling',
      head: 'Divide by α_t + σ_t',
      eq: 'c = 1 / (α_t + σ_t)',
      note1: 't_lin = σ_t / (α_t + σ_t)',
      note2: 'Preserves log-SNR λ_t!',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Linear Path Image',
      head: 'Matched Noise Level',
      eq: '(1−t_lin)x₀ + t_lin ε',
      note1: 'Same posterior p(x₀|x_t)',
      note2: 'Schedule = clock speed',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'All Gaussian paths traverse the exact same distributions indexed by λ ∈ (−∞, +∞)' }
  ),

  'ch17-diff-cond-vs-marg-vel': () => compare2(
    'Single-Pair Conditional Velocity vs. Posterior-Averaged Marginal Velocity',
    {
      head: 'Conditional Velocity ẋ_t',
      sub: 'Single training pair (x₀, ε)',
      rows: [
        'ẋ_t = α̇_t x₀ + σ̇_t ε',
        'Linear path: ẋ_t = ε − x₀ (constant)',
        'Chords from different x₀ cross!',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Marginal Velocity Field u_t(x)',
      sub: 'Posterior average at (x_t = x, t)',
      rows: [
        'u_t(x) = E[α̇_t x₀ + σ̇_t ε | x_t = x]',
        'u_t(x) = α̇_t x̂₀(x) + σ̇_t ε̂(x)',
        'Single vector at (x, t): never crosses!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'The probability cloud p_t(x) moves under the posterior-averaged velocity u_t(x)' }
  ),

  'ch17-diff-cfm-identity': () => pipeline3(
    'Conditional Flow Matching (CFM): Learning u_t Without Evaluating It',
    {
      tag: 'CFM Training Loss',
      head: 'Per-Example Target',
      eq: 'E‖u_θ − (α̇x₀+σ̇ε)‖²',
      note1: 'Sample (x₀, ε) in O(1)',
      note2: 'Linear target: ε − x₀',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Condition on x_t',
      head: 'Orthogonal Split',
      eq: 'E‖u_θ − u_t‖² + C',
      note1: 'Cross term vanishes!',
      note2: 'C = conditional variance',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Identical Gradients',
      head: '∇_θ C = 0',
      eq: 'u*_{θ}(x, t) = u_t(x)',
      note1: 'Recovers marginal field',
      note2: 'Loss plateaus at floor C',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Just as DSM learns ∇log p_σ from −ε/σ, CFM learns u_t(x) from α̇_t x₀ + σ̇_t ε' }
  ),

  'ch17-diff-target-dictionary': () => matrix2x2(
    'Five Equivalent Targets from One Posterior Oracle (x_t = α_t x̂₀ + σ_t ε̂)',
    {
      head: '1. Clean Point & 2. Noise',
      eq: 'x̂₀ = E[x₀|x_t],  ε̂ = E[ε|x_t]',
      sub: 'Linked by x_t = α_t x̂₀ + σ_t ε̂',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: '3. Marginal Score Field',
      eq: '∇log p_t(x_t) = −ε̂(x_t) / σ_t',
      sub: 'Tweedie: α_t x̂₀ = x_t + σ_t² ∇log p_t',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: '4. Angular Velocity v',
      eq: 'v̂ = α_t ε̂ − σ_t x̂₀',
      sub: 'x̂₀ = (α_t x_t − σ_t v̂)/(α_t² + σ_t²)',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: '5. Marginal Velocity u_t',
      eq: 'u_t = α̇_t x̂₀ + σ̇_t ε̂',
      sub: 'Linear path: u_t = ε̂ − x̂₀',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Any trained predictor immediately determines the other four via affine algebra'
  ),

  'ch17-diff-score-velocity-id': () => compare2(
    'The Score–Velocity Identity Across Forward Schedules',
    {
      head: 'Linear Path (α_t = 1−t, σ_t = t)',
      sub: 'α̇_t = −1,  σ̇_t = +1',
      rows: [
        'u_t(x) = ε̂(x) − x̂₀(x)',
        'u_t(x) = −[x + t ∇log p_t(x)] / (1−t)',
        'Score & velocity are affine twins!',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Variance-Preserving (VP) Path',
      sub: 'α_t² + σ_t² = 1,  α̇_t/α_t = −½β(t)',
      rows: [
        'σ_t σ̇_t = −α_t α̇_t = +½β(t) α_t²',
        'u_t(x) = −½β(t)[x + ∇log p_t(x)]',
        'Matches continuous reverse-SDE drift',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'General identity: u_t(x) = (α̇_t/α_t)x − (σ_t σ̇_t − σ_t² α̇_t/α_t) ∇log p_t(x)' }
  ),

  'ch17-diff-target-conditioning': () => compare2(
    'Practical Conditioning: Singularity at Pure Noise (α₁ = 0)',
    {
      head: 'Noise Prediction (\'eps\')',
      sub: 'Recovers x̂₀ = (x_t − σ_t ε̂) / α_t',
      rows: [
        'Divides by α_t to reconstruct x̂₀!',
        'Blows up if schedule reaches α₁ = 0',
        'Requires VP schedule with α₁ ≈ 0.007',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Velocity (\'velocity\') & \'v\'',
      sub: 'Linear: x̂₀ = x_t − t û  (no division!)',
      rows: [
        'Cosine v: x̂₀ = α_t x_t − σ_t v̂',
        'Well-conditioned at both t=0 and t=1',
        'Allows exact pure-noise endpoint α₁=0',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'On the linear path, ‖û − u‖² = ‖ε̂ − ε‖²/(1−t)² = ‖x̂₀ − x₀‖²/t² balances both ends' }
  ),

  'ch17-diff-loss-floors': () => stack3(
    'Analytical Loss Floors: floor(t) = c_t² E[tr Cov(x₀ | x_t)]',
    {
      badge: 'Clean Point x₀',
      head: 'c_t = 1  ·  Floor: 0 at t → 0  to  tr Cov(x₀) at t → 1',
      sub: 'At t = 0, x_t = x₀ is known (floor = 0); at t = 1, posterior is prior',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Noise ε (VP)',
      head: 'c_t = −α_t/σ_t  ·  Floor: d = 2 at t → 0  to  0 at t → 1',
      sub: 'Average floor ≈ 0.54; at t → 1, x_t ≈ ε is known (floor → 0)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'Velocity ε − x₀',
      head: 'c_t = −1/t  ·  Floor: d = 2 at t → 0  to  ≈ 9 near t = 0.9',
      sub: 'Average floor ≈ 4.91 (9× higher than VP noise from c_t² = 1/t²)',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'Never compare raw loss values across targets; rank models by excess over floor'
  ),

  // ── integration.md (9 diagrams) ─────────────────────────────────────
  'ch17-diff-transport-weights': () => compare2(
    'Why ODE/SDE Integration Recovers Exact Mode Weights (0.5, 0.3, 0.2)',
    {
      head: 'Fixed-Level Langevin MCMC',
      sub: 'Local score ∇log p(x) at small σ',
      rows: [
        'Zero flux across 10–12 nat valleys',
        'Chains stay trapped in initial mode',
        'Returns initial [0.33, 0.33, 0.33]!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Reverse-Time Transport (t: 1 → 0)',
      sub: 'Starts from known prior p₁ = N(0, I)',
      rows: [
        'Follows marginal velocity dx/dt = u_t',
        'No trajectories created or destroyed',
        'Mass conservation ⇒ exact π_k!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Transporting a known Gaussian prior along p_t sidesteps Kramers barrier crossing' }
  ),

  'ch17-diff-continuity-flux': () => pipeline3(
    'Conservation of Probability Mass: The Continuity Equation',
    {
      tag: 'Probability Cloud',
      head: 'Time-Varying p_t(x)',
      eq: '∫ p_t(x) dx = 1',
      note1: 'Particles move at u_t(x)',
      note2: 'Mass flux J = p_t u_t',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Divergence Theorem',
      head: 'Net Inward Flux',
      eq: '∂_{t} p_t = −∇·J',
      note1: 'Rate of density change',
      note2: 'Equals −∑ ∂_{i}(p_t u_{t,i})',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Transport Theorem',
      head: 'Exact Marginals',
      eq: 'x₁ ~ p₁ ⇒ x_t ~ p_t',
      note1: 'Integrate dx/dt = u_t(x)',
      note2: 'Arrives at x₀ ~ p₀!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Any velocity field u_t satisfying ∂_{t} p_t = −∇·(p_t u_t) transports p₁ to p₀ exactly' }
  ),

  'ch17-diff-continuity-proof': () => pipeline3(
    'Two-Step Proof That Posterior-Averaged u_t(x) Satisfies Continuity',
    {
      tag: 'Step 1 · Single x₀',
      head: 'Conditional Cloud',
      eq: '∂_{t} p_c = −∇·J_c',
      note1: 'J_c = p_t(x|x₀) u_t(x|x₀)',
      note2: 'Exact Gaussian transport',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Step 2 · Bayes Rule',
      head: 'Average Over x₀ ~ p',
      eq: 'E[J_c] = p_t u_t',
      note1: 'Factors p_t(x) outside E',
      note2: 'p(x₀)p_t(x|x₀)=p_t p(x₀|x)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Step 3 · Linearity',
      head: 'Marginal Continuity',
      eq: '∂_{t} p_t = −∇·J',
      note1: '∂_{t} and ∇· commute with E',
      note2: 'Proves exact transport!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Averaging conditional fluxes over p(x₀) turns conditional velocity into marginal u_t(x)' }
  ),

  'ch17-diff-noncrossing-ode': () => compare2(
    'Crossing Conditional Chords vs. Non-Crossing Probability-Flow ODE',
    {
      head: 'Conditional Lines (x₀, ε)',
      sub: 'x_t = (1 − t)x₀ + t ε',
      rows: [
        'Straight line for each pair (x₀, ε)',
        'Chords from different x₀ intersect',
        'Not a single-valued vector field!',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'Probability-Flow ODE dx/dt = u_t',
      sub: 'Posterior-averaged velocity u_t(x)',
      rows: [
        'Unique vector u_t(x) at each (x, t)',
        'Picard–Lindelöf: paths NEVER cross!',
        'Curved deterministic bijection x₁ ↔ x₀',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    { text: 'Marginal ODE trajectories curve around intersections to partition N(0, I) into basins' }
  ),

  'ch17-diff-euler-mean-collapse': () => pipeline3(
    'Why a Single Euler Step From Pure Noise Collapses to the Data Mean',
    {
      tag: 'At Pure Noise t = 1',
      head: 'x₁ = ε ⊥ x₀',
      eq: 'p(x₀ | x₁) = p(x₀)',
      note1: 'No info about which x₀',
      note2: 'Posterior mean = E[x₀]',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Initial Velocity u₁',
      head: 'Points to Global Mean',
      eq: 'u₁(x₁) = x₁ − E[x₀]',
      note1: 'Since ε̂(x₁) = x₁',
      note2: 'and x̂₀(x₁) = E[x₀]',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '1-Step Euler (K = 1)',
      head: 'Total Collapse!',
      eq: 'x₁ − u₁(x₁) = E[x₀]',
      note1: 'Every start x₁ lands on',
      note2: 'E[x₀] = (−0.50, −0.70)!',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    { text: 'Curved ODE trajectories require multi-step integration (or a flow map §17.8)' }
  ),

  'ch17-diff-ve-coordinates': () => pipeline3(
    'Path-Agnostic Coordinates: Rescaling x̄_t = x_t / α_t',
    {
      tag: 'Divide by Signal α_t',
      head: 'VE Coordinates',
      eq: 'x̄_t = x₀ + σ̄_t ε',
      note1: 'Noise ratio σ̄_t = σ_t/α_t',
      note2: 'σ̄_t = exp(−λ_t / 2)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Universal Flow ODE',
      head: 'Slope Is Just ε̂!',
      eq: 'dx̄ / dσ̄ = ε̂(x̄, σ̄)',
      note1: 'Linear drift absorbed!',
      note2: 'Same ODE on every path',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Euler Step in (x̄, σ̄)',
      head: 'Multiply by α_s',
      eq: 'x_s=α_s x̂₀+σ_s ε̂',
      note1: 'Equals DDIM at η = 0!',
      note2: 'Exact on 1-point data',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Every schedule reduces to dx̄/dσ̄ = ε̂, so one solver integrates any path or target' }
  ),

  'ch17-diff-disentangle-error': () => stack3(
    'Separating Solver Discretization Error from Network Error (2,000 Starts)',
    {
      badge: '10 Euler Steps',
      head: 'Network: Spread 0.45, Strays 3.0%  ·  Oracle: Spread 0.46',
      sub: 'Identical mode contraction → 10-step error is solver stride!',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      badge: '100 Euler Steps',
      head: 'Network: Weights [0.504, 0.302, 0.194] · Spread 0.48',
      sub: 'Matches ground-truth data ([0.506, 0.300, 0.194], spread 0.48)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      badge: '20,000-Start Test',
      head: '99.5%–99.8% Mode Agreement Between MLP and Oracle',
      sub: '100-step ODE and 200-step ancestral match weights within 1 SE',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    'Comparing against the exact 2D oracle proves 10-step error is discretization, not MLP fit'
  ),

  'ch17-diff-ancestral-split': () => pipeline3(
    'First-Order Anatomy of a DDPM Ancestral Step (β = 1 − α_t²/α_s²)',
    {
      tag: 'DDPM Reverse Step',
      head: 'Expand (1−β)⁻¹/²',
      eq: '(x_t + β s_t)/√(1−β)',
      note1: '+ √β z,  z ~ N(0, I)',
      note2: '1/√(1−β) ≈ 1 + β/2',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Half 1 · Transport',
      head: 'Flow ODE Euler Step',
      eq: '+(β/2)(x_t + s_t)',
      note1: 'Equals −(t − s) u_t(x_t)',
      note2: 'Moves marginal p_t → p_s',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Half 2 · Corrector',
      head: 'Langevin MCMC Step',
      eq: '+(β/2) s_t + √β z',
      note1: 'Step size h = β on p_t',
      note2: 'Damps drift errors!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Every DDPM ancestral step = 1 Euler transport step + 1 Langevin correction step' }
  ),

  'ch17-diff-sde-vs-ode': () => compare2(
    'Anderson\'s Time Reversal: Reverse SDE vs. Probability-Flow ODE',
    {
      head: 'Reverse-Time SDE (Stochastic)',
      sub: 'dx = [f − g² ∇log p_t]dt + g dW̄',
      rows: [
        'Full score drift −g(t)² ∇log p_t',
        'Injects Brownian noise g(t) dW̄',
        'Continuously mixes & damps errors',
      ],
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: 'Probability-Flow ODE (Deterministic)',
      sub: 'dx/dt = f − ½ g² ∇log p_t = u_t(x)',
      rows: [
        'Halves score coefficient: −½ g(t)²',
        'Zero stochastic term (dW̄ = 0)',
        'Invertible bijection x₁ ↔ x₀ & fast ODE',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    { text: 'Both processes share the exact same marginals p_t(x) at every t ∈ [0, 1]' }
  ),

  // ── image-diffusion.md (8 diagrams) ─────────────────────────────────
  'ch17-diff-scaling-4pillars': () => matrix2x2(
    'Four Systems Components for Scaling Diffusion to Images',
    {
      head: '1. Multi-Scale U-Net Backbone',
      eq: '28×28 → 14×14 → 7×7 + Skips',
      sub: 'Global context at t→1, pixel edges at t→0',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: '2. Weight EMA & Label Dropout',
      eq: 'θ̄ ← β θ̄ + (1−β)θ,  p_drop = 0.1',
      sub: 'Damps SGD jitter; trains u_c & u_∅ jointly',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: '3. Calibrated Fréchet Distance',
      eq: 'FD(p, q) vs. 1/n Real Floor',
      sub: '5,000-sample evaluation in FeatureCNN',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: '4. Classifier-Free Guidance',
      eq: 'ũ = u_∅ + γ(u_c − u_∅)',
      sub: 'Steers samples toward requested class c',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'The training loss E‖u_θ(x_t, t, c̃) − (ε − x₀)‖² and Euler ODE solver remain unchanged!'
  ),

  'ch17-diff-unet-profile': () => compare2(
    'Systems Profile of the Denoising U-Net (1.20M Params, 247M MACs)',
    {
      head: 'Where Parameters Live (Capacity)',
      sub: '9 C_in C_out weights (independent of H×W)',
      rows: [
        '28×28 stages (64 ch): 17% of params',
        '14×14 stages (128 ch): 45% of params',
        '7×7 bottleneck: 38% of params!',
      ],
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: 'Where FLOPs Go (Arithmetic)',
      sub: 'Scales with spatial pixels H × W',
      rows: [
        '28×28 stages (784 px): 59% of MACs!',
        '14×14 stages (196 px): 33% of MACs',
        '7×7 bottleneck: only 8% of MACs',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    { text: 'Put channel capacity at low resolution (7×7 costs 16× less compute per weight)' }
  ),

  'ch17-diff-label-dropout': () => pipeline3(
    'Joint Conditional & Unconditional Training via 10% Label Dropout',
    {
      tag: 'Minibatch Labels c',
      head: '10 Garment Classes',
      eq: 'c ∈ {0, 1, ..., 9}',
      note1: '128 images per batch',
      note2: 'Draw coin p_drop = 0.1',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '11-Row Embedding',
      head: 'Null Token ∅ = 10',
      eq: '90% c  /  10% ∅',
      note1: '11 × d embedding table',
      note2: 'Added to time embed t',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Single U-Net u_θ',
      head: 'Two Vector Fields',
      eq: 'u_θ(c) & u_θ(∅)',
      note1: 'Shared conv weights',
      note2: 'Ready for CFG at test!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: '10% label dropout trains the unconditional baseline u_∅ with zero extra parameters' }
  ),

  'ch17-diff-ema-smoothing': () => compare2(
    'Why Exponential Moving Average (EMA) Lowers FD by 0.5 Units',
    {
      head: 'Live Adam Iterates θ_k',
      sub: 'Noisy minibatch draws (x₀, t, ε)',
      rows: [
        'Wanders in a cloud around optimum',
        'Sensitive to last minibatch noise',
        '50-step Euler FD = 1.25–1.38',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'EMA Shadow Weights θ̄_k',
      sub: 'θ̄_k = β θ̄_{k−1} + (1−β)θ_k',
      rows: [
        'Half-life n_{1/2}=1,000 (β ≈ 0.9993)',
        'Averages out stochastic gradient jitter',
        '50-step Euler FD = 0.80–0.83 (±0.02)!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Set half-life n_{1/2} to ~5–10% of the training budget so early underfit weights decay' }
  ),

  'ch17-diff-fd-floor': () => compare2(
    'Calibrating Fréchet Distance: The +O(1/n) Finite-Sample Bias Floor',
    {
      head: 'Why Disjoint Real Sets Score > 0',
      sub: 'Sample covariance Σ̂ fluctuates by O(n⁻¹/²)',
      rows: [
        'Matrix square root (Σ_p Σ_q)¹/²',
        'Upward trace bias scales as +O(1 / n)',
        'n = 1,000 real floor: FD = 0.87 ± 0.18',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'Three Iron Laws of FD Evaluation',
      sub: 'n = 5,000 real floor drops 5× to 0.17 ± 0.05',
      rows: [
        '1. Never compare FD across different n!',
        '2. Report the real-vs.-real floor at n',
        '3. Respect the multi-seed noise spread',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    { text: 'An FD of 0.82 at n = 5,000 is 5× the real floor (0.17); at n = 1,000 it equals the floor!' }
  ),

  'ch17-diff-cfg-derivation': () => pipeline3(
    'Classifier-Free Guidance (CFG) via Bayes\' Rule on Scores',
    {
      tag: '1. Bayes Score Identity',
      head: 'Prior p(c) Vanishes',
      eq: '∇ln p(c|x) = s_c−s_∅',
      note1: 's_c = ∇log p_t(x | c)',
      note2: 's_∅ = ∇log p_t(x)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Scale by γ ≥ 0',
      head: 'Guided Score s̃',
      eq: 's_∅ + γ(s_c − s_∅)',
      note1: 'γ = 0: unconditional',
      note2: 'γ > 1: extrapolates!',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Affine Invariance',
      head: 'Same Rule for All!',
      eq: 'u_∅ + γ(u_c − u_∅)',
      note1: 'Weights (1−γ) + γ = 1',
      note2: 'Holds for x̂₀, ε̂, v̂, u_t',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'CFG steers generation without an external classifier by contrasting u_c against u_∅' }
  ),

  'ch17-diff-guided-overshoot': () => compare2(
    'Static Tilted Gaussian vs. Dynamic Guided ODE Overshoot (γ = 4)',
    {
      head: 'Static Tilted Density at t = 0',
      sub: 'p̃₀(x) ∝ p₀(x)¹⁻ᵞ p₀(x | c)ᵞ',
      rows: [
        'Precision: 1/v = γ/s² − (γ−1)/S²',
        'Width narrows 2×: s=0.50 → √v=0.26',
        'Mean shifts outward: 1.00 → 1.10',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Guided Flow ODE (t: 1 → 0)',
      sub: 'Integrates ũ_t = u_∅ + γ(u_c − u_∅)',
      rows: [
        'Compounds tilt across all times t ∈ [0,1]',
        'Width collapses 25×: 0.50 → 0.02!',
        'Mean overshoots further to 1.23!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    { text: 'Integrating a guided field does NOT sample the static tilted distribution p̃₀(x)!' }
  ),

  'ch17-diff-curvature-ceiling': () => stack3(
    'Why High-Noise Guidance Breaks the Gaussian Curvature Ceiling',
    {
      badge: 'Curvature Bound',
      head: 'Blurred p_t = p₀ * N(0, σ_t² I) Obeys −∇²log p_t ⪯ (1/σ_t²) I',
      sub: 'Convolving with variance σ_t² bounds precision above by 1 / σ_t²',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'CFG Violation',
      head: 'For γ > S²/(S² − s²) = 8/7, Precision Exceeds 1/t² Near t = 1',
      sub: 'At γ = 4, guided score violates the ceiling across all t ∈ (0.24, 1]!',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      badge: 'Modern Remedies',
      head: '1. Guidance Intervals (2024)  ·  2. Autoguidance (2024)',
      sub: 'Apply γ > 1 only in mid-noise band, or guide against a weaker net',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Disabling guidance at high noise (t → 1) prevents violent early variance contraction'
  ),

  // ── samplers.md (5 diagrams) ────────────────────────────────────────
  'ch17-diff-one-net-many-samplers': () => pipeline3(
    'Why One Trained Network Supports Infinitely Many Reverse Samplers',
    {
      tag: 'Training Objective',
      head: 'Single-Time Marginals',
      eq: 'p_t(x_t | x₀)',
      note1: 'Never uses joint (x_s, x_t)',
      note2: 'Learns posterior (x̂₀, ε̂)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Non-Markov Family',
      head: 'Any Marginal Match',
      eq: 'q_{σ̃}(x_s | x_t, x₀)',
      note1: 'Preserves N(α_s x₀, σ_s²I)',
      note2: 'For any σ̃ ∈ [0, σ_s]',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Three Fast Levers',
      head: '20–50 Evaluations',
      eq: 'η · Order · Grid',
      note1: 'Deterministic η = 0',
      note2: '2nd-order Heun / AB2',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Cutting 1,000 steps to 20 NFE requires zero retraining of the denoising network' }
  ),

  'ch17-diff-ddim-noise-split': () => pipeline3(
    'The DDIM Family: Splitting Target Noise Variance σ_s² at Step t → s',
    {
      tag: 'Predicted Clean x̂₀',
      head: 'Signal Component',
      eq: 'α_s x̂₀(x_t, t)',
      note1: 'Deterministic mean',
      note2: 'Anchored at time s',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Kept Noise ε̂',
      head: 'Variance σ_s² − σ̃²',
      eq: '√(σ_s² − σ̃²) · ε̂',
      note1: 'Reuses inferred noise',
      note2: 'η = 0 ⇒ keeps 100% (σ̃=0)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Fresh Noise ξ',
      head: 'Variance σ̃²',
      eq: '+ σ̃ ξ,  ξ ~ N(0, I)',
      note1: 'σ̃ = η σ_s √(1 − κ²)',
      note2: 'Total var = σ_s² always!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Orthogonal variances (σ_s² − σ̃²) + σ̃² = σ_s² preserve the marginal for every η ≥ 0' }
  ),

  'ch17-diff-heun-2nd-order': () => pipeline3(
    'Heun\'s Second-Order Predictor–Corrector on dx̄/dσ̄ = ε̂(x̄, σ̄)',
    {
      tag: '1. Start Slope',
      head: 'Evaluate at σ̄_t',
      eq: 'd₁ = ε̂(x̄_t, σ̄_t)',
      note1: '1st network evaluation',
      note2: 'Step Δ = σ̄_s − σ̄_t < 0',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Euler Trial Step',
      head: 'Propose Endpoint x̄\'',
      eq: 'x̄\' = x̄_t + Δ · d₁',
      note1: 'Evaluate d₂ = ε̂(x̄\', σ̄_s)',
      note2: '2nd network evaluation',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Trapezoid Average',
      head: 'O(Δ²) Update',
      eq: 'x̄_t + ½Δ(d₁ + d₂)',
      note1: '4.3× smaller error',
      note2: 'Hits plateau in 20 NFE!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Averaging start and trial slopes cancels first-order curvature error (O(h) → O(h²))' }
  ),

  'ch17-diff-multistep-ab2': () => pipeline3(
    '2nd-Order Adams–Bashforth Multistep Solver (1 NFE per Step)',
    {
      tag: 'Cached Past Slope',
      head: 'From Previous Step',
      eq: 'ε̂\' at step Δ\'',
      note1: 'Zero extra network calls!',
      note2: 'Stored from step k−1',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Current Slope',
      head: 'Evaluate at σ̄_t',
      eq: 'ε̂_t = ε̂(x̄_t, σ̄_t)',
      note1: '1 NFE at current state',
      note2: 'Slope change δ = ε̂_t − ε̂\'',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Linear Extrapolation',
      head: 'Midpoint Estimate',
      eq: 'x̄_t + Δ(ε̂_t + r δ)',
      note1: 'r = Δ/(2Δ\') gives O(Δ²)',
      note2: 'Best solver under CFG!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Reusing the previous slope gives 2nd-order accuracy at 1 NFE/step (DPM-Solver++)' }
  ),

  'ch17-diff-roundtrip-inversion': () => compare2(
    'Deterministic ODE Inversion (t: 0 → 1) and Round-Trip Error',
    {
      head: 'Endpoint Slope Mismatch',
      sub: 'Why finite-step round-trip MSE > 0',
      rows: [
        'Forward t→t\': uses slope ε̂ at lower t',
        'Reverse t\'→t: uses slope ε̂ at upper t\'',
        'MSE drops >100× from 10 to 200 steps',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Code Norm Contraction',
      sub: 'Inverted code std ≈ 0.7–0.85 < 1',
      rows: [
        'Forward Euler under-amplifies variance',
        'Same rate on U-Net & exact 2D oracle',
        'Use Slerp to preserve radius r = ‖z‖!',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'Round-trip error is solver discretization error, not neural vector-field roughness' }
  ),

  // ── design-space.md (7 diagrams) ────────────────────────────────────
  'ch17-diff-three-likelihood-views': () => stack3(
    'Three Connections Between Denoising Regression and Likelihood',
    {
      badge: '1. Reverse ELBO',
      head: 'Variational Bound on −log p_θ(x₀) (Sohl-Dickstein 2015)',
      sub: 'Telescopes into a log-SNR weighted sum of noise errors across k',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: '2. Flow ODE NLL',
      head: 'Exact Log-Density via Change of Variables (Song 2021)',
      sub: 'log p_θ(x₀) = log N(x₁; 0, I) + ∫₀¹ ∇ · u_t(x_t) dt along ODE path',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: '3. w(λ) Integral',
      head: 'Canonical Loss: ½ ∫ w(λ) E‖ε̂_θ(x_λ, λ) − ε‖² dλ',
      sub: 'Unifies ELBO (w ≡ 1), DDPM, v-prediction (exp(−λ/2)), flow & EDM',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Both the continuous ELBO and ODE flow NLL match our 2D mixture entropy to 0.005 nats'
  ),

  'ch17-diff-elbo-jensen': () => pipeline3(
    'Variational Bound (ELBO) on the Reverse Markov Chain',
    {
      tag: 'Forward Chain q',
      head: 'Fixed Gaussian Steps',
      eq: 'q(x_{1:T} | x₀)',
      note1: 'No learnable weights',
      note2: 'Ends at x_T ≈ N(0, I)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Reverse Chain p_θ',
      head: 'Learned Transitions',
      eq: 'p(x_T) ∏ p_θ(k−1|k)',
      note1: 'Gaussian mean μ_θ(x_k,k)',
      note2: 'Step variance ν_k I',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Jensen Inequality',
      head: 'Upper Bound L(x₀)',
      eq: '−log p_θ ≤ L(x₀)',
      note1: 'Gap = KL(q ‖ p_θ | x₀)',
      note2: 'E_p[L] ≥ H(p)+KL(p‖p_θ)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Forward KL(p ‖ p_θ) inside E_p[L(x₀)] heavily penalizes dropping any data mode' }
  ),

  'ch17-diff-elbo-telescoping': () => stack3(
    'Telescoping L(x₀) via Bayes\' Rule on q(x_k | x_{k−1}, x₀)',
    {
      badge: 'Prior Term L_T',
      head: 'KL( q(x_T | x₀)  ‖  p(x_T) )   where p(x_T) = N(0, I)',
      sub: 'Measures how close terminal state x_T is to pure Gaussian noise',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'Denoising L_{k−1}',
      head: '∑_{k=2}^{T} E_q[ KL( q(x_{k−1} | x_k, x₀)  ‖  p_θ ) ]',
      sub: 'Compares each reverse step against Gaussian N(μ̃_k, β̃_k I)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: 'Reconstruction L₀',
      head: 'E_{q(x₁ | x₀)}[ −log p_θ(x₀ | x₁) ]',
      sub: 'Final decoding log-likelihood from lowest noise x₁ to clean x₀',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'Conditioning on x₀ makes every forward posterior q(x_{k−1} | x_k, x₀) an exact Gaussian!'
  ),

  'ch17-diff-elbo-logsnr-integral': () => pipeline3(
    'From Discrete Gaussian KL to the Schedule-Invariant Log-SNR Integral',
    {
      tag: '1. Posterior Mean μ̃_k',
      head: 'Substitute x₀(x_k, ε)',
      eq: 'KL ∝ ω_k ‖ε − ε̂_θ‖²',
      note1: 'Gaussian KL becomes',
      note2: 'weighted noise MSE!',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Step Weight ω_k',
      head: 'At Variance ν_k = β̃_k',
      eq: 'ω_k ≈ ½ Δλ_k',
      note1: 'Half the log-SNR drop',
      note2: 'Spikes near clean k ≤ 10',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Limit T → ∞',
      head: 'Riemann Sum → ∫ dλ',
      eq: '½ ∫ E‖ε − ε̂_θ‖² dλ',
      note1: 'Weight w(λ) ≡ 1 (flat!)',
      note2: 'Invariant to schedule!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Continuous-time ELBO depends only on endpoints [λ_min, λ_max], never on the path' }
  ),

  'ch17-diff-canonical-w-lambda': () => stack3(
    'Unifying Published Objectives via w(λ) = 2 c(t) p(t) |dt / dλ|',
    {
      badge: 'Flat w(λ) = 1',
      head: 'Continuous ELBO / Variational Diffusion (Kingma et al.)',
      sub: 'Weights all log-SNR scales equally; optimal for exact NLL / bits',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: 'w(λ) = exp(−λ/2)',
      head: 'Linear-Path Flow Matching  =  Cosine-Path v-Prediction!',
      sub: 'Monotone decreasing; suppresses high-λ noise for lower FD',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      badge: 'Bell Bump w(λ)',
      head: 'DDPM Simple (λ ≈ −0.9)  ·  EDM Log-Normal (λ ≈ +2.4)',
      sub: 'Focuses capacity on mid-noise scales where structure forms',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    'Changing target c(t), path λ_t, or time sampling p(t) simply reshapes one curve w(λ)'
  ),

  'ch17-diff-monotone-bounds': () => compare2(
    'Why Monotonically Decreasing Weightings w\'(λ) ≤ 0 Are Likelihood Bounds',
    {
      head: 'Step Basis w_μ(λ) = 1{λ ≤ μ}',
      sub: 'Truncates integral above log-SNR μ',
      rows: [
        'Stops reverse integration at level μ',
        'Exact ELBO on noised data x_μ!',
        'Drops high-SNR fine detail λ > μ',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Any Monotone Decreasing w(λ)',
      sub: 'w(λ) = ∫ w_μ(λ) (−w\'(μ)) dμ',
      rows: [
        'Positive mixture of step ELBOs',
        'Bound on noise-augmented data!',
        'Includes Flow Matching w(λ) = e^(−λ/2)',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Non-monotone bumps (DDPM L_simple, linear ε) are not bounds on any data scale' }
  ),

  'ch17-diff-edm-precond': () => pipeline3(
    'EDM Preconditioning: Unit-Variance Inputs and Targets Across σ̄',
    {
      tag: 'Input Scale c_in',
      head: 'Unit-Variance Input',
      eq: '1 / √(σ̄² + σ_d²)',
      note1: 'Normalizes x̄ = x₀ + σ̄ε',
      note2: 'to variance 1 for F_θ',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Skip Weight c_skip',
      head: 'Wiener Coefficient',
      eq: 'σ_d² / (σ̄² + σ_d²)',
      note1: 'σ̄ → 0: c_skip → 1',
      note2: 'σ̄ → ∞: c_skip → 0',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Output Scale c_out',
      head: 'Adaptive Target',
      eq: 'σ̄σ_d / √(σ̄²+σ_d²)',
      note1: 'σ̄ → 0: F_θ predicts −ε',
      note2: 'σ̄ → ∞: F_θ predicts x₀',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'D_θ(x̄; σ̄) = c_skip x̄ + c_out F_θ(c_in x̄; σ̄) keeps activations O(1) across 4 decades' }
  ),

  // ── latent-diffusion.md (6 diagrams) ────────────────────────────────
  'ch17-diff-pixel-compute-wall': () => compare2(
    'Why Pixel-Space Diffusion Hits a High-Resolution Compute Wall',
    {
      head: 'Pixel-Space Bottlenecks',
      sub: '512×512×3 = 786,432 values',
      rows: [
        '50 steps touch every pixel 50 times',
        'Pixel self-attention scales as (H·W)²',
        'Most bits model invisible fine grain!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Two Architectural Breakthroughs',
      sub: 'Compress first, then scale tokens',
      rows: [
        '1. Latent Diffusion: 8× spatial VAE',
        '   (64× fewer sites, >13× fewer FLOPs)',
        '2. DiT: Patch tokens + adaLN-Zero',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Latent diffusion shifts high-frequency texture synthesis to a single decoder pass' }
  ),

  'ch17-diff-kl-vae-reg': () => pipeline3(
    'KL-Regularized Autoencoder for a Diffusable Latent Space',
    {
      tag: 'Encoder E(x)',
      head: '28×28×1 → 7×7×8',
      eq: 'μ(x), s(x) ∈ 7×7×4',
      note1: '4× spatial compression',
      note2: 'z = μ + s ⊙ ξ (196-D)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Light KL Penalty',
      head: 'β = 10⁻⁴ Anchor',
      eq: '½(μ²+s²−1−ln s²)',
      note1: 'Prevents heavy tails',
      note2: 'Avoids s → 0 collapse',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Decoder D(z)',
      head: '7×7×4 → 28×28×1',
      eq: '‖x − D(z)‖² + β·KL',
      note1: 'Sharp recon FD ≈ 0.7',
      note2: 'Smooth latent manifold',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Light β = 10⁻⁴ keeps reconstructions sharp while eliminating heavy-tailed outliers' }
  ),

  'ch17-diff-latent-scale-factor': () => compare2(
    'Why Rescaling Latents by c = 1 / std(z) Is Mandatory Before Diffusion',
    {
      head: 'Unscaled Latents (std = ς ≠ 1)',
      sub: 'Example: Stable Diffusion raw ς ≈ 5.5',
      rows: [
        'z_t = (1 − t)z₀ + t ε with Var(z₀) = ς²',
        'True SNR is ς²(1−t)²/t² (30× higher!)',
        'Shifts log-SNR clock by +log 30 ≈ +3.4!',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Normalized Latents z₀ ← c · z',
      sub: 'c = 1/ς (1.05 here; 0.18215 in SD)',
      rows: [
        'Restores unit signal variance Var(z₀) = 1',
        'Matches unit Gaussian noise ε ~ N(0, I)',
        'Decode at end via x̂₀ = D(ẑ₀ / c)',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Always check empirical latent std and kurtosis (≤ 0) across the training set first' }
  ),

  'ch17-diff-dit-patchify': () => pipeline3(
    'Diffusion Transformer (DiT): Patchify → Isotropic Blocks → Unpatchify',
    {
      tag: '1. PatchEmbedding',
      head: 'Stride-p Conv (p×p)',
      eq: 'L = (H/p)(W/p)',
      note1: '28×28, p=4 → L=49',
      note2: '+ Learned 2D pos embed',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. N × DiTBlock',
      head: 'Isotropic Width d',
      eq: 'O(L² d + L d²) MACs',
      note1: 'Self-Attn + MLP',
      note2: 'Steered by adaLN-Zero',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Linear Unpatchify',
      head: 'Zero-Init Projection',
      eq: 'ℝ^{d} → p×p×C',
      note1: 'Each token → p×p patch',
      note2: 'Starts at exact 0!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Halving patch size p quadruples token count L and multiplies self-attention by 16×!' }
  ),

  'ch17-diff-cross-vs-mmdit': () => compare2(
    'Text Prompt Conditioning: Cross-Attention vs. Joint MM-DiT',
    {
      head: 'Cross-Attention (LDM / SD 1–2)',
      sub: 'Queries Q from image, K & V from text',
      rows: [
        'softmax(Q Kᵀ / √d) V ∈ ℝ^{m × d}',
        'Text tokens X_text are frozen',
        'Cache K, V once outside 50-step loop!',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Joint Multimodal DiT (SD3 / Flux)',
      sub: 'Concatenate [Y_img; X_text] (m + n)',
      rows: [
        'Full bidirectional attention over m + n',
        'Separate adaLN-Zero per modality',
        'Text and image tokens update together',
      ],
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    { text: 'Cross-attention caches K, V once; MM-DiT enables deep bidirectional fusion' }
  ),

  'ch17-diff-backbone-matrix': () => matrix2x2(
    'Architectural Decision Matrix: Space and Backbone Selection',
    {
      head: 'Low Resolution (≤ 64×64)',
      eq: 'Pixel Space + U-Net',
      sub: 'Fastest & lowest FD (no VAE gap)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      head: 'High Resolution (512²+) · Conv',
      eq: 'Latent Space (8× VAE) + U-Net',
      sub: 'Cuts spatial grid & MACs by 64×',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'High Resolution (512²+) · Scale',
      eq: 'Latent Space + DiT (p = 2)',
      sub: 'Steepest compute-scaling improvement',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: 'Large-Patch Pixel Transformers',
      eq: 'Pixel Space + Clean x̂₀ Target',
      sub: 'Projects out high-D orthogonal noise',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'Convolutional U-Nets win at small compute budgets; DiTs dominate as GFLOPs scale up'
  ),

  // ── few-step.md (8 diagrams) ────────────────────────────────────────
  'ch17-diff-tangent-overshoot': () => compare2(
    'Instantaneous Tangent Velocity vs. Integrated Chord Displacement',
    {
      head: 'Velocity Network u_θ(x_t, t)',
      sub: 'Predicts local tangent dx_{t} / dt',
      rows: [
        'At t = 1: u₁(x₁) = x₁ − E[x₀]',
        '1-step Euler: x₁ − u₁(x₁) = E[x₀]!',
        'Curved paths need 20–50 steps',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: 'Two Routes to 1–4 Step Generation',
      sub: 'Replace local tangent with global jump',
      rows: [
        '1a. Straighten paths (Reflow)',
        '1b. Learn flow map Φ_{t→s}(x_t)',
        '2. Match output density (DMD / ADD)',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Velocity is a derivative, not a displacement; 1-step jumping requires ∫ u_τ dτ' }
  ),

  'ch17-diff-reflow-straightening': () => pipeline3(
    'Rectified Flow & Reflow: Straightening ODE Trajectories',
    {
      tag: '1. Standard CFM',
      head: 'Independent Pairs',
      eq: 'x₀ ~ p₀,  z ~ N(0,I)',
      note1: 'Chords (x₀, z) cross!',
      note2: 'ODE trajectories bend',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      tag: '2. Teacher ODE Coupling',
      head: 'Generate Endpoints',
      eq: 'x̂₀ = Φ_{1→0}(z)',
      note1: 'Deterministic pairing',
      note2: 'Non-crossing paths!',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '3. Retrain on (x̂₀, z)',
      head: 'Reflowed Velocity',
      eq: 'E‖u_θ − (z − x̂₀)‖²',
      note1: 'Nearly straight paths',
      note2: 'Inherits teacher errors',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Reflow straightens paths by replacing independent pairs with ODE-coupled pairs' }
  ),

  'ch17-diff-1d-flow-map': () => compare2(
    '1D Gaussian Data N(m = 2, ς² = 0.5²): 1-Step Euler vs. Flow Map',
    {
      head: '1-Step Euler Jump (Instantaneous)',
      sub: 'Uses tangent at t = 1: u₁(x) = x − 2',
      rows: [
        'x̂₀ = x₁ − u₁(x₁) = x₁ − (x₁ − 2) = 2',
        'Maps every x₁ ~ N(0, 1) to constant 2!',
        'Output variance = 0 (total collapse)',
      ],
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      head: '1-Step Flow Map (Average Velocity)',
      sub: 'Uses ū(x, 0, 1) = (1 − ς)x − m = 0.5x − 2',
      rows: [
        'Φ_{1→0}(x₁) = 2 + 0.5 x₁',
        'Preserves standardized z-score!',
        'Exact data N(2, 0.5²) in 1 step!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Average velocity ū(x_t, s, t) accounts for trajectory curvature over [s, t]' }
  ),

  'ch17-diff-three-flowmap-ids': () => stack3(
    'Three Invariance Identities Pinning Down the Flow Map Φ_{t→s}(x_t)',
    {
      badge: '1. Compositional',
      head: 'Semigroup Rule: Φ_{r→s}( Φ_{t→r}(x_t) ) = Φ_{t→s}(x_t)',
      sub: 'Two student passes at midpoint r; no derivatives needed',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      badge: '2. Lagrangian',
      head: 'Moving Endpoint s: ∂_{s} Φ_{t→s}(x_t) = u_s( Φ_{t→s} )',
      sub: 'Evaluates velocity u_s at predicted landing point Φ_{t→s}(x_t)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      badge: '3. Eulerian (JVP)',
      head: 'Moving Start t: ∂_{t} Φ_{t→s} + (∇_x Φ_{t→s}) u_t(x_t) = 0',
      sub: 'Evaluates u_t at start x_t via 1 JVP (Consistency & MeanFlow)',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    'With boundary condition Φ_{t→t}(x) = x (ū(x, t, t) = u_t(x)), each identity is exact'
  ),

  'ch17-diff-consistency-renoise': () => compare2(
    'Endpoint Consistency Models (s = 0) vs. Two-Time Flow Maps (s ≤ t)',
    {
      head: 'Consistency Model f_θ(x_t, t)',
      sub: 'Only jumps all the way to clean s = 0',
      rows: [
        '1st step: x̂₀ = f_θ(x₁, 1)',
        '2nd step MUST inject fresh noise ε\':',
        'x_{0.5} = 0.5 x̂₀ + 0.5 ε\' (stochastic)',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'Two-Time Flow Map ū_θ(x_t, s, t)',
      sub: 'Jumps between any times t → r → s',
      rows: [
        '1st jump: x_{0.5} = Φ_{1 → 0.5}(x₁)',
        '2nd jump: x₀ = Φ_{0.5 → 0}(x_{0.5})',
        'Purely deterministic ODE trajectory!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Two-time flow maps preserve the deterministic bijection z ↔ x₀ across jumps' }
  ),

  'ch17-diff-meanflow-jvp': () => pipeline3(
    'Deriving the Two-Time MeanFlow Identity via Product Rule',
    {
      tag: '1. Integral Form',
      head: 'Displacement on [s, t]',
      eq: '(t−s) ū = ∫ u_τ dτ',
      note1: 'Along trajectory x_τ',
      note2: 'Hold s fixed, vary t',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Differentiate in t',
      head: 'Product & Chain Rule',
      eq: 'ū + (t−s)dū/dt = u_t',
      note1: 'dū/dt = ∂_{t} ū + (∇ū)u_t',
      note2: '1 forward-mode JVP!',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. MeanFlow Target',
      head: 'Fixed-Point Identity',
      eq: 'u_t − (t−s) dū/dt',
      note1: 'Exact at s = t: ū = u_t',
      note2: 'Wrap target in sg[·]!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Use slow time frequencies ω ≤ 1 so ∂_{t} sin(ωt) = ω cos(ωt) avoids JVP noise' }
  ),

  'ch17-diff-unified-fm-loss': () => matrix2x2(
    'Four Few-Step Paradigms Unified by One Loss L_FM(θ)',
    {
      head: 'Consistency Distillation (CD)',
      eq: 'u_ref = Teacher û_φ,  s = 0',
      sub: 'Low-variance target; needs re-noising',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Flow-Map Distillation',
      eq: 'u_ref = Teacher û_φ,  0 ≤ s ≤ t',
      sub: 'Deterministic jumps; preserves spread!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      head: 'Consistency Training (CT)',
      eq: 'u_ref = ε − x₀ (No Teacher), s = 0',
      sub: 'Standalone training; high JVP variance',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: 'MeanFlow (Teacher-Free)',
      eq: 'u_ref = ε − x₀,  0 ≤ s ≤ t',
      sub: 'No teacher run, but contracts spread 25%',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'Stop-gradient sg[u_ref − (t−s)dū/dt] is mandatory so gradients do not collapse ∇_x ū_θ'
  ),

  'ch17-diff-dmd-two-scores': () => pipeline3(
    'Distribution Matching Distillation (DMD): Two Score Networks',
    {
      tag: '1-Step Generator',
      head: 'Noise to Image',
      eq: 'x̂₀ = g_θ(z) ~ q_θ',
      note1: 'Corrupt to time t:',
      note2: 'x_t = (1−t)x̂₀ + tε',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Score Difference',
      head: 'Fake vs. Real Score',
      eq: 's_fake − s_real',
      note1: 'Online fake score net',
      note2: '− Frozen teacher score',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'KL Gradient Step',
      head: 'Matches Marginals',
      eq: '∇_θ KL(q_{θ,t}‖p_t)',
      note1: 'DMD2 adds GAN critic',
      note2: 'Loses ODE bijection',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'DMD pushes 1-step samples along ∇log(p_t/q_{θ,t})—the optimal critic gradient!' }
  ),

  // ── discrete-diffusion.md (8 diagrams) ──────────────────────────────
  'ch17-diff-discrete-motivation': () => compare2(
    'Autoregressive Left-to-Right (GPT) vs. Masked Discrete Diffusion',
    {
      head: 'Autoregressive (GPT)',
      sub: 'Causal chain: ∏_{i} p(x^i | x^{<i})',
      rows: [
        'Fixed left-to-right causal order',
        'Reveals 1 token per forward pass',
        '1,024 tokens ⇒ 1,024 sequential calls',
      ],
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    {
      head: 'Masked Discrete Diffusion',
      sub: 'Absorbing chain: x_0 → ... → [M]^L',
      rows: [
        'Bidirectional attention over all L sites',
        'Unmasks ≈ L/S tokens per step',
        'Any-order generation & infilling!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Discrete tokens have no Euclidean x₀ + σε; corruption uses transition matrices Q_t' }
  ),

  'ch17-diff-transition-matrix': () => pipeline3(
    'Discrete Markov Forward Chains via Row-Stochastic Matrices Q_t',
    {
      tag: '1-Step Transition',
      head: 'Row-Stochastic Q_t',
      eq: 'Cat(x_{t−1} Q_t)',
      note1: 'One-hot row x ∈ {0,1}^{K}',
      note2: '[Q_t]_{ij} = P(i → j)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 't-Step Marginal',
      head: 'Matrix Product Q̄_t',
      eq: 'Q̄_t = Q₁ Q₂ ··· Q_t',
      note1: 'q(x_t | x₀) = x₀ Q̄_t',
      note2: 'Sample x_t in O(1)!',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Forward Posterior',
      head: 'Bayes Elementwise ⊙',
      eq: 'x_{t}Q_{t}ᵀ ⊙ x₀Q̄_{t−1}',
      note1: '÷ Z, Z = x₀ Q̄_t x_tᵀ',
      note2: 'Exact categorical dist',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Matrix multiplication Q̄_t = Q₁···Q_t replaces Gaussian variance addition' }
  ),

  'ch17-diff-binary-flip-bayes': () => pipeline3(
    'Two-Step Binary Flip Posterior: x₀ = (1, 0) → x₁ → x₂ = (0, 1)',
    {
      tag: 'Forward Prior',
      head: 'Step 1: x₀ Q̄₁',
      eq: '(0.9,  0.1)',
      note1: '90% stayed 0 at t=1',
      note2: '10% flipped to 1',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: 'Backward Likelihood',
      head: 'Step 2: x₂ Q₂ᵀ',
      eq: '(0.1,  0.9)',
      note1: 'Prob of reaching x₂=(0,1)',
      note2: 'from state 0 vs. state 1',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: 'Normalized Posterior',
      head: 'Hadamard Product ⊙',
      eq: '(0.50,  0.50)',
      note1: '(0.09, 0.09) / 0.18',
      note2: 'Flip equally likely at 1,2',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Under uniform flips, even visible tokens carry uncertainty about when/if they flipped' }
  ),

  'ch17-diff-absorbing-posterior': () => compare2(
    'Two-Point Posterior of the Absorbing (Masking) Chain',
    {
      head: 'Case 1: Visible Token (x_t^i ≠ [M])',
      sub: 'Token survived all t steps unmasked',
      rows: [
        'Absorbing [M] never turns back to word',
        'Guaranteed 100% clean: x_t^i = x₀ⁱ!',
        'Reverse copies x_{t−1}^{i} = x_t^i (KL=0)',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Case 2: Masked Token (x_t^i = [M])',
      sub: 'Only two predecessors: x₀ⁱ or [M]',
      rows: [
        'Unmask to x₀ⁱ with prob w_t',
        'Stay [M] with prob 1 − w_t',
        'Network predicts p_θ(x₀ⁱ | x_t)!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Visible tokens are never revised; all learning and sampling happen at masked sites' }
  ),

  'ch17-diff-masked-elbo-ce': () => pipeline3(
    'Why the Masked Diffusion ELBO Simplifies to Weighted Cross-Entropy',
    {
      tag: '1. Prior & Visible Sites',
      head: 'Vanish Identically!',
      eq: 'L_T = 0,  KL_{vis} = 0',
      note1: 'At t=T: all [M] (det.)',
      note2: 'Visible tokens copy',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      tag: '2. Masked Site KL',
      head: 'Stay Prob Cancels',
      eq: '−w_t ln p_θ(x₀|x_t)',
      note1: '(1 − w_t) term = 0',
      note2: 'w_t = (ᾱ_{t−1}−ᾱ_t)/(1−ᾱ_t)',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      tag: '3. Linear Schedule',
      head: 'ᾱ_t = 1 − t / T',
      eq: 'w_t = 1 / t',
      note1: 'E[# masks] = L · (t/T)',
      note2: 'Weighted count = L / T!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Every step t contributes the exact same expected weighted mask count L / T' }
  ),

  'ch17-diff-time-agnostic': () => compare2(
    'Continuous-Time Limit (T → ∞) and Time-Agnostic Optimality',
    {
      head: 'Schedule-Invariant Integral',
      sub: 'Change variable to mask rate u = 1 − ᾱ(τ)',
      rows: [
        'L_VLB → ∫₀¹ (1 / u) E_u[ℓ(x₀, x)] du',
        'Jacobian −ᾱ\'(τ) dτ = du cancels ᾱ(τ)!',
        'Continuous ELBO is schedule-invariant',
      ],
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: 'Time-Agnostic Bayes Denoiser',
      sub: 'Why optimal p*(x₀ⁱ | x_t) ignores time t',
      rows: [
        'Mask coin flips are independent of x₀',
        'x_t already shows which sites are [M]',
        'Step index t adds 0 bits of info!',
      ],
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    { text: 'Unlike Gaussian noise, discrete masks [M] explicitly tag which tokens are corrupted' }
  ),

  'ch17-diff-parallel-indep-error': () => pipeline3(
    'The Parallel Independence Error on a Token Pair (X¹, X²) ~ p',
    {
      tag: 'Different Jumps',
      head: 'Prob = 1 − 1 / S',
      eq: 'X¹ first, then X²|X¹',
      note1: 'Chain rule of prob',
      note2: 'Exact draw from p!',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      tag: 'Same Jump Collision',
      head: 'Prob = S·(1/S)² = 1/S',
      eq: '([M],[M]) → (X¹, X²)',
      note1: 'Unmasked simultaneously',
      note2: 'Drawn from p¹ ⊗ p²!',
      fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    },
    {
      tag: 'Exact Output Mixture',
      head: '1/S Error Law',
      eq: 'TV = 0.314 / S',
      note1: 'Mixes p and p¹ ⊗ p²',
      note2: 'ELBO floor: H + I / T',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    { text: 'Simultaneous unmasking draws from the product of marginals p¹ ⊗ p²' }
  ),

  'ch17-diff-sequence-unification': () => matrix2x2(
    'Four Sequence Paradigms as Special Cases of Masked Diffusion',
    {
      head: '1. Autoregressive (GPT)',
      eq: 'Causal Prefix · 1 Token / Call',
      sub: 'L calls; exact joint factorization (TV = 0)',
      fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    },
    {
      head: '2. Order-Agnostic AR',
      eq: 'u ~ U(0, 1) · 1 Token / Call',
      sub: 'Random permutation σ; TV = 0 in L calls',
      fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    },
    {
      head: '3. Masked Diffusion (MDLM / LLaDA)',
      eq: 'u ~ U(0, 1) · L/S Tokens / Call',
      sub: 'S ≪ L parallel jumps; O(1/S) indep. error',
      fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    },
    {
      head: '4. Masked LM (BERT)',
      eq: 'Fixed Single Mask Rate u = 0.15',
      sub: '1 time slice; cannot generate from u = 1.0',
      fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    },
    'Masked diffusion trains across all u ∈ (0, 1] and trades step count S for parallel speed'
  ),
};
