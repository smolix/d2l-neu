// Slide diagrams for Chapter 35 (MDL Probability and Statistics): ch35-mdl-prob.mjs
import { C, TOKENS, FS, FM, grid, tx, baselineY, arrow, block, chip, rc, svg } from "./engine.mjs";

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const circ = (cx, cy, r, fill, stroke, sw = 2, dash = false) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const path = (d, stroke, fill = "none", sw = 2.5, dash = false) =>
  `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// ============================================================================
// 0. index.md (4 diagrams)
// ============================================================================

function coinThread() {
  const W = 540, H = 280;
  let o = "";
  o += box(20, 14, 500, 48, C.lamber, C.amber, 8, 2);
  o += tx(270, 29, "Running Coin Benchmark: 9 Heads in 13 Flips", { fs: 15, fw: 700, fill: D.orange.dark });
  o += tx(270, 49, "Empirical bias θ̂ = 9/13 ≈ 0.692  (Test accuracy = 1 flip/example)", { mono: true, fs: 12, fw: 600, fill: C.ink });

  // Left: Forward Probability
  o += box(20, 82, 234, 110, C.lblue, C.blue, 8, 2);
  o += tx(137, 104, "Forward: Probability (§36.1–4)", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(137, 128, "Known parameter θ = 0.5", { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(137, 150, "→ P(X ≥ 9 | θ=0.5) = 13.3%", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(137, 174, "Predicts outcome frequencies", { fs: 12, fill: C.muted });

  // Right: Inverse Statistics
  o += box(286, 82, 234, 110, C.lgreen, C.green, 8, 2);
  o += tx(403, 104, "Inverse: Statistics (§36.5–8)", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(403, 128, "Observed data D: 9H, 4T", { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(403, 150, "→ θ̂ = 0.69 ± 0.13, Beta(10,5)", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(403, 174, "Infers parameters & error bars", { fs: 12, fill: C.muted });

  o += arrow(256, 125, 284, 125, C.purple);
  o += arrow(284, 150, 256, 150, C.purple);

  // Bottom bar: Two distinct error bars
  o += box(20, 208, 500, 58, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 226, "Frequentist Sampling Spread vs. Bayesian Posterior Spread", { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(270, 248, "CI: scatter across undrawn datasets  |  Credible: p(θ | D) given data", { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function fourPillars() {
  const W = 540, H = 290;
  let o = "";
  const cards = [
    { x: 18, y: 16, w: 242, h: 116, fill: C.lblue, st: C.blue, dk: D.blue.dark,
      t: "1. Distributions (§36.1–4)", s1: "PDFs, Jacobians, Exp Family", s2: "h(x) exp(ηᵀT(x) − A(η))", s3: "Payoff: Flows & GLM gradients" },
    { x: 280, y: 16, w: 242, h: 116, fill: C.lamber, st: C.amber, dk: D.orange.dark,
      t: "2. Estimation & MLE (§36.5–6)", s1: "Bias–Var, p-values, Power, CIs", s2: "I(θ) = Var(s) = E[−∇²log p]", s3: "Payoff: Losses, SEs & L2/L1" },
    { x: 18, y: 156, w: 242, h: 116, fill: D.purple.tint, st: C.purple, dk: D.purple.dark,
      t: "3. Latents & Bayes (§36.7–8)", s1: "log p(x) = ELBO(q) + KL(q||p)", s2: "EM, Laplace, IS, MCMC, VI", s3: "Payoff: GMMs, VAEs & posteriors" },
    { x: 280, y: 156, w: 242, h: 116, fill: C.lgreen, st: C.green, dk: D.green.dark,
      t: "4. Guarantees (§36.9–11)", s1: "Chernoff, Hoeffding, Thin Shell", s2: "Rademacher R_n(F) & SVD", s3: "Payoff: Bounds & Naive Bayes" },
  ];
  cards.forEach(c => {
    o += box(c.x, c.y, c.w, c.h, c.fill, c.st, 9, 2);
    o += tx(c.x + c.w / 2, c.y + 22, c.t, { fs: 13.5, fw: 700, fill: c.dk });
    o += tx(c.x + c.w / 2, c.y + 48, c.s1, { fs: 12.5, fw: 600, fill: C.ink });
    o += tx(c.x + c.w / 2, c.y + 72, c.s2, { mono: true, fs: 11.5, fw: 600, fill: c.dk });
    o += tx(c.x + c.w / 2, c.y + 96, c.s3, { fs: 12, fw: 600, fill: C.muted });
  });
  o += arrow(262, 74, 278, 74, C.blue);
  o += arrow(401, 134, 139, 154, C.amber);
  o += arrow(262, 214, 278, 214, C.purple);
  return svg(W, H, o);
}

function roadmapP1() {
  const W = 540, H = 294;
  let o = "";
  const rows = [
    ["§36.1 Random Vars", "F'=p, F⁻¹(U), Markov/Chebyshev", "Dart p(r)=2r; Cauchy", C.lblue, C.blue, D.blue.dark],
    ["§36.2 Joint Dist", "Marginals, Eve's law, |det J|⁻¹", "Dart Cov=0; Log-normal", C.lblue, C.blue, D.blue.dark],
    ["§36.3 Distributions", "Poisson limit, CLT, MVN cond.", "9/13 coin; λ=4 server", C.lblue, C.blue, D.blue.dark],
    ["§36.4 Exp Family", "∇A=E[T], ∇²A=Cov(T), MaxEnt", "Loaded die; Gumbel-max", D.teal.tint, D.teal.base, D.teal.dark],
    ["§36.5 Statistics", "MSE=Bias²+Var, power, bootstrap", "912 vs 921/1000 paired", C.lamber, C.amber, D.orange.dark],
    ["§36.6 Max Lik", "CE=H+KL, Fisher I(θ), MAP", "Coin NLL; L2/L1 priors", C.lamber, C.amber, D.orange.dark],
    ["§36.7–8 EM & Bayes", "ELBO+KL=log p; Laplace, MCMC", "500-pt GMM; 20-pt LR", D.purple.tint, C.purple, D.purple.dark],
  ];
  rows.forEach(([sec, thm, ex, fill, st, dk], i) => {
    const y = 10 + i * 40;
    o += box(14, y, 138, 34, fill, st, 6, 1.8);
    o += tx(83, y + 17, sec, { mono: true, fs: 11.5, fw: 700, fill: dk });
    o += box(158, y, 224, 34, "#FFFFFF", st, 6, 1.5);
    o += tx(270, y + 17, thm, { fs: 11.5, fw: 600, fill: C.ink });
    o += box(388, y, 138, 34, N.panel, C.gray, 6, 1.5);
    o += tx(457, y + 17, ex, { fs: 11, fw: 600, fill: C.muted });
  });
  return svg(W, H, o);
}

function roadmapP2() {
  const W = 540, H = 276;
  let o = "";
  const stages = [
    { y: 16, t: "§36.9 Concentration Inequalities", sub: "Chernoff e^{λX} → Hoeffding 2exp(−2nt²) · Thin shell at radius √d", fill: C.lgreen, st: C.green, dk: D.green.dark },
    { y: 102, t: "§36.10 Uniform Convergence & Generalization", sub: "Union bound √(log|F|/n) · Rademacher R_n(G) · SVD double descent", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 188, t: "§36.11 Naive Bayes Synthesis Case Study", sub: "784 binary pixels · Laplace Beta(1,1) smoothing · Calibration ECE", fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  stages.forEach((s, i) => {
    o += box(20, s.y, 500, 66, s.fill, s.st, 8, 2);
    o += tx(270, s.y + 22, s.t, { fs: 14.5, fw: 700, fill: s.dk });
    o += tx(270, s.y + 46, s.sub, { fs: 12, fw: 600, fill: C.ink });
    if (i < 2) o += arrow(270, s.y + 68, 270, stages[i + 1].y - 2, s.st);
  });
  return svg(W, H, o);
}

// ============================================================================
// 1. mdl-random-variables.md (8 diagrams)
// ============================================================================

function dartBins() {
  const W = 540, H = 266;
  let o = "";
  // Left: concentric rings on unit disk around r = 0.5
  const cx = 124, cy = 128, R = 90;
  o += circ(cx, cy, R, N.panel, C.gray, 2);
  o += circ(cx, cy, R * 0.55, C.lblue, C.blue, 1.5);
  o += circ(cx, cy, R * 0.45, "#FFFFFF", C.blue, 1.5);
  o += circ(cx, cy, R * 0.50, "none", D.orange.base, 2, true);
  o += circ(cx, cy, 3, C.ink, C.ink, 1);
  o += tx(cx, cy - R * 0.5 - 12, "r = 0.5 ring", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(cx, 240, "Unit Disk (Area = π)", { fs: 13.5, fw: 700, fill: C.ink });

  // Right: Shrinking bins table-cards keeping Prob/width = 1
  const rx = 246;
  o += tx(384, 24, "Shrinking Bins Around r = 0.5", { fs: 15, fw: 700, fill: D.blue.dark });
  const bins = [
    ["[0.45, 0.55]", "ε = 0.1", "P = 0.1", "P/ε = 1", C.lblue, C.blue],
    ["[0.495, 0.505]", "ε = 0.01", "P = 0.01", "P/ε = 1", C.lblue, C.blue],
    ["[0.4995, 0.5005]", "ε = 0.001", "P = 0.001", "P/ε = 1", C.lgreen, C.green],
    ["Exact point R = 0.5", "ε = 0", "P = 0", "lim P/ε = p(0.5) = 1", C.lamber, C.amber],
  ];
  bins.forEach(([b, w, p, r, fill, st], i) => {
    const y = 42 + i * 52;
    o += box(rx, y, 276, 44, fill, st, 7, 1.8);
    o += tx(rx + 138, y + 14, `${b}   (${w})`, { mono: true, fs: 12.5, fw: 700, fill: C.ink });
    o += tx(rx + 138, y + 32, `${p}   →   ${r}`, { mono: true, fs: 12.5, fw: 700, fill: i === 3 ? D.orange.dark : D.blue.dark });
  });
  return svg(W, H, o);
}

function dartDensity() {
  const W = 540, H = 266;
  let o = "";
  // Axes for p(r) = 2r on [0, 1]
  const x0 = 56, y0 = 212, gw = 200, gh = 160;
  // Shaded triangle under p(r) = 2r
  o += `<polygon points="${x0},${y0} ${x0 + gw},${y0 - gh} ${x0 + gw},${y0}" fill="${C.lblue}" stroke="none"/>`;
  // Shaded strip [r, r+eps]
  const rx1 = x0 + 0.6 * gw, rx2 = x0 + 0.72 * gw;
  const ry1 = y0 - 0.6 * gh, ry2 = y0 - 0.72 * gh;
  o += `<polygon points="${rx1},${y0} ${rx1},${ry1} ${rx2},${ry2} ${rx2},${y0}" fill="${C.lamber}" stroke="${C.amber}" stroke-width="1.5"/>`;
  o += line(x0, y0, x0 + gw + 18, y0, C.ink, 2);
  o += line(x0, y0, x0, y0 - gh - 18, C.ink, 2);
  o += line(x0, y0, x0 + gw, y0 - gh, C.blue, 3);
  o += tx(x0 + gw / 2, 24, "Dart Radial Density p(r) = 2r", { fs: 15, fw: 700, fill: D.blue.dark });
  o += tx(x0 - 22, y0 - gh, "2.0", { mono: true, fs: 12.5, fill: C.ink });
  o += tx(x0 - 22, y0 - gh / 2, "1.0", { mono: true, fs: 12.5, fill: C.ink });
  o += tx(x0, y0 + 20, "0", { mono: true, fs: 12.5, fill: C.ink });
  o += tx(x0 + gw * 0.66, y0 + 20, "r", { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw, y0 + 20, "1.0", { mono: true, fs: 12.5, fill: C.ink });
  o += tx(x0 + gw - 36, y0 - gh - 12, "p(r) = 2r", { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });

  // Right panel: Why density can exceed 1
  const bx = 290;
  o += box(bx, 34, 232, 92, C.lamber, C.amber, 8, 2);
  o += tx(bx + 116, 56, "Strip Area ≈ ε · p(r)", { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(bx + 116, 80, "P(r ≤ R < r+ε) = (r+ε)² − r²", { mono: true, fs: 12, fill: C.ink });
  o += tx(bx + 116, 102, "Ratio (2rε + ε²)/ε → 2r", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });

  o += box(bx, 142, 232, 96, C.lgreen, C.green, 8, 2);
  o += tx(bx + 116, 164, "Density Is a Rate, Not a Prob!", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(bx + 116, 188, "At r = 0.9: p(0.9) = 1.8 > 1", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(bx + 116, 212, "Total area ∫₀¹ 2r dr = 1.0", { mono: true, fs: 12.5, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
}

function mixedRelu() {
  const W = 540, H = 266;
  let o = "";
  const panels = [
    { x: 16, t: "1. Purely Discrete", sub: "100% mass on atoms", ex: "Class label Y ∈ {1..K}", col: C.blue, bg: C.lblue, dk: D.blue.dark },
    { x: 194, t: "2. Purely Continuous", sub: "0 atoms; smooth p(x)", ex: "Weights W ~ N(0, 1)", col: C.green, bg: C.lgreen, dk: D.green.dark },
    { x: 372, t: "3. Mixed (ReLU)", sub: "Atom P(Y=0)=½ + PDF", ex: "Y = max(0, X)", col: C.amber, bg: C.lamber, dk: D.orange.dark },
  ];
  panels.forEach((p, idx) => {
    o += box(p.x, 16, 152, 234, "#FFFFFF", p.col, 8, 2);
    o += box(p.x, 16, 152, 46, p.bg, p.col, 8, 2);
    o += tx(p.x + 76, 36, p.t, { fs: 13.5, fw: 700, fill: p.dk });
    o += tx(p.x + 76, 52, p.sub, { fs: 11.5, fw: 600, fill: C.ink });
    const ay = 186, ax1 = p.x + 18, ax2 = p.x + 134, mid = p.x + 76;
    o += line(ax1, ay, ax2, ay, C.ink, 1.8);
    if (idx === 0) {
      [ax1 + 20, mid, ax2 - 20].forEach((sx, j) => {
        const h = [48, 76, 34][j];
        o += line(sx, ay, sx, ay - h, C.blue, 3.5);
        o += circ(sx, ay - h, 4.5, C.blue, C.blue, 1);
      });
    } else if (idx === 1) {
      let d = `M ${ax1} ${ay - 4}`;
      for (let s = 0; s <= 24; s++) {
        const u = (s / 24) * 4 - 2;
        const px = ax1 + (s / 24) * (ax2 - ax1);
        const py = ay - 78 * Math.exp(-0.5 * u * u);
        d += ` L ${px.toFixed(1)} ${py.toFixed(1)}`;
      }
      o += path(d + ` L ${ax2} ${ay} Z`, "none", C.lgreen, 0);
      o += path(d, C.green, "none", 2.5);
    } else {
      // Atom at 0 + right half Gaussian
      const xZero = p.x + 44;
      let d = `M ${xZero} ${ay - 72}`;
      for (let s = 1; s <= 18; s++) {
        const u = (s / 18) * 2.2;
        const px = xZero + (s / 18) * (ax2 - xZero);
        const py = ay - 72 * Math.exp(-0.5 * u * u);
        d += ` L ${px.toFixed(1)} ${py.toFixed(1)}`;
      }
      o += path(d + ` L ${ax2} ${ay} L ${xZero} ${ay} Z`, "none", C.lamber, 0);
      o += path(d, C.amber, "none", 2.5);
      o += arrow(xZero, ay, xZero, ay - 92, D.red.base);
      o += tx(xZero + 38, ay - 94, "Atom 0.5", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
    }
    o += tx(p.x + 76, 212, p.ex, { mono: true, fs: 11.5, fw: 600, fill: p.dk });
    o += tx(p.x + 76, 232, idx === 2 ? "E[Y²] = 1/2 (He init)" : idx === 0 ? "Σ P(X=xᵢ) = 1" : "P(X=x) = 0 ∀x", { fs: 11.5, fill: C.muted });
  });
  return svg(W, H, o);
}

function cdfTypes() {
  const W = 540, H = 264;
  let o = "";
  const panels = [
    { x: 16, t: "Continuous CDF", sub: "Dart F(r) = r²", col: C.blue, dk: D.blue.dark },
    { x: 194, t: "Discrete Staircase", sub: "Coin jumps at 0, 1", col: C.amber, dk: D.orange.dark },
    { x: 372, t: "Mixed ReLU CDF", sub: "Jump ½ + Φ(x) climb", col: C.purple, dk: D.purple.dark },
  ];
  panels.forEach((p, idx) => {
    o += box(p.x, 16, 152, 232, "#FFFFFF", p.col, 8, 2);
    o += tx(p.x + 76, 36, p.t, { fs: 13.5, fw: 700, fill: p.dk });
    o += tx(p.x + 76, 54, p.sub, { mono: true, fs: 11.5, fw: 600, fill: C.muted });
    const ax1 = p.x + 24, ax2 = p.x + 136, ay0 = 196, ay1 = 92;
    o += line(ax1, ay0, ax2, ay0, C.gray, 1.5);
    o += line(ax1, ay1, ax2, ay1, N.hairline, 1.5, true);
    o += tx(ax1 - 10, ay0, "0", { mono: true, fs: 11.5, fill: C.muted });
    o += tx(ax1 - 10, ay1, "1", { mono: true, fs: 11.5, fill: C.muted });
    if (idx === 0) {
      let d = `M ${ax1} ${ay0}`;
      for (let s = 1; s <= 20; s++) {
        const u = s / 20;
        const px = ax1 + u * (ax2 - ax1);
        const py = ay0 - (u * u) * (ay0 - ay1);
        d += ` L ${px.toFixed(1)} ${py.toFixed(1)}`;
      }
      o += path(d, C.blue, "none", 2.8);
      o += tx(p.x + 76, 222, "Smooth, no jumps", { fs: 12, fw: 600, fill: D.blue.dark });
    } else if (idx === 1) {
      const xA = ax1 + 32, xB = ax1 + 78, yMid = (ay0 + ay1) / 2;
      o += line(ax1, ay0, xA, ay0, C.amber, 2.8);
      o += line(xA, yMid, xB, yMid, C.amber, 2.8);
      o += line(xB, ay1, ax2, ay1, C.amber, 2.8);
      o += circ(xA, yMid, 4, C.amber, C.amber, 1.5);
      o += circ(xB, ay1, 4, C.amber, C.amber, 1.5);
      o += tx(p.x + 76, 222, "Right-continuous jumps", { fs: 12, fw: 600, fill: D.orange.dark });
    } else {
      const xZero = ax1 + 44, yMid = (ay0 + ay1) / 2;
      o += line(ax1, ay0, xZero, ay0, C.purple, 2.8);
      let d = `M ${xZero} ${yMid}`;
      for (let s = 1; s <= 16; s++) {
        const u = s / 16;
        const px = xZero + u * (ax2 - xZero);
        const py = yMid - (1 - Math.exp(-2.4 * u)) * (yMid - ay1);
        d += ` L ${px.toFixed(1)} ${py.toFixed(1)}`;
      }
      o += path(d, C.purple, "none", 2.8);
      o += circ(xZero, yMid, 4, C.purple, C.purple, 1.5);
      o += tx(p.x + 76, 222, "F(0) = 0.5 jump at 0", { fs: 12, fw: 600, fill: D.purple.dark });
    }
  });
  return svg(W, H, o);
}

function meanLotus() {
  const W = 540, H = 260;
  let o = "";
  // Top: Dart density balance beam
  o += box(18, 14, 504, 122, "#FFFFFF", C.blue, 8, 2);
  o += tx(154, 34, "Balance Point on Dart p(r) = 2r", { fs: 13.5, fw: 700, fill: D.blue.dark });
  const x0 = 44, y0 = 104, gw = 200, gh = 54;
  o += `<polygon points="${x0},${y0} ${x0 + gw},${y0 - gh} ${x0 + gw},${y0}" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`;
  const xMean = x0 + (2 / 3) * gw;
  const xMed = x0 + 0.707 * gw;
  o += line(xMed, y0, xMed, y0 - 0.707 * gh, D.purple.base, 2, true);
  // Fulcrum triangle at E[R] = 2/3
  o += `<polygon points="${xMean},${y0 + 2} ${xMean - 9},${y0 + 18} ${xMean + 9},${y0 + 18}" fill="${D.orange.base}" stroke="${D.orange.dark}" stroke-width="1.5"/>`;
  o += tx(x0, y0 + 18, "0", { mono: true, fs: 12, fill: C.ink });
  o += tx(xMean - 18, y0 + 26, "μ = 2/3", { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw + 18, y0 + 18, "1.0", { mono: true, fs: 12, fill: C.ink });

  // Legend notes inside top right
  o += box(284, 28, 222, 92, C.lamber, C.amber, 7, 1.5);
  o += tx(395, 48, "Mean vs. Median on Disk", { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(395, 72, "Mean E[R] = 2/3 ≈ 0.667", { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(395, 96, "Median F⁻¹(0.5) ≈ 0.707", { mono: true, fs: 11.5, fw: 600, fill: D.purple.dark });

  // Bottom: LOTUS pipeline
  o += box(18, 152, 504, 92, C.lgreen, C.green, 8, 2);
  o += tx(270, 174, "Law of the Unconscious Statistician (LOTUS)", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(270, 198, "E[g(X)] = ∫ g(x) p(x) dx  (weight g(x) directly by p(x)!)", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(270, 222, "No need to derive transformed density p_Y(y) of Y = g(X) first", { fs: 12, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
}

function varAffine() {
  const W = 540, H = 264;
  let o = "";
  // Top row: Original X with mean mu and spread sigma
  o += box(20, 16, 500, 104, C.lblue, C.blue, 8, 2);
  o += tx(146, 38, "Original Variable X", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(146, 62, "Var(X) = E[X²] − μ² = σ²", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(146, 88, "Dart: E[R²]=1/2, μ=2/3 → Var=1/18", { mono: true, fs: 11.5, fill: C.muted });

  // Mini bell & spread bar on right of top box
  const bx = 360, by = 82;
  o += line(280, by, 480, by, C.ink, 1.8);
  o += box(bx - 30, by - 36, 60, 36, "#FFFFFF", C.blue, 6, 2);
  o += tx(bx, by - 18, "±σ", { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(bx, by + 18, "μ", { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  // Bottom row: Affine map cX + d
  o += box(20, 140, 500, 108, C.lamber, C.amber, 8, 2);
  o += tx(150, 162, "Affine Map Y = cX + d", { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(150, 188, "Shift d drops out; scale c squares:", { fs: 12, fw: 600, fill: C.ink });
  o += tx(150, 214, "Var(cX+d) = c²Var(X), σ_Y = |c|σ_X", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });

  // Stretched & shifted bar on right of bottom box
  const bx2 = 400, by2 = 208;
  o += line(280, by2, 504, by2, C.ink, 1.8);
  o += box(bx2 - 64, by2 - 36, 128, 36, "#FFFFFF", C.amber, 6, 2);
  o += tx(bx2, by2 - 18, "±|c|σ (scaled)", { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(bx2, by2 + 18, "cμ + d (shifted)", { mono: true, fs: 12, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function markovStep() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 54, y0 = 208, gw = 230, gh = 160;
  const xt = x0 + 130, yt = y0 - 90;
  // Shaded step rectangle t * 1_{x >= t}
  o += box(xt, yt, x0 + gw - xt, y0 - yt, C.lamber, C.amber, 0, 2);
  // Axes
  o += line(x0, y0, x0 + gw + 16, y0, C.ink, 2);
  o += line(x0, y0, x0, y0 - gh - 14, C.ink, 2);
  // Line y = x
  o += line(x0, y0, x0 + gw, y0 - gh, C.blue, 3);
  o += tx(x0 + gw - 36, y0 - gh - 12, "y = x", { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(xt + 48, yt + 44, "t · 1{x ≥ t}", { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(x0, y0 + 20, "0", { mono: true, fs: 12.5, fill: C.ink });
  o += tx(xt, y0 + 20, "t", { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(x0 - 18, yt, "t", { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw / 2, 22, "Markov's Step Majorant (x ≥ 0)", { fs: 15, fw: 700, fill: D.blue.dark });

  // Right explanation cards
  const rx = 308;
  o += box(rx, 28, 214, 102, C.lblue, C.blue, 8, 2);
  o += tx(rx + 107, 50, "Pointwise Inequality", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(rx + 107, 74, "X ≥ t · 1{X ≥ t}  always", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(rx + 107, 100, "⇒ E[X] ≥ t · P(X ≥ t)", { mono: true, fs: 13, fw: 700, fill: D.blue.dark });

  o += box(rx, 144, 214, 96, C.lgreen, C.green, 8, 2);
  o += tx(rx + 107, 166, "Ancestor of Tail Bounds", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(rx + 107, 190, "On (X−μ)² → Chebyshev", { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(rx + 107, 214, "On exp(λX) → Chernoff", { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function cauchyTail() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 36, y0 = 196, gw = 232, gh = 112, mid = x0 + gw / 2;
  // Cauchy vs Gaussian curve
  let dG = "", dC = "";
  for (let s = 0; s <= 40; s++) {
    const u = (s / 40) * 6 - 3;
    const px = x0 + (s / 40) * gw;
    const pyG = y0 - gh * Math.exp(-0.5 * u * u);
    const pyC = y0 - (gh * 0.78) / (1 + u * u);
    dG += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${pyG.toFixed(1)} `;
    dC += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${pyC.toFixed(1)} `;
  }
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  o += path(dG, C.blue, "none", 2.5);
  o += path(dC, D.red.base, "none", 2.8);
  o += tx(mid, 22, "Gaussian vs. Cauchy Tails", { fs: 14.5, fw: 700, fill: C.ink });
  // Place labels well separated horizontally at y = 48 above the curves
  o += tx(x0 + 44, 48, "Gaussian e^{-x²/2}", { mono: true, fs: 10.5, fw: 700, fill: D.blue.dark });
  o += tx(x0 + gw - 42, 48, "Cauchy 1/(π(1+x²))", { mono: true, fs: 10.5, fw: 700, fill: D.red.dark });
  o += tx(mid, 224, "Algebraic 1/x² tail never dies out!", { fs: 12, fw: 700, fill: D.red.dark });

  // Right panel: moment divergence
  const rx = 286;
  o += box(rx, 24, 238, 102, D.red.tint, D.red.base, 8, 2);
  o += tx(rx + 119, 46, "Second Moment E[X²] = ∞", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(rx + 119, 72, "∫ x²p(x)dx = (2/π)(L−arctan L)", { mono: true, fs: 11, fill: C.ink });
  o += tx(rx + 119, 98, "L=10 → 5.4, L=1000 → 636 → ∞", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  o += box(rx, 142, 238, 102, C.lamber, C.amber, 8, 2);
  o += tx(rx + 119, 164, "Mean Does Not Exist!", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(rx + 119, 188, "∫ |x|p(x)dx = (1/π)log(1+L²) → ∞", { mono: true, fs: 11, fill: C.ink });
  o += tx(rx + 119, 214, "Lopsided [-L,2L] → (log 2)/π", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// ============================================================================
// 2. mdl-joint-distributions.md (8 diagrams)
// ============================================================================

function dartThreeQuestions() {
  const W = 540, H = 266;
  let o = "";
  const cx = 118, cy = 142, R = 76;
  o += circ(cx, cy, R, C.lblue, C.blue, 2);
  o += line(cx - R, cy, cx + R, cy, C.gray, 1.5, true);
  o += line(cx, cy - R, cx, cy + R, C.gray, 1.5, true);
  // Chord at Y = 0.75
  const cyChord = cy - 0.75 * R, hw = 0.66 * R;
  o += line(cx - hw, cyChord, cx + hw, cyChord, D.red.base, 3.5);
  o += circ(cx, cyChord, 4.5, D.orange.base, D.orange.dark, 1.5);
  o += tx(cx, 38, "Y = 0.9 ⇒ |X| ≤ 0.44", { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(cx, 242, "Dart on Unit Disk (p = 1/π)", { fs: 13, fw: 700, fill: D.blue.dark });

  const rx = 236;
  const items = [
    ["1. Independent?", "NO — circular rim couples |X| ≤ √(1−Y²)", D.red.tint, D.red.base, D.red.dark],
    ["2. Predictable?", "E[X | Y = y] = 0 at every height y!", C.lamber, C.amber, D.orange.dark],
    ["3. Correlated?", "Cov(X, Y) = 0 and ρ = 0 by symmetry", C.lgreen, C.green, D.green.dark],
  ];
  items.forEach(([t, sub, fill, st, dk], i) => {
    const y = 22 + i * 78;
    o += box(rx, y, 286, 64, fill, st, 8, 2);
    o += tx(rx + 143, y + 22, t, { fs: 14, fw: 700, fill: dk });
    o += tx(rx + 143, y + 46, sub, { fs: 12, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

function jointDisk() {
  const W = 540, H = 264;
  let o = "";
  const cx = 124, cy = 136, R = 78;
  o += box(cx - R, cy - R, 2 * R, 2 * R, N.panel, C.gray, 4, 1.5, true);
  o += circ(cx, cy, R, C.lblue, C.blue, 2);
  // Small epsilon square inside disk
  o += box(cx + 16, cy - 32, 20, 20, C.lamber, C.amber, 2, 2);
  o += tx(cx + 26, cy - 44, "ε×ε", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  // Label p = 0 above the square
  o += tx(cx, 36, "Outside disk: p(x,y) = 0", { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(cx - 18, cy + 20, "p = 1/π", { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(cx, 240, "Bounding Square vs. Unit Disk", { fs: 13, fw: 700, fill: C.ink });

  const rx = 244;
  o += box(rx, 24, 278, 102, C.lblue, C.blue, 8, 2);
  o += tx(rx + 139, 46, "Probability per Unit Area", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(rx + 139, 72, "P(X∈[x,x+ε], Y∈[y,y+ε]) ≈ ε²p(x,y)", { mono: true, fs: 11, fw: 600, fill: C.ink });
  o += tx(rx + 139, 98, "Total volume ∬ p(x,y) dx dy = 1", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += box(rx, 142, 278, 102, D.red.tint, D.red.base, 8, 2);
  o += tx(rx + 139, 164, "Flat ≠ Factorized!", { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(rx + 139, 188, "p(x,y) = (1/π) · 1{x² + y² ≤ 1}", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(rx + 139, 214, "Indicator 1{x²+y²≤1} ≠ a(x)b(y)", { mono: true, fs: 12, fw: 600, fill: D.red.dark });
  return svg(W, H, o);
}

function bayesConvolution() {
  const W = 540, H = 266;
  let o = "";
  // Top: Chain rule & Bayes rule
  o += box(18, 16, 504, 108, C.lblue, C.blue, 8, 2);
  o += tx(270, 38, "Chain Rule & Continuous Bayes' Rule", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += box(30, 56, 218, 52, "#FFFFFF", C.blue, 6, 1.8);
  o += tx(139, 74, "Joint = Cond × Marginal", { fs: 12, fw: 700, fill: C.muted });
  o += tx(139, 94, "p(x,y) = p(x|y)p(y) = p(y|x)p(x)", { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  o += arrow(252, 82, 280, 82, C.blue);
  o += box(284, 56, 224, 52, "#FFFFFF", C.purple, 6, 1.8);
  o += tx(396, 74, "Bayes Inversion", { fs: 12, fw: 700, fill: D.purple.dark });
  o += tx(396, 94, "p(x|y) = p(y|x)p(x) / p(y)", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  // Bottom: Independence & Convolution
  o += box(18, 140, 242, 110, C.lgreen, C.green, 8, 2);
  o += tx(139, 162, "Independence X ⊥ Y", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(139, 188, "p(x | y) = p(x)  ∀y", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(139, 212, "⇔ p(x, y) = p_X(x) p_Y(y)", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(139, 234, "Ex: p(x,y)=4xy=(2x)(2y) on [0,1]²", { fs: 11, fill: C.muted });

  o += box(280, 140, 242, 110, C.lamber, C.amber, 8, 2);
  o += tx(401, 162, "Sum S = X + Y (Convolution)", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(401, 188, "Joint of (X, S): p_X(x) p_Y(s−x)", { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(401, 214, "p_{X+Y}(s) = ∫ p_X(x)p_Y(s−x)dx", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(401, 234, "Marginalizes out x along x+y=s", { fs: 11, fill: C.muted });
  return svg(W, H, o);
}

function evesLaw() {
  const W = 540, H = 266;
  let o = "";
  o += box(20, 16, 500, 62, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 36, "Law of Total Variance (Eve's Law: E–V–V–E)", { fs: 15, fw: 700, fill: D.purple.dark });
  o += tx(270, 60, "Var(X)  =  E[ Var(X | Y) ]  +  Var( E[X | Y] )", { mono: true, fs: 14, fw: 700, fill: C.ink });

  // Left branch: Unexplained within-strip variance
  o += box(20, 98, 236, 152, C.lblue, C.blue, 8, 2);
  o += tx(138, 122, "1. E[ Var(X | Y) ]", { mono: true, fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(138, 146, "Unexplained (Within-Strip)", { fs: 13.5, fw: 700, fill: C.ink });
  o += tx(138, 172, "E[X²] − E[(E[X | Y])²]", { mono: true, fs: 12.5, fw: 600, fill: D.blue.dark });
  o += tx(138, 200, "Spread left inside each slice Y=y", { fs: 12, fill: C.muted });
  o += tx(138, 224, "Mixture benchmark: 1.00 (28%)", { mono: true, fs: 12, fw: 700, fill: C.ink });

  // Right branch: Explained between-strip variance
  o += box(284, 98, 236, 152, C.lgreen, C.green, 8, 2);
  o += tx(402, 122, "2. Var( E[X | Y] )", { mono: true, fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(402, 146, "Explained by Y (Between-Strip)", { fs: 13.5, fw: 700, fill: C.ink });
  o += tx(402, 172, "E[(E[X | Y])²] − (E[X])²", { mono: true, fs: 12.5, fw: 600, fill: D.green.dark });
  o += tx(402, 200, "How much slice means m(Y) move", { fs: 12, fill: C.muted });
  o += tx(402, 224, "Mixture benchmark: 2.56 (72%)", { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += tx(270, 174, "+", { fs: 24, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

function optimalMse() {
  const W = 540, H = 264;
  let o = "";
  o += box(20, 14, 500, 56, C.lblue, C.blue, 8, 2);
  o += tx(270, 34, "Why L2 Regression Predicts the Conditional Mean m(Y) = E[X | Y]", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(270, 56, "Split residual:  X − h(Y) = (X − m(Y)) + (m(Y) − h(Y))", { mono: true, fs: 12.5, fw: 600, fill: C.ink });

  // Orthogonal decomposition boxes
  o += box(20, 88, 234, 108, C.lamber, C.amber, 8, 2);
  o += tx(137, 110, "Irreducible Noise Floor", { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(137, 136, "E[ (X − m(Y))² ] = E[Var(X|Y)]", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(137, 162, "Does not depend on predictor h!", { fs: 12.5, fw: 600, fill: D.orange.dark });
  o += tx(137, 182, "Cannot be reduced by any model", { fs: 11.5, fill: C.muted });

  o += tx(270, 142, "+", { fs: 24, fw: 700, fill: C.ink });

  o += box(286, 88, 234, 108, C.lgreen, C.green, 8, 2);
  o += tx(403, 110, "Reducible Approximation Gap", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(403, 136, "E[ (E[X | Y] − h(Y))² ] ≥ 0", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(403, 162, "Vanishes iff h*(Y) = E[X | Y]!", { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(403, 182, "Cross term = 0 in every Y-strip", { fs: 11.5, fill: C.muted });

  // R^2 ceiling bar
  o += box(20, 210, 500, 42, D.purple.tint, C.purple, 8, 1.8);
  o += tx(270, 231, "Explained Variance Ceiling:  R² ≤ Var(E[X | Y]) / Var(X)", { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

function sumVarMc() {
  const W = 540, H = 268;
  let o = "";
  o += box(20, 14, 500, 46, N.panel, C.gray, 8, 1.8);
  o += tx(270, 37, "Sample Mean Variance:  Var(X̄_n) = (1/n²) Σᵢ Σⱼ Cov(Xᵢ, Xⱼ)", { mono: true, fs: 12, fw: 700, fill: C.ink });

  // Left: Independent i.i.d. draws
  o += box(20, 76, 236, 176, C.lgreen, C.green, 8, 2);
  o += tx(138, 100, "Independent Draws (ρ = 0)", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(138, 126, "Off-diagonals Cov(Xᵢ, Xⱼ) = 0", { fs: 12, fw: 600, fill: C.ink });
  o += box(38, 142, 200, 44, "#FFFFFF", C.green, 6, 1.8);
  o += tx(138, 164, "Var(X̄_n) = σ² / n", { mono: true, fs: 14, fw: 700, fill: D.green.dark });
  o += tx(138, 206, "SE = σ / √n → 0 as n → ∞", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(138, 232, "4× samples halves Monte Carlo error", { fs: 11.5, fill: C.muted });

  // Right: Correlated ensemble draws
  o += box(284, 76, 236, 176, C.lamber, C.amber, 8, 2);
  o += tx(402, 100, "Correlated Draws (Corr = ρ)", { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(402, 126, "n(n−1) cross terms = ρσ²", { fs: 12, fw: 600, fill: C.ink });
  o += box(302, 142, 200, 44, "#FFFFFF", C.amber, 6, 1.8);
  o += tx(402, 164, "Var(Ȳ_n) = ρσ² + (1−ρ)σ²/n", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(402, 206, "Irreducible floor: → ρσ² > 0", { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(402, 232, "Why ensembles need diverse models", { fs: 11.5, fill: C.muted });
  return svg(W, H, o);
}

function uncorrDep() {
  const W = 540, H = 266;
  let o = "";
  const panels = [
    { x: 16, t: "1. Circular Disk", sub: "x² + y² ≤ 1", why0: "Odd symmetry E[XY]=0", dep: "|X| ≤ √(1−Y²)", col: C.blue, bg: C.lblue, dk: D.blue.dark },
    { x: 194, t: "2. Parabola Y = X²", sub: "X ~ N(0, 1)", why0: "E[X³] = 0 ⇒ Cov=0", dep: "Y fixed by X!", col: C.amber, bg: C.lamber, dk: D.orange.dark },
    { x: 372, t: "3. Random Sign", sub: "Y = S·X (S ∈ {±1})", why0: "E[S]E[X²] = 0", dep: "|Y| = |X| always!", col: C.purple, bg: D.purple.tint, dk: D.purple.dark },
  ];
  panels.forEach((p, idx) => {
    o += box(p.x, 16, 152, 234, "#FFFFFF", p.col, 8, 2);
    o += box(p.x, 16, 152, 46, p.bg, p.col, 8, 2);
    o += tx(p.x + 76, 35, p.t, { fs: 13.5, fw: 700, fill: p.dk });
    o += tx(p.x + 76, 52, p.sub, { mono: true, fs: 11.5, fw: 600, fill: C.ink });
    const cx = p.x + 76, cy = 128;
    o += line(cx - 48, cy, cx + 48, cy, N.hairline, 1.5);
    o += line(cx, cy - 44, cx, cy + 44, N.hairline, 1.5);
    if (idx === 0) {
      o += circ(cx, cy, 38, C.lblue, C.blue, 2);
    } else if (idx === 1) {
      let d = "";
      for (let s = -10; s <= 10; s++) {
        const u = s / 10;
        const px = cx + u * 40;
        const py = cy + 28 - (u * u) * 56;
        d += (s === -10 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
      }
      o += path(d, C.amber, "none", 3);
    } else {
      o += line(cx - 38, cy - 38, cx + 38, cy + 38, C.purple, 2.8);
      o += line(cx - 38, cy + 38, cx + 38, cy - 38, C.purple, 2.8);
    }
    o += tx(cx, 198, p.why0, { fs: 11.5, fw: 600, fill: C.ink });
    o += tx(cx, 222, p.dep, { mono: true, fs: 12, fw: 700, fill: p.dk });
  });
  return svg(W, H, o);
}

function flowCoupling() {
  const W = 540, H = 266;
  let o = "";
  o += box(20, 14, 500, 44, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 36, "Affine Coupling Flow:  log p_Y(y) = log p_X(x) − log|det J_g(x)|", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  // Top path: x1 -> y1 = x1
  o += box(24, 78, 86, 44, C.lblue, C.blue, 7, 2);
  o += tx(67, 100, "x₁", { mono: true, fs: 16, fw: 700, fill: D.blue.dark });
  o += arrow(114, 100, 290, 100, C.blue);
  o += tx(202, 84, "identity pass", { fs: 12.5, fw: 600, fill: D.blue.dark });
  o += box(294, 78, 106, 44, C.lgreen, C.green, 7, 2);
  o += tx(347, 100, "y₁ = x₁", { mono: true, fs: 14, fw: 700, fill: D.green.dark });

  // Neural net s(x1), t(x1)
  o += box(136, 128, 128, 42, C.lamber, C.amber, 7, 2);
  o += tx(200, 149, "NN: s(x₁), t(x₁)", { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += arrow(114, 108, 134, 144, C.amber);
  o += arrow(200, 172, 200, 198, C.amber);

  // Bottom path: x2 -> y2 = x2 exp(s(x1)) + t(x1)
  o += box(24, 196, 86, 44, C.lblue, C.blue, 7, 2);
  o += tx(67, 218, "x₂", { mono: true, fs: 16, fw: 700, fill: D.blue.dark });
  o += arrow(114, 218, 168, 218, C.blue);
  o += box(172, 198, 56, 40, "#FFFFFF", C.amber, 6, 2);
  o += tx(200, 218, "×eˢ+t", { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += arrow(232, 218, 290, 218, C.green);
  o += box(294, 196, 106, 44, C.lgreen, C.green, 7, 2);
  o += tx(347, 218, "y₂", { mono: true, fs: 15, fw: 700, fill: D.green.dark });

  // Right card: Triangular Jacobian
  o += box(410, 78, 114, 162, N.panel, C.gray, 8, 2);
  o += tx(467, 102, "Triangular J_g", { fs: 13, fw: 700, fill: C.ink });
  o += tx(467, 132, "[ 1    0 ]", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(467, 154, "[ *   eˢ ]", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(467, 190, "log|det J_g|", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  o += tx(467, 214, "= s(x₁) O(1)!", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ============================================================================
// 3. mdl-distributions.md (5 diagrams)
// ============================================================================

function lossDistMap() {
  const W = 540, H = 276;
  let o = "";
  const rows = [
    ["Binary p̂ ∈ (0,1)", "Bernoulli(p̂)", "BCE: −y log p̂ − (1−y)log(1−p̂)", C.lblue, C.blue, D.blue.dark],
    ["Softmax p̂ ∈ Δ^{K−1}", "Categorical(p̂)", "CE: −Σ_k y_k log p̂_k", D.purple.tint, C.purple, D.purple.dark],
    ["Mean ŷ ∈ ℝ", "Gaussian N(ŷ, σ²)", "MSE: (y − ŷ)² / (2σ²)", C.lgreen, C.green, D.green.dark],
    ["Median ŷ ∈ ℝ", "Laplace(ŷ, b)", "MAE (L1): |y − ŷ| / b", C.lamber, C.amber, D.orange.dark],
  ];
  o += tx(86, 20, "Network Output", { fs: 13.5, fw: 700, fill: C.muted });
  o += tx(242, 20, "Target Distribution", { fs: 13.5, fw: 700, fill: C.muted });
  o += tx(426, 20, "Training Loss −log p(y | ŷ)", { fs: 13.5, fw: 700, fill: C.muted });
  rows.forEach(([out, dist, loss, fill, st, dk], i) => {
    const y = 34 + i * 58;
    o += box(14, y, 144, 46, N.panel, C.gray, 7, 1.8);
    o += tx(86, y + 23, out, { mono: true, fs: 12, fw: 700, fill: C.ink });
    o += arrow(160, y + 23, 174, y + 23, st);
    o += box(176, y, 132, 46, fill, st, 7, 2);
    o += tx(242, y + 23, dist, { mono: true, fs: 12.5, fw: 700, fill: dk });
    o += arrow(310, y + 23, 324, y + 23, st);
    o += box(326, y, 200, 46, "#FFFFFF", st, 7, 2);
    o += tx(426, y + 23, loss, { mono: true, fs: 11.5, fw: 700, fill: dk });
  });
  return svg(W, H, o);
}

function bernoulliVar() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 54, y0 = 208, gw = 216, gh = 148;
  // Parabola p(1-p)
  let d = "";
  for (let s = 0; s <= 40; s++) {
    const p = s / 40;
    const px = x0 + p * gw;
    const py = y0 - 4 * p * (1 - p) * gh;
    d += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d + `L ${x0 + gw},${y0} L ${x0},${y0} Z`, "none", C.lblue, 0);
  o += line(x0, y0, x0 + gw + 14, y0, C.ink, 2);
  o += line(x0, y0, x0, y0 - gh - 14, C.ink, 2);
  o += path(d, C.blue, "none", 3);

  // Peak at p = 0.5 and coin at p = 9/13
  const xHalf = x0 + 0.5 * gw, yHalf = y0 - gh;
  const pCoin = 9 / 13, xCoin = x0 + pCoin * gw, yCoin = y0 - 4 * pCoin * (1 - pCoin) * gh;
  o += line(xHalf, y0, xHalf, yHalf, C.blue, 1.5, true);
  o += circ(xHalf, yHalf, 5, C.blue, "#FFFFFF", 1.5);
  o += line(xCoin, y0, xCoin, yCoin, D.orange.base, 1.5, true);
  o += circ(xCoin, yCoin, 5, D.orange.base, "#FFFFFF", 1.5);

  o += tx(x0 + gw / 2, 22, "Bernoulli Variance Var(X) = p(1 − p)", { fs: 15, fw: 700, fill: D.blue.dark });
  o += tx(x0 - 24, yHalf, "0.25", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(x0, y0 + 20, "0", { mono: true, fs: 12, fill: C.ink });
  o += tx(xHalf, y0 + 20, "0.5", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(xCoin + 8, y0 + 20, "9/13", { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw, y0 + 20, "1", { mono: true, fs: 12, fill: C.ink });

  const rx = 296;
  o += box(rx, 28, 226, 98, C.lblue, C.blue, 8, 2);
  o += tx(rx + 113, 50, "Algebraic Switch PMF", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(rx + 113, 74, "P(X=x) = pˣ(1−p)¹⁻ˣ, x∈{0,1}", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(rx + 113, 100, "X² = X ⇒ E[X²] = E[X] = p", { mono: true, fs: 12, fw: 600, fill: D.blue.dark });

  o += box(rx, 142, 226, 102, C.lamber, C.amber, 8, 2);
  o += tx(rx + 113, 164, "Fair vs. Empirical Coin", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(rx + 113, 188, "Fair p=0.5: max Var = 0.250", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(rx + 113, 214, "Coin p=9/13: Var ≈ 0.213", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

function binomialGrid() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 52, C.lblue, C.blue, 8, 2);
  o += tx(270, 32, "Binomial(n, p): Sum of n Independent Bernoulli(p) Trials", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(270, 52, "X = X₁ + X₂ + ··· + Xₙ   ⇒   E[X] = np,   Var(X) = np(1 − p)", { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  // Left: n=2 four outcomes table
  o += box(18, 82, 236, 168, "#FFFFFF", C.blue, 8, 2);
  o += tx(136, 104, "Smallest Case: n = 2 Flips", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(136, 130, "k=0 (TT):  (1 − p)²", { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(136, 156, "k=1 (HT, TH):  2p(1 − p)", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(136, 182, "k=2 (HH):  p²", { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(136, 218, "General: C(n,k) pᵏ(1−p)ⁿ⁻ᵏ", { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });

  // Right: Independent vs Cloned flips contrast
  o += box(276, 82, 246, 168, D.red.tint, D.red.base, 8, 2);
  o += tx(399, 104, "Why Var = np(1−p) Needs Indep!", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(399, 132, "13 Independent Fair Flips:", { fs: 12.5, fw: 700, fill: C.ink });
  o += tx(399, 152, "Var(X) = 13 × 0.25 = 3.25 (σ=1.80)", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(399, 184, "1 Fair Flip Copied 13 Times:", { fs: 12.5, fw: 700, fill: C.ink });
  o += tx(399, 204, "Var(13 X₁) = 13² × 0.25 = 42.25!", { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(399, 232, "13× larger variance (X ∈ {0, 13})", { fs: 11.5, fill: C.muted });
  return svg(W, H, o);
}

function poissonSlicing() {
  const W = 540, H = 266;
  let o = "";
  // Top: Minute sliced into n subintervals
  o += box(18, 14, 504, 88, C.lblue, C.blue, 8, 2);
  o += tx(270, 34, "Slicing 1 Minute Into n Tiny Subintervals (pₙ = λ/n = 4/n)", { fs: 14.5, fw: 700, fill: D.blue.dark });
  const sx0 = 36, sy = 50, sw = 28, sg = 3;
  const active = new Set([2, 6, 10, 13]);
  for (let i = 0; i < 15; i++) {
    const x = sx0 + i * (sw + sg);
    const hit = active.has(i);
    o += box(x, sy, sw, 26, hit ? C.lamber : "#FFFFFF", hit ? C.amber : C.blue, 4, 1.5);
    o += tx(x + sw / 2, sy + 13, hit ? "1" : "0", { mono: true, fs: 12, fw: 700, fill: hit ? D.orange.dark : C.muted });
  }
  o += tx(270, 90, "As n → ∞: at most 1 arrival per slice, independent trials with E[X] = n(λ/n) = λ", { fs: 12, fw: 600, fill: C.ink });

  // Bottom: 4-factor limit
  o += box(18, 118, 504, 132, C.lgreen, C.green, 8, 2);
  o += tx(270, 140, "Four-Factor Rare-Event Limit: Binomial(n, λ/n) → Poisson(λ)", { fs: 14, fw: 700, fill: D.green.dark });
  const fBoxes = [
    [30, "n(n−1)···(n−k+1)/nᵏ", "→ 1", C.lgray, C.gray],
    [164, "λᵏ / k!", "kept!", C.lamber, C.amber],
    [274, "(1 − λ/n)ⁿ", "→ e⁻ᵞ", C.lamber, C.amber],
    [390, "(1 − λ/n)⁻ᵏ", "→ 1", C.lgray, C.gray],
  ];
  fBoxes.forEach(([x, f, lim, bg, st]) => {
    o += box(x, 156, 118, 52, bg, st, 6, 1.8);
    o += tx(x + 59, 173, f, { mono: true, fs: 10.5, fw: 700, fill: C.ink });
    o += tx(x + 59, 194, lim, { mono: true, fs: 12.5, fw: 700, fill: st === C.amber ? D.orange.dark : C.muted });
  });
  o += tx(270, 230, "Result: P(X = k) = λᵏ e⁻ᵞ / k!   with   E[X] = Var(X) = λ", { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function uniformCenter() {
  const W = 540, H = 264;
  let o = "";
  // Top: Unif(a, b) shifted to [-c, c]
  o += box(20, 16, 236, 112, C.lblue, C.blue, 8, 2);
  o += tx(138, 38, "Uniform U(a, b)", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += box(64, 56, 148, 36, "#FFFFFF", C.blue, 4, 2);
  o += tx(138, 74, "p(x) = 1 / (b − a)", { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(64, 108, "a", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(138, 108, "μ = (a+b)/2", { mono: true, fs: 11.5, fw: 600, fill: C.muted });
  o += tx(212, 108, "b", { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += arrow(260, 72, 280, 72, C.purple);

  o += box(284, 16, 236, 112, D.purple.tint, C.purple, 8, 2);
  o += tx(402, 38, "Centered on [−c, c], c=(b−a)/2", { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(402, 68, "Var(X) = ∫₋c^c y²/(2c) dy = c²/3", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(402, 98, "Var(X) = (b − a)² / 12", { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });

  // Bottom: Canonical benchmarks
  o += box(20, 144, 500, 104, C.lgreen, C.green, 8, 2);
  o += tx(270, 166, "Key Uniform Benchmarks in Deep Learning", { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(270, 194, "• Base RNG U(0, 1):  E[U] = 1/2,  Var(U) = 1/12 ≈ 0.0833", { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(270, 222, "• Unit-Variance U(−√3, √3):  E[X] = 0,  Var(X) = (2√3)²/12 = 1.0", { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ============================================================================
// 4. mdl-exponential-family.md (9 diagrams)
// ============================================================================

function fourRegressions() {
  const W = 540, H = 268;
  let o = "";
  const models = [
    ["Linear Regression", "Gaussian N(η, 1)", "μ = η", C.lblue, C.blue, D.blue.dark],
    ["Logistic Regression", "Bernoulli(p)", "μ = sigmoid(η)", C.lgreen, C.green, D.green.dark],
    ["Softmax Regression", "Categorical(p)", "μ = softmax(η)", D.purple.tint, C.purple, D.purple.dark],
    ["Poisson Regression", "Poisson(λ)", "μ = exp(η)", C.lamber, C.amber, D.orange.dark],
  ];
  models.forEach(([m, fam, mu, fill, st, dk], i) => {
    const y = 14 + i * 52;
    o += box(16, y, 274, 44, fill, st, 7, 1.8);
    o += tx(153, y + 15, `${m}  (${fam})`, { fs: 13, fw: 700, fill: dk });
    o += tx(153, y + 33, `Canonical link: ${mu}`, { mono: true, fs: 12, fw: 600, fill: C.ink });
    o += arrow(294, y + 22, 326, 118, st);
  });

  // Converging right hub
  o += box(330, 56, 194, 124, "#FFFFFF", D.red.base, 10, 2.5);
  o += tx(427, 82, "One Universal Rule!", { fs: 15, fw: 700, fill: D.red.dark });
  o += tx(427, 112, "∇_W (−log p(y|x))", { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(427, 138, "= x (μ(x) − T(y))ᵀ", { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(427, 162, "input × (pred − target)", { fs: 12.5, fw: 600, fill: C.muted });

  o += box(16, 224, 508, 32, N.panel, C.gray, 6, 1.5);
  o += tx(270, 240, "Proof in 1 line: ∇_η(A(η) − ηᵀT(y)) = ∇A(η) − T(y) = E[T(y)] − T(y)", { mono: true, fs: 12, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function bernoulliLogit() {
  const W = 540, H = 264;
  let o = "";
  o += box(20, 14, 500, 62, C.lblue, C.blue, 8, 2);
  o += tx(270, 34, "Algebraic Separation of Outcome x and Parameter p", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(270, 58, "log P(X=x) = x log(p/(1−p)) + log(1−p) = η·x − log(1 + e^η)", { mono: true, fs: 12, fw: 700, fill: C.ink });

  // Bijection between p and eta
  o += box(20, 96, 198, 98, C.lamber, C.amber, 8, 2);
  o += tx(119, 118, "Mean Probability p", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(119, 144, "p ∈ (0, 1)", { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(119, 172, "Coin: p̂ = 9/13 ≈ 0.692", { mono: true, fs: 11.5, fw: 600, fill: D.orange.dark });

  o += arrow(222, 130, 318, 130, C.purple);
  o += tx(270, 114, "logit(p)", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  o += arrow(318, 162, 222, 162, C.green);
  o += tx(270, 180, "sigmoid(η)", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  o += box(322, 96, 198, 98, C.lgreen, C.green, 8, 2);
  o += tx(421, 118, "Natural Parameter η", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(421, 144, "η ∈ (−∞, +∞)", { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(421, 172, "Coin: η̂ = log(9/4) ≈ 0.811", { mono: true, fs: 11.5, fw: 600, fill: D.green.dark });

  o += box(20, 210, 500, 42, D.purple.tint, C.purple, 8, 1.8);
  o += tx(270, 231, "Log-partition A(η) = log(1 + e^η)  ⇒  A'(η) = sigmoid(η) = p", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

function expAnatomy() {
  const W = 540, H = 266;
  let o = "";
  o += box(20, 14, 500, 50, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 39, "p(x | η)  =  h(x) · exp( ηᵀ T(x)  −  A(η) )", { mono: true, fs: 15.5, fw: 700, fill: D.purple.dark });

  const parts = [
    { x: 18, y: 80, w: 242, h: 82, t: "1. Base Measure h(x) ≥ 0", s: "Fixes support {x : h(x) > 0}", s2: "Independent of parameters η", fill: C.lgray, st: C.gray, dk: C.ink },
    { x: 280, y: 80, w: 242, h: 82, t: "2. Sufficient Statistic T(x)", s: "Data summary vector in ℝᵈ", s2: "n draws compress to Σᵢ T(xᵢ)", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 18, y: 174, w: 242, h: 80, t: "3. Natural Parameter η ∈ ℝᵈ", s: "Multiplies T(x) linearly", s2: "Convex domain {η : A(η) < ∞}", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 280, y: 174, w: 242, h: 80, t: "4. Log-Partition A(η)", s: "log ∫ h(x) exp(ηᵀT(x)) dx", s2: "Normalizer & cumulant generator", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  parts.forEach(p => {
    o += box(p.x, p.y, p.w, p.h, p.fill, p.st, 8, 2);
    o += tx(p.x + p.w / 2, p.y + 20, p.t, { fs: 13.5, fw: 700, fill: p.dk });
    o += tx(p.x + p.w / 2, p.y + 44, p.s, { mono: true, fs: 12, fw: 600, fill: C.ink });
    o += tx(p.x + p.w / 2, p.y + 64, p.s2, { fs: 12, fill: C.muted });
  });
  return svg(W, H, o);
}

function outsideExp() {
  const W = 540, H = 266;
  let o = "";
  // Left: Moving support Unif(0, theta)
  o += box(18, 16, 242, 234, C.lamber, C.amber, 8, 2);
  o += tx(139, 38, "1. Moving Support: Unif(0, θ)", { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(139, 62, "p(x|θ) = (1/θ)·1{0 ≤ x ≤ θ}", { mono: true, fs: 12, fw: 700, fill: C.ink });
  // Mini number line for data {0.1, 0.2, 0.9}
  o += box(32, 82, 214, 72, "#FFFFFF", C.amber, 6, 1.5);
  o += line(48, 120, 230, 120, C.ink, 2);
  o += circ(188, 120, 5, D.orange.base, D.orange.dark, 1.5);
  o += tx(188, 100, "MLE = max = 0.9", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += circ(164, 120, 5, D.red.base, D.red.dark, 1.5);
  o += tx(114, 140, "2x̄ = 0.8 (impossible!)", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(139, 180, "Support depends on θ (not h(x))", { fs: 12, fw: 600, fill: C.ink });
  o += tx(139, 204, "Sufficient stat is maxᵢ xᵢ, not a sum", { fs: 12, fw: 600, fill: D.orange.dark });
  o += tx(139, 228, "Converges at rate 1/n, not 1/√n", { fs: 11.5, fill: C.muted });

  // Right: Heavy-tailed Student-t
  o += box(280, 16, 242, 234, D.red.tint, D.red.base, 8, 2);
  o += tx(401, 38, "2. Heavy Tails: Student-t(μ)", { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(401, 62, "p(x|μ) ∝ (1 + (x−μ)²/ν)⁻⁽ᵛ⁺¹⁾/²", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  // Mini double-well non-convex NLL curve
  o += box(294, 82, 214, 72, "#FFFFFF", D.red.base, 6, 1.5);
  let d = "";
  for (let s = 0; s <= 32; s++) {
    const u = (s / 32) * 4 - 2;
    const px = 310 + (s / 32) * 182;
    const py = 136 - 24 * Math.exp(-2 * (u + 1) * (u + 1)) - 28 * Math.exp(-2 * (u - 1) * (u - 1));
    d += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d, D.red.base, "none", 2.5);
  o += tx(401, 98, "Non-convex NLL (2 local minima!)", { fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(401, 180, "No fixed-dim sufficient statistic", { fs: 12, fw: 600, fill: C.ink });
  o += tx(401, 204, "(Pitman–Koopman–Darmois thm)", { fs: 12, fw: 600, fill: D.red.dark });
  o += tx(401, 228, "Clusters create multiple NLL wells", { fs: 11.5, fill: C.muted });
  return svg(W, H, o);
}

function gibbsTangent() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 52, y0 = 152, gw = 206;
  // Axes for u > 0
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  o += line(x0, 28, x0, 232, C.ink, 1.8);
  // Tangent line u - 1 and curve log u
  let dLine = "", dLog = "";
  for (let s = 2; s <= 40; s++) {
    const u = (s / 40) * 2.5;
    const px = x0 + (u / 2.5) * gw;
    const pyL = y0 - (u - 1) * 64;
    const pyC = y0 - Math.log(u) * 64;
    dLine += (s === 2 ? "M" : "L") + `${px.toFixed(1)},${pyL.toFixed(1)} `;
    dLog += (s === 2 ? "M" : "L") + `${px.toFixed(1)},${pyC.toFixed(1)} `;
  }
  o += path(dLine, C.blue, "none", 2.2, true);
  o += path(dLog, D.orange.base, "none", 3);
  const xOne = x0 + (1 / 2.5) * gw;
  o += circ(xOne, y0, 5, D.green.base, "#FFFFFF", 1.5);
  o += tx(xOne, y0 + 20, "u = 1", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(x0 + 136, 52, "u − 1 (tangent)", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(x0 + 178, 128, "log u", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw / 2, 20, "Concavity Bound: log u ≤ u − 1", { fs: 14, fw: 700, fill: C.ink });

  const rx = 282;
  o += box(rx, 24, 240, 104, C.lblue, C.blue, 8, 2);
  o += tx(rx + 120, 46, "Gibbs' Inequality Proof", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(rx + 120, 72, "Set u = p(x)/q(x) in log u ≤ u−1:", { mono: true, fs: 11, fw: 600, fill: C.ink });
  o += tx(rx + 120, 96, "−KL(q||p) ≤ Σ(p − q) = 0", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  o += box(rx, 142, 240, 104, C.lgreen, C.green, 8, 2);
  o += tx(rx + 120, 164, "Global MaxEnt Optimality", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(rx + 120, 188, "If E_q[T] = E_p[T], then:", { fs: 12, fw: 600, fill: C.ink });
  o += tx(rx + 120, 214, "H_h[q] = H_h[p] − KL(q||p)", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function cumulantDerivs() {
  const W = 540, H = 266;
  let o = "";
  const stages = [
    { x: 18, w: 152, t: "Log-Partition A(η)", f: "log ∫ h(x)e^{ηᵀT(x)}dx", ex: "Bernoulli: log(1+e^η)", ex2: "Poisson: e^η", fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 194, w: 152, t: "1st Deriv ∇A(η)", f: "= E[ T(x) ]  (Mean)", ex: "A'(η) = sigmoid(η) = p", ex2: "A'(η) = e^η = λ", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 370, w: 152, t: "2nd Deriv ∇²A(η)", f: "= Cov(T(x)) ⪰ 0", ex: "A''(η) = p(1 − p)", ex2: "A''(η) = e^η = λ", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  stages.forEach((s, i) => {
    o += box(s.x, 20, s.w, 168, s.fill, s.st, 8, 2);
    o += tx(s.x + s.w / 2, 44, s.t, { fs: 13.5, fw: 700, fill: s.dk });
    o += tx(s.x + s.w / 2, 74, s.f, { mono: true, fs: 11, fw: 700, fill: C.ink });
    o += line(s.x + 12, 96, s.x + s.w - 12, 96, s.st, 1, true);
    o += tx(s.x + s.w / 2, 122, s.ex, { mono: true, fs: 10.5, fw: 600, fill: s.dk });
    o += tx(s.x + s.w / 2, 152, s.ex2, { mono: true, fs: 11, fw: 700, fill: C.ink });
    if (i < 2) {
      o += arrow(s.x + s.w + 3, 104, stages[i + 1].x - 3, 104, C.ink);
      o += tx(s.x + s.w + 12, 86, "∇_η", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    }
  });
  o += box(18, 204, 504, 46, C.lamber, C.amber, 8, 2);
  o += tx(270, 227, "Since Cov(T) ⪰ 0 everywhere, A(η) and NLL nA(η) − ηᵀΣT(xᵢ) are ALWAYS CONVEX!", { fs: 12, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

function boundaryMinimal() {
  const W = 540, H = 266;
  let o = "";
  // Left: 13/13 heads boundary divergence
  o += box(18, 16, 242, 234, D.red.tint, D.red.base, 8, 2);
  o += tx(139, 38, "1. Boundary Mean (13/13 Heads)", { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(139, 62, "T̄ = 1 ∉ sigmoid(ℝ) = (0, 1)", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += box(34, 80, 210, 82, "#FFFFFF", D.red.base, 6, 1.5);
  let d1 = "";
  for (let s = 0; s <= 24; s++) {
    const u = (s / 24) * 5 - 1;
    const px = 46 + (s / 24) * 184;
    const py = 152 - 36 * Math.log(1 + Math.exp(-u)) / Math.log(1 + Math.exp(1));
    d1 += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d1, D.red.base, "none", 2.5);
  o += tx(139, 96, "L(η) = 13 log(1+e⁻ⁿ) → 0", { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  o += tx(139, 184, "No finite minimizer: η̂ → +∞", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(139, 208, "Same runaway on separable", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(139, 228, "logistic regression! Fix: L2 prior", { fs: 11.5, fw: 700, fill: D.green.dark });

  // Right: Non-minimal softmax logits
  o += box(280, 16, 242, 234, C.lblue, C.blue, 8, 2);
  o += tx(401, 38, "2. Non-Minimal Softmax Logits", { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(401, 62, "One-hot sum: 1ᵀy = Σ_k y_k = 1", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += box(296, 80, 210, 82, "#FFFFFF", C.blue, 6, 1.5);
  o += line(316, 128, 486, 114, C.blue, 3);
  o += tx(401, 96, "Line of minimizers z + c·1", { mono: true, fs: 11, fw: 700, fill: D.blue.dark });
  o += tx(401, 148, "∇²A(z)·1 = 0 (flat along 1)", { mono: true, fs: 11, fw: 600, fill: C.ink });
  o += tx(401, 184, "softmax(z + c·1) = softmax(z)", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += tx(401, 208, "Exploited by log-sum-exp:", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(401, 228, "subtract c = max_k z_k for stability!", { fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function glmHead() {
  const W = 540, H = 260;
  let o = "";
  // Pipeline: x -> h_theta(x) -> eta = W^T h -> mu = grad A(eta) -> NLL
  const nodes = [
    { x: 16, w: 102, t: "Features", s: "h_θ(x) ∈ ℝᵖ", fill: N.panel, st: C.gray, dk: C.ink },
    { x: 142, w: 114, t: "Linear Head", s: "η = Wᵀx", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 280, w: 118, t: "Mean Map", s: "μ = ∇A(η)", fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 422, w: 102, t: "NLL Loss", s: "A(η)−ηᵀT(y)", fill: C.lamber, st: C.amber, dk: D.orange.dark },
  ];
  nodes.forEach((n, i) => {
    o += box(n.x, 34, n.w, 72, n.fill, n.st, 8, 2);
    o += tx(n.x + n.w / 2, 58, n.t, { fs: 13, fw: 700, fill: n.dk });
    o += tx(n.x + n.w / 2, 84, n.s, { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    if (i < 3) o += arrow(n.x + n.w + 3, 70, nodes[i + 1].x - 3, 70, C.blue);
  });

  // Backward VJP bar
  o += box(16, 134, 508, 108, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 158, "Canonical Link Reverse Gradient: Always Prediction Minus Observation!", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(270, 186, "∂L/∂η = ∇A(η) − T(y) = μ(x) − T(y)   ⇒   ∇_W L = x (μ(x) − T(y))ᵀ", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(270, 216, "Swapping the family swaps only ∇A(η): identity, sigmoid, softmax, or exp", { fs: 12, fw: 600, fill: D.red.dark });
  return svg(W, H, o);
}

function conjugateAdd() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 38, "Exponential-Family Conjugacy: Prior + Likelihood = Add Counts!", { fs: 14, fw: 700, fill: D.purple.dark });

  // Prior + Data -> Posterior
  o += box(18, 78, 148, 96, C.lblue, C.blue, 8, 2);
  o += tx(92, 100, "Beta(a, b) Prior", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(92, 126, "θ^{a−1}(1−θ)^{b−1}", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(92, 152, "Unif: Beta(1, 1)", { mono: true, fs: 11.5, fill: D.blue.dark });

  o += tx(180, 126, "×", { fs: 22, fw: 700, fill: C.ink });

  o += box(194, 78, 148, 96, C.lamber, C.amber, 8, 2);
  o += tx(268, 100, "Data Likelihood", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(268, 126, "θˣ (1 − θ)^{n−x}", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(268, 152, "9 Heads, 4 Tails", { mono: true, fs: 11.5, fill: D.orange.dark });

  o += arrow(346, 126, 370, 126, C.green);

  o += box(374, 78, 148, 96, C.lgreen, C.green, 8, 2);
  o += tx(448, 100, "Posterior", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(448, 126, "Beta(a+x, b+n−x)", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(448, 152, "Beta(10, 5)", { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  // Bottom comparison: Mean vs Mode
  o += box(18, 190, 504, 62, "#FFFFFF", C.green, 8, 2);
  o += tx(270, 210, "Posterior Mean (Laplace Rule of Succession) vs. Posterior Mode (MAP)", { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(270, 234, "Mean = (a+x)/(a+b+n) = 10/15 = 2/3  |  Mode = (a+x−1)/(a+b+n−2) = 9/13", { mono: true, fs: 11, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ============================================================================
// 5. mdl-statistics.md (4 diagrams)
// ============================================================================

function pairedVsUnpaired() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, C.lblue, C.blue, 8, 2);
  o += tx(270, 38, "Model A (91.2%) vs. Model B (92.1%) on 1,000 Questions: Gap = +0.9%", { fs: 13.5, fw: 700, fill: D.blue.dark });

  // Left: Unpaired independent test sets
  o += box(18, 78, 240, 172, D.red.tint, D.red.base, 8, 2);
  o += tx(138, 100, "Unpaired (Separate Sets)", { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(138, 126, "Var(θ̂_B − θ̂_A) = Var(A) + Var(B)", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(138, 152, "SE = √(0.90² + 0.85²)% = 1.24%", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(138, 182, "Gap / SE = 0.9 / 1.24 = 0.73 se", { mono: true, fs: 11, fw: 600, fill: C.ink });
  o += tx(138, 210, "Needs 27,000 total questions", { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(138, 232, "for 80% power at 1% gap!", { fs: 11.5, fill: C.muted });

  // Right: Paired shared test set
  o += box(282, 78, 240, 172, C.lgreen, C.green, 8, 2);
  o += tx(402, 100, "Paired (Same 1,000 Questions)", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(402, 126, "959 agreements (dᵢ = 0) drop out!", { fs: 12, fw: 700, fill: C.ink });
  o += tx(402, 152, "Only 41 disagreements (25 vs 16)", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(402, 182, "Var(B−A) = Var(A)+Var(B)−2Cov(A,B)", { mono: true, fs: 10.5, fw: 600, fill: C.ink });
  o += tx(402, 210, "Needs only 3,100 paired questions", { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(402, 232, "9× reduction in labeling cost!", { fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function besselDof() {
  const W = 540, H = 266;
  let o = "";
  // Top: n=2 geometry showing why x_bar is closer than mu
  o += box(18, 14, 504, 114, C.lblue, C.blue, 8, 2);
  o += tx(270, 34, "Why Centering at Sample Mean x̄ Shrinks Squared Deviations (n = 2)", { fs: 13.5, fw: 700, fill: D.blue.dark });
  const yL = 78;
  o += line(48, yL, 492, yL, C.ink, 2);
  // Points x1, x_bar, mu, x2
  const x1 = 104, x2 = 284, xBar = (x1 + x2) / 2, xMu = 384;
  o += circ(x1, yL, 5.5, C.blue, "#FFFFFF", 1.5);
  o += circ(x2, yL, 5.5, C.blue, "#FFFFFF", 1.5);
  o += circ(xBar, yL, 5.5, D.orange.base, "#FFFFFF", 1.5);
  o += circ(xMu, yL, 5.5, D.red.base, "#FFFFFF", 1.5);
  o += tx(x1, yL - 18, "x₁", { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(xBar, yL - 18, "x̄ (midpoint)", { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(x2, yL - 18, "x₂", { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(xMu, yL - 18, "true μ", { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(270, 108, "Exact identity:  Σᵢ(xᵢ − x̄)² = Σᵢ(xᵢ − μ)² − n(x̄ − μ)²", { mono: true, fs: 12, fw: 700, fill: C.ink });

  // Bottom: Expectation & n-1 denominator
  o += box(18, 142, 240, 108, C.lamber, C.amber, 8, 2);
  o += tx(138, 164, "Plug-In s₀² (ddof=0, ÷ n)", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(138, 190, "E[Σ(xᵢ−x̄)²] = nσ² − σ² = (n−1)σ²", { mono: true, fs: 11, fw: 600, fill: C.ink });
  o += tx(138, 218, "Bias = −σ²/n  (−50% at n=2!)", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  o += box(282, 142, 240, 108, C.lgreen, C.green, 8, 2);
  o += tx(402, 164, "Unbiased s² (ddof=1, ÷ n−1)", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(402, 190, "s² = Σ(xᵢ − x̄)² / (n − 1)", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(402, 218, "E[s²] = σ²  (n−1 degrees of freedom)", { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function mcVsGrid() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 46, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 37, "High-Dimensional Integration Error for N Evaluations in ℝᵈ", { fs: 14, fw: 700, fill: D.purple.dark });

  const cols = [
    { x: 18, t: "d = 1 Dimension", g1: "Left Grid: N⁻¹", g2: "Midpoint: N⁻² (Wins!)", mc: "MC: N⁻¹/²", col: C.blue, bg: C.lblue, dk: D.blue.dark },
    { x: 194, t: "d = 4 Crossover", g1: "Left Grid: N⁻¹/⁴", g2: "Midpoint: N⁻¹/² (Tie)", mc: "MC: N⁻¹/² (Tie)", col: C.amber, bg: C.lamber, dk: D.orange.dark },
    { x: 370, t: "d = 100 (Deep ML)", g1: "Left Grid: N⁻¹/¹⁰⁰", g2: "Midpoint: N⁻²/¹⁰⁰", mc: "MC: N⁻¹/² (Wins!)", col: C.green, bg: C.lgreen, dk: D.green.dark },
  ];
  cols.forEach(c => {
    o += box(c.x, 76, 152, 174, "#FFFFFF", c.col, 8, 2);
    o += box(c.x, 76, 152, 38, c.bg, c.col, 8, 2);
    o += tx(c.x + 76, 95, c.t, { fs: 13, fw: 700, fill: c.dk });
    o += tx(c.x + 76, 134, c.g1, { mono: true, fs: 11, fill: C.muted });
    o += tx(c.x + 76, 166, c.g2, { mono: true, fs: 11, fw: 700, fill: C.ink });
    o += box(c.x + 10, 190, 132, 44, c.bg, c.col, 6, 1.5);
    o += tx(c.x + 76, 212, c.mc, { mono: true, fs: 11.5, fw: 700, fill: c.dk });
  });
  return svg(W, H, o);
}

function bonferroniFdr() {
  const W = 540, H = 266;
  let o = "";
  // Left: Uncorrected 20-sweep FWER explosion
  o += box(18, 16, 240, 234, D.red.tint, D.red.base, 8, 2);
  o += tx(138, 38, "Uncorrected Sweep (m = 20)", { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(138, 64, "Each null model tested at α = 0.05", { fs: 12.5, fw: 600, fill: C.ink });
  o += box(36, 84, 204, 72, "#FFFFFF", D.red.base, 6, 1.8);
  o += tx(138, 106, "P(≥ 1 false breakthrough)", { fs: 12.5, fw: 700, fill: C.ink });
  o += tx(138, 134, "= 1 − 0.95²⁰ = 64.2%!", { mono: true, fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(138, 184, "Cherry-picking the best seed or", { fs: 12, fill: C.ink });
  o += tx(138, 204, "hyperparameter turns pure noise", { fs: 12, fill: C.ink });
  o += tx(138, 224, "into a 'significant' p < 0.05!", { fs: 12, fw: 700, fill: D.red.dark });

  // Right: Bonferroni & BH remedies
  o += box(282, 16, 240, 110, C.lblue, C.blue, 8, 2);
  o += tx(402, 38, "1. Bonferroni (FWER ≤ α)", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(402, 64, "Threshold each test at pⱼ ≤ α / m", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(402, 88, "P(∪{pⱼ ≤ α/m}) ≤ m·(α/m) = α", { mono: true, fs: 11.5, fw: 600, fill: D.blue.dark });
  o += tx(402, 110, "Valid for ANY correlation!", { fs: 11.5, fw: 700, fill: D.blue.dark });

  o += box(282, 140, 240, 110, C.lgreen, C.green, 8, 2);
  o += tx(402, 162, "2. Benjamini–Hochberg (FDR ≤ α)", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(402, 188, "Sort p₍₁₎ ≤ ··· ≤ p₍ₘ₎; reject up to", { fs: 12, fw: 600, fill: C.ink });
  o += tx(402, 210, "largest k with p₍ₖ₎ ≤ (k / m) α", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(402, 232, "Higher power for large m ≫ 100", { fs: 11.5, fill: C.muted });
  return svg(W, H, o);
}

// ============================================================================
// 6. mdl-maximum-likelihood.md (9 diagrams)
// ============================================================================

function coinThreeQ() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 52, C.lamber, C.amber, 8, 2);
  o += tx(270, 32, "Observed Coin Flips: HHHTHTTHHHHHT  (n_H = 9, n_T = 4)", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 52, "One Likelihood Function P(X | θ) = θ⁹(1 − θ)⁴ Answers Three Core Questions", { fs: 12, fw: 600, fill: C.ink });

  const qs = [
    { x: 18, t: "1. Point Estimate", q: "Why pick θ̂ = 9/13?", a: "Maximizes P(X | θ)", a2: "(2.7× fair coin prob)", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: "2. Error Bar", q: "Is θ = 0.5 ruled out?", a: "Curvature ℓ''(θ̂) = 61", a2: "⇒ SE = 1/√61 = 0.128", fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 370, t: "3. Scarce Data", q: "What if 3 flips = HHH?", a: "MLE = 1 (vetoes T!)", a2: "Prior → MAP = 4/5", fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  qs.forEach(c => {
    o += box(c.x, 82, 152, 168, c.fill, c.st, 8, 2);
    o += tx(c.x + 76, 106, c.t, { fs: 13.5, fw: 700, fill: c.dk });
    o += tx(c.x + 76, 134, c.q, { fs: 12, fw: 700, fill: C.ink });
    o += line(c.x + 14, 154, c.x + 138, 154, c.st, 1, true);
    o += tx(c.x + 76, 182, c.a, { mono: true, fs: 11.5, fw: 700, fill: c.dk });
    o += tx(c.x + 76, 210, c.a2, { fs: 11.5, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

function likVsProb() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 38, "One Formula P(X | θ), Two Opposite Perspectives", { fs: 15, fw: 700, fill: D.purple.dark });

  // Left: Probability
  o += box(18, 78, 240, 172, C.lblue, C.blue, 8, 2);
  o += tx(138, 100, "1. Probability Distribution", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(138, 126, "Hold parameter θ FIXED", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(138, 150, "Vary dataset outcomes X", { fs: 12.5, fw: 600, fill: D.blue.dark });
  o += box(38, 170, 200, 38, "#FFFFFF", C.blue, 6, 1.5);
  o += tx(138, 189, "∑_X P(X | θ) = 1", { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(138, 228, "Forward data generation", { fs: 12, fill: C.muted });

  // Right: Likelihood
  o += box(282, 78, 240, 172, C.lamber, C.amber, 8, 2);
  o += tx(402, 100, "2. Likelihood Function L(θ)", { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(402, 126, "Hold observed data X FIXED", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(402, 150, "Vary candidate parameters θ", { fs: 12.5, fw: 600, fill: D.orange.dark });
  o += box(302, 170, 200, 38, "#FFFFFF", C.amber, 6, 1.5);
  o += tx(402, 189, "θ̂_MLE = argmax_θ P(X | θ)", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(402, 228, "Does NOT integrate to 1 over θ!", { fs: 12, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

function underflowNll() {
  const W = 540, H = 266;
  let o = "";
  // Top: Product underflow
  o += box(18, 16, 504, 102, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 38, "Raw Product Likelihood:  ∏ᵢ₌₁ⁿ p(xᵢ | θ)  →  Floating-Point Underflow!", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(270, 66, "For n = 10⁹ examples:  (1/2)¹⁰⁹ = 0.0 in float32 & float64", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(270, 94, "Product-rule gradient requires n products of n − 1 factors (no minibatching!)", { fs: 11.5, fw: 600, fill: D.red.dark });

  o += arrow(270, 120, 270, 144, C.green);
  o += tx(346, 133, "take −log(·)", { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  // Bottom: NLL sum
  o += box(18, 148, 504, 104, C.lgreen, C.green, 8, 2);
  o += tx(270, 170, "Negative Log-Likelihood (NLL):  ℓ(θ) = −∑ᵢ₌₁ⁿ log p(xᵢ | θ)", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(270, 198, "−log((1/2)¹⁰⁹) = 10⁹ log 2 ≈ 6.93 × 10⁸ nats  (safe float32!)", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(270, 226, "Additive gradient ∇ℓ(θ) = −∑ᵢ ∇log p(xᵢ|θ) enables unbiased minibatch SGD", { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function ceKlSplit() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 50, C.lblue, C.blue, 8, 2);
  o += tx(270, 32, "Average NLL Equals Cross-Entropy From Empirical Data to Model", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(270, 52, "(1/n) ℓ(θ) = −∑_x p̂_data(x) log p_θ(x) = CE(p̂_data, p_θ)", { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  // Stacked horizontal bar showing CE = H + KL
  const bx = 34, by = 86, hBar = 54;
  o += box(bx, by, 256, hBar, C.lgreen, C.green, 6, 2);
  o += tx(bx + 128, by + 18, "Empirical Entropy Floor H(p̂_data)", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(bx + 128, by + 38, "Fixed by data (0.617 nats for 9/13)", { mono: true, fs: 11.5, fw: 600, fill: C.ink });

  o += box(bx + 260, by, 212, hBar, C.lamber, C.amber, 6, 2);
  o += tx(bx + 366, by + 18, "KL Gap D_KL(p̂_data || p_θ) ≥ 0", { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(bx + 366, by + 38, "Reduced by training (0 at MLE)", { mono: true, fs: 11.5, fw: 600, fill: C.ink });

  // Bottom cards
  o += box(18, 160, 242, 88, "#FFFFFF", C.green, 8, 1.8);
  o += tx(139, 182, "Gibbs' Inequality Guarantee", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(139, 206, "CE(p̂, p_θ) ≥ H(p̂_data) always", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(139, 228, "Touches floor iff p_θ = p̂_data", { fs: 12, fill: C.muted });

  o += box(280, 160, 242, 88, "#FFFFFF", C.amber, 8, 1.8);
  o += tx(401, 182, "MLE = Forward KL Projection", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(401, 206, "argmin_θ CE = argmin_θ D_KL(p̂||p_θ)", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(401, 228, "Matches empirical data frequencies", { fs: 12, fill: C.muted });
  return svg(W, H, o);
}

function scoreZero() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 38, "Per-Example Score s(x; θ) = ∇_θ log p(x | θ) Has Zero Mean at θ*", { fs: 14, fw: 700, fill: D.purple.dark });

  // Left: Coin upward vs downward pulls canceling at theta* = 9/13
  o += box(18, 78, 242, 172, C.lblue, C.blue, 8, 2);
  o += tx(139, 100, "Tug-of-War at True θ*", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(139, 126, "Heads (x=1, prob θ*):", { fs: 12, fw: 700, fill: C.ink });
  o += tx(139, 146, "pulls right by s(1;θ*) = +1/θ*", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(139, 176, "Tails (x=0, prob 1−θ*):", { fs: 12, fw: 700, fill: C.ink });
  o += tx(139, 196, "pulls left by s(0;θ*) = −1/(1−θ*)", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(139, 228, "E[s] = θ*(1/θ*) − 1 = 0!", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  // Right: General 3-step proof from normalization
  o += box(280, 78, 242, 172, C.lgreen, C.green, 8, 2);
  o += tx(401, 100, "Proof From ∫ p(x | θ) dx = 1", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(401, 130, "1. Differentiate ∫ p(x|θ)dx = 1:", { fs: 12, fw: 600, fill: C.ink });
  o += tx(401, 152, "∇_θ ∫ p dx = ∫ ∇_θ p(x|θ) dx = 0", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(401, 180, "2. Log-derivative trick ∇p = p·∇log p:", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(401, 202, "∫ s(x;θ) p(x|θ) dx = E_θ[s] = 0", { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  o += tx(401, 230, "(Requires fixed support to swap ∇, ∫)", { fs: 11, fill: C.muted });
  return svg(W, H, o);
}

function fisherDuality() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, C.lamber, C.amber, 8, 2);
  o += tx(270, 38, "The Fisher Information Identity: Gradient Variance = Expected Curvature", { fs: 13.5, fw: 700, fill: D.orange.dark });

  // Left: First-order score variance
  o += box(18, 80, 214, 118, C.lblue, C.blue, 8, 2);
  o += tx(125, 104, "1st-Order View: Score Spread", { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(125, 134, "Var_θ[ s(x; θ) ]", { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  o += tx(125, 158, "= E_θ[ s(x;θ) s(x;θ)ᵀ ]", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(125, 182, "How wildly gradients fluctuate", { fs: 11.5, fill: C.muted });

  // Center equality badge
  o += box(240, 116, 60, 46, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 139, "I(θ) =", { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });

  // Right: Second-order NLL curvature
  o += box(308, 80, 214, 118, C.lgreen, C.green, 8, 2);
  o += tx(415, 104, "2nd-Order View: NLL Bowl", { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(415, 134, "E_θ[ −∇²_θ log p(x|θ) ]", { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(415, 158, "= Expected Hessian of NLL", { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(415, 182, "How sharply loss pins down θ", { fs: 11.5, fill: C.muted });

  // Bottom proof line
  o += box(18, 212, 504, 40, N.panel, C.gray, 7, 1.5);
  o += tx(270, 232, "Proof: Differentiate ∫ s·p dx = 0  ⇒  E[∇²log p] + E[s sᵀ] = 0", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function nllSeReadout() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 204, gw = 216;
  // NLL parabola around theta_hat = 0.692
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  let d = "";
  for (let s = 0; s <= 40; s++) {
    const u = (s / 40) * 2 - 1;
    const px = x0 + (s / 40) * gw;
    const py = 168 - u * u * 116;
    d += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d, C.blue, "none", 3);
  const xMin = x0 + gw / 2, yMin = 168;
  const xL = xMin - 48, xR = xMin + 48, yHalf = yMin - 48 * 48 / 180 * 3.2;
  o += line(xL, yHalf, xR, yHalf, D.orange.base, 2.5);
  o += line(xL, yHalf, xL, y0, D.orange.base, 1.5, true);
  o += line(xR, yHalf, xR, y0, D.orange.base, 1.5, true);
  o += circ(xMin, yMin, 5, C.blue, "#FFFFFF", 1.5);
  o += tx(xMin, y0 + 20, "θ̂ = 0.69", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(xMin, yHalf - 14, "±1 se = ±1/√ℓ''(θ̂)", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(xMin, 22, "NLL Curvature Readout", { fs: 14.5, fw: 700, fill: D.blue.dark });

  const rx = 282;
  o += box(rx, 24, 240, 104, C.lblue, C.blue, 8, 2);
  o += tx(rx + 120, 46, "Coin Curvature at θ̂ = 9/13", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(rx + 120, 72, "ℓ''(θ̂) = 9/θ̂² + 4/(1−θ̂)² = 61.0", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(rx + 120, 98, "se(θ̂) = 1 / √61.0 = 0.128", { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });

  o += box(rx, 142, 240, 104, C.lamber, C.amber, 8, 2);
  o += tx(rx + 120, 164, "95% Confidence Interval", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(rx + 120, 190, "0.69 ± 1.96(0.128) = [0.44, 0.94]", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(rx + 120, 216, "Contains 0.5! (±0.05 needs 330 flips)", { fs: 11.5, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

function asymptoticNorm() {
  const W = 540, H = 266;
  let o = "";
  const steps = [
    { y: 14, t: "Step 1: Linearize Score Sₙ(θ̂) = 0 Around True θ*", f: "0 ≈ Sₙ(θ*) + Sₙ'(θ*)(θ̂ − θ*)  ⇒  √n(θ̂ − θ*) ≈ [Sₙ(θ*)/√n] / [−Sₙ'(θ*)/n]", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 98, t: "Step 2: Apply CLT to Numerator & LLN to Denominator", f: "Numerator Sₙ(θ*)/√n → N(0, I(θ*))  (CLT)  |  Denominator −Sₙ'(θ*)/n → I(θ*)  (LLN)", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 182, t: "Step 3: Ratio Variance I(θ*) / I(θ*)² = I(θ*)⁻¹", f: "√n (θ̂ − θ*)  →ᵈ  N( 0, I(θ*)⁻¹ )   ⇒   Var(θ̂) ≈ (n I(θ*))⁻¹  (Cramér–Rao Optimal)", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s, i) => {
    o += box(18, s.y, 504, 68, s.fill, s.st, 8, 2);
    o += tx(270, s.y + 22, s.t, { fs: 13.5, fw: 700, fill: s.dk });
    o += tx(270, s.y + 48, s.f, { mono: true, fs: 10.5, fw: 700, fill: C.ink });
    if (i < 2) o += arrow(270, s.y + 69, 270, steps[i + 1].y - 2, s.st);
  });
  return svg(W, H, o);
}

function mapHhh() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 46, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 37, "Small-Sample Collapse on 3 Flips = HHH: Raw MLE vs. Beta(2,2) MAP", { fs: 13.5, fw: 700, fill: D.red.dark });

  // Left: MLE collapse
  o += box(18, 76, 240, 174, "#FFFFFF", D.red.base, 8, 2);
  o += tx(138, 100, "Raw MLE (No Prior)", { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(138, 128, "n_H = 3,  n_T = 0", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += box(36, 144, 204, 40, D.red.tint, D.red.base, 6, 1.8);
  o += tx(138, 164, "θ̂_MLE = 3 / 3 = 1.00", { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(138, 204, "Declares tails impossible (P=0)!", { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(138, 228, "Next tail incurs NLL = −log 0 = ∞", { mono: true, fs: 11, fill: C.ink });

  // Right: MAP with Beta(2,2) prior
  o += box(282, 76, 240, 174, C.lgreen, C.green, 8, 2);
  o += tx(402, 100, "MAP With Beta(2, 2) Prior", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(402, 128, "+1 imaginary H, +1 imaginary T", { fs: 12, fw: 700, fill: C.ink });
  o += box(300, 144, 204, 40, "#FFFFFF", C.green, 6, 1.8);
  o += tx(402, 164, "θ̂_MAP = (3+1)/(3+2) = 0.80", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(402, 204, "Keeps θ̂ away from boundary {0,1}", { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(402, 228, "Next tail costs −log(0.2) = 1.61 nats", { mono: true, fs: 10.5, fill: C.ink });
  return svg(W, H, o);
}

// ============================================================================
// 7. mdl-latent-em-elbo.md (10 diagrams)
// ============================================================================

function lostLabels() {
  const W = 540, H = 266;
  let o = "";
  // Top: Complete data (colored two clusters)
  o += box(18, 14, 504, 106, C.lgreen, C.green, 8, 2);
  o += tx(270, 34, "(a) Complete Data (Labels zᵢ ∈ {1, 2} Observed): Closed-Form 1-Line Fit!", { fs: 13.5, fw: 700, fill: D.green.dark });
  const y1 = 76;
  o += line(44, y1, 310, y1, C.ink, 1.8);
  [72, 86, 98, 110, 124].forEach(px => { o += circ(px, y1, 5.5, C.blue, "#FFFFFF", 1.5); });
  [216, 232, 246, 262].forEach(px => { o += circ(px, y1, 5.5, D.orange.base, "#FFFFFF", 1.5); });
  o += tx(98, y1 + 22, "300 pts: N(−2, 0.8²)", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += tx(238, y1 + 22, "200 pts: N(1.5, 0.6²)", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(416, 76, "π̂_k = n_k/n,  μ̂_k = x̄_k", { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  // Bottom: Observed unlabeled data (gray dots)
  o += box(18, 136, 504, 114, C.lamber, C.amber, 8, 2);
  o += tx(270, 158, "(b) Missing Labels (Latent zᵢ Lost): Log of a Sum Entangles All Parameters", { fs: 13.5, fw: 700, fill: D.orange.dark });
  const y2 = 198;
  o += line(44, y2, 310, y2, C.ink, 1.8);
  [72, 86, 98, 110, 124, 216, 232, 246, 262].forEach(px => { o += circ(px, y2, 5.5, C.gray, "#FFFFFF", 1.5); });
  o += tx(170, y2 + 24, "500 unlabeled gray observations xᵢ", { fs: 12, fw: 700, fill: C.muted });
  o += tx(416, 192, "Guess soft qᵢ(z) ∈ [0,1]", { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(416, 216, "→ Climb ELBO lower bound!", { fs: 12, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

function latentDag() {
  const W = 540, H = 260;
  let o = "";
  // Left: Directed graphical model theta -> z_i -> x_i
  o += box(18, 18, 224, 224, "#FFFFFF", C.blue, 8, 2);
  o += tx(130, 40, "Generative Latent DAG", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += box(36, 62, 188, 162, N.panel, C.gray, 8, 1.5, true);
  o += tx(192, 210, "i = 1..n", { mono: true, fs: 12, fw: 700, fill: C.muted });

  o += circ(130, 96, 22, "#FFFFFF", C.purple, 2.2, true);
  o += tx(130, 96, "zᵢ", { mono: true, fs: 16, fw: 700, fill: D.purple.dark });
  o += tx(66, 96, "hidden", { fs: 11.5, fw: 600, fill: D.purple.dark });

  o += arrow(130, 120, 130, 152, C.blue);

  o += circ(130, 178, 22, C.lblue, C.blue, 2.2);
  o += tx(130, 178, "xᵢ", { mono: true, fs: 16, fw: 700, fill: D.blue.dark });
  o += tx(66, 178, "observed", { fs: 11.5, fw: 700, fill: D.blue.dark });

  // Right: Joint vs Marginal formulas
  const rx = 260;
  o += box(rx, 18, 262, 102, C.lblue, C.blue, 8, 2);
  o += tx(rx + 131, 40, "1. Complete Joint Density", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(rx + 131, 68, "p(x, z | θ) = p(z | θ) p(x | z, θ)", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(rx + 131, 94, "Prior over causes × conditional", { fs: 12, fill: C.muted });

  o += box(rx, 138, 262, 104, C.lamber, C.amber, 8, 2);
  o += tx(rx + 131, 160, "2. Marginal Likelihood (Evidence)", { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(rx + 131, 188, "p(x | θ) = ∑_z p(z | θ) p(x | z, θ)", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(rx + 131, 216, "Sums over all hidden paths to x", { fs: 12, fill: C.ink });
  return svg(W, H, o);
}

function oracleVsMixture() {
  const W = 540, H = 266;
  let o = "";
  // Left: Oracle log of product
  o += box(18, 16, 240, 234, C.lgreen, C.green, 8, 2);
  o += tx(138, 40, "Complete-Data Oracle (zᵢ Known)", { fs: 13, fw: 700, fill: D.green.dark });
  o += box(28, 60, 220, 46, "#FFFFFF", C.green, 6, 1.8);
  o += tx(138, 83, "log [ π_{zᵢ} N(xᵢ; μ_{zᵢ}, σ²) ]", { mono: true, fs: 10.5, fw: 700, fill: D.green.dark });
  o += tx(138, 128, "Logarithm hits exp(·) directly!", { fs: 12, fw: 700, fill: C.ink });
  o += tx(138, 156, "Decouples into K quadratics:", { fs: 11.5, fill: C.muted });
  o += tx(138, 184, "π̂_k = n_k / n", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(138, 208, "μ̂_k = (1/n_k) ∑_{zᵢ=k} xᵢ", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(138, 232, "σ̂²_k = (1/n_k) ∑_{zᵢ=k}(xᵢ−μ̂_k)²", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  // Right: Observed mixture log of sum
  o += box(282, 16, 240, 234, D.red.tint, D.red.base, 8, 2);
  o += tx(402, 40, "Observed Mixture (zᵢ Hidden)", { fs: 13, fw: 700, fill: D.red.dark });
  o += box(292, 60, 220, 46, "#FFFFFF", D.red.base, 6, 1.8);
  o += tx(402, 83, "log [ ∑_k π_k N(xᵢ; μ_k, σ²_k) ]", { mono: true, fs: 10.5, fw: 700, fill: D.red.dark });
  o += tx(402, 128, "Sum trapped INSIDE logarithm!", { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(402, 156, "Logarithm cannot reach exp(·)", { fs: 11.5, fill: C.ink });
  o += tx(402, 184, "∇ℓ = 0 couples all 3K params", { fs: 12, fw: 700, fill: C.ink });
  o += tx(402, 208, "across every data point xᵢ", { fs: 11.5, fill: C.ink });
  o += tx(402, 232, "⇒ No closed-form MLE!", { fs: 12, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

function softResp() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 198, gw = 236, gh = 136, mid = x0 + gw / 2;
  // Two Gaussians N(-1, 1) and N(1, 1)
  let d1 = "", d2 = "";
  for (let s = 0; s <= 40; s++) {
    const u = (s / 40) * 6 - 3;
    const px = x0 + (s / 40) * gw;
    const py1 = y0 - gh * Math.exp(-0.5 * (u + 1) * (u + 1));
    const py2 = y0 - gh * Math.exp(-0.5 * (u - 1) * (u - 1));
    d1 += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py1.toFixed(1)} `;
    d2 += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py2.toFixed(1)} `;
  }
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  o += path(d1, C.blue, "none", 2.8);
  o += path(d2, D.orange.base, "none", 2.8);

  // Vertical slice at x = 0.5
  const xObs = mid + (0.5 / 3) * (gw / 2);
  const yH1 = y0 - gh * Math.exp(-1.125);
  const yH2 = y0 - gh * Math.exp(-0.125);
  o += line(xObs, y0, xObs, yH2, D.purple.base, 2, true);
  o += circ(xObs, yH1, 5, C.blue, "#FFFFFF", 1.5);
  o += circ(xObs, yH2, 5, D.orange.base, "#FFFFFF", 1.5);

  o += tx(mid, 22, "Two Components N(−1,1) & N(+1,1) at x = 0.5", { fs: 13.5, fw: 700, fill: C.ink });
  o += tx(mid - 56, 54, "k = 1", { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(mid + 64, 54, "k = 2", { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(xObs, y0 + 20, "x = 0.5", { mono: true, fs: 12, fw: 700, fill: D.purple.dark });

  const rx = 296;
  o += box(rx, 28, 226, 108, D.purple.tint, C.purple, 8, 2);
  o += tx(rx + 113, 50, "Soft Bayes Responsibility", { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(rx + 113, 76, "r₁ = 1 / (1 + e) ≈ 0.27", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(rx + 113, 102, "r₁ = 27% left,  r₂ = 73% right", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  o += box(rx, 150, 226, 92, C.lgreen, C.green, 8, 2);
  o += tx(rx + 113, 172, "Smooth Uncertainty Scaling", { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(rx + 113, 196, "At x = 0:  50% / 50% tie", { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(rx + 113, 220, "At x = 3:  0.2% / 99.8% sure", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function jensenElbo() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 206, gw = 216;
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  o += line(x0, 32, x0, y0, C.ink, 1.8);
  // Concave log u curve
  let d = "";
  for (let s = 2; s <= 40; s++) {
    const u = (s / 40) * 3;
    const px = x0 + (s / 40) * gw;
    const py = 180 - Math.log(u + 0.3) * 90;
    d += (s === 2 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d, C.blue, "none", 3);
  // Secant chord between u1 and u2
  const u1 = 0.45, u2 = 2.55;
  const x1 = x0 + (u1 / 3) * gw, y1 = 180 - Math.log(u1 + 0.3) * 90;
  const x2 = x0 + (u2 / 3) * gw, y2 = 180 - Math.log(u2 + 0.3) * 90;
  o += line(x1, y1, x2, y2, D.orange.base, 2.2, true);
  const xm = (x1 + x2) / 2, ymChord = (y1 + y2) / 2;
  const ymCurve = 180 - Math.log((u1 + u2) / 2 + 0.3) * 90;
  o += line(xm, ymCurve, xm, ymChord, D.red.base, 3);
  o += circ(xm, ymCurve, 5, C.blue, "#FFFFFF", 1.5);
  o += circ(xm, ymChord, 5, D.orange.base, "#FFFFFF", 1.5);
  o += tx(x0 + 96, 52, "Curve: log E_q[p/q]", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += tx(x0 + 138, 184, "Chord: E_q[log(p/q)]", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw / 2, 20, "Jensen's Inequality on Concave log(·)", { fs: 13.5, fw: 700, fill: C.ink });

  const rx = 282;
  o += box(rx, 26, 240, 104, C.lblue, C.blue, 8, 2);
  o += tx(rx + 120, 48, "1. Multiply & Divide by q(z)", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(rx + 120, 74, "log p(x|θ) = log ∑_z q(z)[p/q]", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(rx + 120, 100, "= log E_q[ p(x,z|θ) / q(z) ]", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  o += box(rx, 144, 240, 104, C.lgreen, C.green, 8, 2);
  o += tx(rx + 120, 166, "2. Push log Inside via Jensen", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(rx + 120, 192, "≥ ∑_z q(z) log[ p(x,z|θ) / q(z) ]", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(rx + 120, 218, "=: L(q, θ)  (ELBO Lower Bound)", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function elboAnatomy() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 50, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 39, "L(q, θ)  =  E_{q(z)}[ log p(x, z | θ) ]   +   H(q)", { mono: true, fs: 15, fw: 700, fill: D.purple.dark });

  // Left: Expected complete-data log-likelihood
  o += box(18, 82, 240, 168, C.lblue, C.blue, 8, 2);
  o += tx(138, 106, "1. Expected Complete Log-Lik", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(138, 134, "∑_z q(z) log p(x, z | θ)", { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(138, 164, "• Tractable oracle objective!", { fs: 12.5, fw: 600, fill: D.blue.dark });
  o += tx(138, 190, "• Soft weights q(z) stand in", { fs: 12, fill: C.ink });
  o += tx(138, 210, "  for missing oracle labels", { fs: 12, fill: C.ink });
  o += tx(138, 234, "• Optimized over θ in M-step", { fs: 12, fw: 700, fill: D.blue.dark });

  // Right: Entropy bonus H(q)
  o += box(282, 82, 240, 168, C.lamber, C.amber, 8, 2);
  o += tx(402, 106, "2. Shannon Entropy Bonus H(q)", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(402, 134, "−∑_z q(z) log q(z) ≥ 0", { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(402, 164, "• Penalizes overconfident hard", { fs: 12.5, fw: 600, fill: D.orange.dark });
  o += tx(402, 186, "  0/1 guesses (where H(q) = 0)", { fs: 12, fill: C.ink });
  o += tx(402, 210, "• Independent of parameters θ", { fs: 12, fill: C.ink });
  o += tx(402, 234, "  (drops out in M-step!)", { fs: 12, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

function elboGapBars() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 12, 504, 42, D.purple.tint, C.purple, 8, 1.8);
  o += tx(270, 33, "At x = 0.5:  True log p(x) = −1.42 nats  =  ELBO L(q) + D_KL(q || p)", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  const cases = [
    ["1. Exact Posterior q = (0.27, 0.73)", "E_q[log p] = −2.00, H(q) = +0.58", "ELBO = −1.42", "KL Gap = 0.00 (Tight!)", C.lgreen, C.green, D.green.dark],
    ["2. Uniform Guess q = (0.50, 0.50)", "E_q[log p] = −2.23, H(q) = +0.69", "ELBO = −1.54", "KL Gap = 0.12 nats", C.lblue, C.blue, D.blue.dark],
    ["3. Hard Guess q = (0.00, 1.00)", "E_q[log p] = −1.73, H(q) = +0.00", "ELBO = −1.73", "KL Gap = 0.31 (2.6×!)", D.red.tint, D.red.base, D.red.dark],
  ];
  cases.forEach(([title, sub, elbo, kl, fill, st, dk], i) => {
    const y = 66 + i * 64;
    o += box(18, y, 504, 54, fill, st, 7, 2);
    o += tx(168, y + 17, title, { fs: 13, fw: 700, fill: dk });
    o += tx(168, y + 38, sub, { mono: true, fs: 11, fw: 600, fill: C.ink });
    o += box(334, y + 8, 178, 38, "#FFFFFF", st, 6, 1.5);
    o += tx(423, y + 19, elbo, { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    o += tx(423, y + 36, kl, { mono: true, fs: 11, fw: 700, fill: dk });
  });
  return svg(W, H, o);
}

function emLoop() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 44, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 36, "EM as Coordinate Ascent on Total ELBO  ∑ᵢ₌₁ⁿ L(qᵢ, θ)", { fs: 14.5, fw: 700, fill: D.purple.dark });

  // E-step box on left
  o += box(20, 80, 222, 166, C.lblue, C.blue, 9, 2);
  o += tx(131, 104, "E-Step: Optimize qᵢ", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(131, 128, "Hold θ⁽ᵗ⁾ fixed", { fs: 12, fw: 600, fill: C.muted });
  o += box(34, 144, 194, 42, "#FFFFFF", C.blue, 6, 1.8);
  o += tx(131, 165, "qᵢ⁽ᵗ⁾(z) ← p(z | xᵢ, θ⁽ᵗ⁾)", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += tx(131, 206, "Closes KL gap to 0:", { fs: 12, fw: 700, fill: C.ink });
  o += tx(131, 226, "∑ᵢ L(qᵢ⁽ᵗ⁾, θ⁽ᵗ⁾) = log P(X|θ⁽ᵗ⁾)", { mono: true, fs: 10.5, fw: 700, fill: D.blue.dark });

  // Arrows between E and M
  o += arrow(246, 134, 294, 134, C.blue);
  o += tx(270, 118, "r_{ik}", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += arrow(294, 192, 246, 192, C.green);
  o += tx(270, 210, "θ⁽ᵗ⁺¹⁾", { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  // M-step box on right
  o += box(298, 80, 222, 166, C.lgreen, C.green, 9, 2);
  o += tx(409, 104, "M-Step: Optimize θ", { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(409, 128, "Hold qᵢ⁽ᵗ⁾ fixed (H(qᵢ) drops)", { fs: 12, fw: 600, fill: C.muted });
  o += box(312, 144, 194, 42, "#FFFFFF", C.green, 6, 1.8);
  o += tx(409, 165, "max_θ ∑ᵢ E_{qᵢ}[log p(xᵢ,z|θ)]", { mono: true, fs: 10.5, fw: 700, fill: D.green.dark });
  o += tx(409, 206, "Climbs lower bound:", { fs: 12, fw: 700, fill: C.ink });
  o += tx(409, 226, "log P(X|θ⁽ᵗ⁺¹⁾) ≥ log P(X|θ⁽ᵗ⁾)", { mono: true, fs: 10.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function gmmSpikes() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 204, gw = 216;
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  // Broad component 1 covering n-1 points
  let d1 = "";
  for (let s = 0; s <= 32; s++) {
    const u = (s / 32) * 4 - 2;
    const px = x0 + (s / 32) * gw;
    const py = y0 - 52 * Math.exp(-0.5 * u * u);
    d1 += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d1, C.blue, "none", 2.5);
  // Sharp spike component 2 on x_1
  const xSpike = x0 + 164;
  let d2 = `M ${xSpike - 16},${y0} L ${xSpike - 4},${y0 - 20} L ${xSpike},${y0 - 156} L ${xSpike + 4},${y0 - 20} L ${xSpike + 16},${y0}`;
  o += path(d2, D.red.base, "none", 2.8);
  o += circ(xSpike, y0, 5, D.red.base, "#FFFFFF", 1.5);
  o += tx(xSpike, y0 + 20, "μ₂ = x₁", { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(xSpike - 44, y0 - 142, "σ₂ = ε → 0", { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(x0 + 76, y0 - 64, "Comp 1 (n−1 pts)", { fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(x0 + gw / 2, 22, "Degenerate GMM Variance Spike", { fs: 14.5, fw: 700, fill: D.red.dark });

  const rx = 282;
  o += box(rx, 26, 240, 104, D.red.tint, D.red.base, 8, 2);
  o += tx(rx + 120, 48, "Unbounded Mixture Likelihood", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(rx + 120, 74, "At x₁: p(x₁) ≥ π₂ / (√(2π) ε) → +∞", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(rx + 120, 100, "⇒ NLL ℓ(θ) ≤ log ε + C → −∞!", { mono: true, fs: 12, fw: 700, fill: D.red.dark });

  o += box(rx, 144, 240, 104, C.lgreen, C.green, 8, 2);
  o += tx(rx + 120, 166, "Practical Remedies", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(rx + 120, 192, "• Variance floor σ²_k ≥ σ²_min > 0", { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(rx + 120, 218, "• Inverse-Gamma prior on σ²_k", { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function vaeAmortized() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 44, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 36, "Variational Autoencoder (VAE): Amortized Joint SGD on the ELBO", { fs: 14.5, fw: 700, fill: D.purple.dark });

  // Pipeline: x_i -> Encoder q_phi(z|x) -> mu, sigma -> z = mu + sigma*eps -> Decoder p_theta(x|z)
  const nodes = [
    { x: 16, w: 76, t: "Input xᵢ", s: "data", fill: N.panel, st: C.gray, dk: C.ink },
    { x: 110, w: 124, t: "Encoder q_φ(z|x)", s: "μ_φ(x), log σ_φ(x)", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 252, w: 128, t: "Reparameterize", s: "z = μ + σ ⊙ ε", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 398, w: 126, t: "Decoder p_θ(x|z)", s: "reconstruct x̂ᵢ", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  nodes.forEach((n, i) => {
    o += box(n.x, 76, n.w, 68, n.fill, n.st, 8, 2);
    o += tx(n.x + n.w / 2, 98, n.t, { fs: 13, fw: 700, fill: n.dk });
    o += tx(n.x + n.w / 2, 122, n.s, { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    if (i < 3) o += arrow(n.x + n.w + 2, 110, nodes[i + 1].x - 2, 110, C.purple);
  });

  // Bottom comparison: Exact EM vs Amortized VAE
  o += box(16, 162, 508, 88, "#FFFFFF", C.purple, 8, 2);
  o += tx(270, 184, "Why Amortized Variational Inference Replaces Exact EM", { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(270, 208, "Exact posterior p(z|x,θ) is intractable for deep nonlinear decoders p_θ(x|z)", { fs: 12, fw: 600, fill: C.ink });
  o += tx(270, 230, "Encoder φ shares weights across all xᵢ: joint SGD on ∑ᵢ L(q_φ(·|xᵢ), θ)", { mono: true, fs: 11, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// ============================================================================
// 8. mdl-bayesian-computation.md (3 diagrams)
// ============================================================================

function predictiveIntegral() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 38, "Predictive Integral: p(y*|x*, D) = ∫ p(y*|x*, θ) p(θ|D) dθ", { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });

  // Left: Unnormalized target
  o += box(18, 78, 240, 172, C.lblue, C.blue, 8, 2);
  o += tx(138, 102, "Known Up to a Constant", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(138, 128, "p̃(θ) = p(D | θ) p(θ)", { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  o += tx(138, 156, "Evaluated in O(n) at any θ,", { fs: 12.5, fill: C.ink });
  o += tx(138, 178, "plus gradient ∇log p̃(θ) and", { fs: 12.5, fill: C.ink });
  o += tx(138, 200, "negative Hessian H(θ)", { fs: 12.5, fill: C.ink });
  o += tx(138, 228, "Missing: p(D) = ∫ p̃(θ) dθ", { mono: true, fs: 12, fw: 700, fill: D.red.dark });

  // Right: How 4 methods cancel p(D)
  o += box(282, 78, 240, 172, C.lgreen, C.green, 8, 2);
  o += tx(402, 102, "How p(D) Cancels Out", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(402, 130, "1. Laplace: ∇log p(D) = 0 at mode", { fs: 12, fw: 600, fill: C.ink });
  o += tx(402, 156, "2. Importance: ∑ w_s h_s / ∑ w_s", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(402, 182, "3. MCMC: ratio p̃(θ') / p̃(θ)", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(402, 208, "4. VI: shifts ELBO by const log p(D)", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(402, 232, "Every method uses only p̃(θ)!", { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function predictiveShift() {
  const W = 540, H = 274;
  let o = "";
  o += tx(270, 20, "Walking From MAP Plug-In (−5.49) to Exact Predictive (−3.98) at x* = −3", { fs: 13.5, fw: 700, fill: C.ink });
  const steps = [
    ["1. MAP Plug-In Log-Prob", "−5.49", "mode θ_MAP = (−2.14, 1.12)", C.lgray, C.gray, C.ink],
    ["2. Shift to Mean Logit μ", "−6.06  (−0.57)", "skewness steepens mean slope", D.red.tint, D.red.base, D.red.dark],
    ["3. Variance Boost μ + s²/2", "−3.19  (+2.87)", "logit spread s²/2 = +2.87", C.lgreen, C.green, D.green.dark],
    ["4. Skewed Logit log E[eᵃ]", "−3.66  (−0.47)", "short upper tail trims MGF", C.lamber, C.amber, D.orange.dark],
    ["5. Saturation log E[σ(a)]", "−3.98  (−0.32)", "Net: +1.51 nats (4.6× P!)", D.purple.tint, C.purple, D.purple.dark],
  ];
  steps.forEach(([label, val, note, fill, st, dk], i) => {
    const y = 34 + i * 46;
    o += box(16, y, 184, 38, fill, st, 6, 1.8);
    o += tx(108, y + 19, label, { fs: 11.5, fw: 700, fill: dk });
    o += box(206, y, 118, 38, "#FFFFFF", st, 6, 1.8);
    o += tx(265, y + 19, val, { mono: true, fs: 11.5, fw: 700, fill: dk });
    o += box(330, y, 194, 38, N.panel, C.gray, 6, 1.5);
    o += tx(427, y + 19, note, { fs: 11, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

function detailedBalance() {
  const W = 540, H = 260;
  let o = "";
  o += box(18, 14, 504, 46, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 37, "Detailed Balance (Reversibility): π(θ) T(θ'|θ) = π(θ') T(θ|θ')", { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });

  // Two state nodes with equal opposing fluxes
  o += circ(116, 134, 46, C.lblue, C.blue, 2.5);
  o += tx(116, 124, "State θ", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(116, 146, "mass π(θ)", { mono: true, fs: 11.5, fw: 700, fill: C.ink });

  o += circ(424, 134, 46, C.lgreen, C.green, 2.5);
  o += tx(424, 124, "State θ'", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(424, 146, "mass π(θ')", { mono: true, fs: 11.5, fw: 700, fill: C.ink });

  o += arrow(168, 114, 372, 114, C.blue);
  o += tx(270, 96, "Forward: π(θ) g(θ'|θ) α(θ,θ')", { mono: true, fs: 10.5, fw: 700, fill: D.blue.dark });

  o += arrow(372, 154, 168, 154, C.green);
  o += tx(270, 174, "Reverse: π(θ') g(θ|θ') α(θ',θ)", { mono: true, fs: 10.5, fw: 700, fill: D.green.dark });

  // Bottom bar: Why both equal min{...} and imply stationarity
  o += box(18, 200, 504, 48, C.lamber, C.amber, 8, 1.8);
  o += tx(270, 216, "Both fluxes equal min{ π(θ)g(θ'|θ), π(θ')g(θ|θ') } (symmetric in θ ↔ θ')", { mono: true, fs: 11, fw: 700, fill: D.orange.dark });
  o += tx(270, 236, "Integrating over θ gives ∫ π(θ)T(θ'|θ)dθ = π(θ') (Stationarity)", { mono: true, fs: 11, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// ============================================================================
// 9. mdl-concentration-generalization.md (9 diagrams)
// ============================================================================

function threeIntervals() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 12, 504, 42, C.lblue, C.blue, 8, 1.8);
  o += tx(270, 33, "Classifier With 830 Errors in n = 10,000 Test Images (p̂ = 8.3% Error)", { fs: 13.5, fw: 700, fill: D.blue.dark });

  const rows = [
    ["1. Asymptotic CLT", "±0.54% (δ=0.05)", "±1.35% (δ=10⁻⁶)", "Exact only as n→∞; plug-in σ̂", C.lblue, C.blue, D.blue.dark],
    ["2. Chebyshev (σ²/nt²)", "±2.24% (δ=0.05)", "±500% (USELESS!)", "1/δ scaling explodes at 10⁻⁶", D.red.tint, D.red.base, D.red.dark],
    ["3. Hoeffding (2e⁻²ⁿᵗ²)", "±1.36% (δ=0.05)", "±2.69% (δ=10⁻⁶)", "Finite-n guarantee; √log(2/δ)", C.lamber, C.amber, D.orange.dark],
    ["4. Bernstein (Var-Aware)", "±0.79% (δ=0.05)", "±1.62% (δ=10⁻⁶)", "Finite-n + uses small p(1−p)!", C.lgreen, C.green, D.green.dark],
  ];
  rows.forEach(([m, c95, c1m, note, fill, st, dk], i) => {
    const y = 64 + i * 48;
    o += box(16, y, 168, 40, fill, st, 6, 1.8);
    o += tx(100, y + 20, m, { fs: 11.5, fw: 700, fill: dk });
    o += box(190, y, 152, 40, "#FFFFFF", st, 6, 1.8);
    o += tx(266, y + 13, `95%: ${c95}`, { mono: true, fs: 10.5, fw: 700, fill: C.ink });
    o += tx(266, y + 29, `10⁻⁶: ${c1m}`, { mono: true, fs: 10.5, fw: 700, fill: dk });
    o += box(348, y, 176, 40, N.panel, C.gray, 6, 1.5);
    o += tx(436, y + 20, note, { fs: 11, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

function fairCoinMgf() {
  const W = 540, H = 266;
  let o = "";
  const steps = [
    { y: 14, t: "Step 1: Single Centered Flip Zᵢ ∈ {−½, +½} Has Gaussian-Bounded MGF", f: "E[e^{λZᵢ}] = cosh(λ/2) = ∑ (λ/2)²ᵏ/(2k)! ≤ ∑ (λ²/8)ᵏ/k! = e^{λ²/8}", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 98, t: "Step 2: Chernoff on Sum of n Independent Flips Factorizes Into Product", f: "P(p̂ − ½ ≥ t) ≤ e^{-λnt} ∏ᵢ E[e^{λZᵢ}] ≤ exp( −λnt + nλ²/8 )", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 182, t: "Step 3: Minimize Parabola Over λ > 0 at Vertex λ* = 4t", f: "−(4t)nt + n(4t)²/8 = −2nt²   ⇒   P( |p̂ − ½| ≥ t ) ≤ 2 exp( −2 n t² )", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s, i) => {
    o += box(18, s.y, 504, 68, s.fill, s.st, 8, 2);
    o += tx(270, s.y + 22, s.t, { fs: 13, fw: 700, fill: s.dk });
    o += tx(270, s.y + 48, s.f, { mono: true, fs: 11, fw: 700, fill: C.ink });
    if (i < 2) o += arrow(270, s.y + 69, 270, steps[i + 1].y - 2, s.st);
  });
  return svg(W, H, o);
}

function hoeffdingTilting() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 38, "Hoeffding's Lemma: E[e^{λX}] ≤ exp(λ²(b−a)²/8) for X ∈ [a,b], E[X]=0", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  // Left: Max variance on [a, b]
  o += box(18, 78, 240, 172, C.lblue, C.blue, 8, 2);
  o += tx(138, 100, "1. Max Variance on [a, b]", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(138, 126, "Midpoint c = (a + b) / 2:", { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(138, 150, "|Y − c| ≤ (b − a) / 2  always", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += box(36, 168, 204, 40, "#FFFFFF", C.blue, 6, 1.5);
  o += tx(138, 188, "Var(Y) ≤ (b − a)² / 4", { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(138, 228, "Attained by 50/50 atoms at {a, b}", { fs: 11.5, fill: C.muted });

  // Right: Tilted distribution & Taylor
  o += box(282, 78, 240, 172, C.lgreen, C.green, 8, 2);
  o += tx(402, 100, "2. Tilted Curvature ψ''(λ)", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(402, 126, "ψ(λ) = log E[e^{λX}], ψ(0)=ψ'(0)=0", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(402, 150, "Tilted q_λ(dx) ∝ e^{λx}p(dx) on [a,b]", { mono: true, fs: 10.5, fw: 600, fill: D.green.dark });
  o += box(300, 168, 204, 40, "#FFFFFF", C.green, 6, 1.5);
  o += tx(402, 188, "ψ''(λ) = Var_{q_λ}(X) ≤ (b−a)²/4", { mono: true, fs: 10.5, fw: 700, fill: D.green.dark });
  o += tx(402, 228, "Taylor: ψ(λ) = ½λ²ψ''(ξ) ≤ λ²(b−a)²/8", { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function subexpWindow() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 204, gw = 216;
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  const xMid = x0 + 96, xAsym = x0 + 188;
  o += line(xMid, 32, xMid, y0, C.gray, 1.5);
  // Shaded admissible window |lambda| <= 1/alpha
  o += box(xMid - 48, 36, 96, y0 - 36, C.lgreen, C.green, 0, 1.5, true);
  // Vertical pole at lambda = 1/2
  o += line(xAsym, 32, xAsym, y0, D.red.base, 2, true);
  // Curve blowing up near xAsym
  let d = "";
  for (let s = 0; s <= 32; s++) {
    const u = (s / 32) * 1.35 - 0.55; // u in [-0.55, 0.80], pole at 0.92
    const px = xMid + u * 96;
    const val = u * u * 56 / Math.max(0.12, 0.92 - u);
    const py = Math.max(36, y0 - 16 - val);
    d += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d, C.blue, "none", 2.8);
  o += tx(xMid, y0 + 20, "λ=0", { mono: true, fs: 12, fill: C.ink });
  o += tx(xMid + 48, y0 + 20, "1/α", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(xAsym + 6, y0 + 20, "λ=1/2", { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(xMid, 22, "Squared Gaussian MGF Pole", { fs: 14, fw: 700, fill: C.ink });
  o += tx(xMid, 54, "|λ| ≤ 1/α", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(xAsym + 24, 72, "pole", { mono: true, fs: 11, fw: 700, fill: D.red.dark });

  const rx = 282;
  o += box(rx, 24, 240, 106, C.lblue, C.blue, 8, 2);
  o += tx(rx + 120, 46, "Small Deviation (t ≤ ν²/α)", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(rx + 120, 72, "Vertex λ* = t/ν² lies inside window", { fs: 12, fw: 600, fill: C.ink });
  o += tx(rx + 120, 98, "Gaussian tail: 2 exp(−nt²/(2ν²))", { mono: true, fs: 11, fw: 700, fill: D.blue.dark });

  o += box(rx, 144, 240, 106, C.lamber, C.amber, 8, 2);
  o += tx(rx + 120, 166, "Large Deviation (t > ν²/α)", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(rx + 120, 192, "Clamped at boundary λ* = 1/α", { fs: 12, fw: 600, fill: C.ink });
  o += tx(rx + 120, 218, "Exponential tail: 2 exp(−nt/(2α))", { mono: true, fs: 11, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

function randomOrtho() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 198, gw = 216, gh = 136, mid = x0 + gw / 2;
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  // Narrow bell of cosine similarity in R^784
  let d = "";
  for (let s = 0; s <= 40; s++) {
    const u = (s / 40) * 6 - 3;
    const px = x0 + (s / 40) * gw;
    const py = y0 - gh * Math.exp(-1.4 * u * u);
    d += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${py.toFixed(1)} `;
  }
  o += path(d + `L ${x0 + gw},${y0} L ${x0},${y0} Z`, "none", C.lblue, 0);
  o += path(d, C.blue, "none", 2.8);
  o += line(mid, y0, mid, y0 - gh, D.blue.dark, 1.5, true);
  o += tx(mid, 22, "Cosine Similarity cos(x, y) in ℝ⁷⁸⁴", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(mid, y0 + 20, "0 (90°)", { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(x0 + 24, y0 + 20, "−0.10", { mono: true, fs: 11.5, fill: C.muted });
  o += tx(x0 + gw - 24, y0 + 20, "+0.10", { mono: true, fs: 11.5, fill: C.muted });
  o += tx(mid, 238, "SD = 1/√784 = 0.036  (89.9° ± 2.0°)", { mono: true, fs: 12, fw: 700, fill: C.ink });

  const rx = 282;
  o += box(rx, 24, 240, 104, C.lblue, C.blue, 8, 2);
  o += tx(rx + 120, 46, "Unequal-Range Hoeffding", { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(rx + 120, 72, "Yᵢ = uᵢxᵢ ∈ [−|uᵢ|, |uᵢ|], ∑(2|uᵢ|)²=4", { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  o += tx(rx + 120, 98, "P(|cos(x,u)| ≥ ε) ≤ 2 exp(−dε²/2)", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  o += box(rx, 144, 240, 104, C.lgreen, C.green, 8, 2);
  o += tx(rx + 120, 166, "Embedding Baseline in ℝᵈ", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(rx + 120, 192, "Random vectors: E|cos| ≈ √(2/πd)", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(rx + 120, 218, "At d=784: cos=0.30 is >8σ signal!", { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function exponentialPacking() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 38, "Exact Orthogonality vs. Quasi-Orthogonal Packing in ℝᵈ", { fs: 15, fw: 700, fill: D.purple.dark });

  // Left: Exact orthogonality
  o += box(18, 78, 232, 172, C.lblue, C.blue, 8, 2);
  o += tx(134, 102, "Strict Orthogonality", { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(134, 128, "cos(xᵢ, xⱼ) = 0  for all i ≠ j", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += box(38, 148, 192, 42, "#FFFFFF", C.blue, 6, 1.8);
  o += tx(134, 169, "Max capacity: m = d", { mono: true, fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(134, 212, "In ℝ⁷⁸⁴: at most 784", { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(134, 232, "mutually perpendicular axes", { fs: 12, fill: C.muted });

  // Right: Quasi-orthogonal packing
  o += box(270, 78, 252, 172, C.lgreen, C.green, 8, 2);
  o += tx(396, 102, "Quasi-Orthogonal (|cos| < ε)", { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(396, 128, "Union bound over m(m−1)/2 pairs:", { fs: 12, fw: 600, fill: C.ink });
  o += box(288, 144, 216, 46, "#FFFFFF", C.green, 6, 1.8);
  o += tx(396, 167, "m ≤ √δ · exp( d ε² / 4 )", { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(396, 210, "In ℝ⁷⁸⁴ (δ=0.05): m ≈ 570 at ε=0.2,", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(396, 232, "and m ≈ 10,000,000 at ε = 0.3!", { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function depthNormDrift() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, C.lblue, C.blue, 8, 2);
  o += tx(270, 38, "He Initialization (W_k ~ N(0, 2/d)) Across K ReLU Layers of Width d", { fs: 14, fw: 700, fill: D.blue.dark });

  // Layer chain h0 -> h1 -> ... -> hK
  const lx = [24, 152, 280, 408];
  const labs = ["h₀", "h₁ = ReLU(W₁h₀)", "h₂ = ReLU(W₂h₁)", "h_K (depth K)"];
  const subs = ["‖h₀‖²", "ρ₁ = ‖h₁‖²/‖h₀‖²", "ρ₂ = ‖h₂‖²/‖h₁‖²", "∏_{k=1}^K ρ_k"];
  lx.forEach((x, i) => {
    o += box(x, 78, 108, 62, i === 3 ? C.lamber : "#FFFFFF", i === 3 ? C.amber : C.blue, 7, 2);
    o += tx(x + 54, 98, labs[i], { mono: true, fs: 11.5, fw: 700, fill: i === 3 ? D.orange.dark : D.blue.dark });
    o += tx(x + 54, 122, subs[i], { mono: true, fs: 11, fw: 600, fill: C.ink });
    if (i < 3) o += arrow(x + 110, 109, lx[i + 1] - 2, 109, C.blue);
  });

  // Bottom two panels: Single layer vs Depth accumulation
  o += box(18, 156, 240, 96, C.lgreen, C.green, 8, 2);
  o += tx(138, 178, "Single Layer Multiplier ρ_k", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(138, 204, "E[ρ_k] = 1,   Var(ρ_k) = 5 / d", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(138, 228, "At d = 512: SD(ρ_k) = √(5/512) ≈ 10%", { mono: true, fs: 10.5, fw: 600, fill: D.green.dark });

  o += box(282, 156, 240, 96, D.red.tint, D.red.base, 8, 2);
  o += tx(402, 178, "Depth Drift: Aspect Ratio K / d", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(402, 204, "Var(log ‖h_K‖²/‖h₀‖²) ≈ 5 K / d", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(402, 228, "Needs K ≪ d (or LayerNorm / ResNet!)", { fs: 11.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

function slerpVsLerp() {
  const W = 540, H = 266;
  let o = "";
  // Left: Quarter circle shell of radius sqrt(d)
  const cx = 44, cy = 222, R = 164;
  o += box(18, 16, 232, 234, "#FFFFFF", C.blue, 8, 2);
  // Outer and inner dashed shell arc
  o += path(`M ${cx},${cy - R} A ${R},${R} 0 0,1 ${cx + R},${cy}`, C.green, "none", 10);
  o += path(`M ${cx},${cy - R} A ${R},${R} 0 0,1 ${cx + R},${cy}`, D.green.dark, "none", 2.5);
  // Endpoints z1 (top) and z2 (right)
  const x1 = cx + 22, y1 = cy - Math.sqrt(R * R - 22 * 22);
  const x2 = cx + Math.sqrt(R * R - 22 * 22), y2 = cy - 22;
  // Linear chord
  o += line(x1, y1, x2, y2, D.red.base, 2.5, true);
  const xm = (x1 + x2) / 2, ym = (y1 + y2) / 2;
  o += circ(xm, ym, 5, D.red.base, "#FFFFFF", 1.5);
  // Slerp midpoint on arc
  const xs = cx + R * Math.cos(Math.PI / 4), ys = cy - R * Math.sin(Math.PI / 4);
  o += circ(xs, ys, 5.5, D.green.base, "#FFFFFF", 1.5);
  o += circ(x1, y1, 5, C.blue, "#FFFFFF", 1.5);
  o += circ(x2, y2, 5, C.blue, "#FFFFFF", 1.5);
  o += tx(x1 + 32, y1 - 14, "z₁ (√d)", { mono: true, fs: 11, fw: 700, fill: D.blue.dark });
  o += tx(x2 - 24, y2 + 22, "z₂ (√d)", { mono: true, fs: 11, fw: 700, fill: D.blue.dark });
  o += tx(xm - 28, ym + 26, "Lerp: √(d/2)", { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  o += tx(xs + 28, ys - 14, "Slerp: √d", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  // Right: Comparison cards
  const rx = 268;
  o += box(rx, 16, 254, 110, D.red.tint, D.red.base, 8, 2);
  o += tx(rx + 127, 38, "Linear Chord: (1−γ)z₁ + γz₂", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(rx + 127, 64, "At γ = 1/2:  ½(z₁+z₂) ~ N(0, ½I_d)", { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(rx + 127, 88, "Norm drops to √(d/2) = 0.71 √d!", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(rx + 127, 110, "Plunges into empty interior hole", { fs: 11.5, fill: C.ink });

  o += box(rx, 140, 254, 110, C.lgreen, C.green, 8, 2);
  o += tx(rx + 127, 162, "Spherical Arc (Slerp, θ ≈ 90°)", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(rx + 127, 188, "[sin((1−γ)θ)z₁ + sin(γθ)z₂] / sin θ", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(rx + 127, 212, "Keeps ‖z̃(γ)‖ ≈ √d for all γ ∈ [0,1]", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(rx + 127, 234, "Stays on the decoder's training shell!", { fs: 11.5, fill: C.ink });
  return svg(W, H, o);
}

function tailBoundsLadder() {
  const W = 540, H = 276;
  let o = "";
  const rungs = [
    ["1. Markov (X ≥ 0, mean μ)", "P(X ≥ t) ≤ μ / t", "n ∝ 1 / δ", C.lgray, C.gray, C.ink],
    ["2. Chebyshev (Var(Xᵢ) = σ²)", "P(|X̄−μ| ≥ t) ≤ σ² / (nt²)", "n ∝ 1 / (δ t²)", C.lblue, C.blue, D.blue.dark],
    ["3. Sub-Exp (|λ| ≤ 1/α)", "2 exp(−½n min(t²/ν², t/α))", "Gauss → Exp tail", C.lamber, C.amber, D.orange.dark],
    ["4. Hoeffding (Xᵢ ∈ [a, b])", "2 exp(−2nt² / (b−a)²)", "n ∝ (b−a)²log(2/δ)/t²", D.purple.tint, C.purple, D.purple.dark],
    ["5. Bernstein (|Xᵢ−μ|≤M, σ²)", "2 exp(−nt²/[2(σ²+Mt/3)])", "n ∝ σ²log(2/δ)/t²", C.lgreen, C.green, D.green.dark],
  ];
  o += tx(270, 18, "The Concentration Ladder: Stronger Moment Assumptions ⇒ Exponential Tails", { fs: 13, fw: 700, fill: C.ink });
  rungs.forEach(([name, bound, scale, fill, st, dk], i) => {
    const y = 30 + i * 48;
    o += box(14, y, 174, 40, fill, st, 6, 1.8);
    o += tx(101, y + 20, name, { fs: 11, fw: 700, fill: dk });
    o += box(194, y, 182, 40, "#FFFFFF", st, 6, 1.8);
    o += tx(285, y + 20, bound, { mono: true, fs: 10.5, fw: 700, fill: C.ink });
    o += box(382, y, 144, 40, N.panel, C.gray, 6, 1.5);
    o += tx(454, y + 20, scale, { mono: true, fs: 10, fw: 700, fill: dk });
  });
  return svg(W, H, o);
}

// ============================================================================
// 10. mdl-generalization-bounds.md (9 diagrams)
// ============================================================================

function uniformSup() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 204, gw = 220;
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  // True risk R(f) and tube R(f) +- t, plus empirical R_hat(f)
  let dTrue = "", dUp = "", dLo = "", dEmp = "";
  for (let s = 0; s <= 36; s++) {
    const u = (s / 36) * 2 - 1;
    const px = x0 + (s / 36) * gw;
    const rTrue = 144 - u * u * 64;
    const rEmp = rTrue + 18 * Math.sin(s * 0.65);
    dTrue += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${rTrue.toFixed(1)} `;
    dUp += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${(rTrue - 26).toFixed(1)} `;
    dLo += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${(rTrue + 26).toFixed(1)} `;
    dEmp += (s === 0 ? "M" : "L") + `${px.toFixed(1)},${rEmp.toFixed(1)} `;
  }
  o += path(dUp, C.green, "none", 1.5, true);
  o += path(dLo, C.green, "none", 1.5, true);
  o += path(dTrue, C.blue, "none", 2.8);
  o += path(dEmp, D.orange.base, "none", 2.5);
  o += tx(x0 + gw / 2, 20, "Uniform Tube: sup_f |R̂_S(f)−R(f)| ≤ t", { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(x0 + 68, 44, "True R(f) ± t band", { fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(x0 + 150, 188, "Empirical R̂_S(f)", { fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw / 2, y0 + 22, "Function class F (includes winner f̂)", { fs: 11.5, fw: 600, fill: C.ink });

  const rx = 284;
  o += box(rx, 24, 238, 104, D.red.tint, D.red.base, 8, 2);
  o += tx(rx + 119, 46, "Why Pointwise Fails on f̂(S)", { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(rx + 119, 72, "f̂ = argmin_f R̂_S(f) depends on S", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(rx + 119, 98, "Picks the largest downward dip!", { fs: 12, fw: 700, fill: D.red.dark });

  o += box(rx, 144, 238, 104, C.lgreen, C.green, 8, 2);
  o += tx(rx + 119, 166, "Uniform Convergence Remedy", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(rx + 119, 192, "|R̂_S(f̂)−R(f̂)| ≤ sup_f |R̂_S(f)−R(f)|", { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  o += tx(rx + 119, 218, "Covers EVERY f ∈ F simultaneously!", { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function unionLogCost() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, C.lblue, C.blue, 8, 2);
  o += tx(270, 38, "Finite Pool |F| on n = 10,000 Examples (δ = 0.05): Hoeffding vs. Chebyshev", { fs: 13, fw: 700, fill: D.blue.dark });

  // Left: Hoeffding union bound (log|F|)
  o += box(18, 78, 242, 172, C.lgreen, C.green, 8, 2);
  o += tx(139, 100, "Hoeffding + Union Bound", { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(139, 126, "t = √( [log(2/δ) + log|F|] / (2n) )", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(139, 156, "• |F| = 1 model:      ±1.36%", { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(139, 180, "• |F| = 2,000 models: ±2.38%", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(139, 204, "• |F| = 10⁶ models:   ±2.96%", { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(139, 230, "Doubling |F| costs +3,466 examples", { fs: 11.5, fill: C.muted });

  // Right: Chebyshev union bound (sqrt(|F|))
  o += box(280, 78, 242, 172, D.red.tint, D.red.base, 8, 2);
  o += tx(401, 100, "Chebyshev + Union Bound", { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(401, 126, "t = √( |F| / (4 n δ) )", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(401, 156, "• |F| = 1 model:      ±2.24%", { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(401, 180, "• |F| = 2,000 models: ±100%!", { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(401, 204, "• |F| = 10⁶ models:   ±2,236%!", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(401, 230, "Polynomial tails collapse under search", { fs: 11.5, fill: D.red.dark });
  return svg(W, H, o);
}

function leaderboardAttack() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 48, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 38, "Non-Adaptive Grid (√(log k / n)) vs. Adaptive Leaderboard Attack (√(k / n))", { fs: 13, fw: 700, fill: D.red.dark });

  // Top: Non-adaptive pre-registered grid
  o += box(18, 76, 504, 76, C.lgreen, C.green, 8, 2);
  o += tx(270, 96, "(a) Non-Adaptive Pre-Registered Pool: Fix All k Models BEFORE Seeing Scores", { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(270, 120, "f₁,...,f_k chosen independent of holdout S ⇒ Max gap ≤ √(log(2k/δ)/(2n))", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(270, 140, "Safe even for k = 1,000,000 candidates!", { fs: 12, fw: 600, fill: D.green.dark });

  // Bottom: Adaptive Blum-Hardt boosting attack
  o += box(18, 166, 504, 84, C.lamber, C.amber, 8, 2);
  o += tx(270, 186, "(b) Adaptive Blum–Hardt Attack: Inspect Holdout Score After Each Query", { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 210, "Submit k random coin-flip vectors → keep those with error ≤ ½ → majority vote", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(270, 234, "Holdout error drops to ½ − Ω(√(k/n)) while population error stays 50%!", { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

function rademacherNoise() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 50, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 32, "Empirical Rademacher Complexity: Best Correlation With Random Signs", { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(270, 52, "R̂_S(F) = E_{ε ∈ {±1}ⁿ}[ sup_{f ∈ F} (1/n) ∑ᵢ₌₁ⁿ εᵢ f(xᵢ) ]", { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  // Three capacity tiers
  const tiers = [
    { x: 18, t: "Singleton {f}", r: "R̂_S = 0", sub: "No freedom to chase ε", sub2: "E[εᵢ] = 0 cancels out", fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 194, t: "Two Constants {±1}", r: "E|ε̄| ≈ √(2/πn)", sub: "n=2: 1/2,  n=4: 3/8", sub2: "n=100: 0.080 (O(n⁻¹/²))", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 370, t: "Shattering Class", r: "R̂_S = 1.0 (max!)", sub: "Matches f(xᵢ) = εᵢ ∀ε", sub2: "Memorizes pure noise!", fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  tiers.forEach(c => {
    o += box(c.x, 82, 152, 168, c.fill, c.st, 8, 2);
    o += tx(c.x + 76, 106, c.t, { fs: 13.5, fw: 700, fill: c.dk });
    o += box(c.x + 12, 122, 128, 38, "#FFFFFF", c.st, 6, 1.8);
    o += tx(c.x + 76, 141, c.r, { mono: true, fs: 10.5, fw: 700, fill: c.dk });
    o += tx(c.x + 76, 184, c.sub, { mono: true, fs: 10.5, fw: 600, fill: C.ink });
    o += tx(c.x + 76, 212, c.sub2, { fs: 11, fill: C.muted });
  });
  return svg(W, H, o);
}

function symmetrizationGhost() {
  const W = 540, H = 266;
  let o = "";
  const steps = [
    { y: 14, t: "Step 1 (Ghost Sample S'): Replace Population Risk R(g) = E_{S'}[R̂_{S'}(g)]", f: "E_S[sup_{g∈G}(R(g)−R̂_S(g))] ≤ E_{S,S'}[sup_{g∈G} (1/n) ∑ᵢ (g(z'ᵢ) − g(zᵢ))]", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 98, t: "Step 2 (Exchangeable Swap zᵢ ↔ z'ᵢ): Multiply Differences by εᵢ ∈ {±1}", f: "Since zᵢ, z'ᵢ are i.i.d.: = E_{S,S',ε}[sup_g (1/n) ∑ᵢ εᵢ(g(z'ᵢ) − g(zᵢ))]", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 182, t: "Step 3 (Split Suprema + McDiarmid Concentration): Factor of 2 Emerges", f: "≤ E_{S',ε}[sup_g (1/n)∑ εᵢg(z'ᵢ)] + E_{S,ε}[sup_g (1/n)∑ (−εᵢ)g(zᵢ)] = 2 R_n(G)", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s, i) => {
    o += box(18, s.y, 504, 68, s.fill, s.st, 8, 2);
    o += tx(270, s.y + 22, s.t, { fs: 12.5, fw: 700, fill: s.dk });
    o += tx(270, s.y + 48, s.f, { mono: true, fs: 10.5, fw: 700, fill: C.ink });
    if (i < 2) o += arrow(270, s.y + 69, 270, steps[i + 1].y - 2, s.st);
  });
  return svg(W, H, o);
}

function rampMargin() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 44, y0 = 204, gw = 216, gh = 132, xZero = x0 + 84, xGam = x0 + 156;
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  o += line(xZero, 34, xZero, y0 + 8, C.gray, 1.5, true);
  // 0-1 step loss (red dashed)
  o += line(x0, y0 - gh, xZero, y0 - gh, D.red.base, 2.5, true);
  o += line(xZero, y0 - gh, xZero, y0, D.red.base, 2.5, true);
  o += line(xZero, y0, x0 + gw, y0, D.red.base, 2.5, true);
  // Ramp loss phi_gamma(u) (blue solid)
  o += path(`M ${x0},${y0 - gh} L ${xZero},${y0 - gh} L ${xGam},${y0} L ${x0 + gw},${y0}`, C.blue, "none", 3.2);
  o += tx(x0 + 34, y0 - gh - 14, "loss = 1", { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(xGam + 34, y0 - gh / 2, "slope −1/γ", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += tx(xZero, y0 + 22, "0", { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xGam, y0 + 22, "γ", { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(x0 + gw - 16, y0 + 22, "u=yf(x)", { mono: true, fs: 11.5, fill: C.muted });
  o += tx(x0 + gw / 2, 22, "(1/γ)-Lipschitz Ramp Loss φ_γ(u)", { fs: 14, fw: 700, fill: D.blue.dark });

  const rx = 282;
  o += box(rx, 24, 240, 104, C.lblue, C.blue, 8, 2);
  o += tx(rx + 120, 46, "Talagrand Contraction Lemma", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(rx + 120, 72, "R̂_S(φ_γ ∘ F) ≤ (1/γ) R̂_S(F)", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(rx + 120, 98, "≤ B r / (γ √n)", { mono: true, fs: 13, fw: 700, fill: D.blue.dark });

  o += box(rx, 144, 240, 104, C.lgreen, C.green, 8, 2);
  o += tx(rx + 120, 166, "Dimension-Free Margin Bound", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(rx + 120, 192, "P(y⟨w,x⟩≤0) ≤ R̂_γ + 2Br/(γ√n)", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += tx(rx + 120, 218, "Large γ/B ⇒ low test error!", { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function svdInterpolant() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 38, "Minimum-Norm SVD Interpolant: w = Φ⁺y = ∑ᵢ (⟨uᵢ, y⟩ / σᵢ) vᵢ", { mono: true, fs: 12, fw: 700, fill: D.purple.dark });

  const regimes = [
    { x: 18, t: "1. Underparam (p < n)", s: "Projects y onto p dirs", s2: "Residual absorbs noise", s3: "σ_min moderately large", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: "2. Threshold (p = n)", s: "Zero residual (exact fit)", s2: "Tiny σ_n ≈ 0.02 pinch!", s3: "1/σ_n explodes ‖w‖₂!", fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 370, t: "3. Overparam (p ≫ n)", s: "Extra room in ℝᵖ lifts σ_n", s2: "‖w_{p+1}‖₂ ≤ ‖w_p‖₂", s3: "Second descent of MSE!", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  regimes.forEach(r => {
    o += box(r.x, 80, 152, 170, r.fill, r.st, 8, 2);
    o += tx(r.x + 76, 104, r.t, { fs: 12.5, fw: 700, fill: r.dk });
    o += line(r.x + 12, 120, r.x + 140, 120, r.st, 1, true);
    o += tx(r.x + 76, 148, r.s, { fs: 11.5, fw: 600, fill: C.ink });
    o += tx(r.x + 76, 180, r.s2, { fs: 11.5, fw: 700, fill: r.dk });
    o += tx(r.x + 76, 214, r.s3, { mono: true, fs: 11, fw: 700, fill: C.ink });
  });
  return svg(W, H, o);
}

function cleanVsNoise() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 50, C.lblue, C.blue, 8, 2);
  o += tx(270, 32, "Linearity of Pseudoinverse on Noisy Targets y = y₀ + ξ (Cov(ξ) = σ²_ξ Iₙ)", { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(270, 52, "w = Φ⁺ y₀ + Φ⁺ ξ   ⇒   E_ξ[MSE_test] = Clean Error + Noise Error", { mono: true, fs: 12, fw: 700, fill: C.ink });

  // Left: Clean target approximation error
  o += box(18, 82, 240, 168, C.lamber, C.amber, 8, 2);
  o += tx(138, 106, "1. Clean Approximation Error", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += box(32, 122, 212, 42, "#FFFFFF", C.amber, 6, 1.5);
  o += tx(138, 143, "‖Φ_te Φ⁺ y₀ − y_te‖² / n_te", { mono: true, fs: 10.5, fw: 700, fill: D.orange.dark });
  o += tx(138, 188, "• Independent of label noise σ_ξ", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(138, 212, "• Also spikes at p = n if target y₀", { fs: 11.5, fill: C.ink });
  o += tx(138, 232, "  has energy along u_n!", { fs: 11.5, fw: 700, fill: D.orange.dark });

  // Right: Noise amplification error
  o += box(282, 82, 240, 168, C.lgreen, C.green, 8, 2);
  o += tx(402, 106, "2. Noise Amplification Error", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += box(296, 122, 212, 42, "#FFFFFF", C.green, 6, 1.5);
  o += tx(402, 143, "(σ²_ξ/n) ∑ᵢ (‖Φ_te vᵢ‖²/n_te)/(σ²ᵢ/n)", { mono: true, fs: 9.5, fw: 700, fill: D.green.dark });
  o += tx(402, 188, "• Average test-to-train feature", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(402, 210, "  energy ratio across n directions", { fs: 11.5, fill: C.ink });
  o += tx(402, 232, "• Governs benign overfitting!", { fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function benignDirections() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 46, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 37, "Test/Train Energy Ratio: (‖Φ_te v_min‖²/n_te) ÷ (‖Φ v_min‖²/n)", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });

  // Left: Threshold p = n = 40 accidental pinch
  o += box(18, 76, 240, 174, D.red.tint, D.red.base, 8, 2);
  o += tx(138, 100, "Threshold p = n = 40 (Catastrophic)", { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(138, 126, "Train denominator σ²_min/n ≈ 10⁻⁵", { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  o += tx(138, 150, "Test numerator stays O(1) normal!", { fs: 11.5, fw: 700, fill: C.ink });
  o += box(36, 168, 204, 40, "#FFFFFF", D.red.base, 6, 1.8);
  o += tx(138, 188, "Energy Ratio ≈ 1,000×!", { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  o += tx(138, 228, "Accidental pinch on 40 train points", { fs: 11.5, fill: C.ink });

  // Right: Wide limit p = 400 benign overfitting
  o += box(282, 76, 240, 174, C.lgreen, C.green, 8, 2);
  o += tx(402, 100, "Wide Limit p = 400 (Benign!)", { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(402, 126, "Noise absorbed into genuinely weak", { fs: 11.5, fw: 600, fill: C.ink });
  o += tx(402, 148, "directions where test energy is ALSO", { fs: 11.5, fw: 600, fill: C.ink });
  o += box(300, 168, 204, 40, "#FFFFFF", C.green, 6, 1.8);
  o += tx(402, 188, "Mean Energy Ratio ≈ 0.9×!", { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(402, 228, "Small numerator cancels denominator", { fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ============================================================================
// 11. mdl-naive-bayes.md (5 diagrams)
// ============================================================================

function genVsDiscRoute() {
  const W = 540, H = 264;
  let o = "";
  // Top: Generative route
  o += box(18, 16, 504, 108, C.lblue, C.blue, 8, 2);
  o += tx(270, 38, "1. Generative Route (Naive Bayes): Model How Each Class Generates x", { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += box(32, 54, 144, 52, "#FFFFFF", C.blue, 6, 1.8);
  o += tx(104, 72, "Fit Class Model", { fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(104, 92, "p(x | y) & p(y)", { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += arrow(180, 80, 206, 80, C.blue);
  o += box(210, 54, 148, 52, "#FFFFFF", C.purple, 6, 1.8);
  o += tx(284, 72, "Bayes Numerator", { fs: 12, fw: 700, fill: D.purple.dark });
  o += tx(284, 92, "p(y) ∏ᵢ p(xᵢ | y)", { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  o += arrow(362, 80, 388, 80, C.purple);
  o += box(392, 54, 116, 52, "#FFFFFF", C.green, 6, 1.8);
  o += tx(450, 72, "Invert to Label", { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(450, 92, "argmax_y s_y(x)", { mono: true, fs: 11, fw: 700, fill: D.green.dark });

  // Bottom: Discriminative route
  o += box(18, 140, 504, 108, C.lamber, C.amber, 8, 2);
  o += tx(270, 162, "2. Discriminative Route (Softmax): Model Boundary p(y | x) Directly", { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += box(32, 178, 214, 52, "#FFFFFF", C.amber, 6, 1.8);
  o += tx(139, 196, "Skip Modeling p(x | y)", { fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(139, 216, "Saves capacity on input correlations", { fs: 11, fill: C.ink });
  o += arrow(250, 204, 286, 204, C.amber);
  o += box(290, 178, 218, 52, "#FFFFFF", C.green, 6, 1.8);
  o += tx(399, 196, "Optimize Conditional NLL", { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(399, 216, "argmin_{W,b} −∑ᵢ log p(yᵢ | xᵢ)", { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function spamOdds() {
  const W = 540, H = 264;
  let o = "";
  o += box(18, 14, 504, 46, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 37, "Two-Word Spam Filter on x = (\"free\" = 1, \"meeting\" = 0)", { fs: 14, fw: 700, fill: D.purple.dark });

  // Multiplicative chain of odds
  const boxes = [
    { x: 16, w: 108, t: "Prior Odds", f: "0.4 / 0.6", v: "= 2 / 3", fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 148, w: 112, t: "Has \"free\"", f: "(3/4) / (1/6)", v: "× 4.5", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 284, w: 116, t: "No \"meeting\"", f: "(4/4) / (2/6)", v: "× 3.0", fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 424, w: 100, t: "Post. Odds", f: "9 to 1", v: "P = 0.90", fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  boxes.forEach((b, i) => {
    o += box(b.x, 78, b.w, 92, b.fill, b.st, 8, 2);
    o += tx(b.x + b.w / 2, 100, b.t, { fs: 12.5, fw: 700, fill: b.dk });
    o += tx(b.x + b.w / 2, 126, b.f, { mono: true, fs: 11.5, fw: 700, fill: C.ink });
    o += tx(b.x + b.w / 2, 150, b.v, { mono: true, fs: 12.5, fw: 700, fill: b.dk });
    if (i < 3) {
      const sym = i < 2 ? "×" : "=";
      o += tx(b.x + b.w + 12, 124, sym, { fs: 20, fw: 700, fill: C.ink });
    }
  });

  // Bottom bar: Log-odds additive view
  o += box(16, 188, 508, 62, "#FFFFFF", C.purple, 8, 1.8);
  o += tx(270, 208, "In Log Space, Multiplying Likelihood Ratios Becomes Adding Log-Odds:", { fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(270, 232, "log(9) = log(2/3) + log(4.5) + log(3.0) = −0.41 + 1.50 + 1.10 = +2.20 nats", { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function dupEvidence() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 38, "Why Correlated Features Cause Extreme Overconfidence (ECE = 0.29)", { fs: 13.5, fw: 700, fill: D.red.dark });

  // Left: True single clue
  o += box(18, 78, 236, 172, C.lgreen, C.green, 8, 2);
  o += tx(136, 102, "True Posterior (1 Clue, r = 4)", { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(136, 128, "Copies x₁ = x₂ = ··· = x_k", { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(136, 150, "carry zero new information", { fs: 12, fill: C.ink });
  o += box(36, 168, 200, 42, "#FFFFFF", C.green, 6, 1.8);
  o += tx(136, 189, "True Odds = 4¹ = 4 (P = 80%)", { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  o += tx(136, 228, "Well-calibrated 80% accuracy", { fs: 12, fw: 700, fill: D.green.dark });

  // Right: Naive Bayes k=5 copies
  o += box(276, 78, 246, 172, C.lamber, C.amber, 8, 2);
  o += tx(399, 102, "Naive Bayes (k = 5 Copies)", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(399, 128, "Multiplies likelihood ratio k times:", { fs: 12, fw: 600, fill: C.ink });
  o += tx(399, 150, "log-odds = log(π₁/π₀) + k log r(x)", { mono: true, fs: 11, fw: 700, fill: C.ink });
  o += box(296, 168, 206, 42, "#FFFFFF", D.red.base, 6, 1.8);
  o += tx(399, 189, "NB Odds = 4⁵ = 1024 (99.9%!)", { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  o += tx(399, 228, "Fix: Temperature scaling s_y(x) / T", { fs: 11.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

function missingPixels() {
  const W = 540, H = 266;
  let o = "";
  o += box(18, 14, 504, 48, C.lgreen, C.green, 8, 2);
  o += tx(270, 38, "Occluded Features x_U: Exact Marginalization vs. Zero-Imputation", { fs: 13, fw: 700, fill: D.green.dark });

  // Left: Exact marginalization
  o += box(18, 78, 242, 172, C.lblue, C.blue, 8, 2);
  o += tx(139, 100, "1. Exact NB Marginalization", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(139, 126, "Sum out hidden j ∈ U:", { fs: 12, fw: 600, fill: C.ink });
  o += tx(139, 150, "∑_{xⱼ} p(xⱼ|y) = θ_{jy} + 1 − θ_{jy} = 1", { mono: true, fs: 9.8, fw: 700, fill: D.blue.dark });
  o += box(34, 168, 210, 42, "#FFFFFF", C.blue, 6, 1.8);
  o += tx(139, 189, "s_y(x_V) = log p(y) + ∑_{i∈V} log p", { mono: true, fs: 9.2, fw: 700, fill: C.ink });
  o += tx(139, 228, "Hidden factors simply drop out!", { fs: 12, fw: 700, fill: D.blue.dark });

  // Right: Pitfall of setting x_j = 0 in W^T x + b
  o += box(280, 78, 242, 172, D.red.tint, D.red.base, 8, 2);
  o += tx(401, 100, "2. Pitfall: Setting xⱼ = 0 in Wᵀx+b", { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(401, 126, "b_y = log p(y) + ∑_j log(1−θ_{jy})", { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  o += tx(401, 150, "already includes log(1 − θ_{jy})!", { fs: 12, fw: 700, fill: D.red.dark });
  o += box(296, 168, 210, 42, "#FFFFFF", D.red.base, 6, 1.8);
  o += tx(401, 189, "Treats occluded pixel as BLANK", { fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(401, 228, "background rather than unobserved!", { fs: 11.5, fill: C.ink });
  return svg(W, H, o);
}

function ngJordanCrossover() {
  const W = 540, H = 266;
  let o = "";
  const x0 = 52, y0 = 206, gw = 208;
  o += line(x0, y0, x0 + gw, y0, C.ink, 1.8);
  o += line(x0, 32, x0, y0, C.ink, 1.8);
  // Naive Bayes curve: fast drop to 29.4% plateau
  const dNB = `M ${x0 + 12},64 Q ${x0 + 36},134 ${x0 + 80},138 L ${x0 + gw - 8},140`;
  // Logistic Regression curve: slower drop to 16.9% plateau
  const dLR = `M ${x0 + 12},44 Q ${x0 + 82},116 ${x0 + 132},166 L ${x0 + gw - 8},176`;
  o += path(dNB, C.blue, "none", 3);
  o += path(dLR, D.orange.base, "none", 3);
  o += circ(x0 + 86, 138, 5, D.purple.base, "#FFFFFF", 1.5);
  o += tx(x0 + gw / 2, 20, "Ng & Jordan (2002) Crossover", { fs: 14, fw: 700, fill: C.ink });
  o += tx(x0 + 148, 118, "Naive Bayes (29.4%)", { fs: 11.5, fw: 700, fill: D.blue.dark });
  o += tx(x0 + 148, 192, "Logistic / Softmax (16.9%)", { fs: 11.5, fw: 700, fill: D.orange.dark });
  o += tx(x0 + gw / 2, y0 + 22, "log(Training Sample Size n)", { fs: 12, fw: 600, fill: C.ink });

  const rx = 282;
  o += box(rx, 24, 240, 104, C.lblue, C.blue, 8, 2);
  o += tx(rx + 120, 46, "Small n ≪ d: Naive Bayes Wins", { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(rx + 120, 72, "1D parameters decoupled:", { fs: 12, fw: 600, fill: C.ink });
  o += tx(rx + 120, 98, "m = O(log d) samples  (low variance)", { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  o += box(rx, 144, 240, 104, C.lamber, C.amber, 8, 2);
  o += tx(rx + 120, 166, "Large n ≫ d: Discriminative Wins", { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(rx + 120, 192, "No independence bias:", { fs: 12, fw: 600, fill: C.ink });
  o += tx(rx + 120, 218, "n = O(d) samples  (83.1% vs 70.6%!)", { mono: true, fs: 11.5, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

export const diagrams = {
  // 0. index.md
  "ch35-prob-coin-thread": coinThread,
  "ch35-prob-four-pillars": fourPillars,
  "ch35-prob-roadmap-p1": roadmapP1,
  "ch35-prob-roadmap-p2": roadmapP2,

  // 1. mdl-random-variables.md
  "ch35-prob-dart-bins": dartBins,
  "ch35-prob-dart-density": dartDensity,
  "ch35-prob-mixed-relu": mixedRelu,
  "ch35-prob-cdf-types": cdfTypes,
  "ch35-prob-mean-lotus": meanLotus,
  "ch35-prob-var-affine": varAffine,
  "ch35-prob-markov-step": markovStep,
  "ch35-prob-cauchy-tail": cauchyTail,

  // 2. mdl-joint-distributions.md
  "ch35-prob-dart-three-questions": dartThreeQuestions,
  "ch35-prob-joint-disk": jointDisk,
  "ch35-prob-bayes-convolution": bayesConvolution,
  "ch35-prob-eves-law": evesLaw,
  "ch35-prob-optimal-mse": optimalMse,
  "ch35-prob-sum-var-mc": sumVarMc,
  "ch35-prob-uncorr-dep": uncorrDep,
  "ch35-prob-flow-coupling": flowCoupling,

  // 3. mdl-distributions.md
  "ch35-prob-loss-dist-map": lossDistMap,
  "ch35-prob-bernoulli-var": bernoulliVar,
  "ch35-prob-binomial-grid": binomialGrid,
  "ch35-prob-poisson-slicing": poissonSlicing,
  "ch35-prob-uniform-center": uniformCenter,

  // 4. mdl-exponential-family.md
  "ch35-prob-four-regressions": fourRegressions,
  "ch35-prob-bernoulli-logit": bernoulliLogit,
  "ch35-prob-exp-anatomy": expAnatomy,
  "ch35-prob-outside-exp": outsideExp,
  "ch35-prob-gibbs-tangent": gibbsTangent,
  "ch35-prob-cumulant-derivs": cumulantDerivs,
  "ch35-prob-boundary-minimal": boundaryMinimal,
  "ch35-prob-glm-head": glmHead,
  "ch35-prob-conjugate-add": conjugateAdd,

  // 5. mdl-statistics.md
  "ch35-prob-paired-vs-unpaired": pairedVsUnpaired,
  "ch35-prob-bessel-dof": besselDof,
  "ch35-prob-mc-vs-grid": mcVsGrid,
  "ch35-prob-bonferroni-fdr": bonferroniFdr,

  // 6. mdl-maximum-likelihood.md
  "ch35-prob-coin-three-q": coinThreeQ,
  "ch35-prob-lik-vs-prob": likVsProb,
  "ch35-prob-underflow-nll": underflowNll,
  "ch35-prob-ce-kl-split": ceKlSplit,
  "ch35-prob-score-zero": scoreZero,
  "ch35-prob-fisher-duality": fisherDuality,
  "ch35-prob-nll-se-readout": nllSeReadout,
  "ch35-prob-asymptotic-norm": asymptoticNorm,
  "ch35-prob-map-hhh": mapHhh,

  // 7. mdl-latent-em-elbo.md
  "ch35-prob-lost-labels": lostLabels,
  "ch35-prob-latent-dag": latentDag,
  "ch35-prob-oracle-vs-mixture": oracleVsMixture,
  "ch35-prob-soft-resp": softResp,
  "ch35-prob-jensen-elbo": jensenElbo,
  "ch35-prob-elbo-anatomy": elboAnatomy,
  "ch35-prob-elbo-gap-bars": elboGapBars,
  "ch35-prob-em-loop": emLoop,
  "ch35-prob-gmm-spikes": gmmSpikes,
  "ch35-prob-vae-amortized": vaeAmortized,

  // 8. mdl-bayesian-computation.md
  "ch35-prob-predictive-integral": predictiveIntegral,
  "ch35-prob-predictive-shift": predictiveShift,
  "ch35-prob-detailed-balance": detailedBalance,

  // 9. mdl-concentration-generalization.md
  "ch35-prob-three-intervals": threeIntervals,
  "ch35-prob-fair-coin-mgf": fairCoinMgf,
  "ch35-prob-hoeffding-tilting": hoeffdingTilting,
  "ch35-prob-subexp-window": subexpWindow,
  "ch35-prob-random-ortho": randomOrtho,
  "ch35-prob-exponential-packing": exponentialPacking,
  "ch35-prob-depth-norm-drift": depthNormDrift,
  "ch35-prob-slerp-vs-lerp": slerpVsLerp,
  "ch35-prob-tail-bounds-ladder": tailBoundsLadder,

  // 10. mdl-generalization-bounds.md
  "ch35-prob-uniform-sup": uniformSup,
  "ch35-prob-union-log-cost": unionLogCost,
  "ch35-prob-leaderboard-attack": leaderboardAttack,
  "ch35-prob-rademacher-noise": rademacherNoise,
  "ch35-prob-symmetrization-ghost": symmetrizationGhost,
  "ch35-prob-ramp-margin": rampMargin,
  "ch35-prob-svd-interpolant": svdInterpolant,
  "ch35-prob-clean-vs-noise": cleanVsNoise,
  "ch35-prob-benign-directions": benignDirections,

  // 11. mdl-naive-bayes.md
  "ch35-prob-gen-vs-disc-route": genVsDiscRoute,
  "ch35-prob-spam-odds": spamOdds,
  "ch35-prob-dup-evidence": dupEvidence,
  "ch35-prob-missing-pixels": missingPixels,
  "ch35-prob-ng-jordan-crossover": ngJordanCrossover,
};
