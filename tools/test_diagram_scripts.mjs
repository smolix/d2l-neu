// Unit checks for diagrams/scripts.mjs (TeX-style sub/superscripts in
// diagram labels).  Run: node tools/test_diagram_scripts.mjs
import { scriptText, scriptSvg } from '../diagrams/scripts.mjs';
import assert from 'node:assert/strict';

const show = (s) => s
  .replace(/<tspan baseline-shift="sub"[^>]*>/g, '_(')
  .replace(/<tspan baseline-shift="super"[^>]*>/g, '^(')
  .replace(/<\/tspan>/g, ')');
const cases = [
  ['n_b', 'n_(b)'], ['R_emp[f]', 'R_(emp)[f]'], ['x_{k+1}', 'x_(k+1)'],
  ['Σ_{i=1}^{n}', 'Σ_(i=1)^(n)'], ['∑_{s=1}^t', '∑_(s=1)^(t)'], ['2^m', '2^(m)'],
  ['W_hh^T', 'W_(hh)^(T)'], ['T^0.75', 'T^(0.75)'], ['ℝ^d', 'ℝ^(d)'],
  ['ℝ^(m×n)', 'ℝ^(m×n)'], ['e^(−λ/2)', 'e^(−λ/2)'], ['X^(L)', 'X^((L))'],
  ['h^(1)', 'h^((1))'], ['S_(t−1)', 'S_(t−1)'], ['‖G‖_F', '‖G‖_(F)'],
  ['max_a Q', 'max_(a) Q'], ['(n_h−k_h+1)', '(n_(h)−k_(h)+1)'],
  // literal: code names and escapes
  ['batch_size', 'batch_size'], ['requires_grad', 'requires_grad'],
  ['X_train_std', 'X_train_std'], ['max_len', 'max_len'], ['min_freq', 'min_freq'],
  ['c\\_attn', 'c_attn'],
];
for (const [inp, want] of cases) assert.equal(show(scriptText(inp)), want, inp);

// mono labels with code syntax stay literal; mono formulas convert
const mono = (t) => `<text font-family="'JetBrains Mono', monospace">${t}</text>`;
assert.equal(scriptSvg(mono('y_hat = self.net(X)')), mono('y_hat = self.net(X)'));
assert.equal(scriptSvg(mono("{'loss': l_t}")), mono("{'loss': l_t}"));
assert.ok(scriptSvg(mono('w_h = 1.1')).includes('baseline-shift="sub"'));
assert.ok(scriptSvg(mono('p_T(x) = p_S(x)')).includes('baseline-shift="sub"'));
console.log(`ok: ${cases.length + 4} checks`);
