// TeX-style sub/superscripts in diagram labels → SVG tspans.
//
// Labels are written the way authors type math: n_b, R_emp[f], x_{k+1},
// Σ_{i=1}^{n_in}, 2^m, ℝ^{|B|×1}, W_hh^T. Drawn literally, the `_`, `^` and
// braces show in the figure. `scriptSvg` rewrites the text of every <text>
// element of a finished diagram so these become lowered / raised, smaller
// runs (<tspan baseline-shift="sub|super" font-size="70%">), which the
// outliner (tools/figstyle/outline_svg.py) and every browser draw.
//
// Code must stay literal: `g_fn = jax.grad(f)` or `x_train` in a code panel
// is Python, not math. So
//   * sans labels are math-capable: they are prose and formulas;
//   * mono labels are converted when they read as a formula: they carry a
//     braced script (`_{`, `^{`), a bare `^`, or a math-only symbol
//     (Σ ∂ ∇ ∈ ℝ ≤ ⇒ Greek, Unicode super/subscripts, ...), or they have no
//     code syntax at all (`s_t`, `w_h = 1.1`, `p_T(x) = p_S(x)`). A mono
//     label with code syntax (see CODE) is left alone;
//   * a bare `_` subscript needs a one-letter base (not preceded by a
//     letter, digit, `_` or `.`) and a script word that does not run on into
//     another `_`: n_b, w_ij, R_emp, b_noise, x_k. So `batch_size`,
//     `requires_grad`, `X_train_std` stay as written.
// `x^(…)` and `x_(…)` treat the parentheses as grouping and drop them, as
// TeX braces would. A label that should stay literal anyway is written with `\_` / `\^`, which
// draws a plain `_` / `^`.

const SCRIPT_SIZE = '70%';
const NESTED_SIZE = '85%';  // a script inside a script (n_in inside ^{...})

// Letters that may carry a bare subscript: Latin, Greek, ℓ, ∇, precomposed
// accented Latin (ŷ ȳ ŵ x̄ come precomposed or as base + combining mark).
const BASE = /[A-Za-zα-ωΑ-Ωϵϑϕℓŷȳŵĉĥĝŝẑḡ∇]/u;
const IDENT = /[A-Za-z0-9_.]/;
// Code syntax in a mono label: attribute access (y.reshape, self.loss),
// keywords, quotes, ==, slices, comments, a call of a snake_case name
// (squared_loss(, stop_grad() or of a long lowercase name that is not a math
// function (linreg( but not log( / softmax().
const MATH_FN = 'log|exp|max|min|mean|sum|softmax|tanh|sigmoid|argmax|argmin|sqrt|det|diag|sign|relu|concat|floor|ceil|clip|norm|prod|cos|sin|var|cov|logsumexp';
const CODE = new RegExp(String.raw`[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_]|\bself\b|\breturn\b|\bimport\b|\bdef\b|'[A-Za-z_][A-Za-z0-9_]*'|#|==|\[:|:\]|:,|[A-Za-z]{2,}_[A-Za-z0-9_]+\(|\b(?!(?:${MATH_FN})\()[a-z]{4,}\(`);
const MATH_HINT = /[Σ∑∏∂∇∈∉ℝℕℤℚℂ≈≤≥≠⇒⇔→←↦·×√∞∫⊙⊗⊕∀∃∝‖α-ωΑ-Ωℓ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻₀-₉ᵀᵢⱼₖₘₙₜ]|_\{|\^\{|[A-Za-z0-9)\]]\^[A-Za-z0-9(−-]/u;

function tsp(kind, inner, nested) {
  return `<tspan baseline-shift="${kind}" font-size="${nested ? NESTED_SIZE : SCRIPT_SIZE}">${inner}</tspan>`;
}

// Matching close brace for the `{` at s[i]; -1 if unbalanced.
function closeBrace(s, i) {
  let d = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === '\\') { j++; continue; }
    if (s[j] === '{') d++;
    else if (s[j] === '}' && --d === 0) return j;
  }
  return -1;
}

// Convert the scripts in one run of (already XML-escaped) label text.
export function scriptText(s, nested) {
  const chars = Array.from(s);   // code points, so Greek / ℝ index cleanly
  const str = (a, b) => chars.slice(a, b).join('');
  let out = '';
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    if (c === '\\' && (chars[i + 1] === '_' || chars[i + 1] === '^')) {
      out += chars[++i];   // escaped: draw literally
      continue;
    }
    if (c !== '_' && c !== '^') { out += c; continue; }
    const kind = c === '_' ? 'sub' : 'super';
    const prev = chars[i - 1], prev2 = chars[i - 2];
    // braced script: x_{k+1}, Σ_{i=1}, ℝ^{n×d}, 1_{j=k}
    if (chars[i + 1] === '{' && prev !== undefined && prev !== ' ') {
      const j = closeBrace(chars, i + 1);
      if (j > 0) {
        out += tsp(kind, scriptText(str(i + 2, j), true), nested);
        i = j;
        continue;
      }
    }
    if (kind === 'super') {
      // bare ^: 2^m, (1+g²)^N, W^T, 2^-1, e^(x)
      if (prev !== undefined && /[A-Za-z0-9)\]²³ᵀ'α-ω]/u.test(prev)) {
        const rest = str(i + 1, chars.length);
        const m = rest.match(/^[−-]?[A-Za-z0-9α-ω]{1,6}(?![A-Za-z0-9_])/u);
        const g = !m && rest.match(/^\(([^()]{1,16})\)/);   // e^(−λ/2): parens only group
        if (m || g) {
          out += tsp('super', scriptText(m ? m[0] : g[1], true), nested);
          i += Array.from(m ? m[0] : g[0]).length;
          continue;
        }
      }
      out += c; continue;
    }
    // bare _: one-letter base (or a primed one, g'_j), short script
    let baseOk = false;
    if (prev !== undefined && BASE.test(prev) && (prev2 === undefined || !IDENT.test(prev2))) baseOk = true;
    if (prev === "'" && prev2 !== undefined && BASE.test(prev2)) baseOk = true;
    // a combining accent on the base (x̄_k): look one further back
    if (prev !== undefined && /[̀-ͯ]/.test(prev) && prev2 !== undefined && BASE.test(prev2)
        && (chars[i - 3] === undefined || !IDENT.test(chars[i - 3]))) baseOk = true;
    if (baseOk) {
      const rest = str(i + 1, chars.length);
      const m = rest.match(/^[A-Za-z0-9α-ωℓ∅]{1,12}(?![A-Za-z0-9_])/u);
      const g = !m && rest.match(/^\(([^()]{1,16})\)/);     // S_(t−1): parens only group
      if (m || g) {
        out += tsp('sub', m ? m[0] : scriptText(g[1], true), nested);
        i += Array.from(m ? m[0] : g[0]).length;
        continue;
      }
    }
    out += c;
  }
  return out;
}

// Rewrite every <text> of an SVG string. Text nodes between tags are
// converted; existing markup (tspans written by a module) is kept.
export function scriptSvg(svgStr) {
  return svgStr.replace(/(<text\b([^>]*)>)([\s\S]*?)(<\/text>)/g, (all, open, attrs, inner, close) => {
    const mono = /Mono|monospace/.test(attrs);
    const plain = inner.replace(/<[^>]+>/g, '');
    if (!/[_^]/.test(plain)) return all;
    if (mono && !MATH_HINT.test(plain) && CODE.test(plain)) {
      // a code label: only unescape \_ / \^
      return open + inner.replace(/\\([_^])/g, '$1') + close;
    }
    const body = inner.split(/(<[^>]+>)/).map((seg) => (seg.startsWith('<') ? seg : scriptText(seg))).join('');
    return open + body + close;
  });
}
