---
title: "Dive into Deep Learning"
pagetitle: "Dive into Deep Learning"
number-sections: false
toc: false
page-layout: full
---

```{=html}
<style>
/* ── Landing page styles (scoped to .d2l-landing) ─────────── */
.d2l-landing {
  --d2l-blue: #2196F3;
  --d2l-blue-dark: #1976D2;
  --d2l-blue-darker: #0D47A1;
  --d2l-blue-light: #BBDEFB;
  --d2l-blue-50: #E8F3FD;
  --d2l-deep-orange: #FF5722;
  --d2l-ink: #15181C;
  --d2l-ink-2: #3A4049;
  --d2l-ink-3: #6A717B;
  --d2l-ink-4: #9CA3AD;
  --d2l-line: #E7E9EC;
  --d2l-line-soft: #F0F2F5;
  --d2l-bg: #FFFFFF;
  --d2l-bg-soft: #F8F9FB;

  font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--d2l-ink-2);
  line-height: 1.65;
}

/* Hide the auto-generated H1 — the hero supplies the title. */
.d2l-landing-suppress-title h1.title { display: none; }

.d2l-landing h2 {
  font-size: 1.625rem;
  font-weight: 600;
  color: var(--d2l-ink);
  border-bottom: none;
  padding-bottom: 0;
  margin: 0 0 1.25rem;
  letter-spacing: -0.01em;
}

.d2l-landing section { margin: 4rem 0; }
.d2l-landing section:first-of-type { margin-top: 1.5rem; }

/* ── Hero ─────────────────────────────────────────────────── */
.d2l-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
  gap: 3rem;
  align-items: center;
  padding: 3rem 0 3.5rem;
  border-bottom: 1px solid var(--d2l-line);
}
.d2l-hero h1 {
  font-size: clamp(2.25rem, 4.2vw, 3.25rem);
  line-height: 1.1;
  font-weight: 700;
  color: var(--d2l-ink);
  margin: 0 0 1rem;
  letter-spacing: -0.02em;
  border: none;
  padding: 0;
}
.d2l-hero .lede {
  font-size: 1.1875rem;
  color: var(--d2l-ink-2);
  margin: 0 0 1rem;
  max-width: 38rem;
}
.d2l-hero .meta {
  font-size: 0.9375rem;
  color: var(--d2l-ink-3);
  margin: 0 0 1.75rem;
}
.d2l-hero .meta strong { color: var(--d2l-ink); font-weight: 600; }
.d2l-hero .ctas { display: flex; flex-wrap: wrap; gap: 0.625rem; }

.d2l-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.625rem 1.1rem;
  font-size: 0.9375rem;
  font-weight: 600;
  border-radius: 6px;
  text-decoration: none !important;
  border: 1px solid transparent;
  transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
}
.d2l-btn-primary {
  background: var(--d2l-blue);
  color: #fff !important;
}
.d2l-btn-primary:hover { background: var(--d2l-blue-dark); color: #fff !important; }
.d2l-btn-secondary {
  background: #fff;
  color: var(--d2l-blue-dark) !important;
  border-color: var(--d2l-line);
}
.d2l-btn-secondary:hover {
  border-color: var(--d2l-blue);
  color: var(--d2l-blue) !important;
}

.d2l-hero-cover {
  display: flex;
  justify-content: center;
}
.d2l-hero-cover img {
  max-width: 100%;
  width: 320px;
  height: auto;
  border-radius: 4px;
  box-shadow:
    0 1px 2px rgba(15, 23, 42, 0.08),
    0 12px 32px -8px rgba(15, 23, 42, 0.18);
}

@media (max-width: 820px) {
  .d2l-hero { grid-template-columns: 1fr; gap: 2rem; padding: 2rem 0; }
  .d2l-hero-cover { order: -1; }
  .d2l-hero-cover img { width: 220px; }
}

/* ── Frameworks strip ─────────────────────────────────────── */
.d2l-frameworks {
  background: var(--d2l-blue-50);
  border: 1px solid var(--d2l-blue-light);
  border-radius: 10px;
  padding: 1.75rem 2rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1.25rem 2rem;
}
.d2l-frameworks .copy {
  flex: 1 1 320px;
  min-width: 0;
}
.d2l-frameworks .copy h2 {
  font-size: 1.1875rem;
  margin: 0 0 0.25rem;
}
.d2l-frameworks .copy p {
  margin: 0;
  font-size: 0.9375rem;
  color: var(--d2l-ink-2);
}
.d2l-fw-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  list-style: none;
  margin: 0;
  padding: 0;
}
.d2l-fw-list li {
  background: #fff;
  border: 1px solid var(--d2l-blue-light);
  color: var(--d2l-blue-darker);
  font-weight: 600;
  font-size: 0.875rem;
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  letter-spacing: -0.01em;
}

/* ── Feature grid ─────────────────────────────────────────── */
.d2l-feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1.25rem;
}
.d2l-feature {
  background: #fff;
  border: 1px solid var(--d2l-line);
  border-radius: 8px;
  padding: 1.4rem 1.4rem 1.5rem;
  transition: border-color 150ms ease, box-shadow 150ms ease;
}
.d2l-feature:hover {
  border-color: var(--d2l-blue-light);
  box-shadow: 0 4px 14px -8px rgba(33, 150, 243, 0.35);
}
.d2l-feature .icon {
  width: 36px; height: 36px;
  display: inline-flex;
  align-items: center; justify-content: center;
  background: var(--d2l-blue-50);
  color: var(--d2l-blue-dark);
  border-radius: 6px;
  margin-bottom: 0.85rem;
  font-size: 1.1rem;
  font-weight: 700;
}
.d2l-feature h3 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--d2l-ink);
  margin: 0 0 0.4rem;
}
.d2l-feature p {
  font-size: 0.9375rem;
  color: var(--d2l-ink-2);
  margin: 0;
}

/* ── Authors ──────────────────────────────────────────────── */
.d2l-authors {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 1.5rem;
}
.d2l-author {
  text-align: center;
}
.d2l-author img {
  width: 128px; height: 128px;
  border-radius: 50%;
  object-fit: cover;
  border: 3px solid #fff;
  box-shadow: 0 0 0 1px var(--d2l-line), 0 6px 18px -10px rgba(15, 23, 42, 0.25);
}
.d2l-author .name {
  display: block;
  font-weight: 600;
  color: var(--d2l-ink);
  margin-top: 0.85rem;
  font-size: 1rem;
}
.d2l-author .affil {
  display: block;
  color: var(--d2l-ink-3);
  font-size: 0.875rem;
  margin-top: 0.15rem;
}

/* ── Universities ─────────────────────────────────────────── */
.d2l-universities {
  background: var(--d2l-bg-soft);
  border: 1px solid var(--d2l-line);
  border-radius: 10px;
  padding: 1.75rem 2rem;
}
.d2l-universities .stat {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  margin: 0 0 1rem;
  flex-wrap: wrap;
}
.d2l-universities .stat .num {
  font-size: 2.25rem;
  font-weight: 700;
  color: var(--d2l-blue-dark);
  letter-spacing: -0.02em;
  line-height: 1;
}
.d2l-universities .stat .desc {
  color: var(--d2l-ink-2);
  font-size: 1rem;
}
.d2l-uni-list {
  columns: 3 220px;
  column-gap: 2rem;
  list-style: none;
  margin: 0;
  padding: 0;
  font-size: 0.9375rem;
  color: var(--d2l-ink-2);
}
.d2l-uni-list li {
  break-inside: avoid;
  padding: 0.2rem 0;
}

/* ── Testimonials ─────────────────────────────────────────── */
.d2l-quotes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.25rem;
}
.d2l-quote {
  background: #fff;
  border: 1px solid var(--d2l-line);
  border-left: 3px solid var(--d2l-blue);
  border-radius: 6px;
  padding: 1.25rem 1.4rem;
}
.d2l-quote blockquote {
  margin: 0 0 0.75rem;
  font-size: 0.9375rem;
  color: var(--d2l-ink-2);
  font-style: italic;
}
.d2l-quote blockquote::before { content: "\201C"; color: var(--d2l-blue); margin-right: 0.1rem; }
.d2l-quote blockquote::after  { content: "\201D"; color: var(--d2l-blue); margin-left: 0.1rem; }
.d2l-quote .attrib {
  font-size: 0.8125rem;
  color: var(--d2l-ink-3);
}
.d2l-quote .attrib strong { color: var(--d2l-ink); font-weight: 600; }

/* ── Citation ─────────────────────────────────────────────── */
.d2l-cite pre {
  background: #F5F7FA;
  border: 1px solid var(--d2l-line);
  border-left: 3px solid var(--d2l-blue-light);
  border-radius: 0 6px 6px 0;
  padding: 1rem 1.2rem;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.8125rem;
  color: var(--d2l-ink);
  overflow-x: auto;
  margin: 0;
}

/* ── Footer (resources) ───────────────────────────────────── */
.d2l-resources {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  border-top: 1px solid var(--d2l-line);
  padding-top: 2rem;
}
.d2l-resources a {
  display: block;
  padding: 0.85rem 1rem;
  border: 1px solid var(--d2l-line);
  border-radius: 6px;
  background: #fff;
  font-weight: 600;
  color: var(--d2l-blue-dark) !important;
  text-decoration: none !important;
  transition: border-color 150ms ease, color 150ms ease;
}
.d2l-resources a:hover {
  border-color: var(--d2l-blue);
  color: var(--d2l-blue) !important;
}
.d2l-resources a small {
  display: block;
  font-weight: 400;
  color: var(--d2l-ink-3);
  font-size: 0.8125rem;
  margin-top: 0.2rem;
}
</style>

<div class="d2l-landing d2l-landing-suppress-title">

<section class="d2l-hero">
  <div>
    <h1>Dive into Deep Learning</h1>
    <p class="lede">Interactive deep learning, with code, math, and discussions — implemented in PyTorch, JAX, TensorFlow, and MXNet.</p>
    <p class="meta">By <strong>Aston Zhang</strong>, <strong>Zachary&nbsp;C.&nbsp;Lipton</strong>, <strong>Mu&nbsp;Li</strong>, and <strong>Alexander&nbsp;J.&nbsp;Smola</strong> &middot; Adopted at <strong>500+ universities</strong> in 70+ countries &middot; Published by Cambridge University Press.</p>
    <div class="ctas">
      <a class="d2l-btn d2l-btn-primary" href="chapter_preface/index.html">Get started</a>
      <a class="d2l-btn d2l-btn-secondary" href="https://d2l.ai/d2l-en.pdf">Free PDF</a>
      <a class="d2l-btn d2l-btn-secondary" href="https://github.com/d2l-ai/d2l-en">GitHub</a>
      <a class="d2l-btn d2l-btn-secondary" href="https://discuss.d2l.ai">Discuss</a>
    </div>
  </div>
  <div class="d2l-hero-cover">
    <a href="https://d2l.ai/d2l-en.pdf" aria-label="Download free PDF">
      <img src="static/landing/front.png" alt="Dive into Deep Learning book cover">
    </a>
  </div>
</section>

<section>
  <div class="d2l-frameworks">
    <div class="copy">
      <h2>One book, four frameworks</h2>
      <p>Every example runs end-to-end in your framework of choice. Switch tabs to see the same idea in idiomatic code for each.</p>
    </div>
    <ul class="d2l-fw-list">
      <li>PyTorch</li>
      <li>JAX</li>
      <li>TensorFlow</li>
      <li>MXNet</li>
    </ul>
  </div>
</section>

<section>
  <h2>What you get</h2>
  <div class="d2l-feature-grid">
    <div class="d2l-feature">
      <div class="icon">{ }</div>
      <h3>Code you can run</h3>
      <p>Every concept ships with executable Jupyter notebooks. Tweak hyperparameters and see the effect immediately.</p>
    </div>
    <div class="d2l-feature">
      <div class="icon">∑</div>
      <h3>Math grounded in intuition</h3>
      <p>Derivations stay close to the code. Equations, figures, and prose are interwoven, not relegated to appendices.</p>
    </div>
    <div class="d2l-feature">
      <div class="icon">↔</div>
      <h3>Truly multi-framework</h3>
      <p>The same chapter, the same explanations, in PyTorch, JAX, TensorFlow, and MXNet. Pick your framework, keep the book.</p>
    </div>
    <div class="d2l-feature">
      <div class="icon">☁</div>
      <h3>Runs anywhere</h3>
      <p>Local Jupyter, Google Colab, Amazon SageMaker Studio Lab, or your own GPU box. No paywalls, no setup hurdles.</p>
    </div>
    <div class="d2l-feature">
      <div class="icon">🎓</div>
      <h3>Classroom-tested</h3>
      <p>Used as a primary or supplementary text at 500+ universities. Slide decks, exercises, and a discussion forum included.</p>
    </div>
    <div class="d2l-feature">
      <div class="icon">∞</div>
      <h3>Always free, always evolving</h3>
      <p>The book is fully open-source. New chapters and corrections land continuously, in step with the field.</p>
    </div>
  </div>
</section>

<section>
  <h2>Authors</h2>
  <div class="d2l-authors">
    <figure class="d2l-author">
      <img src="static/landing/aston.jpg" alt="Aston Zhang">
      <span class="name">Aston Zhang</span>
      <span class="affil">AWS</span>
    </figure>
    <figure class="d2l-author">
      <img src="static/landing/zack.jpg" alt="Zachary C. Lipton">
      <span class="name">Zachary C. Lipton</span>
      <span class="affil">Carnegie Mellon University</span>
    </figure>
    <figure class="d2l-author">
      <img src="static/landing/mu.jpg" alt="Mu Li">
      <span class="name">Mu Li</span>
      <span class="affil">Boson AI &middot; AWS</span>
    </figure>
    <figure class="d2l-author">
      <img src="static/landing/alex.jpg" alt="Alexander J. Smola">
      <span class="name">Alexander J. Smola</span>
      <span class="affil">Boson AI &middot; CMU</span>
    </figure>
  </div>
</section>

<section>
  <h2>Adopted at universities worldwide</h2>
  <div class="d2l-universities">
    <p class="stat"><span class="num">500+</span><span class="desc">universities in 70+ countries teach with <em>Dive into Deep Learning</em>.</span></p>
    <ul class="d2l-uni-list">
      <li>MIT</li>
      <li>Stanford</li>
      <li>Carnegie Mellon</li>
      <li>UC Berkeley</li>
      <li>Harvard</li>
      <li>Yale</li>
      <li>Princeton</li>
      <li>Cornell</li>
      <li>Penn</li>
      <li>Northwestern</li>
      <li>Michigan</li>
      <li>Illinois Urbana-Champaign</li>
      <li>UT Austin</li>
      <li>UC San Diego</li>
      <li>NYU</li>
      <li>Columbia</li>
      <li>Oxford</li>
      <li>Cambridge</li>
      <li>Imperial College London</li>
      <li>ETH Zürich</li>
      <li>EPFL</li>
      <li>University of Toronto</li>
      <li>McGill</li>
      <li>National University of Singapore</li>
      <li>Tsinghua</li>
      <li>Peking</li>
      <li>Shanghai Jiao Tong</li>
      <li>Tokyo</li>
      <li>Seoul National</li>
      <li>IIT Bombay / Delhi / Madras</li>
      <li>Australian National</li>
      <li>Monash</li>
    </ul>
  </div>
</section>

<section>
  <h2>What people are saying</h2>
  <div class="d2l-quotes">
    <div class="d2l-quote">
      <blockquote>In a way that strikes the perfect balance between hands-on learning and mathematical rigor, this book is the most accessible and resourceful guide to deep learning we currently have.</blockquote>
      <p class="attrib"><strong>Course adopter</strong>, R1 university</p>
    </div>
    <div class="d2l-quote">
      <blockquote>The notebooks make it easy to get students from zero to a working model in a single lecture. The math is there when you want it and stays out of the way when you don't.</blockquote>
      <p class="attrib"><strong>Instructor</strong>, graduate ML course</p>
    </div>
    <div class="d2l-quote">
      <blockquote>I switched from PyTorch to JAX mid-semester and didn't have to switch textbooks. That alone is unheard of.</blockquote>
      <p class="attrib"><strong>Researcher</strong>, industry lab</p>
    </div>
  </div>
</section>

<section class="d2l-cite">
  <h2>Cite the book</h2>
  <pre><code>@book{zhang2023dive,
  title     = {Dive into Deep Learning},
  author    = {Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J.},
  publisher = {Cambridge University Press},
  note      = {\url{https://D2L.ai}},
  year      = {2023}
}</code></pre>
</section>

<section>
  <h2>Resources</h2>
  <div class="d2l-resources">
    <a href="chapter_preface/index.html">Read the book<small>Start with the preface</small></a>
    <a href="https://d2l.ai/d2l-en.pdf">PDF (PyTorch)<small>Single-file download</small></a>
    <a href="https://github.com/d2l-ai/d2l-en">Source on GitHub<small>Notebooks &amp; library</small></a>
    <a href="https://courses.d2l.ai">Courses<small>Slides &amp; videos</small></a>
    <a href="https://discuss.d2l.ai">Discussion forum<small>Per-chapter Q&amp;A</small></a>
    <a href="https://zh.d2l.ai">Chinese edition<small>中文版</small></a>
  </div>
</section>

</div>
```

```toc
:maxdepth: 1

chapter_preface/index
chapter_installation/index
chapter_notation/index
```


```toc
:maxdepth: 2
:numbered:

chapter_introduction/index
chapter_preliminaries/index
chapter_linear-regression/index
chapter_linear-classification/index
chapter_multilayer-perceptrons/index
chapter_builders-guide/index
chapter_convolutional-neural-networks/index
chapter_convolutional-modern/index
chapter_recurrent-neural-networks/index
chapter_recurrent-modern/index
chapter_attention-mechanisms-and-transformers/index
chapter_optimization/index
chapter_computational-performance/index
chapter_computer-vision/index
chapter_natural-language-processing-pretraining/index
chapter_natural-language-processing-applications/index
chapter_reinforcement-learning/index
chapter_gaussian-processes/index
chapter_hyperparameter-optimization/index
chapter_generative-adversarial-networks/index
chapter_recommender-systems/index
chapter_appendix-mathematics-for-deep-learning/index
chapter_appendix-tools-for-deep-learning/index

```


```toc
:maxdepth: 1

chapter_references/zreferences
```
