<script>
(function () {
  var MAP = [
    ['chapter_preface',                    'i-type'],
    ['chapter_installation',               'i-terminal'],
    ['chapter_notation',                   'i-hash'],
    ['chapter_introduction',               'i-compass'],
    ['chapter_preliminaries',              'i-book'],
    ['chapter_linear-regression',          'i-flow'],
    ['chapter_linear-classification',      'i-flow'],
    ['chapter_multilayer-perceptrons',     'i-layers'],
    ['chapter_builders-guide',             'i-cube'],
    ['chapter_convolutional-neural-networks', 'i-grid'],
    ['chapter_convolutional-modern',       'i-grid'],
    ['chapter_recurrent-neural-networks',  'i-flow'],
    ['chapter_recurrent-modern',           'i-flow'],
    ['chapter_attention-mechanisms-and-transformers', 'i-sparkle'],
    ['chapter_optimization',               'i-gauge'],
    ['chapter_computational-performance',  'i-gauge'],
    ['chapter_computer-vision',            'i-eye'],
    ['chapter_natural-language-processing-pretraining', 'i-message'],
    ['chapter_natural-language-processing-applications', 'i-message'],
    ['chapter_reinforcement-learning',     'i-target'],
    ['chapter_gaussian-processes',         'i-flow'],
    ['chapter_hyperparameter-optimization','i-gauge'],
    ['chapter_generative-adversarial-networks','i-sparkle'],
    ['chapter_recommender-systems',        'i-target'],
    ['chapter_appendix-mathematics-for-deep-learning', 'i-hash'],
    ['chapter_appendix-tools-for-deep-learning', 'i-tool'],
    ['references',                         'i-cite']
  ].sort(function(a, b) { return b[0].length - a[0].length; });

  function iconFor(href) {
    if (!href) return null;
    for (var i = 0; i < MAP.length; i++) {
      if (href.indexOf('/' + MAP[i][0]) !== -1 || href.indexOf(MAP[i][0] + '/') !== -1 || href.indexOf(MAP[i][0] + '.') !== -1) {
        return MAP[i][1];
      }
    }
    return null;
  }

  function svgMarkup(id) {
    return '<svg class="d2l-icon" aria-hidden="true"><use href="#' + id + '"/></svg>';
  }

  function decorateTopLevel() {
    document.querySelectorAll(
      '#quarto-sidebar .sidebar-item > .sidebar-item-container > .sidebar-item-text'
    ).forEach(function(a) {
      var id = iconFor(a.getAttribute('href'));
      if (id && !a.querySelector('.d2l-icon')) {
        a.insertAdjacentHTML('afterbegin', svgMarkup(id));
      }
    });

    document.querySelectorAll('.sidebar-item-section').forEach(function(section) {
      var first = section.querySelector('.sidebar-item-container li a');
      var header = section.querySelector(':scope > .sidebar-item-container > .sidebar-item-text');
      if (!first || !header || header.querySelector('.d2l-icon')) return;
      var id = iconFor(first.getAttribute('href'));
      if (id) header.insertAdjacentHTML('afterbegin', svgMarkup(id));
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', decorateTopLevel);
  } else {
    decorateTopLevel();
  }
})();
</script>
