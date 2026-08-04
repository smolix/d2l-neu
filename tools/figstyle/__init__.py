"""figstyle — the unified d2l figure toolkit.

One token layer (figstyle.tokens), three renderers:

* ``figstyle.svg``   — deterministic pure-SVG composer for block/flow
  diagrams (outlined text, tight canvas).
* ``figstyle.mpl``   — matplotlib theme + primitives for mathematical
  illustrations (``use_style()`` then draw as usual).
* ``figstyle.export``— emits tokens as JSON and as ``diagrams/tokens.mjs``
  so the JS slide-diagram pipeline shares the same palette.

See docs/figure-style-guide.md for the house style these implement.
"""

from . import tokens  # noqa: F401
