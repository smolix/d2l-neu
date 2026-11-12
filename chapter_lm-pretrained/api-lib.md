# API Client Scaffolding (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It will hold the scaffolding of
:numref:`sec_lm_api` that the section does not show: `ChatResult`,
`ReplayMiss`, `CostCapExceeded`, the `ChatAPI` class body (the section
shows its `async def achat`, merged in by `@d2l.add_to_class`),
`price_from_listing`, `openrouter_api`, `LocalChatServer` and
`FakeChatServer`. Every block is tagged `%%tab pytorch`; `openai`, `httpx`
and `asyncio` are imported inside the functions that use them, so
`import d2l.torch` works without them. Stub until the section's library code
is written (stage 2, step 6).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_api`
