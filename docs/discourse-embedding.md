# Embedding Discourse topics on each chapter page

**Status:** Plan / blocked on topic-seeding (see [§ Seeding](#seeding-the-topics-on-d2ldiscoursegroup)).

## Why

Today every chapter page in the rendered book ends with a plain link of
the form

    [Discussions](https://d2l.discourse.group/t/<id>)

The reader clicks through to a separate site (Discourse) for the
discussion. The original `d2l.ai` instead **embeds the discussion topic
directly at the bottom of each chapter** — the comment thread renders
in-page, reading is anonymous, and the "reply" button takes the user
into the Discourse auth flow only when they actually want to post.

We want the same experience on `d2l.smola.org`, now backed by
`d2l.discourse.group`.

## Goal

Replace the trailing `<a href="https://d2l.discourse.group/t/<id>">Discussions</a>`
anchor on every chapter page with Discourse's official embedding
snippet, so the matching topic renders inline.

## How embedding works in modern Discourse (3.x+)

Discourse ships a first-class embed feature. Two pieces are needed:

### 1. One-time server-side configuration

`https://d2l.discourse.group/admin/customize/embedding`

- **Embeddable host**: `d2l.smola.org`
- **Embed user**: the Discourse account that "authors" any
  auto-created topics. A bot-style account (e.g. `d2l_book`) is
  customary so the topic ownership is clear.
- **Category**: the Discourse category newly-created embedded topics
  land in. We already have most topics from the old `discuss.d2l.ai`
  migration; this category is only used for any *new* page whose topic
  doesn't exist yet.
- **Path whitelist** (optional): restrict which URLs under
  `d2l.smola.org` can spawn topics. We can scope to
  `chapter_*/*.html` so the slides + landing page never trigger
  topic creation.
- **Embedded CSS** (under
  `Admin → Customize → Themes → Embedded CSS`): override Discourse's
  default colours / type so the embedded thread matches the book
  (d2l blue accent, Source Sans 3, white background).

This is a five-minute config and is **not** something the build can
do for us — it's a Discourse-server admin step.

### 2. Per-page embed snippet (replaces the link)

For each chapter page, append:

```html
<div id="discourse-comments"></div>
<script type="text/javascript">
  DiscourseEmbed = {
    discourseUrl: 'https://d2l.discourse.group/',
    topicId: <id>,
  };
  (function() {
    var d = document.createElement('script'); d.async = true;
    d.src = DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
    (document.head || document.body).appendChild(d);
  })();
</script>
```

`embed.js` is ~2 KB; it injects an `<iframe>` that points at
`https://d2l.discourse.group/embed/topic/<id>` and forwards reading
state via `postMessage`. Reading is anonymous; replying triggers
the user's existing Discourse auth.

If a page's topic doesn't exist yet, swap `topicId` for
`discourseEmbedUrl: '<canonical book URL>'` and Discourse will look up
or auto-create a topic the first time someone views the page (subject
to the path whitelist and the embed user's permissions).

### 3. Lazy loading (nice-to-have)

The 2-KB `embed.js` itself is cheap, but it loads an iframe that pulls
the full topic thread. For a page where most readers never scroll to
the bottom, this is wasted bandwidth. Gate it on visibility:

```js
var slot = document.getElementById('discourse-comments');
new IntersectionObserver(function (entries, obs) {
  if (entries.some(function (e) { return e.isIntersecting; })) {
    obs.disconnect();
    /* load embed.js as above */
  }
}, { rootMargin: '300px' }).observe(slot);
```

The build can emit this version unconditionally — it's a strict
improvement for any page longer than a screen-height.

## Implementation plan in this repo

Mirror what `tools/add_cfasync.py` already does for `data-cfasync`:
walk `_book/**/*.html` after the Quarto render finishes and rewrite
the discussion link in place. **No source `.md` edits are needed** —
the link text + URL pattern is stable across all chapters.

### Sketch: `tools/add_discourse_embed.py`

```python
import re, sys
from pathlib import Path

LINK_RE = re.compile(
    r'<a\s+[^>]*href="https://d2l\.discourse\.group/t/(\d+)"[^>]*>'
    r'\s*Discussions?\s*</a>')

EMBED_TEMPLATE = """\
<div id="discourse-comments" data-topic-id="{topic_id}"></div>
<script>
(function() {{
  var slot = document.getElementById('discourse-comments');
  if (!slot || !('IntersectionObserver' in window)) {{
    /* old browser: load immediately */
    return load();
  }}
  new IntersectionObserver(function(entries, obs) {{
    if (entries.some(function(e) {{ return e.isIntersecting; }})) {{
      obs.disconnect(); load();
    }}
  }}, {{ rootMargin: '300px' }}).observe(slot);
  function load() {{
    window.DiscourseEmbed = {{
      discourseUrl: 'https://d2l.discourse.group/',
      topicId: {topic_id},
    }};
    var d = document.createElement('script'); d.async = true;
    d.src = window.DiscourseEmbed.discourseUrl + 'javascripts/embed.js';
    (document.head || document.body).appendChild(d);
  }}
}})();
</script>
"""

def rewrite(path: Path) -> int:
    text = path.read_text(encoding='utf-8')
    out, n = LINK_RE.subn(
        lambda m: EMBED_TEMPLATE.format(topic_id=m.group(1)), text)
    if n:
        path.write_text(out, encoding='utf-8')
    return n

if __name__ == '__main__':
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '_book')
    total = sum(rewrite(p) for p in root.rglob('*.html'))
    print(f'add_discourse_embed: replaced {total} link(s)')
```

Wire into the Makefile next to `add_cfasync.py`, e.g.:

```make
python3 tools/add_cfasync.py _book; \
python3 tools/add_discourse_embed.py _book; \
```

Pages without a `Discussions` link (the landing page, the slides
index, the front-matter index) are untouched.

## Seeding the topics on d2l.discourse.group

This is the part we have to do **before** flipping the switch, and
the reason this note is being filed as a plan rather than as a
follow-up to the URL swap.

The current `[Discussions](https://d2l.discourse.group/t/<id>)`
links carry IDs that came from the **old** `discuss.d2l.ai` site.
The new `d2l.discourse.group` instance does not (yet) have those
topics — only the IDs that have been manually migrated. Switching
on embedding before the topics are seeded would render an empty
"Topic not found" iframe at the bottom of every chapter.

### Plan

For every chapter source file in this repo:

1. Extract the discussion-thread ID from its
   `[Discussions](https://d2l.discourse.group/t/<id>)` line.
2. Fetch the corresponding archived topic from the old
   `discuss.d2l.ai` instance — easiest via Discourse's JSON API:
   `https://discuss.d2l.ai/t/<id>.json` returns the topic title,
   original-post Markdown, all replies with author + timestamps.
   (The old site is read-only; we don't need to keep it online once
   the migration is done, but we do need the JSON snapshot.)
3. Post the topic into `d2l.discourse.group` via its REST API
   (`POST /posts.json` for the first post, then more `/posts.json`
   calls for replies, or `POST /t/<id>/posts.json` for bulk import).
   The Discourse `discourse_api` Ruby gem or the Python
   `pydiscourse` client both work; or curl + the admin API key.
4. **Pin the new topic IDs to the same IDs as the old site** if
   possible — easier with the Discourse admin import (which lets
   you preserve `topic_id`). Otherwise build a mapping
   `old_id → new_id` and rewrite the link bodies in the chapter
   `.md` files to point at the new ID before the next render.
5. Verify a small set of topics render correctly when embedded on
   a staging page before going site-wide.

This is a one-shot migration, expected to take "a bit of time" per
the user. Tracking it as a separate work item from the embed wiring.

### Suggested order of operations

1. Stand up an embed-user account + category on
   `d2l.discourse.group`.
2. Migrate the old topics into the new instance (steps 1–4 above).
3. Add `tools/add_discourse_embed.py` and wire it into `make html`.
4. Build, upload, smoke-test on a single chapter URL.
5. Flip on path whitelisting in the embedding admin so unwanted
   pages (e.g. the index) can't accidentally create topics.

## References

- Discourse embedding docs:
  <https://meta.discourse.org/t/embed-a-discussion-from-discourse-on-another-site/31875>
- Embed.js source:
  <https://github.com/discourse/discourse/blob/main/public/javascripts/embed.js>
- Embed iframe-side renderer:
  <https://github.com/discourse/discourse/blob/main/app/assets/javascripts/discourse/app/templates/embed.hbs>
- d2l.ai's implementation (reference): inspect the bottom of any
  chapter page, e.g. <https://d2l.ai/chapter_preliminaries/ndarray.html>
- Old Discourse REST API for fetching topics:
  <https://docs.discourse.org/#tag/Topics/operation/getTopic>
