# Forum discussions on book pages

Every section page ends with a link to its topic on https://d2l.discourse.group, as a `[Discussions](…/t/<id>)` line
just before the page's `<!-- slides -->` marker.

## Topics
- **Older pages** carry one topic per framework, each link inside a `:begin_tab:` block.
- **Pages added since the 2026 rebuild** carry one untabbed link to a single topic shared by all framework tabs:
  - the topic is titled "<page title> (<chapter title>)" and sits in the English Version category;
  - its first post is the page's d2l.ai URL.
- **New pages:** `python3 tools/create_forum_topics.py` lists the pages without a link (a dry run). With `--create`, it
  creates their topics and inserts the links.
  - It needs an admin API key in `~/.config/d2l/discourse.env` (`DISCOURSE_API_KEY=...`, mode 600).
  - Never commit the key.
- The PDF build strips the links (`tools/gen_pdf.py`).

## Embed
The Discourse script in `_d2l-tabs.html` turns the link into an embedded comment stream:
- **Topic:** it hides the link, and loads the topic of the active framework tab (or the page's single topic).
- **Placement:** below the previous/next page navigation, so that a long discussion does not push those links out of
  reach.
- **Tab switches:** it reloads when the reader switches framework tabs.

Posting through the Discourse MCP server is described in [discourse.md](discourse.md).
