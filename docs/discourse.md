# Posting to d2l.discourse.group via the Discourse MCP

**Status (2026-06-08):** not yet wired up — the API key is the remaining
step (planned for 2026-06-09). Everything else below is ready to run once
the key exists.

Goal: let Claude Code read the per-chapter discussion threads and post
replies (e.g. "this suggestion is now incorporated", or answers to
questions) **under review**, instead of copy-pasting by hand.

Related: in-page comment embedding is a separate concern — see
[discourse-embedding.md](discourse-embedding.md).

## Which MCP

Use the **official** server, [`@discourse/mcp`](https://github.com/discourse/discourse-mcp)
(shipped Oct 2025). It wraps the Discourse REST API and is safety-first:

- **Read-only by default.** Reading topics/posts needs no auth and no flags.
- **Writes are opt-in**, rate-limited, and bounded by the API key's scopes.
  Write tools include `discourse_create_post`, `discourse_create_topic`,
  `discourse_update_topic`, and **`discourse_save_draft`** (stage a reply as
  a Discourse draft you eyeball in the forum UI before publishing).
- Tether to one instance with `--site https://d2l.discourse.group` (hides the
  multi-site selector).

## Setup (do this once)

### 1. Create an API key  ← the part to do tomorrow

Two options:

- **Admin key** — `d2l.discourse.group/admin/api/keys`. Scope it narrowly
  (e.g. *create posts* only) and bind it to the user account that should
  appear as the author of replies (`api_username`).
- **User API key** (no admin needed) — run and follow the browser prompt:
  ```bash
  npx @discourse/mcp@latest generate-user-api-key --site https://d2l.discourse.group
  ```

### 2. Store credentials in a gitignored profile

Keep the key out of the shell history, process list, and MCP config by
putting it in a profile file (not on the command line):

```json
// .discourse-mcp.json   (repo root; chmod 600; gitignored)
{
  "auth_pairs": [
    {
      "site": "https://d2l.discourse.group",
      "api_key": "PASTE_ADMIN_KEY_HERE",
      "api_username": "your_username"
    }
  ],
  "allow_writes": true,
  "read_only": false
}
```

For a **user** API key, swap the auth pair for:
`{ "site": "...", "user_api_key": "KEY", "user_api_client_id": "CLIENT_ID" }`.

Then:
```bash
chmod 600 .discourse-mcp.json
echo '.discourse-mcp.json' >> .gitignore   # if not already ignored
```

### 3. Register the MCP server with Claude Code, then restart

```bash
claude mcp add discourse -- npx -y @discourse/mcp@latest \
  --site https://d2l.discourse.group \
  --profile /home/smola/d2l-neu/.discourse-mcp.json
```

The Discourse tools load on the **next** Claude Code session start. Verify
with `claude mcp list` (should show `discourse`).

Read-only smoke test (no key needed) if you want to confirm the server runs
before adding the key:
```bash
npx -y @discourse/mcp@latest --site https://d2l.discourse.group --log_level debug
```

## How replies should be made (review-gated)

Even with writes enabled, treat posting as **outward-facing and
irreversible** (public, indexed even if later deleted). Workflow:

1. **Scope** — do *not* blanket-reply to all ~166 linked threads. Pick the
   threads with an incorporated suggestion or an answerable question (a
   triage pass over recent commits/fixes vs. the linked topics works).
2. **Read** each target thread (`GET /t/<id>.json`, unauthenticated) to see
   the actual suggestion/question.
3. **Draft** a reply, cross-referenced to what actually changed in the book.
4. **Review** — either `discourse_save_draft` (review in-forum) or paste the
   draft here for sign-off.
5. **Post** with `discourse_create_post` only after approval.

Pick the **author account** deliberately (a `d2l_book` bot vs. a personal
account) — it's whatever `api_username` / the user key resolves to.

## Useful flags

| Flag | Purpose |
|------|---------|
| `--site <url>` | Tether to one Discourse instance |
| `--allow_writes` + `--read_only=false` | Enable write tools (both required) |
| `--auth_pairs '[…]'` / `--profile <file>` | Credentials (prefer the profile file) |
| `--log_level debug` | Verbose request/response logging |

## Sources

- Blog: <https://blog.discourse.org/2025/10/discourse-mcp-is-here/>
- Repo: <https://github.com/discourse/discourse-mcp>
- Discussion: <https://meta.discourse.org/t/is-there-an-official-discourse-model-context-protocol-mcp/364859>
