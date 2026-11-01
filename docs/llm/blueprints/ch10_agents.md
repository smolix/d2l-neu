# Chapter 10: Agents and Tool Use — Blueprint

**Label:** `chap_lm_agents` · **Arc:** V. Acting · **Sections:** 10.1 `sec_lm_tool_calling`, 10.2 `sec_lm_structured_outputs`, 10.3 `sec_lm_agent_loops`, 10.4 `sec_lm_agentic_rl`, 10.5 `sec_lm_agent_eval` · **Tiers:** T0 (10.2), T1 (10.1, 10.3, 10.5), T3 with a T1 reduction (10.4); T2 optional extensions in 10.5

## Chapter overview

### Draft introduction (for the chapter opener)

A language model that only emits text can be asked a question; a model that emits *actions* can be given a job. This chapter turns the pretrained and post-trained models of Chapters 1–9 into agents: systems that call tools, read what comes back, and decide what to do next over many turns. We build the pieces in the order a practitioner meets them. First, how a tool schema is rendered into a chat template and how a call is parsed back out of generated text (10.1). Second, how to guarantee that the call parses, by masking the vocabulary with an automaton, and what that masking does to the model's distribution (10.2). Third, the loop itself: a sandboxed shell, an interface, a context policy, and honest cost accounting (10.3). Fourth, how an agent is trained with reinforcement learning when part of every trajectory is written by the environment (10.4). Fifth, how to evaluate and attack an agent without fooling ourselves (10.5). Every measurement comes with an interval, every sandbox with its limits.

### Learning objectives (measurable)

After this chapter a reader can:

1. Render the same tool schema through the Qwen3 and Qwen3.5 chat templates, explain each difference, and parse generated calls with a parser that returns errors instead of raising; report call accuracy per BFCL-style category with paired-bootstrap intervals (10.1).
2. Build a token-level finite-state masker from a regex or JSON schema, verify it mask-for-mask against XGrammar, and compute exactly, on a finite language, the distribution that masked sampling produces and its KL divergence from the true conditional (10.2).
3. Write a ≤ ~100-line agent loop, run it in a resource-limited sandbox whose escape routes are listed and tested, and report how interface and context policy change success, steps, tokens and latency (10.3).
4. Derive the policy gradient of a multi-turn episode in which only the agent's tokens are actions, implement the loss mask, and check it to 1e-5 against per-turn gradients; train a search agent with GRPO and compare it with prompting and single-turn RAG (10.4).
5. Report pass@1, pass@k and pass^k with cluster-bootstrap intervals, separate task variance from trial variance, plot success against cost, and measure attack success under indirect prompt injection with and without a defense (10.5).

### Prerequisites

Book labels:

- `sec_decoding` — sampling from logits; 10.2 replaces the sampler's logits step by a masked one.
- `sec_mdp`, `sec_imitation` (`subsec_il_error_bounds`) — episodes, horizon, compounding error; 10.3 measures per-step hazard and 10.1's fine-tuning is behavior cloning.
- `sec_rl_sequences` (`tab_rl_notation_contract`, `eq_seq_factorization`, `subsec_rl_sequences_hacking`, exercise on tool outputs in the trajectory) — 10.4 extends this exercise into a section.
- `sec_policygradient`, `sec_baselines` (`eq_traj_prob`, the zero-mean lemma), `sec_ppo` — used verbatim in 10.4.
- `sec_mdl-statistics` — bootstrap and confidence intervals (10.5).

Earlier sections of this part:

- 1.1, 1.2: `d2l.load_hf`, `d2l.generate`.
- 1.3: chat templates, special tokens. 10.1 does not re-teach them.
- 1.5: `d2l.EvalSuite`, `d2l.paired_bootstrap`.
- 2.4–2.5: `d2l.BookRetriever` as the search tool, `d2l.book_qa` and the single-turn RAG baseline.
- 3.6: byte-level vocabularies (10.2).
- 7.2 and 7.4: assistant-only SFT and LoRA (10.1).
- 8.5 and 8.6: GRPO, verifiable rewards and rollout systems (10.4).

### Section dependency graph

```
        1.3 ─▶ 10.1 ──────────────┐
 sec_decoding ─▶ 10.2 ◀── 10.1   │   (10.2 constrains the 10.1 tool arguments)
                  │               ▼
                  └────────────▶ 10.3 ──▶ 10.5
                                  │         ▲
        8.5, 8.6, 2.4 ──────────▶ 10.4 ────┘   (10.4 reuses 10.1's turn masks and 10.3's Episode/Sandbox)
```

10.1 → 10.2 is a soft dependency: 10.2 runs without 10.1's fine-tuned adapter. 10.3 forward-references nothing from 10.4. 10.1 forward-references 10.3's `d2l.run_python` as a black box (see 10.1, "Builds on (this part)").

### Running examples and shared artifacts

Used:

- R1 (`d2l.GPT`) is not needed here. Released models are used through Transformers.
- R3 (the book as a corpus) is the retrieval tool of 10.1 and the search environment of 10.4, plus the corpus that 10.5 poisons.

Produced (each has a `DATA_HUB` key, committed by the section that owns it):

| Artifact | Section | Format and size class |
|---|---|---|
| `lm-bfcl-subset` | 10.1 | JSON list of BFCL v3 item ids per category, the dataset revision, and the seed; data are fetched from the pinned revision |
| `lm-tool-traces-synth` | 10.1 | JSONL, 3–4k verified synthetic tool traces, with a manifest |
| `lm-lora-tools-qwen3-1p7b` | 10.1 | LoRA adapter (tens of MB) |
| `lm-repair30` (+ `lm-repair30-dev10`) | 10.3 | tar.gz of 30 repair tasks (10 modules × 3 injected bugs) and a 10-task development set; hidden tests included |
| `lm-agent-runs-ablation` | 10.3 | JSONL trajectories of the interface × context grid |
| `lm-book-search-qa` | 10.4 | JSONL train/dev/test questions over R3, with gold passage ids and short answers |
| `lm-search-agent-logs` | 10.4 | T3 training logs and rollout samples; T1 0.6B checkpoint |
| `lm-injection-suite` | 10.5 | JSON: injection templates, poisoned-chunk patches for R3, injected repair variants |
| `lm-agent-eval-trials` | 10.5 | per-(task, trial) outcome matrices with token and latency fields |

Shared code (details in each section's **Exports**): the `d2l.ToolRegistry` family (10.1), `d2l.TokenFSM` (10.2), `d2l.Sandbox` and `d2l.Episode` (10.3), `d2l.Trajectory` and `d2l.masked_pg_loss` (10.4), `d2l.pass_pow_k` and `d2l.agent_report` (10.5).

### Compute plan

Budgets are caps to design against, not measurements. Every number labeled "to be measured" is filled in by the first authoring run.

| § | Tier | Budget cap | What runs | Committed artifacts |
|---|---|---|---|---|
| 10.1 | T1 | ≤ ~15 min per notebook run on one Ada GPU, ≤ 15 GiB | Qwen3-1.7B (3.4 GB bf16 weights) greedy evaluation on ~450 items; LoRA fine-tune on ≤ 4k short traces | `lm-bfcl-subset`, `lm-tool-traces-synth`, `lm-lora-tools-qwen3-1p7b` |
| 10.2 | T0 | minutes on CPU with Qwen3-0.6B; optional GPU | FSM construction is pure Python; the generation experiment uses ≤ 100 prompts | none (a precomputed mask-cache timing table is printed, not stored) |
| 10.3 | T1 | ≤ ~15 min for the interactive cells; the headline grid is a one-off offline run (hours) whose trajectories are committed | Qwen3-4B (8 GB bf16) with batched episode stepping | `lm-repair30`, `lm-repair30-dev10`, `lm-agent-runs-ablation` |
| 10.4 | T3 (reduced T1) | T3: ≤ ~2 h on 8× A100 40 GB (≤ 16 A100-hours; one run, limited access); T1: ≤ ~15 min with Qwen3-0.6B | GRPO of a search agent | `lm-book-search-qa`, `lm-search-agent-logs` |
| 10.5 | T1 (T2 optional) | ≤ ~15 min on committed trials plus fresh injection runs; T2: ≤ a few hours | pass^k analysis, injection experiments, optional τ²-bench and SWE-bench Verified subsets | `lm-injection-suite`, `lm-agent-eval-trials` |

## Assignment

v5 attaches no assignment to Chapter 10: its table lists A1–A8, and A7 covers Chapters 7–8 only. This blueprint therefore ships none (decided 2026-10-29: Chapter 10 has no assignment for now). The content most suitable for a graded assignment is the 10.3 grid (machine-checkable: sandbox self-test passes, `repair30` reference patches pass, masks match per-turn gradients to 1e-5); revisit once the 10.3 sandbox is confirmed on the build host.

## Teaching sources

All of v5's chapter-level sources are kept; "take" says what to borrow.

- [Berkeley LLM Agents (Fall 2024)](https://llmagents-learning.org/f24) and [Advanced LLM Agents (Spring 2025)](https://rdi.berkeley.edu/adv-llm-agents/sp25) — course arc (tools, planning, code agents, agentic RL, safety); take the ordering of topics and the guest-lecture reading lists.
- [HF Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) and [MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction) — runnable smolagents and MCP examples; take the "thought, action, observation" cycle in 10.3 and the MCP walkthrough in 10.1.
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — workflows versus agents; take the vocabulary and the "simplest thing that works" principle for the 10.3 ablation design.
- Anthropic, [Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — context as a finite resource, compaction, notes, subagents; take the four context policies of 10.3.
- Lilian Weng, [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — planning, memory, tool use taxonomy; take the memory taxonomy for the 10.3 "external memory" paragraph.
- Chip Huyen, [Agents](https://huyenchip.com/2025/01/07/agents.html) — planning, tool selection and failure modes; take the failure taxonomy for 10.3.
- Added by this blueprint (verified 28 Oct 2026): Anthropic, [Writing effective tools for AI agents](https://www.anthropic.com/research/writing-tools-for-agents) (already a v5 source for 10.3; reused in 10.1 for tool descriptions); Qwen, [Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html) (model-specific template and parser conventions).

## Sections

### 10.1 Tool calling

- **Label:** `sec_lm_tool_calling` · **Tier:** T1 (cells for templates, schemas, parsing and MCP are T0: tokenizer only) · **Budget:** render/parse/MCP cells under 1 min on CPU; one greedy evaluation of one model on the ~450-item set ≤ ~5 min on T1 (batch 32, ≤ 256 new tokens; to be measured); LoRA fine-tune ≤ ~8 min (≤ 3k traces, ≤ 768 tokens, 1 epoch; trim the trace count to fit); peak memory below 12 GiB expected for Qwen3-1.7B (3.4 GB bf16 weights, LoRA with gradient checkpointing; to be measured) · **Codex units:** none (v5 marks the section `[new]`; the Codex plan excludes tool-use loops).
- **Builds on (book):**
  - `sec_decoding`: greedy and temperature decoding; not re-taught.
  - `sec_imitation`: SFT on traces is behavior cloning; the section states this in one paragraph and points to `subsec_il_error_bounds` for why multi-turn tool use compounds errors (the 10.3 hazard curve measures it).
  - `sec_rl_sequences`: the exercise "Tool outputs in the trajectory" is the seed of 10.4; a margin note points forward.
- **Builds on (this part):**
  - 1.1/1.2: `d2l.load_hf`, `d2l.generate` (`d2l.generate(model, tokenizer, prompts, max_new_tokens, **sampling) -> list[d2l.Generation]`, per 01_interfaces.md; `Generation` has `text, token_ids, logprobs, finish_reason`. It returns completions only and stops at EOS (`finish_reason='eos'`; `<|im_end|>` for Qwen3 instruct). Call sites use `[g.text for g in d2l.generate(...)]`).
  - 1.3: chat templates, role markers, special tokens, `apply_chat_template`. This section does not re-teach them. It only adds the `tools` argument.
  - 1.5: `d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None) -> d2l.BootstrapResult` (read `.mean` and `.ci`); per-item results are kept as a `d2l.EvalResult` (`.per_item`, `.scores()`), or a plain DataFrame for tool-call categories that are not `EvalSuite` tasks.
  - 2.4: `d2l.BookRetriever` (`.search(query, k, mode=...) -> list[d2l.Hit]`; `Hit` is a NamedTuple with `chunk_id, score, rank, text, heading_path, scores`, not a dict), wrapped as a tool.
  - 7.2 and 7.4: assistant-only loss and LoRA ideas (use `peft` here; do not depend on `d2l.LoRALinear`'s constructor).
  - Forward, same chapter: 10.3's `d2l.run_python(code, timeout=5.0)`, used as a black box for the `python` tool, with a margin note that all code execution in this chapter goes through the 10.3 sandbox.
- **Exports** (`#@save`; `d2l.` prefix omitted in signatures):
  - `function_to_schema(fn) -> dict`: HF/OpenAI-style `{"type":"function","function":{name, description, parameters}}` from type hints and a Google-style docstring.
  - `@dataclass ToolCall(name: str, arguments: dict, raw: str, error: str | None)`.
  - `parse_tool_calls(text, fmt='qwen3', schemas=None) -> tuple[str, list[ToolCall]]`: `fmt in {'qwen3','qwen35'}`. It returns the text outside the calls plus the calls. Malformed calls are returned with `error` set; the function never raises. `schemas` enables type coercion for the XML format.
  - `validate_args(schema: dict, args: dict) -> list[str]`: JSON-Schema subset (`type`, `required`, `enum`, `items`, `properties`, `minimum/maximum`, `additionalProperties`), returning error strings.
  - `ToolRegistry`: `.add(fn)`, `.schemas -> list[dict]`, `.execute(call) -> str` (validates, runs, truncates to 2,000 characters, converts exceptions to `"Error: ..."`; never raises), `.log` (name, arguments, status, seconds).
  - `calculator(expression: str) -> str` (AST whitelist), `search_book(query: str, k: int = 3) -> str` (formats `BookRetriever.search` hits as `[{h.chunk_id}] {' > '.join(h.heading_path)}: {h.text[:300]}`).
  - `tool_chat(model, tokenizer, messages, registry, fmt='qwen3', max_rounds=4, enable_thinking=False, **sampling) -> tuple[list[dict], dict]`: the round-trip loop; the dict holds per-round tokens, seconds and error codes.
  - `turn_samples(messages, tools, tokenizer, fmt) -> list[tuple[list[int], list[int]]]`: one `(prompt_ids, target_ids)` per assistant turn, with prompts rendered exactly as `tool_chat` renders them at inference (also the 10.4 oracle).
  - `load_bfcl(category, n=None, seed=0, revision=BFCL_REVISION) -> list[dict]` (items with `id`, `messages`, `tools` converted to JSON Schema, `gold` or `None` for irrelevance); `bfcl_match(calls, gold) -> bool`; `classify_call(calls, item) -> str` (code from `tab_lm_tool_errors`).
  - `make_tool_traces(n, seed, families=TRAIN_FAMILIES) -> list[dict]`: the synthetic generator with three verification stages.
  - Data: `DATA_HUB['lm-bfcl-subset']`, `['lm-tool-traces-synth']`, `['lm-lora-tools-qwen3-1p7b']`.
- **Models and data:**
  - `Qwen/Qwen3-1.7B` (revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0, vocab 151,936, 28 layers, hidden 2,048); development with `Qwen/Qwen3-0.6B` (`c1899de289a04d12100db370d81485cdf75e47ca`); comparison model `Qwen/Qwen3.5-2B` (`15852e8c16360a2fea060d615a32b45270f8a8fc`, Apache-2.0, vocab 248,320). Chat templates read from each repo's `tokenizer_config.json` on 28 Oct 2026.
  - `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` (Apache-2.0). Files `BFCL_v3_{simple,multiple,parallel,parallel_multiple,irrelevance}.json` are JSON-lines (about 400, 200, 200, 200 and 240 items by line count; the loader reports exact counts), gold in `possible_answer/BFCL_v3_*.json` (irrelevance has no gold: the correct behavior is to call nothing). Multi-turn files are left to 10.5.
  - Synthetic traces: generated in the notebook, no external data. The `search_book` family draws queries from R3 headings.
- **Packages:** `transformers>=4.56` (verify `get_json_schema` and chat-template `tools` support; `tokenizer.parse_response` exists in newer versions for models that define a response schema), `peft>=0.17` (0.21.2 current on PyPI), `jsonschema` (test oracle), `bfcl-eval` (PyPI 2026.3.23, Apache-2.0; oracle for the AST checker), `mcp` (official SDK; optional interop cell; PyPI major version 2 at the time of writing, so pin and check the API).

#### Narrative arc

1. **The question.** Opening cell: 200 seeded products of a 3-digit by a 2-digit integer, Qwen3-1.7B non-thinking, greedy, no tools. Print exact-match accuracy (to be measured; the expected direction is "well below 100%"; Qwen3-1.7B is a small model). Then add the `calculator` tool and rerun through `tool_chat`: print accuracy and the mean prompt-token overhead the schema adds (a number the cell measures with `len(tokenizer(...))` of the rendered system block with and without tools). Second symptom, printed in the same cell: one rendered prompt, so the reader sees that a "tool" is text in a system message.
2. **10.1.1 What the template does with `tools`.**
   - `tokenizer.apply_chat_template(messages, tools=[...], add_generation_prompt=True, enable_thinking=False, tokenize=False)`, printed for Qwen3-1.7B. Annotate in the text: the `# Tools` header, one `tool | tojson` line per tool inside `<tools></tools>`, the instruction to answer with `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`.
   - Special tokens: `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>` are single added tokens (cell prints their ids; they appear in `added_tokens_decoder`). Consequence used in 10.2: the call envelope costs a fixed number of tokens, and a mask can force it.
   - Results travel as role `tool` in the API but are rendered as a `user` turn holding one `<tool_response>` block per result; adjacent tool messages are merged into one user turn (this is how parallel results return). The template's `last_query_index` scan skips user turns that begin with `<tool_response>`, so they do not count as new queries.
   - `tab_lm_tool_formats` (Qwen3-1.7B versus Qwen3.5-2B, read from the two `chat_template` strings): tool header text; call syntax (JSON object inside `<tool_call>` versus `<function=name>` with `<parameter=key>` blocks); argument encoding (JSON values versus raw strings, with mappings and sequences JSON-encoded); thinking default (Qwen3 thinks unless `enable_thinking=False`; Qwen3.5's generation prompt appends an empty think block unless `enable_thinking=True`); vocab sizes 151,936 versus 248,320; where reasoning may appear relative to a call (Qwen3.5's instructions allow reasoning before, not after, the call).
   - **Prefix stability check** (cell `tool-calling-prefix-stability`). Render `m = [user, assistant(tool_call), tool]` and ask whether `render(m[:2])` is a string prefix of `render(m + [final assistant])`. Blueprint author's check with a jinja2 re-implementation of the HF environment on 28 Oct 2026: Qwen3 no (a last-position assistant turn is rendered with an empty `<think>` block that disappears once more messages follow), Qwen3.5 in non-thinking mode yes. The authoring agent must re-run it with the real tokenizers. Consequences stated in the text: cache reuse across turns in serving (13.3), the SFT sample construction below, and the 10.4 rollout design.
3. **10.1.2 From a function to a schema.**
   - `function_to_schema` on `calculator`, `search_book`, `unit_convert`; side by side with `transformers.utils.get_json_schema`.
   - Tool descriptions are prompt text. One cell measures the token cost per tool, and a short paragraph takes three rules from *Writing effective tools for AI agents*: name by action, describe when not to use it, return what the next step needs. A `measure` exercise tests a description rewrite.
4. **10.1.3 Parsing, and what goes wrong.**
   - `parse_tool_calls` for both formats. Qwen3: DOTALL match of `<tool_call>\s*(.*?)\s*</tool_call>`, `json.loads`, and decode `arguments` again if it is a string (double encoding is common in small models). Qwen3.5: read `<function=NAME>` and each `<parameter=KEY>` block, strip one leading and one trailing newline, coerce each string by the schema's `type` (`"5"` to 5), JSON-decode values for `array` and `object`.
   - An unterminated `<tool_call>` (generation hit `max_new_tokens`) returns `error='truncated'`.
   - `tab_lm_tool_errors`: E_NO_CALL, E_SPURIOUS (a call where none is expected: BFCL irrelevance), E_SYNTAX, E_TRUNC, E_UNKNOWN_TOOL, E_SCHEMA (missing required, wrong type, unknown argument, enum violation), E_VALUE (schema-valid, wrong value), E_SELECT (wrong tool), E_COUNT (wrong number of parallel calls), E_EXEC (the tool raised). Columns: code, definition, detector (`parse`, `validate_args`, `bfcl_match`, or the registry), one example, and what the harness returns to the model.
   - The harness returns E_SYNTAX, E_UNKNOWN_TOOL, E_SCHEMA and E_EXEC to the model as a `tool` message (`Error: ...`) and lets it retry once. The notebook reports the repair rate (fraction fixed after one retry; to be measured).
5. **10.1.4 The round trip.**
   - `tool_chat` appends `{"role":"assistant","content":..., "tool_calls":[{"type":"function","function":{"name","arguments":dict}}]}`, then one `{"role":"tool","content":...}` per call, re-renders, and generates again, until a turn has no call or `max_rounds` is reached.
   - `tab_lm_bfcl_cats`: simple (one function, one call), multiple (choose among functions), parallel (one function, several independent calls), parallel_multiple (both), irrelevance (no function applies), multi-turn (10.5). Columns: what it tests, scoring, item count.
   - Parallel calls are several `<tool_call>` blocks in one assistant turn. Dependent ("nested") calls cannot be issued together because the second needs the first's result, so they take two rounds. A cell shows both for `calculator` and `search_book`.
   - Tools: `calculator` (AST whitelist; exponent and result-size caps so that `9**9**9` returns an error in under 10 ms), `python` (through `d2l.run_python`), `search_book` (the 2.4 retriever).
6. **10.1.5 MCP: tools as a protocol.**
   - Message shapes (JSON-RPC 2.0): `initialize`, `tools/list` returning `{name, description, inputSchema}`, `tools/call` with `{name, arguments}` returning `{content:[{type:"text", text}], isError}`. Two transports: stdio and streamable HTTP. Pin the spec revision (the docs landing page resolved to the 2026-07-28 revision on 28 Oct 2026; the three methods used here are the long-stable core, but state the pinned revision in the notebook).
   - From scratch: a ~30-line stdio JSON-RPC server exposing the registry, and a ~30-line client. The client turns `inputSchema` into the `parameters` field of an OpenAI-style tool and forwards each parsed call to `tools/call`. An optional cell connects the official `mcp` client SDK to the same server (wrap in `try/except ImportError`).
   - What a protocol does not solve: every connected server adds descriptions to every prompt (the notebook prints the token cost of 20 tools), and descriptions are untrusted text (pointer to 10.5, tool poisoning).
7. **10.1.6 Where the ability comes from: training data.**
   - Toolformer's filter (D1) with a small demonstration on the calculator.
   - Synthetic API traces: APIGen's three verification stages and ToolACE's complexity grading are the reference designs. `make_tool_traces` produces: single-call 50%, parallel 15%, dependent two-round 15%, tool-available-but-unneeded 10% (teaches abstention), error-recovery 10% (first call invalid, error returned, corrected call). Counts per stage are printed.
   - Families. Train: `calculator`, `unit_convert`, `date_diff`, `search_book`, `python`, `inventory_lookup`. Held out: `convert_currency`, `send_note` and every BFCL function, so that the evaluation separates in-distribution gain from transfer. A cell asserts that no training function name occurs in the BFCL subset.
   - Queries come from at least 25 templates per family with lexical variation; a `measure` exercise adds LLM paraphrase.
   - SFT is behavior cloning on turn samples (D2). LoRA r=16, alpha=32, dropout 0.05, all linear layers, learning rate 2e-4 with cosine decay and 3% warmup, 1 epoch over ≤ 3,000 traces, batch 16 sequences, max length 768, loss on target tokens only, seeds 0 and 1.
8. **10.1.7 Measuring call accuracy** (the headline experiment, below). Close with a paragraph on what the benchmark does not measure: argument values under ambiguity, multi-turn state, and cost.

#### Derivations

**D1. The Toolformer filter is a log-likelihood gain.**

- *Statement.* For a text `x_1..x_n`, a candidate call `c` inserted before position `i` with executed result `r`, let `w_j >= 0` weight the next tokens (decaying with distance from `i`). Define `L(z) = -sum_j w_j log p(x_j | z, x_<j)` for a prefix `z`. Toolformer (threshold written `t_keep` here so that `τ` stays the trajectory of the notation contract) keeps the call iff `min(L(none), L(c)) - L(c, r) >= t_keep`, where `none` is the empty insertion and `L(c)` is the call without its result. Claim: the criterion is a thresholded, weighted pointwise log-likelihood ratio of the continuation, so it keeps only calls whose result makes the following text more predictable than both baselines.
- *Assumptions.* Fixed model `p`; the same continuation tokens in all three terms; `t_keep >= 0`.
- *Steps.*
  1. Write `L(none) - L(c, r) = sum_j w_j [log p(x_j | c, r, x_<j) - log p(x_j | x_<j)]`.
  2. Each bracket is the log-likelihood ratio of token `x_j` with and without the executed call in context; the weighted sum is a weighted sum of pointwise log-likelihood ratios.
  3. The extra baseline `L(c)` rules out gains that come from the call text alone (for example, a quoted question), so the gain must come from `r`.
  4. Hence the filter keeps `(c, r)` iff `r` carries information about the continuation beyond what `c` and the unaided model supply.
- *Final formula.* `keep(c, r) = [ min(L(none), L(c)) - L(c, r) >= t_keep ]`.
- *Numerical check* (cell `tool-calling-toolformer`): 30 seeded arithmetic sentences ("17 items at 23 dollars each cost 391 dollars"); insertion point before the answer; compute `L` with Qwen3-0.6B base-style scoring (`d2l.score_choices` conventions are not needed; use summed token log-probs). Expected direction: a correct result gives a positive gain, a corrupted result gives a negative one. Report the fraction of each sign with a bootstrap interval; the sign split is to be measured.

**D2. Per-turn samples equal the masked full-sequence loss only for prefix-stable templates.**

- *Statement.* Let the conversation be `h_1, y_1, o_1, h_2, y_2, ...` with agent turns `y_m` and environment turns `o_m`. Let `R(m)` be the template's rendering of the history before turn `m` plus the generation prompt. The per-turn objective is `sum_m -log p(y_m | R(m))`. The one-sequence objective renders the whole conversation once and masks non-agent tokens: `sum_m -log p(y_m | S_<m)` where `S_<m` is the prefix of that single rendering. They are equal iff `R(m)` equals `S_<m` token for token.
- *Assumptions.* Greedy tokenization of the same strings; the model is deterministic for a fixed input.
- *Steps.*
  1. For a prefix-stable template, `render(m[:k])` is a prefix of `render(m)` and the generation prompt is the only suffix, so `R(m) = S_<m` for all `m`.
  2. Then every agent token is conditioned on identical contexts in both objectives, and the sums coincide term by term.
  3. For Qwen3 non-thinking, the last assistant turn is rendered with an empty think block, whereas the same turn in the middle of a longer conversation is not, so `R(m) != S_<m`: contexts differ by those tokens.
  4. The two objectives therefore train on different conditionals. Only the per-turn one matches the prompts the model sees at inference.
- *Final formula.* Equality of the two losses holds iff `R(m) == S_<m` for all `m`; otherwise they differ by the effect of the dropped tokens.
- *Numerical check* (cell `tool-calling-turn-samples`): fp32, one trace, `Qwen3.5-0.8B` and `Qwen3-0.6B` (CPU is fine). Compute both objectives. Assert agreement to 1e-4 relative for Qwen3.5 (the check also asserts token equality of `R(m)` and `S_<m`), and assert that Qwen3's token sequences differ and print the loss gap. Fallback if Qwen3.5-0.8B is awkward on the build host: use a hand-built prefix-stable template for the equality case.

#### From-scratch implementation

Budget: about 150 lines for everything in **Exports** except `make_tool_traces`, `load_bfcl` and the SFT loop (these three are counted separately, about 80 lines).

| Component | Lines | Contract |
|---|---|---|
| `function_to_schema` | 20 | Python types to JSON Schema, Google-style docstring parsing |
| `parse_tool_calls` | 35 | both formats; errors as values |
| `validate_args` | 20 | JSON-Schema subset |
| `ToolRegistry`, `calculator` | 25 | never raises; truncation; log |
| `tool_chat` | 20 | rounds, parallel results merged |
| `bfcl_match` | 30 | AST matching |

Tests (all in the notebook, seeded):

1. `function_to_schema` equals `get_json_schema` on 5 functions after sorting keys. Tolerance: exact equality.
2. Parser round trip: 200 seeded `(name, args)` pairs with nested dicts, lists, unicode and numeric strings: `parse(render(call)) == call` for both formats, with the XML format given the schemas. Tolerance: 100%. A documented counterexample: an argument string that contains the text `</tool_call>` breaks the non-greedy match (a pitfall, not a bug to fix).
3. `validate_args` agrees with `jsonschema.validate` on 200 random schemas in the supported subset. Tolerance: 100% agreement of valid/invalid; message text is not compared.
4. `calculator` agrees with `eval` on 1,000 seeded expressions of `+ - * / // % **` and parentheses; rejects `__import__('os')` and `9**9**9` within 10 ms.
5. `bfcl_match` against the `bfcl_eval` checker on 200 outputs (100 correct, 100 perturbed): report agreement and itemize disagreements (expected ≥ 98%; to be measured).
6. `turn_samples` prompts equal the prompts `tool_chat` generated at inference, token for token, for both families on 20 traces.

#### In production

- Templates and schemas: `apply_chat_template(..., tools=...)` and `transformers.utils.get_json_schema` are the oracles for the from-scratch schema code. Where the tokenizer supports `parse_response`, compare it with `parse_tool_calls` on 200 generations (agreement to be measured).
- Serving: vLLM's OpenAI-compatible server with `--enable-auto-tool-choice --tool-call-parser hermes` for Qwen3 (the Qwen docs give this command). Qwen3.5 uses the XML-style format, so the matching parser must be looked up in the vLLM and Qwen documentation at authoring time. The from-scratch parser is the oracle: on the 450 generations, count calls that the server parser and ours read differently.
- MCP: the official SDK is the production client. The from-scratch server is checked by listing its tools through the SDK client.
- Guaranteed parsing is delegated to 10.2 (`TokenFSM` and XGrammar). Say so in a margin note at the end of 10.1.3.

#### Headline experiment

- **Hypothesis.** H1: LoRA fine-tuning on 3k verified synthetic traces raises call accuracy on held-in tool families and reduces E_SYNTAX and E_SCHEMA. H2 (open): effects on BFCL categories and on held-out families are small and may be negative; no direction is asserted.
- **Setup.**
  - Models: Qwen3-1.7B base-instruct (non-thinking, greedy, `max_new_tokens=256`, batch 32) with and without the adapter (seeds 0 and 1); Qwen3.5-2B as a native-XML comparison without fine-tuning.
  - Items: 60 per BFCL category (simple, multiple, parallel, parallel_multiple, irrelevance; seeded sample, ids committed in `lm-bfcl-subset`) = 300; 100 held-in-family items and 50 held-out-family items from fresh seeds. About 450 total.
  - Tier T1.
- **Procedure.** Evaluate base; fine-tune seeds 0, 1; evaluate; evaluate Qwen3.5-2B; compute the taxonomy.
- **Metrics.** Call accuracy (binary per item), format-valid rate, E_* counts, rounds and tokens per item, repair rate after one retry.
- **Plot.** `fig_lm_tc_accuracy`: grouped bars; x = category (simple, multiple, parallel, parallel_multiple, irrelevance, held-in, held-out); series = base, +LoRA seed 0, +LoRA seed 1, Qwen3.5-2B; y = accuracy with 95% bootstrap intervals (items resampled within category). `fig_lm_tc_errors`: stacked bars of E_* counts for base versus +LoRA (seed 0).
- **Expected outcome.** Direction only (H1). The BFCL leaderboard publishes scores for released Qwen3 models; do not copy them into the text. State that the subset's scores are not comparable with the leaderboard because of sampling, prompt format and the AST-only subset.
- **Noise check.** Bootstrap over items with `cluster=<function name>`; two fine-tune seeds; a claim "better" requires a paired-bootstrap interval excluding 0 and the same sign for both seeds. Re-run the base evaluation once to expose batch-size nondeterminism in bf16 (a few item flips are expected; to be measured).
- **T1 reduction.** None needed. If over budget, cut traces to 2k and BFCL items to 40 per category.
- **Artifacts.** `lm-bfcl-subset`, `lm-tool-traces-synth`, `lm-lora-tools-qwen3-1p7b`.

#### Figures and tables

- `fig_lm_tc_template`: annotated rendering of one tool turn for Qwen3 (blocks colored: system tools, call, response, generation prompt). Made by a matplotlib text layout from the real rendering.
- `fig_lm_tc_accuracy`, `fig_lm_tc_errors`: above.
- `fig_lm_tc_mcp`: sequence diagram of `initialize`, `tools/list`, `tools/call` between host, client, server (hand-drawn SVG).
- `tab_lm_tool_formats`, `tab_lm_tool_errors`, `tab_lm_bfcl_cats`: above.
- `tab_lm_tc_trace_stats`: counts per family and per stage (format, execution, semantic) of the synthetic traces, printed by `make_tool_traces`.

#### Pitfalls and misconceptions

- "The model has a tool API." It has a text convention; tools are prompt tokens. Schema size is paid on every turn.
- Qwen3 and Qwen3.5 are not interchangeable: wrong parser, wrong format, silent failure (parser returns no calls).
- History re-rendering changes earlier tokens for Qwen3 (prefix instability), which hurts prefix caches and RL.
- Tool results come back as user turns; a model can mistake result text for instructions (10.5).
- Greedy decoding hides sampling variance but not batch nondeterminism.
- Narrow tool fine-tuning can lower irrelevance accuracy (the model over-calls); check E_SPURIOUS.
- BFCL names contain dots (`math.factorial`) and use `"type": "dict"`, `"float"`, `"tuple"`; convert to JSON Schema (`object`, `number`, `array`) before rendering or the template shows non-standard types.
- A tool that returns 50 KB of text is a context bug, not a tool.

#### Summary (draft bullets)

- Tool calling is a text protocol defined by the chat template: schemas go in the system turn, calls are delimited tokens, results come back as a user turn.
- A parser must return errors as data; the common failures are syntax, schema, wrong tool and spurious calls.
- Templates differ between families and in prefix stability; the training sample construction must reproduce inference prompts.
- Synthetic traces verified by execution teach format and abstention; transfer to unseen functions must be measured, not assumed.
- MCP standardizes how tools are listed and called, not whether they are trustworthy or cheap in tokens.
- Constrained decoding (10.2) turns "usually parses" into "always parses".

#### Exercises

1. **Check.** In `tab_lm_tool_formats`, why does Qwen3.5's parser need the tool schema, while Qwen3's does not? *Sketch:* XML parameter values are raw strings, so `5` and `"5"` are indistinguishable without the declared type; JSON carries types.
2. **Check.** Two parallel calls and a dependent pair of calls look different in the transcript. How many assistant turns and how many `<tool_response>` blocks does each produce, and why can the dependent pair not be merged? *Sketch:* parallel: one assistant turn, one user turn with two response blocks; dependent: two assistant turns, two user turns; the second call's argument is the first call's result.
3. **Derive.** Prove that `min(L(none), L(c)) - L(c, r) >= t_keep` implies `L(none) - L(c, r) >= t_keep`, and say what is lost by dropping the `L(c)` term. *Sketch:* `min(a,b) <= a`, so the left side is at most `L(none) - L(c, r)`; dropping the baseline keeps calls whose benefit comes from the call text, not the result.
4. **Derive.** For a template that inserts an empty think block only after the last user query, give a rule that makes per-turn and one-sequence losses agree without changing the template. *Sketch:* render every assistant turn as if it were last (supply an explicit empty `reasoning_content` and force the block), or build the sequence by appending the sampled token ids rather than re-rendering.
5. **Measure.** Rewrite each tool description of the notebook in the style of the Anthropic guide (when to use, when not to use) and predict the sign of the change in E_SELECT and E_SPURIOUS before measuring. Report with paired bootstrap.
6. **Measure.** Repeat the headline evaluation with `enable_thinking=True`. Predict whether call accuracy rises, and how tokens per item change; measure both.
7. **Open.** Add paraphrases generated by Qwen3-8B to `make_tool_traces`, keep only those that pass all three verification stages, and report whether transfer to held-out families changes.

#### Further reading

- Papers: [Toolformer](https://arxiv.org/abs/2302.04761) (self-supervised insertion and the filter of D1); [Gorilla](https://arxiv.org/abs/2305.15334) (retrieval-aware API calling; origin of BFCL); [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) and its [v3 multi-turn blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) (categories and AST scoring); new: [ToolACE](https://arxiv.org/abs/2409.00920) (verified synthetic function-calling data, complexity grading); new: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (how Qwen3 trains tool use and the thinking switch).
- Tutorials and docs: [HF tool use in chat templates](https://huggingface.co/docs/transformers/chat_extras); [Model Context Protocol](https://modelcontextprotocol.io/); new: [Qwen function calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html); HF [MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction) (from the chapter list).
- Code bases: [Gorilla repository](https://github.com/ShishirPatil/gorilla) (`bfcl-eval`); [peft](https://github.com/huggingface/peft); new: the official MCP Python SDK (install name `mcp`).

#### Risks and open decisions

- **Template drift.** Qwen3 or Qwen3.5 templates may change in later revisions: pin revisions and re-run the prefix-stability cell. Fallback: embed the template string in the notebook.
- **`parse_response`** may not exist for these models in the installed `transformers`: the oracle comparison is then skipped with a printed note.
- **Fine-tune time.** If 3k traces exceed the budget, cut to 2k; if memory exceeds 12 GiB, enable gradient checkpointing and reduce batch to 8 with 2 accumulation steps.
- **Retriever interface** is `d2l.Hit` from 2.4 (01_interfaces.md). If a later revision changes the shape, adapt `search_book` only.
- **MCP SDK major version** may differ from the plan. The from-scratch JSON-RPC server and client are the fallback and do not need the SDK.
- **BFCL revision.** The pinned revision may be superseded (BFCL v4 added agentic categories); the notebook states that v3 single-turn files are used and why.

### 10.2 Structured outputs and constrained decoding

- **Label:** `sec_lm_structured_outputs` · **Tier:** T0 (CPU, Qwen3-0.6B; `--gpu` flag runs Qwen3-1.7B on T1 with larger n) · **Budget:** automaton, schema and mask cells take seconds; building the byte trie of the Qwen3 vocabulary takes a few seconds in pure Python (to be measured; cache it); the exact-distribution experiment needs one forward pass per tree node (tens to hundreds of sequences; minutes on CPU); the headline experiment ≤ ~10 min on CPU for ~120 items, ≤ ~3 min on T1 (to be measured). Peak memory under 4 GB on CPU · **Codex units:** 13.9 (constrained and structured generation). Carried over from Codex 13.9: finite-state constraints before grammars; multi-transition tokens; distribution change; "valid syntax is not a correct answer"; state management and speculative verification; XGrammar, Outlines and SGLang as sources.
- **Builds on (book):** `sec_decoding` (how logits become a token; temperature, top-p; the section changes the logits step only and does not re-teach samplers). Recap of the byte-level BPE vocabulary is one paragraph pointing to `sec_text-sequence`.
- **Builds on (this part):**
  - 3.6: tokenizer anatomy (byte-to-unicode map, merges, special tokens), used to turn tokens into byte strings.
  - 1.1/1.2: `d2l.load_hf`, `d2l.generate` (the unconstrained baseline).
  - 1.5: `d2l.paired_bootstrap`.
  - 10.1 (soft): `d2l.load_bfcl`, `d2l.bfcl_match`, `d2l.parse_tool_calls`, `d2l.validate_args`, `d2l.function_to_schema`, `d2l.classify_call`. The section runs without 10.1's adapter.
- **Exports:**
  - `parse_regex(pattern: str) -> Node`: byte-level regex subset (literals, escapes, classes with ranges and negation, groups, `|`, `* + ?`, `{m,n}`); the token `@{<tool_call>}` denotes one special-token symbol.
  - `schema_to_regex(schema: dict, ws: str = 'single') -> str`: JSON Schema subset to regex. `ws in {'none','single','any'}`.
  - `tool_call_regex(tools: list[dict], fmt: str = 'qwen3', max_calls: int = 3) -> str`: the call envelope with the name restricted to the tool list and arguments restricted to each tool's schema, ending with the end-of-turn special symbol.
  - `TokenFSM(pattern, tokenizer, vocab_size, trigger=None)`: `.start`, `.mask(state) -> BoolTensor[vocab_size]`, `.step(state, token_id) -> int` (−1 if disallowed), `.is_final(state) -> bool`, `.n_states`, `.stats` (states built, mask seconds).
  - `constrained_generate(model, tokenizer, prompt_ids, fsm, max_new_tokens, temperature=0.0, top_p=1.0, seed=0) -> tuple[list[int], dict]`: KV-cached loop; the dict holds per-step `log_Z` (log of the allowed mass), mask seconds and forward seconds.
  - `enumerate_paths(fsm, max_len) -> list[tuple[int, ...]]` for finite languages; `exact_masked_distribution(model, tokenizer, prompt_ids, fsm, max_len) -> dict` with arrays `paths`, `p` (unconstrained path probabilities), `q` (masked-sampling probabilities), scalar `pL`, and `cond = p / pL`.
  - No data artifacts.
- **Models and data:** `Qwen/Qwen3-0.6B` (revision `c1899de289a04d12100db370d81485cdf75e47ca`; logits dimension 151,936, tokenizer 151,669 ids = 151,643 ordinary plus 26 added). The Qwen3 vocabulary contains all 256 single-byte tokens (checked on 28 Oct 2026 from `vocab.json`); this is what makes every byte string tokenizable. BFCL v3 subset ids from `DATA_HUB['lm-bfcl-subset']` (10.1). 40 seeded arithmetic word problems for the reasoning-order probe, generated in the notebook.
- **Packages:** `xgrammar` (PyPI 0.2.8, Apache-2.0; verify the API names below against the installed version), `outlines>=1.0` (PyPI 1.3.3, Apache-2.0), `llguidance` (PyPI 1.9.1, MIT; optional), `jsonschema`; optional `vllm` for the `structured_outputs` request field (older versions call it `guided_decoding`).

#### Narrative arc

1. **The question.** Opening cell: take the 10.1 `calculator` prompt and sample 100 completions from Qwen3-0.6B at temperature 0.7 without constraints. Parse with `parse_tool_calls` and validate with `validate_args`; print the rate of parse failures and of schema violations (to be measured; the expectation is a minority but not zero for a 0.6B model). One failing completion is printed with the offending character highlighted. The learning objective "guaranteed to parse" is stated here.
2. **10.2.1 A constraint is an automaton.**
   - Regular expressions to deterministic automata through derivatives. For a regex `r` and a byte `b`, the derivative `d_b(r)` is a regex for the suffixes that follow `b`. A regex is nullable if it accepts the empty string. A state is a regex; it is dead if it is `∅`.
   - Smart constructors (identities for the empty set and the empty-string regex, associativity, sorted and deduplicated alternation) keep the number of distinct derivatives finite; this is the classical result (Brzozowski 1964; Owens, Reppy and Turon 2009), cited by title.
   - `fig_lm_so_automaton`: the automaton of `-?(0|[1-9][0-9]*)` with each state labelled by its derivative.
   - A cell builds the automaton lazily and prints the number of states for four patterns (integer, quoted enum of 5 words, ISO date, and the `calculator` call).
3. **10.2.2 JSON Schema to regex.**
   - `schema_to_regex` rules: object with fixed property order, required properties first and optional ones as `(, "k": V)?`; string (JSON escapes, control characters excluded, `minLength/maxLength` as counted repetition); integer, number (no leading zeros, optional fraction and exponent, bounded digit count); boolean; null; array with `minItems/maxItems`; `enum` and `const` as alternations of `json.dumps` literals; `anyOf` as alternation. `$ref`, `pattern` keywords and unbounded recursion are out of scope (exercise).
   - Whitespace is a policy: `'none'` (compact), `'single'` (exactly the `, ` and `: ` that Qwen's template prints), `'any'` (bounded optional spaces and newlines). Why it matters is deferred to D2 and measured in the exercises.
   - Unicode: strings are byte-level; bytes ≥ 0x80 are admitted freely, so an output can contain invalid UTF-8. State this limit and point to an exercise.
4. **10.2.3 From bytes to tokens.**
   - Token strings to bytes through the inverse of the byte-to-unicode map (`sec_text-sequence`). Special tokens are single alphabet symbols outside the byte range.
   - A token may span several automaton transitions (for example `"}` or `":` or ` {"`), and one string has many tokenizations. A cell lists, for the state after `{"name": "calc`, which tokens are allowed (`u`, `ul`, `ulator`, ...) and which are not.
   - The index: a byte trie over the whole vocabulary; `mask(state)` walks the trie and the automaton together and prunes dead branches, so the cost per state is the number of live (state, trie-node) pairs, not `|V|`. Masks are computed lazily and cached per state. `fig_lm_so_mask_time` compares first-visit and cached times.
   - The end-of-turn symbol is allowed exactly when the state is nullable. Ids in `[151669, 151936)` (embedding padding) are always masked.
5. **10.2.4 Decoding under a mask.**
   - `constrained_generate`: forward, set disallowed logits to `-inf`, then temperature and top-p, sample, advance. Order matters (mask before top-p). `log_Z` records `log sum_{t allowed} p(t | prefix)`.
   - Triggered grammars: free text until `<tool_call>` is emitted, then the call regex is enforced. This keeps the decision whether to call in the model's hands (see 10.1's irrelevance category) while the arguments are guaranteed.
   - Statefulness: batching needs one state per sequence; speculative decoding needs one state per draft position and rollback on rejection; beam search needs forkable states. A paragraph from Codex 13.9; no code.
6. **10.2.5 What masking does to the distribution.** Derivation D2 in the narrative: the toy example; the exact computation on the real model; non-canonical tokenizations.
7. **10.2.6 In production.** XGrammar, Outlines, llguidance, and a serving-engine request; `tab_lm_so_engines`; mask-for-mask oracle comparison; mask-time comparison; jump-forward (compressed FSM, as in SGLang) in one paragraph: when the automaton has a single path, tokens can be appended without a model call, subject to canonical-tokenization risk.
8. **10.2.7 Valid is not correct.** The headline experiment, the reasoning-order probe, and a closing paragraph: syntax constraints give structure, not truth; semantic checks (10.1's `bfcl_match`, schema `enum`s from retrieved ids) remain necessary.

#### Derivations

**D1. A token-level automaton from a byte-level one.**

- *Statement.* Let `M = (S, Σ, next, s0, F)` be a deterministic byte automaton with the dead state removed (`next` partial). Let the vocabulary be `V`, each token `t ∈ V` with a byte string `s(t)` (special tokens: a single symbol outside the byte range). Define the extended transition `next*(q, w)` on strings. The token automaton has transitions `next_V(q, t) = next*(q, s(t))` when defined. Claim: (i) a token sequence `t_1..t_n` is accepted by `next_V` from `s0` iff the concatenation `s(t_1)...s(t_n)` is accepted by `M`; (ii) if `V` contains every single byte, then every live state of `M` is live in `next_V`, so masking never produces a dead end; (iii) the allowed set at `q` is `Allow(q) = {t : next*(q, s(t)) is defined}` (plus the end symbol if `q ∈ F`).
- *Assumptions.* Token strings are fixed byte strings; the end symbol has no bytes; all 256 bytes are tokens (true for Qwen3; stated as a check).
- *Steps.*
  1. By induction on `n`, `next_V*(s0, t_1..t_n) = next*(s0, s(t_1)...s(t_n))` because `next*` is a monoid action: `next*(q, uv) = next*(next*(q, u), v)`.
  2. Acceptance of the concatenation is membership of that state in `F`, which proves (i).
  3. A state `q` is live iff some string `w` leads from `q` into `F`. Write `w` as bytes; each byte is a token by assumption, so a token path of single-byte tokens reaches `F`, which proves (ii).
  4. (iii) is the definition of `next_V`.
  5. Cost: computing `Allow(q)` naively takes `sum_t |s(t)|` steps. Walking a trie of the vocabulary alongside the automaton takes the number of live (state, trie-node) pairs, because a dead branch prunes every token below it.
- *Final formulas.* `next_V(q, t) = next*(q, s(t))`; `Allow(q) = {t : next*(q, s(t)) ≠ ⊥}`.
- *Caveat.* (i) concerns strings, not tokenizations: several token sequences give the same string, and the model's training data used one canonical tokenization. Masking allows non-canonical ones (for example `t`, `r`, `u`, `e` for `true`).
- *Numerical check* (cell `so-token-automaton-check`): for 10 patterns, run 200 random token paths through `TokenFSM` and assert that decoding the accepted token ids and matching with Python's `re.fullmatch` agree on the resulting string; run 200 random byte-strings both ways to assert the byte automaton and `re` agree. Tolerance: exact. Also assert no reachable state has an empty mask.

**D2. The distribution that masked sampling produces.**

- *Statement.* Let `p(t | h)` be the model's next-token distribution at prefix `h`. Masked sampling draws `t` with probability `p_mask(t | h) = p(t | h) 1[t ∈ Allow(h)] / Z(h)`, where `Z(h) = sum_{t ∈ Allow(h)} p(t | h)`. For a complete token sequence `y` in the language `L` of accepted token sequences, the masked-sampling probability is `p_mask(y) = p(y) / prod_{k} Z(y_<k)`. The true conditional is `p(y | L) = p(y) / p(L)`, with `p(L) = sum_{y' ∈ L} p(y')`. Then `p_mask(y) / p(y | L) = p(L) / prod_k Z(y_<k)`, so `p_mask = p(· | L)` iff `prod_k Z(y_<k)` is constant over `y ∈ L`.
- *Assumptions.* `L` is prefix-free once the end symbol is included; every state has a nonempty allowed set; `p` is fixed.
- *Steps.*
  1. By the chain rule `p(y) = prod_k p(y_k | y_<k)`.
  2. Masked sampling multiplies step `k` by `1/Z(y_<k)` for allowed tokens, so `p_mask(y) = p(y) / prod_k Z(y_<k)`.
  3. The conditional is `p(y)/p(L)` by definition.
  4. Divide: `p_mask(y)/p(y | L) = p(L) / prod_k Z(y_<k)`.
  5. Since `sum_y p_mask(y) = 1`, `p(L) = E_mask[prod_k Z(y_<k)]`. Hence the ratio is `E_mask[W] / W(y)` with `W(y) = prod_k Z(y_<k)`. It equals 1 for all `y` iff `W` is p_mask-almost-surely constant.
  6. KL: `D_KL(p_mask || p(· | L)) = log p(L) − E_mask[sum_k log Z(y_<k)]`. It is nonnegative and zero iff `W` is constant.
  7. Bound. If every constrained step has `Z ≥ 1 − λ` and a sequence has at most `T` constrained steps, then `W ∈ [(1−λ)^T, 1]`, so the ratio lies in `[(1−λ)^T, (1−λ)^{−T}]` and `|log ratio| ≤ T log(1/(1−λ))`. Masking is nearly exact when `λT ≪ 1`, that is, when the model already puts almost all its mass on grammatical tokens (a tuned tool-calling model).
  8. Repair. Let `w(h) = P(complete in L | h)` under `p`. The exact conditional has `p(t | h, L) ∝ p(t | h) w(h, t)`, so masked sampling is exact iff `w(h, t)` is the same for all allowed `t` (all allowed continuations are equally "completable"). Computing `w` is a sum over all completions, which is intractable in general (grammar-aligned decoding approximates it).
- *Worked example (toy, hand-checked).* `L = {ax, by}`; `p(a)=0.5`, `p(b)=0.4`, `p(other)=0.1`; `p(x | a)=0.01`, `p(y | b)=0.9`. Then `p(ax)=0.005`, `p(by)=0.36`, `p(L)=0.365`, true conditionals `0.0137` and `0.9863`. Masked sampling: `Z` is 0.9 at step 1 and 0.01 or 0.9 at step 2, so `p_mask(ax)=0.5556`, `p_mask(by)=0.4444`. The ratio `p_mask/cond` is 40.56 for `ax` and 0.45 for `by`. `D_KL(p_mask || cond) = 1.7028` nats, matching `log p(L) − E_mask[sum log Z]` (checked by the blueprint author on 28 Oct 2026).
- *Numerical check* (cell `so-distribution-toy`): build the toy as a lookup-table language model; compute `p_mask`, `cond`, `W` and the KL by enumeration; sample 200,000 sequences from the masked sampler and assert empirical frequencies within 3 standard errors of `p_mask`. Second cell (`so-distribution-real`): on Qwen3-0.6B take a prompt with a finite answer set of 12–20 multi-token strings (cities, mixed lengths), `enumerate_paths` over the FSM, compute `p(path)` exactly by teacher forcing (tree-structured batching), and compare `p_mask` with `cond`. Report `D_KL(p_mask || cond)`, the largest ratio, and the fraction of `p_mask` mass on non-canonical tokenizations (compare each path with `tokenizer.encode(string)`); all numbers to be measured.
- *When it matters (stated in the text).* (a) The model is not tuned for the format (base models, tiny models). (b) The grammar forces tokens the model finds unlikely, for example a whitespace policy different from its habit. (c) Branches have very different `W`, as in the toy. (d) Long constrained spans. It rarely matters when `Z ≈ 1` throughout.

#### From-scratch implementation

Budget: about 150 lines.

| Component | Lines | Contract |
|---|---|---|
| derivative engine: node types, smart constructors, `nullable`, `deriv`, regex parser | 55 | states are hash-consed regexes; `deriv(r, sym)` over alphabet `range(256 + n_special)` |
| `schema_to_regex`, `tool_call_regex` | 40 | rules listed above |
| `TokenFSM`: byte trie, lazy `mask`, `step`, `is_final`, trigger | 35 | masks cached per state |
| `constrained_generate`, `exact_masked_distribution` | 25 | KV cache from `model(..., use_cache=True)` |

Tests:

1. Regex engine against `re.fullmatch` on 10 patterns × 200 random strings (half sampled from the language, half perturbed). Tolerance: exact.
2. `schema_to_regex`: for 8 schemas, 100 instances generated from `jsonschema`-valid samples must match the regex; 100 mutated instances (wrong type, missing required, extra key) must fail the regex and fail `validate_args`. Tolerance: exact agreement.
3. `TokenFSM` D1 checks (above).
4. Masked greedy decoding of the `calculator` call with a stub model that returns fixed logits gives the expected string.
5. Mask-for-mask against XGrammar (see below).

#### In production

- **XGrammar** (API names as of 0.1–0.2; verify): `tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.vocab_size)`; `compiler = xgr.GrammarCompiler(tokenizer_info)`; `compiled = compiler.compile_regex(pattern)` or `compile_json_schema(schema_json, any_whitespace=False)`; `matcher = xgr.GrammarMatcher(compiled)`; `bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)`; `matcher.fill_next_token_bitmask(bitmask)`; `xgr.apply_token_bitmask_inplace(logits, bitmask)`; `matcher.accept_token(tid)`. A Transformers adapter (`xgr.contrib.hf.LogitsProcessor`) plugs into `model.generate`.
- **Oracle test.** Compile the same regex in both implementations. Along 50 valid completions per pattern (10 patterns), compare the allowed-token sets at every step. Report the fraction of identical masks; itemize differences by cause (end/stop tokens, special tokens, partial UTF-8 bytes, whitespace). Tolerance: identical on all ASCII patterns, differences explained for the rest.
- **Outlines ≥ 1.0:** `model = outlines.from_transformers(hf_model, hf_tokenizer)` and a call with a regex or JSON-schema output type (verify the type names in the installed version). Its index is the whole-vocabulary-per-state design of Willard and Louf, the direct ancestor of `TokenFSM`; XGrammar splits tokens into context-independent ones (cacheable per grammar position) and context-dependent ones (checked at runtime with a pushdown automaton), which is what makes context-free grammars and fast compilation possible. `llguidance` (Rust, lexer plus Earley parser) is the third engine; run it only if installed.
- **Serving engines:** a vLLM request with `structured_outputs` (`json`, `regex`, `choice`, `grammar`) and an OpenAI-style `tools` request with `--tool-call-parser` (10.1). State which backend the engine reports using.
- `tab_lm_so_engines` compares the three engines (automaton type, what is precomputed, JSON Schema coverage, startup cost, where it runs) from their documentation and papers; only qualitative cells. A measured column (mask seconds per step on the same pattern, from-scratch versus XGrammar, first visit and steady state) is added by the notebook.
- Interaction with speculative decoding and batching: one matcher per sequence; XGrammar's matcher supports rollback of accepted tokens, which speculative verification needs (check the installed API).

#### Headline experiment

- **Hypothesis.** Constraining the 10.1 tool call to its schema makes every output parse and validate (by construction); it changes accuracy and spurious-call behavior in ways that depend on the grammar mode. No direction asserted for accuracy.
- **Setup.**
  - Model: Qwen3-0.6B (T0) or Qwen3-1.7B (`--gpu`, T1), non-thinking, greedy.
  - Items: from `lm-bfcl-subset`, 30 per category for simple, multiple, parallel and irrelevance (120 items; 60 per category on the `--gpu` path).
  - Arms: U (unconstrained); C-force (always call; name in the tool list; arguments per schema; `ws='single'`); C-trig (triggered, `ws='single'`); C-compact (`ws='none'`, forced).
- **Procedure.** Generate per arm; parse; validate; match; record tokens, wall time, and per-step `log_Z` for constrained arms.
- **Metrics.** Parse-valid rate; schema-valid rate; `bfcl_match` accuracy; spurious-call rate on irrelevance; generated tokens per item; seconds per item and mask-time share; mean of `sum_k log Z` per item (the forcing cost of D2).
- **Plot** `fig_lm_so_headline`: left, grouped bars (x = arm; two series: schema-valid rate and accuracy; y = rate with 95% bootstrap interval; irrelevance spurious rate as a third hatched series); right, histogram of `sum_k log Z` for C-force versus C-compact.
- **Expected outcome.** Schema-valid = 100% for constrained arms (asserted). C-force has 100% spurious calls on irrelevance (it cannot refuse), which the plot makes visible; C-trig does not. Other effects: to be measured. Literature direction: strict format restrictions can hurt reasoning-heavy tasks (Tam et al., 2408.02442), and local renormalization can distort the sampled distribution (Park et al., 2405.21047).
- **Noise check.** Greedy decoding, so no sampling noise; paired bootstrap (1.5) over items with `cluster=<function name>` between arms. Claims need an interval excluding 0.
- **Reasoning-order probe** (a labelled side probe with n=40, no headline claim unless the interval excludes 0): 40 seeded two-step arithmetic word problems; arms: free-form then `Answer: N`; constrained `{"answer": N}`; constrained `{"reasoning": "<=300 chars", "answer": N}` (reasoning first); unconstrained JSON prompt. Report accuracy and parse validity.
- **T1 reduction.** Not applicable (T0). `--gpu`: 1.7B and double the items.
- **Artifacts.** None committed.

#### Figures and tables

- `fig_lm_so_automaton` (derivative automaton and token overlays), `fig_lm_so_mask_overlay` (next-token probabilities at one tool-call state, allowed tokens colored), `fig_lm_so_distortion` (toy: p, conditional, p_mask as bars; real model: scatter of masked versus conditional on log axes with the identity line), `fig_lm_so_mask_time`, `fig_lm_so_headline`.
- `tab_lm_so_engines`, `tab_lm_so_validity` (arm-by-metric table behind the headline plot), `tab_lm_so_patterns` (state counts for the four patterns).

#### Pitfalls and misconceptions

- Mask shape: the logits have 151,936 columns but the tokenizer has 151,669 ids; pad ids must be masked or they can be sampled (with XGrammar, pass `vocab_size`).
- Tokens are byte strings, not characters; `tokenizer.decode` of a partial UTF-8 token is not the token's bytes.
- Special tokens such as `<tool_call>` are single symbols; matching their text with bytes does nothing.
- `ws='any'` can let the model emit spaces forever; always bound whitespace.
- Mask before temperature and top-p; otherwise top-p includes invalid tokens.
- Validity holds only at the final state: a generation cut off by `max_new_tokens` is a valid prefix, not a valid output (check `is_final`).
- Constrained does not mean correct: valid JSON with a wrong value is the commonest failure afterwards.
- A schema property order other than the model's habit forces low-probability tokens.
- Grammars that cannot express "do not call" make the model hallucinate a call.
- Masking is not conditioning: D2.

#### Summary (draft bullets)

- A constraint is a byte automaton; a token automaton follows by running each token's bytes through it.
- A trie over the vocabulary makes the per-state mask cost proportional to live paths, not to `|V|`.
- Masked sampling is local renormalization, so it samples `p(y)/prod Z`, not `p(y | y ∈ L)`; the two agree only when the product of allowed masses is constant.
- The distortion is small when the model already puts nearly all of its mass on valid tokens at each step, and large when branches differ in completability.
- Triggered grammars constrain arguments and leave the choice to call with the model.
- Production engines add context-free grammars, caching and batching; the from-scratch version is their test oracle.

#### Exercises

1. **Check.** Why is the end-of-turn token allowed only in nullable states, and what happens to the validity guarantee if it is always allowed? *Sketch:* the model may stop inside the object; the output is a valid prefix but not valid JSON.
2. **Check.** For the state after `{"name": "calc` in the `calculator` call grammar, list which of `u`, `ulator`, `ulator"`, `ulator", "arguments"` are allowed and which are not, and say why one string has several tokenizations. *Sketch:* all four are allowed only if they are byte-prefixes of the remaining literal; here `ulator"` and longer ones are allowed, and the grammar does not care how the string is split.
3. **Derive.** In the toy example compute `w(h, t)` for each first token and show that sampling with `q*(t | h) ∝ p(t | h) w(h, t)` reproduces `cond`. *Sketch:* `w(a) = 0.01`, `w(b) = 0.9`; `q*(a) ∝ 0.5 × 0.01`, `q*(b) ∝ 0.4 × 0.9`, giving 0.0137 and 0.9863.
4. **Derive.** Prove the bound `|log(p_mask(y)/p(y | L))| ≤ T log(1/(1−λ))` and give an example with `T = 20`, `λ = 0.05` where the bound exceeds `log 2`. *Sketch:* step 7 of D2; `20 × 0.0513 = 1.03 > 0.69`.
5. **Measure.** Compare `ws in {'none','single','any'}` on the headline items: predict the order of mean `sum_k log Z` and of accuracy first; then measure with paired bootstrap.
6. **Measure.** Run the reasoning-order probe with Qwen3-1.7B and `n=200`; state whether the 40-item result survives.
7. **Open.** Implement one of: UTF-8-strict string automata; `$ref` and recursive schemas via a pushdown stack; jump-forward decoding with canonical retokenization. Compare mask or step time with XGrammar.

#### Further reading

- Papers: [Outlines: Efficient Guided Generation](https://arxiv.org/abs/2307.09702) (index design); [XGrammar](https://arxiv.org/abs/2411.15100) (pushdown automaton and token-mask caching); new: [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) (exactly the distortion of D2 and an approximate fix); new: [Guiding LLMs the Right Way (DOMINO)](https://arxiv.org/abs/2403.06988) (subword misalignment and minimal invasiveness); new: [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442) (format restriction and reasoning); [SGLang](https://arxiv.org/abs/2312.07104) (compressed FSM and jump-forward decoding; from Codex 13.9, verified 28 Oct 2026).
- Docs and tutorials: [XGrammar documentation](https://xgrammar.mlc.ai/docs/); [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/); [Outlines documentation](https://dottxt-ai.github.io/outlines/latest/).
- Code bases: [outlines](https://github.com/dottxt-ai/outlines); [XGrammar](https://github.com/mlc-ai/xgrammar); new: [llguidance](https://github.com/guidance-ai/llguidance) (MIT; Rust engine used by several serving stacks); [guidance](https://github.com/guidance-ai/guidance).

#### Risks and open decisions

- **XGrammar or Outlines API drift** (both release often): pin versions; keep the smoke test that asserts the first fill of a bitmask; fallback: skip the production cells with a printed notice and keep the oracle comparison against `re`.
- **Pure-Python trie walk is slow** for large automata: if mask-building exceeds 1 s per state, precompute only visited states (already lazy) and restrict the headline to the 10.1 tools; fallback: cache masks to disk keyed by `(pattern, tokenizer revision)`.
- **CPU generation speed.** If 120 items exceed 10 minutes on the build host, cut to 20 per category or require `--gpu`.
- **Exact enumeration blows up** if the finite language is large: cap `max_len` and the number of paths at 500; choose answer sets accordingly.
- **Whitespace and property order** may differ for Qwen3.5-style XML calls; XML-format constraints are left to an exercise.

### 10.3 Agent loops and harnesses

- **Label:** `sec_lm_agent_loops` · **Tier:** T1 · **Budget:** interactive cells (sandbox self-test, three scripted episodes, three live episodes with HF generation, analysis of committed runs) ≤ ~15 min on T1. The headline grid (4 cells × 30 tasks × 2 seeds = 240 episodes, plus 120 more if the pre-registered stopping rule below adds a third seed, 360 at most; it was 6 cells × 30 tasks × 3 seeds = 540 episodes) is a one-off offline run with a design cap of 4 GPU-hours on the build box (the earlier 6 GPU-hours for 540 episodes scaled pro rata to 360 episodes; 2.7 GPU-hours if the third seed is not run; to be measured); its trajectories are committed and the notebook analyzes them. its trajectories are committed and the notebook analyzes them. Peak memory: Qwen3-4B bf16 weights 8 GB plus a KV cache of 147,456 bytes per token (36 layers × 8 KV heads × 128 × 2 tensors × 2 bytes, from the model config), so about 5 GB of cache holds roughly 34k tokens, which limits concurrent episodes to a handful at 4–8k tokens each · **Codex units:** none (v5 `[new]`; the Codex plan excludes autonomous tool-use loops).
- **Builds on (book):**
  - `sec_mdp`: episode, horizon, termination versus truncation (`subsec_mdp-termination`: a step limit is truncation); the context is the state (`sec_mdp`, state augmentation).
  - `sec_imitation`: `subsec_il_error_bounds` and `subsec_il_state_divergence` for compounding error; one paragraph, then measured.
  - `sec_kv-cache`: prefill versus decode and what re-reading a context costs; the cost derivation here uses these, without re-deriving them.
  - `sec_decoding`: sampling settings.
- **Builds on (this part):**
  - 10.1: `d2l.ToolRegistry`, `d2l.parse_tool_calls`, `d2l.tool_chat` (this section's `Episode` generalizes it), `d2l.function_to_schema`.
  - 10.2 (optional): `d2l.TokenFSM` to force well-formed calls; not needed for the headline grid.
  - 1.1/1.2: `d2l.load_hf`, `d2l.generate`; 1.5: `d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None)`.
  - Forward pointers: 13.3 (prefix caching), 12.4 (KV compression) for the cost discussion; 10.4, 10.5 consume `Episode`, `Sandbox` and the committed runs.
- **Exports:**
  - `Sandbox(workdir=None, cpu_s=10, mem_mb=2048, fsize_mb=10, nofile=64, extra_procs=64, timeout_s=20, max_out=20000, workdir_cap_mb=50, net='auto', fs='auto')`: `.run(cmd, stdin=None) -> RunResult(returncode, stdout, stderr, timed_out, seconds, truncated)`, `.put(files: dict[str, str])`, `.capabilities() -> dict[str, bool]`, `.close()`, context manager; `Sandbox.selftest() -> DataFrame` runs the battery of `tab_lm_sbx_limits`.
  - `run_python(code: str, timeout: float = 5.0) -> str`: the function 10.1's `python` tool calls; formatted stdout, stderr tail and exit status.
  - `Episode(task, registry, sandbox, system_prompt, policy='full', max_steps=15, max_ctx_tokens=4096)`: `.prompt_text(tokenizer, fmt='qwen3') -> str`, `.observe(model_text: str) -> None`, `.done`, `.status in {'submitted','step_limit','ctx_overflow','format_limit'}`, `.stats` (per-step prompt tokens, generated tokens, observation tokens, tool seconds), `.messages`.
  - `run_batch(episodes, gen_fn, max_new_tokens=512) -> list[Episode]`: lockstep driver; `gen_fn(prompts: list[str]) -> list[str]` (an HF-based and a vLLM-based implementation are shown).
  - `bash_interface(sandbox) -> ToolRegistry`, `aci_interface(sandbox) -> ToolRegistry`, `codeact_interface(sandbox) -> ToolRegistry`.
  - `context_policy(name: str) -> Callable[[list[dict], int, Callable], list[dict]]` for `'full'`, `'mask'`, `'compact'`.
  - `load_repair30(split='test'|'dev') -> list[dict]` (`id`, `module`, `bug_class`, `issue`, `repo_dir`, `visible_tests`, `hidden_tests`, `gold_patch`); `grade_repair(task, workdir) -> dict(passed, f2p, p2p, tampered)`.
  - `failure_labels(episode, task) -> list[str]` (codes F1–F9); `cost_report(episodes) -> DataFrame`.
  - Data: `DATA_HUB['lm-repair30']`, `['lm-repair30-dev10']`, `['lm-agent-runs-ablation']`.
- **Models and data:**
  - Agent model: `Qwen/Qwen3-4B` (revision `1cfa9a7208912126459214e8b04321603b3df60c`, Apache-2.0), non-thinking, temperature 0.7, top-p 0.8, `max_new_tokens=512` per step. Pilot with `Qwen/Qwen3-1.7B` on `dev10`; rule: use the smallest model whose base cell (ACI + full) solves at least 20% of `dev10` (default expectation: 4B). `Qwen/Qwen3-8B` (T2) as an exercise.
  - `repair30`: authored in this book (no external license issue): code tasks, hidden tests and gold patches are book-written code, so `lm-repair30` and `lm-repair30-dev10` are MIT-0 (replacing the earlier Apache-2.0 proposal; decided 2026-10-29); see `tab_lm_repair30`. SWE-bench is discussed, not run, in this section.
- **Packages:** Python ≥ 3.10, `pytest>=8` inside the sandbox, `vllm` (optional, for the offline grid; pin the version used), `psutil` (process counts), `mini-swe-agent` (PyPI 2.4.6, MIT; production comparison). System tools probed at runtime: `bwrap` (bubblewrap), `unshare`, `prlimit`.

#### Narrative arc

1. **The question.** Opening cell: run one `repair30` task with a 12-line scripted "agent" that has the right idea (it applies the gold patch through the interface). It succeeds. Then run the same task with Qwen3-4B for three seeds. Print outcome, steps, prompt tokens summed over steps, and generated tokens. Symptom: the summed prompt tokens exceed the final context length by a large factor (to be measured; the cell prints the ratio), and two of the three runs end in different ways. This frames the chapter's themes: loop, context, cost, variance.
2. **10.3.1 The observe–think–act loop.**
   - ReAct as the pattern: the model alternates reasoning text and an action, receives an observation, and continues; with tool calling (10.1) the "action" is a parsed call. Terminal conditions: `submit`, no-call final answer, step limit (truncation in the MDP sense), context overflow, format-error limit.
   - `Episode` and `run_batch`. A step is: render prompt from messages (policy applied), generate, parse, execute, append. `run_batch` steps all active episodes at once so a batched `gen_fn` sees many prompts; finished episodes drop out.
   - Table `tab_lm_al_notation`: loop objects to the notation contract: task `x`, prompt distribution `μ0`, agent turn `y^(m)`, observation `o^(m)`, trajectory `τ`, policy `π_θ`, reward = hidden-test outcome. The loop is the MDP of `sec_rl_sequences` with nontrivial transitions; forward pointer to 10.4.
3. **10.3.2 A sandbox, with its limits.**
   - Why: the model's shell commands are untrusted (bugs, hallucinated `rm -rf`, and, in 10.5, injected instructions).
   - Specification (below) and `Sandbox.selftest()`, printed as `tab_lm_sbx_limits`: each threat, the mitigation, whether it is active on this host, and the observed outcome (CONTAINED or ESCAPED).
   - The statement that this is accident-grade, not adversary-grade, and where the line is (container with gVisor or a microVM).
4. **10.3.3 The agent–computer interface.**
   - Three interfaces on the same tools: `bash` only (mini-swe-agent's premise: a shell is enough; each command runs in a fresh process, so `cd` does not persist); ACI (SWE-agent's lesson: windowed `view` with line numbers, `find`, exact-match `replace` that returns actionable errors, lint-on-edit, `run_tests`, `submit`); CodeAct (actions are Python snippets that call the ACI functions as a library, so one step can loop or branch).
   - A table `tab_lm_al_interfaces`: tools, return formats, truncation rules, error messages. Observation formatting rules (shared): at most 2,000 characters, head 1,200 plus tail 800 with `[... N characters omitted ...]`, exit code printed.
   - Reference: *Writing effective tools for AI agents*: tool names and errors are part of the prompt.
5. **10.3.4 Context over a long horizon.**
   - Four options, three implemented: `full` (keep everything; overflow ends the episode), `mask` (keep the last `k=3` observations verbatim, replace older ones with `[output omitted: n lines]`, keep all actions and reasoning), `compact` (at 50% of the budget, ask the same model for a summary of files viewed, hypotheses, edits and test status, then replace messages between the task and the last two steps by it). External memory (a `notes` file that survives compaction) and subagents (delegate a search to a fresh context that returns a short report) are described, with an exercise and no grid cell.
   - The context budget is deliberately small (calibration rule below) so the policies have something to do: the tasks are small. State that real software-engineering agents face contexts of 100k tokens or more.
   - Observation masking versus summarization: Lindenbauer et al. report on SWE-bench Verified that masking halves cost relative to the raw agent while matching or slightly exceeding summarization's solve rate; the notebook tests the direction at small scale.
6. **10.3.5 Cost and latency accounting.** Derivations D1 and D2; `cost_report`; the token-ledger table `tab_lm_al_ledger` (per episode: steps, prompt tokens summed, uncached and cached-equivalent, generated tokens, tool seconds, model seconds).
7. **10.3.6 A taxonomy of failures.** `tab_lm_al_failures` with detectors; the detectors are run over the committed runs; hand-label 30 failed episodes and report agreement.
8. **10.3.7 Production harnesses.** mini-swe-agent and smolagents compared with the section's loop on `dev10`; what each adds (environments, trajectory format, multiple executors).
9. **10.3.8 The headline experiment** (below) and a horizon paragraph: errors compound with steps (`sec_imitation`); the section measures recovery after a tool error and the rate of repeated actions.

#### The sandbox specification (precise)

Process model:

- Each call to `Sandbox.run` starts a fresh process: `/bin/bash --noprofile --norc -c <cmd>` (or `python -I file.py` for `run_python`), `shell=False` for the outer `subprocess.Popen`, `cwd=<tempdir>/work`, `start_new_session=True` (own process group; the controller kills the group with `SIGKILL` on timeout), stdin closed unless given.
- Working directory: `tempfile.TemporaryDirectory(prefix='d2l_sbx_')`; the task repository is copied in; `HOME` and `TMPDIR` point inside it; the directory is deleted on `close()`.
- Environment: replaced, not filtered: `PATH` (system bin dirs plus the interpreter's), `HOME`, `TMPDIR`, `LANG=C.UTF-8`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=0`. No variable of the parent is inherited, so tokens such as `HF_TOKEN` or any `*_KEY` are absent (tested).
- Resource limits applied in the child through `preexec_fn` and `resource.setrlimit` (Linux and macOS):
  - `RLIMIT_CPU` = `cpu_s` seconds;
  - `RLIMIT_AS` = `mem_mb` MiB (not enforced on macOS; do not import torch inside the sandbox, because it maps large virtual regions);
  - `RLIMIT_FSIZE` = `fsize_mb` MiB per file;
  - `RLIMIT_NOFILE` = `nofile`;
  - `RLIMIT_CORE` = 0;
  - `RLIMIT_NPROC` is per user, not per sandbox, so it is set to `(current process count of this uid) + extra_procs`; inside a PID namespace the count is per namespace.
- Wall-clock timeout `timeout_s` through `communicate(timeout=...)`, then `os.killpg(pgid, SIGKILL)`. Output goes to files outside the work directory; at most `max_out` bytes are read back (head plus tail); `truncated` records whether it was cut.
- A watchdog thread polls the total size of the work directory every 0.5 s and kills the group above `workdir_cap_mb`. `RLIMIT_FSIZE` limits one file, not the sum.
- Isolation ladder, chosen at construction and reported by `capabilities()`:
  1. `bwrap` present and usable (`bwrap --ro-bind / / true` succeeds): `--unshare-net --unshare-pid --unshare-ipc --die-with-parent`, read-only bind of `/usr`, `/lib*`, `/bin` and the interpreter prefix, a bind of the work directory read-write, `--tmpfs /tmp`, `--proc /proc`, `--dev /dev`. Network, PID and filesystem views are isolated.
  2. else `unshare --user --map-root-user --net --pid --fork --mount-proc` (probe with `unshare -rn true`): network and PID isolated; the filesystem is not.
  3. else resource limits and environment scrubbing only; network and filesystem are not isolated. `capabilities()` says so, and a banner is printed at the top of every notebook that uses the sandbox. `net='auto'` then refuses to run 10.5's exfiltration experiments unless `allow_unsafe=True` (fail closed): on the lowest rung 10.5's exfiltration experiments do not run, and the notebook analyzes the committed outcomes of a host with a higher rung instead.
- Authoring-time check (decided 2026-10-29, because `bwrap` and `unshare` availability on the build host is unknown): before the grid is run, on the build host, run `bwrap --ro-bind / / true` and `unshare -rn true`, record both exit codes and the rung `capabilities()` selects in the manifest of `lm-agent-runs-ablation`, and keep the ladder above unchanged whatever the outcome. If both fail, the grid still runs (rung 3, with the banner and the limits-only caveats in the text) and the exfiltration experiments of 10.5 are generated on a host that reaches rung 1 or 2.
- macOS note for authors: `sandbox-exec` with a deny-network profile exists but is deprecated; use it only on a developer machine. CI runs on Linux.

Honest limits (`tab_lm_sbx_limits`, with the result of each test on this host):

| Threat | Mitigation | Residual risk |
|---|---|---|
| Infinite loop | wall timeout, `RLIMIT_CPU`, group kill | none observed |
| Memory bomb | `RLIMIT_AS` (Linux) | not enforced on macOS; AS limits break some libraries; cgroup `MemoryMax` is more accurate and is optional |
| Fork bomb | `RLIMIT_NPROC` set relative to the uid, PID namespace when available, group kill | without a PID namespace a bomb can still stress the user's process table until the timeout |
| Disk fill | per-file limit, watchdog on directory size | the watchdog polls: a fast writer can overshoot between polls |
| Network access | network namespace | without a namespace, outbound connections are possible; a Python-level monkeypatch of `socket` is not offered because a shell can bypass it |
| Reading host files | mount namespace with read-only binds | without `bwrap`, the process can read anything the user can (a canary file outside the work directory is read in the self-test) |
| Signals to the user's other processes | PID namespace | without it, the sandbox can signal same-uid processes (self-test uses a sacrificial process, never the notebook kernel) |
| Secrets in the environment | scrubbed environment | none for variables; files such as `~/.ssh` are covered by the filesystem row |
| Kernel or namespace exploits, side channels | none | out of scope: use a container with gVisor or a microVM for untrusted code |

Self-test battery (each prints CONTAINED or ESCAPED and asserts that the result matches `capabilities()`; a test that escapes while the capability is reported as present fails the notebook):

1. `while True: pass`: returns within `timeout_s + 1`.
2. `bytearray(4 << 30)`: dies with `MemoryError` or is killed.
3. a shell fork bomb with a 3-second timeout: afterwards no process of that session remains.
4. `dd if=/dev/zero of=big bs=1M count=1024`: stops at `fsize_mb`.
5. 200 files of 1 MiB: killed by the watchdog.
6. outbound connection to a listener the controller opened on a free local port, and to `1.1.1.1:53` with a 2-second timeout: blocked iff `net_blocked`.
7. read of a canary file in a directory the controller created outside the work directory: blocked iff `fs_isolated`.
8. write to a file outside the work directory: blocked iff `fs_isolated`.
9. `env | grep -i -E 'key|token|secret'`: empty.
10. `pkill -f d2l_victim` against a sleeping sacrificial process: survives iff `pid_isolated`.

#### The task suite

`repair30` is ten small Python modules, each with three injected bugs, built the way SWE-smith and SWE-Gym construct tasks, but by hand and tiny.

- Each module has 8–12 pytest tests: 4 visible in `tests/`, the rest hidden in an archive that the grader restores before running. The agent sees the issue text, the code, the visible tests, and a shell or the ACI.
- Validation protocol (each is a test in the build): the clean module passes all tests; every buggy variant fails at least one visible and one hidden test (the fail-to-pass set `f2p`); the pass-to-pass set `p2p` passes before and after the gold patch; the gold patch fixes all `f2p`; an empty patch scores 0; a patch that edits `tests/` is flagged as tampering and its tests are restored.
- Success = all `f2p` and `p2p` hidden tests pass and no tampering. Reporting follows the SWE-bench convention (`f2p` and `p2p`).
- `dev10`: one bug per module of the ten, disjoint from `test30` bug variants but not from modules (state this limit); used for calibration only.

`tab_lm_repair30` (bug classes: a = one-line operator or boundary error; b = a missing branch needing ≥ 3 new lines; c = a wrong state update spanning two functions or, for module 5, a bug in another file):

| # | Module | a | b | c |
|---|---|---|---|---|
| 1 | `lru.py`, an LRU cache | capacity check `>` for `>=` | updating an existing key does not refresh recency | a miss on `get` mutates the order |
| 2 | `intervals.py`, merge and insert | touching intervals not merged | unsorted input not handled | merge mutates its input and drops the last interval |
| 3 | `ledger.py`, transfers with fee and overdraft limit | fee charged to the receiver | failed transfer leaves a partial debit | history not updated on the fee path |
| 4 | `tokenize.py`, an expression tokenizer | decimals split at `.` | unary minus unsupported | digits after whitespace dropped |
| 5 | `calc.py` plus `tokenize.py`, shunting-yard | `^` left-associative | parenthesis popping order | the bug is in `tokenize.py`, the failing tests in `calc` |
| 6 | `ratelimit.py`, a token bucket with a fake clock | integer division in refill | no cap at capacity after refill | `allow(n)` with `n>1` partially consumes |
| 7 | `trie.py`, insert and autocomplete | exact word not returned for its own prefix | delete leaves the end flag | autocomplete not in lexicographic order, limit off by one |
| 8 | `csvlite.py`, quoted fields | `""` not unescaped | trailing empty field dropped | commas inside quotes split fields |
| 9 | `dates.py`, leap years, `add_days`, weekday | century rule | `add_days` across a year boundary | weekday off by one before 1900 |
| 10 | `heap.py`, a min-heap with `decrease_key` | wrong child index in sift-down | `pop` on a one-element heap | `decrease_key` does not sift up |

Authoring note: the author writes the ten clean modules (40–80 lines each) and tests, then the bug patches; difficulty is calibrated on `dev10` with the pilot rule above; none of the 30 bugs may be fixable by deleting the failing test.

#### Derivations

**D1. Token cost of an episode, with and without prefix caching.**

- *Statement.* Let an episode have `T` steps, initial context `c0` (system prompt, tool schemas, task), generated tokens `a_s` and observation tokens `o_s` at step `s`, and no context management. Then: (i) without prefix caching the prefill work is `P_nc = T c0 + Σ_{s=1}^{T-1} (T − s)(a_s + o_s)`; (ii) with perfect prefix caching it is `P_c = c0 + Σ_{s=1}^{T-1} (a_s + o_s)`; (iii) for `a_s + o_s = u` constant, `P_nc = T c0 + u T(T−1)/2` and `P_c = c0 + u(T−1)`.
- *Assumptions.* The context at step `t` is `c0 + Σ_{s<t}(a_s + o_s)`; the cache holds the whole previous prompt; no template drift (10.1 prefix-stability check; Qwen3 history re-rendering breaks (ii) at the first changed token).
- *Steps.*
  1. The prompt at step `t` has `P_t = c0 + Σ_{s<t}(a_s + o_s)` tokens.
  2. Without a cache the engine prefills every prompt: `Σ_t P_t`.
  3. Each increment `a_s + o_s` appears in `P_t` for all `t > s`, i.e. `T − s` times; collecting terms gives (i).
  4. With a cache only the new suffix is prefilled: `P_1 = c0`, then `a_{t−1} + o_{t−1}` new tokens (the generated tokens were already computed during decoding, so with perfect reuse they are not prefilled again; state both conventions and use the cell's flag), which sums to (ii).
  5. (iii) follows by arithmetic series.
  6. Decode attention still reads the whole cache for each generated token, so even (ii) has a quadratic term in memory traffic: `Σ_t a_t (c0 + Σ_{s<t}(a_s + o_s))` (cross-reference `sec_kv-cache`).
- *Final formulas.* As in the statement.
- *Numerical check* (cell `al-cost-identity`): for all committed episodes, recompute `P_nc` from the logged `a_s`, `o_s` and compare with the sum of logged prompt lengths. Tolerance: exact equality for `full`; for `mask` and `compact` the formula holds with `o_s` replaced by the length actually kept (the cell does this). Where vLLM reports cached-token counts, compare `P_c` with measured uncached prompt tokens (agreement to be measured; Qwen3's drifting history may reduce reuse).

**D2. When does compaction pay for itself?**

- *Statement.* Without prefix caching, compact at step `k` with `H = u(k−1)` tokens of history, replacing them with a summary of `S < H` tokens that costs `S` decode tokens, with each decode token priced at `φ` prefill tokens. If `R = T − k + 1` steps remain, the compaction saves prefill work iff `R > φ S / (H − S)`.
- *Assumptions.* Constant `u`; no cache; the summary stays fixed in size; the summary does not change the number of remaining steps.
- *Steps.*
  1. Each remaining prompt is shorter by `H − S` tokens: saving `R (H − S)`.
  2. Cost: generating the summary, `φ S`.
  3. Net benefit positive iff `R (H − S) > φ S`, which gives `R* = φ S / (H − S)`.
  4. With a cache, compaction also invalidates the prefix: add a one-time prefill of `c0 + S`, so `R* = (φ S + c0 + S) / (H − S)` when the saving is in prefill tokens; when the saving is in attention reads or memory, price accordingly (KV memory, `sec_kv-cache`).
  5. Masking costs no decode tokens (`S` is a short placeholder, `φ S ≈ 0`), so `R*` is near 0: it pays as soon as an observation is dropped. This is the cost argument behind the reported advantage of masking.
- *Final formula.* `R* = φ S / (H − S)` (no cache).
- *Numerical check* (cell `al-compaction-breakeven`): simulate a constant-`u` episode with token ledgers for `T` from 5 to 40 and `φ ∈ {1, 4}` (a relative price of decode to prefill tokens: choose values here as scenarios, not as measurements of a GPU); assert the simulated break-even equals the formula within one step. On committed `compact` runs, count how often the compaction occurred after the break-even point.

#### From-scratch implementation

Budget: about 150 lines for the core, plus infrastructure cells that ship as `#@save` code but are folded in the text (stated deviation from the 150-line rule: a sandbox and a task grader are infrastructure, not exposition).

| Component | Lines | Notes |
|---|---|---|
| `Episode` (message list, prompt rendering, observe, termination) | 60 | the "~100-line agent loop" is `Episode` + `run_batch` + the policies |
| `run_batch` | 25 | lockstep batching |
| `context_policy` (`full`, `mask`, `compact`) | 35 | |
| `failure_labels` | 30 | heuristics in `tab_lm_al_failures` |
| infrastructure: `Sandbox` (~80), interfaces (~70), `grade_repair` and loaders (~40) | ~190 | tested by the batteries below |

Tests:

1. Sandbox self-test (above); a failing assertion stops the notebook.
2. Grader: gold patch gives 30/30; unpatched gives 0/30; the tamper patch is flagged on a task where it would otherwise pass.
3. Scripted oracle policy (a fixed list of model texts that applies the gold patch through each interface): 30/30 success under `bash`, `aci` and `codeact`. A scripted random-edit policy: 0/30.
4. Policy unit tests: the first user message and system prompt are preserved; after `mask`, at most `k` observations are verbatim; `compact` leaves at least the last two steps; applying a policy twice is idempotent; the rendered prompt is at most the budget or the episode is marked `ctx_overflow` (never silently truncated).
5. Determinism: replaying scripted outputs gives identical transcripts.
6. D1 identity (above).

#### In production

- **mini-swe-agent** (PyPI 2.4.6, MIT): bash-only agent, the reference for the `bash + full` cell. Run it on `dev10` with a local OpenAI-compatible vLLM endpoint serving Qwen3-4B; compare success with this section's cell (agreement to be measured; differences are explained by system prompt and observation format, which the notebook prints side by side).
- **SWE-agent** for the ACI design; **smolagents** for CodeAct-style agents and its executor options (local restricted interpreter, and sandboxed executors); read its executor documentation for the same caveat as here: a restricted interpreter is not a security boundary.
- **Serving.** vLLM offline engine with prefix caching for the grid; report the cache-hit rate it exposes and compare with D1.
- **Real benchmarks.** SWE-bench Verified is the production counterpart of `repair30`: 500 human-validated tasks from 12 repositories (Django 231, SymPy 75, Sphinx 44, Matplotlib 34, scikit-learn 32, Astropy 22, xarray 22, pytest 19, Pylint 10, requests 8, seaborn 2, Flask 1), each needing a Docker image; the harness recommends x86_64 with about 120 GB free storage, 16 GB RAM and 8 cores. Not run here; the T2 extension is in 10.5.

#### Headline experiment

- **Hypothesis.** H1: the ACI interface raises success over bash-only for a 4B model (the interface absorbs formatting burden). H2: `mask` and `compact` lower summed prompt tokens versus `full` at equal or better success under a small context budget. H3: `mask` matches `compact` on success at lower cost (the direction reported by Lindenbauer et al. at much larger scale). No effect sizes are asserted.
- **Setup.** Model: Qwen3-4B. Tasks: 30 (`test30`). Grid: 4 cells: ACI × 3 policies (full, mask, compact) plus bash + full. Seeds: 2 (0 and 1) with a pre-registered rule that may add a third (sampling at temperature 0.7, top-p 0.8). Budget: `max_steps=15`, `max_ctx_tokens` chosen on `dev10` before the test run from `{3072, 4096, 6144, 8192}`: the smallest value at which ACI + `full` overflows in 15–40% of `dev10` episodes (written into the notebook before the test run; the `dev10` calibration episodes are outside the grid and its cap). 240 episodes (4 × 30 × 2), 360 if seed 2 is added.
- **Why this grid (cuts from the earlier 6-cell, 3-seed, 540-episode design; each cut is justified against H1–H3).** H1 compares the two interfaces at one policy: bash + full against ACI + full. H2 and H3 compare policies within one interface: full, mask, compact on ACI. The interface × policy interaction is not claimed, so the cells bash + mask and bash + compact are dropped (2 cells); the policies are run on ACI because a floor effect in bash (success near 0, a plausible outcome noted below) would make bash the wrong place to compare policies. The 30 tasks stay: the unit of analysis is the task and every interval is clustered by task, so tasks are what buy power, whereas seeds only reduce within-task sampling noise; fewer tasks would widen every interval. Seeds fall from 3 to 2, with a stopping rule fixed before the test run and written into the notebook: seeds 0 and 1 run for all four cells (240 episodes); seed 2 runs for all four cells (+120 episodes) if and only if, on the task-level means of seeds 0–1, at least one claimed contrast (H1: ACI + full against bash + full; H2: mask against full and compact against full, on success and on summed prompt tokens; H3: mask against compact) has a 95% task-cluster interval that includes 0 while its absolute point estimate exceeds the mean absolute between-seed difference of task success in the two cells compared, that is, an effect that may be real but is unresolved. If the rule does not fire the grid stops at 240 episodes and unresolved contrasts are reported as unresolved. No cell or seed beyond these is added. Intervals and the seed-spread yardstick are computed from the seeds actually run.
- **Procedure.** Offline script (`%%writefile run_grid.py`) with the vLLM engine, `run_batch` and a fixed seed schedule; trajectories saved as JSONL (`lm-agent-runs-ablation`); the notebook loads them, grades, labels failures, computes the ledgers.
- **Metrics.** Success (per task: mean over the seeds run), steps used, summed prompt tokens (uncached convention) and generated tokens, wall seconds (tool and model), failure labels, status counts (`ctx_overflow`, `step_limit`).
- **Plots.**
  - `fig_lm_al_grid`: left, success with 95% task-cluster bootstrap intervals for the four cells; right, the four paired contrasts of H1–H3 (ACI + full minus bash + full; mask minus full; compact minus full; mask minus compact, all on ACI).
  - `fig_lm_al_cost`: success versus mean summed prompt tokens per episode, one point per cell, with the Pareto frontier drawn (reused in 10.5).
  - `fig_lm_al_failures`: stacked failure labels per cell.
  - `fig_lm_al_context`: prompt length versus step for one task under the three policies.
  - `fig_lm_al_survival`: fraction of episodes still running at each step, split by outcome.
- **Expected outcome.** Direction only (above). The citation: Lindenbauer et al. (2508.21433). A floor effect (success near 0 in `bash`) is a plausible result; it is reported as such, not tuned away.
- **Noise check.** Unit of analysis is the task (cluster = task); two seeds per task (three if the stopping rule fires) separate sampling noise from task difficulty; paired bootstrap between cells on the same tasks (`d2l.paired_bootstrap(..., cluster=task_id)`); report the seed-to-seed spread of each cell (the mean absolute between-seed difference of task success when only two seeds exist) next to the between-cell differences. Claims need an interval that excludes 0 and a difference larger than the seed spread.
- **T1 reduction.** Interactive cells: 10 `dev10` tasks, 1 seed, 2 cells (bash+full, aci+mask), HF generation, ≤ ~15 min. Full grid: committed.
- **Probes (labelled, no claims).** CodeAct interface on `dev10`, 1 seed; recovery after a tool error (probability that the next action is valid and not a repeat).
- **Artifacts.** `lm-repair30`, `lm-repair30-dev10`, `lm-agent-runs-ablation` (grid configuration, seeds, vLLM version, model revision in the manifest).

#### Figures and tables

- `fig_lm_al_loop`: the loop as a diagram (model, parser, tool, sandbox, observation, context policy).
- `fig_lm_al_grid`, `fig_lm_al_cost`, `fig_lm_al_failures`, `fig_lm_al_context`, `fig_lm_al_survival`: above.
- `tab_lm_al_notation`, `tab_lm_al_interfaces`, `tab_lm_al_ledger`, `tab_lm_al_failures`, `tab_lm_sbx_limits`, `tab_lm_repair30`.
- `tab_lm_al_failures` (detector in brackets): F1 format (≥ 3 consecutive parse errors or ≥ 5 in total) [parser]; F2 action loop (same normalized action ≥ 3 times in a row) [string match]; F3 context overflow [status]; F4 navigation (never opened the file containing the gold patch) [grep of actions against the gold patch's files]; F5 edit failure (≥ 3 rejected edits or syntax errors) [tool statuses]; F6 wrong fix (hidden tests still fail after submit; sub-label F6a if the visible tests pass, i.e. overfitting to visible tests); F7 regression (`p2p` broken); F8 tampering [hash]; F9 no submit at the step limit.

#### Pitfalls and misconceptions

- The sandbox is not a security boundary against a determined attacker; run `Sandbox.selftest()` on the host that will run the experiments and read the table.
- `RLIMIT_AS` and `RLIMIT_NPROC` do not behave the way their names suggest (virtual size, and per-user count).
- Small tasks hide context problems; the budget is lowered on purpose, and the section says so.
- Stateless bash commands: `cd` and shell variables do not persist; models often assume they do.
- A scaffold change can move success as much as a model change; always report the harness with the model.
- Summed prompt tokens depend on the caching convention; state it.
- Chat-template history drift (10.1) silently breaks prefix caches.
- Hidden tests exist to catch overfitting to visible ones; an agent that edits tests must not pass.
- Reporting steps without tokens, or success without cost, hides the trade-off (10.5).

#### Summary (draft bullets)

- An agent is a loop around a model: render, generate, parse, act, observe, until a terminal condition; the MDP of `sec_rl_sequences` with real transitions.
- A sandbox for model-written code needs a process group, resource limits, a scrubbed environment and a work directory; network, PID and filesystem isolation need namespaces or containers, and the notebook tests which of them are active.
- The interface is part of the model's inputs: windowed views, exact-match edits and error messages change success.
- Context is the state; keeping it small by masking or compaction trades tokens for information, and the break-even has a closed form.
- Cost is quadratic in steps without prefix caching; report tokens and seconds with success.
- A failure taxonomy turns trajectories into diagnoses.

#### Exercises

1. **Check.** A run of 12 steps has `c0 = 800`, and each step adds 300 tokens. Compute `P_nc` and `P_c`. *Sketch:* `P_nc = 12·800 + 300·66 = 29,400`; `P_c = 800 + 300·11 = 4,100` (under the convention that generated tokens are not prefilled again, the cell shows both).
2. **Check.** Why is `RLIMIT_NPROC` alone not a fork-bomb defense for a notebook that shares a user with other jobs, and what is the fix? *Sketch:* it counts all processes of the uid, so either the sandbox is starved or the limit is loose; use a PID namespace or a cgroup `pids.max`.
3. **Derive.** Extend D2 to a cache of hit rate `h` on the old prefix: write `R*` and show it increases with `h`. *Sketch:* the saving per step is `(1−h)(H − S)` (only uncached tokens cost), so `R* = (φ S + c0 + S)/((1−h)(H − S))`.
4. **Derive.** Suppose a task needs `m` specific correct steps, each correct with probability `1 − η` independently, and a recovery mechanism catches each error with probability `r`. Give the success probability and the horizon at which it halves. *Sketch:* per-step success `1 − η(1 − r)`, so `P = (1 − η(1 − r))^m`; half at `m = ln 2 / (−ln(1 − η(1 − r)))`; link to `sec_imitation`'s linear versus quadratic error growth in the horizon (`subsec_il_error_bounds`).
5. **Measure.** Predict, then measure, how the `mask` window `k ∈ {1, 3, 6}` changes success and tokens on `dev10` (2 seeds).
6. **Measure.** Replace the ACI `replace` tool by whole-file rewrite; predict the change in F5 and in tokens, then measure.
7. **Open.** Implement an external-memory `notes` tool and a subagent for `find`, and place them on the success–cost plane next to the four cells.

#### Further reading

- Papers: [ReAct](https://arxiv.org/abs/2210.03629) (reasoning and acting); [CodeAct](https://arxiv.org/abs/2402.01030) (code as the action space); [SWE-agent](https://arxiv.org/abs/2405.15793) (the agent–computer interface); new: [The Complexity Trap](https://arxiv.org/abs/2508.21433) (observation masking versus LLM summarization); new: [MemGPT](https://arxiv.org/abs/2310.08560) (external memory and context paging); new: [SWE-smith](https://arxiv.org/abs/2504.21798) (scaled task construction by bug injection; the model for `repair30`).
- Blogs and docs: Anthropic, [Writing effective tools for AI agents](https://www.anthropic.com/research/writing-tools-for-agents); [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents); Python docs, [`resource`](https://docs.python.org/3/library/resource.html) and [`subprocess`](https://docs.python.org/3/library/subprocess.html) (new; exact semantics of the limits).
- Code bases: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent); [smolagents](https://github.com/huggingface/smolagents); [SWE-bench](https://github.com/SWE-bench/SWE-bench) (MIT).

#### Risks and open decisions

- **Docker and bubblewrap may be absent** on the build host. Fallback: the `unshare` rung, then the limits-only rung with the banner; 10.5's exfiltration experiments then refuse to run (fail closed). Authoring-time check for the build owner: `bwrap --ro-bind / / true` and `unshare -rn true` (are unprivileged user namespaces enabled?), recorded as above; the answer is not known yet.
- **The offline grid is a heavy T1 job** (hours). If the build box cannot afford it, run it once on a T2 host and commit the logs; the notebook then analyzes artifacts, as T2+ notebooks do.
- **Floor and ceiling effects.** If Qwen3-4B solves under 10% or over 90% of `dev10` in the base cell, move to the model chosen by the pilot rule or adjust bug difficulty on `dev10` only.
- **vLLM prefix-cache counters** differ between versions; fall back to the formula and say so.
- **Task licensing and authorship.** `repair30` and `dev10` are written for the book and licensed MIT-0 (code and test suites, per the part's licence rule); the manifest of `lm-repair30` says so.
- **Qwen3 template drift** reduces prefix-cache reuse; the section reports it rather than hiding it.

### 10.4 Training agents with reinforcement learning

- **Label:** `sec_lm_agentic_rl` · **Tier:** T3 (reduced T1 configuration: Qwen3-0.6B, same code behind a config flag; T0 for the derivation checks) · **Budget:** T3 headline run ≤ ~2 h on 8× A100 40 GB (design cap 16 A100-hours; to be measured; run once, limited access; memory per GPU: the 1.72B policy (tied embeddings) has 1,720,574,976 × 16 B = 27.5 GB of weights, gradients and AdamW state, which FSDP2 (6.2) shards to 3.44 GB per GPU on 8 trainer GPUs or 4.59 GB on 6, beside a TP=1 vLLM replica of 3.44 GB bf16 plus KV cache wherever the rollout engine is colocated, far inside 38 GiB; an unsharded 27.5 GB replica would not leave room for the engine and activations, so the T3 trainer is the FSDP2 module of 6.2; peaks to be measured); T1 run ≤ ~15 min (40 steps; trim steps to fit) with peak memory under 15 GiB (0.6B: bf16 weights 1.2 GB, AdamW states about 4.8 GB for fp32 moments, activations with checkpointing; to be measured); the T1 probes (mask ablation and verifier-hacking) are offline runs, 10 runs in total, committed; T0 checks take seconds · **Codex units:** none (v5 `[new]`); Codex 8.x mentions AReaL "excluding agent applications", so only the systems part is carried over, through 8.6.
- **Builds on (book):**
  - `sec_rl_sequences`: the token MDP, `eq_seq_factorization`, `tab_rl_notation_contract` (used verbatim), `subsec_rl_sequences_hacking`, and the exercise "Tool outputs in the trajectory" (this section is its full treatment).
  - `sec_policygradient` (`eq_traj_prob`, the score-function identity), `sec_baselines` (the zero-mean lemma, group baselines, RLOO), `sec_ppo` (ratio, clipping), `sec_regularized` (KL).
  - `sec_imitation`: on-policy data repairs the state-distribution shift that behavior cloning (10.1) leaves; `subsec_il_dagger`.
  - `sec_mdp`: partial observability and state augmentation (the observation history is the state).
- **Builds on (this part):**
  - 8.5: GRPO, Dr. GRPO, DAPO-style dynamic sampling, verifiable rewards; 8.6: rollout engines, weight synchronization, asynchronous scheduling.
  - 8.5: `d2l.group_advantages(rewards, std=True, loo=False, eps=1e-8)` with rewards of shape `(G, K)`; `d2l.sample_responses` (or a vLLM engine) for rollouts; 8.6: `d2l.sync_weights(trainer, engine, mode) -> (seconds, bytes)` when a separate rollout engine is used. There is no `GRPOTrainer`: this section writes its own `masked_pg_loss` (below) and a short training loop around it. `d2l.grpo_train(..., reward_fn=...)` covers the single-turn case only; the multi-turn masked case does not fit its batch-level `reward_fn(texts, items)`. T3 may also go through verl or AReaL (see **In production**).
  - 10.1: `d2l.turn_samples`, `d2l.parse_tool_calls`, `d2l.ToolRegistry`, `d2l.search_book`; 10.3: `d2l.Sandbox` and the ledger code (SWE-style environment discussion and throughput cell).
  - 2.4: `d2l.BookRetriever` (`.search(query, k, mode=...) -> list[d2l.Hit]`, fields `chunk_id, score, rank, text, heading_path, scores`); 2.5: `d2l.book_qa(split)` (splits `'train'`, `'dev'`, `'test'`; items with `qid, split, type, question, reference, key_facts, answerable, gold_spans, ...`; gold chunk ids from `d2l.gold_chunk_ids(item, corpus)`, the reference answer from `item['reference']`) and the single-turn RAG baseline (reimplemented locally in about 15 lines so the section does not depend on 2.5's internal names; the editor should check that the prompt matches 2.5's baseline).
  - 1.5: `d2l.paired_bootstrap`.
- **Exports:**
  - `Turn(gen_ids: list[int], gen_logprobs: list[float], obs_ids: list[int])`; `Trajectory(prompt_ids, token_ids, loss_mask, old_logprobs, reward, group_id, meta)` (dataclasses). `token_ids` holds every token after the prompt, agent and environment alike; `loss_mask[t] = 1` exactly for tokens the policy sampled.
  - `assemble_trajectory(prompt_ids, turns, reward, group_id, meta=None) -> Trajectory`: token-in-token-out concatenation.
  - Group advantages come from 8.5's `d2l.group_advantages(rewards, ...)` on a `(G, K)` reward tensor (imported, not redefined). Only if trajectories arrive unsorted does the section add `group_advantages_by_id(rewards: Tensor, group_ids: Tensor, std: bool = True) -> Tensor`, which scatters into `(G, K)` and calls the 8.5 function.
  - `masked_pg_loss(logp, old_logp, adv, loss_mask, clip_eps=0.2, agg='token_mean') -> tuple[Tensor, dict]`; shapes `[B, L]` for log-probabilities and mask, `[B]` or `[B, L]` for `adv`; `agg in {'token_mean', 'seq_mean', 'const'}`.
  - `SearchEnv(retriever, item, max_turns=4, k=3, obs_tokens=200)` with `.reset() -> messages` and `.step(model_text: str) -> tuple[str | None, bool, dict]`; `search_rollout(gen_fn, tokenizer, envs, max_new_tokens=256) -> list[Trajectory]` (batched; `gen_fn(prompt_ids: list[list[int]], max_new_tokens) -> list[tuple[list[int], list[float]]]`).
  - Rewards: `em_reward(pred: str, golds: list[str]) -> float`, `f1_reward(...)`, `sloppy_reward(response: str, golds: list[str]) -> float` (answer string occurs anywhere in the response; the hackable verifier; distinct from 8.7's `sloppy_math`, which inspects the boxed content of a math answer, so both are kept). `golds` is `[item['reference']]` plus aliases.
  - `load_book_search_qa(split) -> list[dict]` (items follow the 2.5 schema, with `reference` as the short answer; `gold` chunk ids are filled by `d2l.gold_chunk_ids(item, corpus)` and stored as `gold_chunks`). Data: `DATA_HUB['lm-book-search-qa']`, `['lm-search-agent-logs']`.
- **Models and data:**
  - T3: `Qwen/Qwen3-1.7B` (revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`); T1: `Qwen/Qwen3-0.6B` (`c1899de289a04d12100db370d81485cdf75e47ca`); question synthesis: `Qwen/Qwen3-8B` (`b968826d9c46dd6066d109eabc6255188de91218`, T2, vLLM); all Apache-2.0. Qwen3.5-0.8B is a prefix-stable alternative (10.1) selectable by a config flag.
  - Environment: R3 through `BookRetriever`. `lm-book-search-qa` is synthesized in this section (recipe below); target sizes at least 3,000 train, 300 dev, 300 test after filtering (to be measured); split by source section, not by question.
- **Packages:** `vllm` (PyPI 0.32.0 at the time of writing; pin), `transformers`, `torch`, optionally `verl` or `areal` for T3 (pin the commit). All T3 code paths run on A100: bf16, no FP8 and no FlashAttention-3.

#### Narrative arc

1. **The question.** Opening cell: run the 10.1 tool loop with `search_book` on 100 `dev` questions using Qwen3-1.7B zero-shot. Print exact match (EM), the mean number of searches, and the fraction of episodes that end without a final answer (to be measured). Next to it, the single-turn RAG baseline (retrieve with the question, answer once). The symptom: prompting alone uses the tool poorly (few or malformed searches, or too many), and the gap to what the task allows is unknown. This is what training is for.
2. **10.4.1 A multi-turn episode as an MDP.**
   - Table `tab_lm_arl_notation`: the symbols of `tab_rl_notation_contract` reused unchanged ($x$, $y$, $\tau$, $\mu_0$, $\pi_\theta$, $\pi_{\textrm{ref}}$, $\hat G_t$, $A$, $K$, $\rho_t$, $\epsilon$, $\beta$, $D_{\textrm{KL}}$, $\mathbf 1(\cdot)$), then the symbols this section adds: $M$ (number of turns), $m$ (turn index), $h_m$ (history before turn $m$), $y^{(m)}$ (tokens the policy samples in turn $m$), $o^{(m)}$ (tokens the environment returns), $P_{\textrm{env}}$, $\kappa_t \in \{0,1\}$ (loss mask), $r_m$ (turn reward). The clip half-width is $\epsilon$ and no numerical constant is called $\epsilon$.
   - The transition kernel of `sec_rl_sequences` was concatenation; here, when the policy emits end-of-turn, the environment appends $o^{(m)} \sim P_{\textrm{env}}(\cdot \mid h_m, y^{(m)})$. A deterministic tool (our retriever is deterministic for a fixed query) still produces tokens the policy did not sample.
   - Margin: the Markov state is the full token history, observations included; truncation versus termination (`subsec_mdp-termination`) for the turn limit.
3. **10.4.2 The policy gradient with environment tokens (derivation D1).** The statement, the mask, the bias of the unmasked loss; the numerical check on a tabular toy and on a tiny transformer.
4. **10.4.3 Token-in-token-out rollouts.**
   - The problem: Qwen3's template re-renders history differently from what was sampled (10.1 prefix-stability check), and decode-then-encode of a sampled sequence need not reproduce the sampled ids. Rule: build training sequences by concatenating the sampled ids and the ids of the observation text, never by re-rendering and re-tokenizing the transcript.
   - Observation formatting: the same text the template would render for a tool result (`\n<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`), tokenized with special-token parsing off for the retrieved text itself. Otherwise a passage containing the string `<|im_end|>` forges a turn boundary (correctness now, an attack in 10.5).
   - The template-drift gap: the trained model is evaluated under the append-only prompts it was trained on and under the standard re-rendered prompts; the notebook reports both (to be measured).
   - `assemble_trajectory`, `search_rollout`, the `SearchEnv` interface; old log-probabilities come from the rollout engine, and a cell recomputes them with the trainer model to measure the train–inference mismatch (mean absolute difference per token; to be measured).
5. **10.4.4 Rewards and credit.**
   - Trajectory-level reward: exact match of the text after `Final answer:`; no answer by the turn limit gets 0 (episodes are not dropped).
   - Turn-level rewards: shaping examples (a retrieval-hit bonus when a returned `Hit.chunk_id` is in `gold_chunk_ids(item, corpus)`) and the reward-to-go weighting for tokens of turn $m$; the causality argument for dropping earlier rewards (exercise).
   - The group baseline: $K = 8$ per question; groups with all-equal rewards carry zero advantage (waste); dynamic sampling (8.5) is used if the notebook measures that more than half of groups are degenerate.
   - `tab_lm_arl_rewards`: EM, F1, retrieval-hit bonus, sloppy contains-match; what each rewards and how it can be gamed.
6. **10.4.5 Environments as code.**
   - Search: in-process function, milliseconds. Software engineering: a container or a sandbox per rollout with repository, tests and a time limit; throughput is bounded by environment seconds per episode. The notebook measures `BookRetriever.search` latency and `Sandbox.run('pytest -q')` latency on `repair30` tasks (10.3) and computes the rollout idle fraction `E / (E + G)` for synchronous rollouts, where `E` is environment seconds and `G` is batched generation seconds per episode.
   - Asynchronous long-horizon rollouts: an episode may span several policy versions; storing behavior log-probabilities per token keeps the ratio $\rho_t$ correct; a staleness cap discards old trajectories. System design is 8.6's; a cell shows the per-token behavior-version bookkeeping, no new infrastructure.
   - Task and environment synthesis: the recipe for `lm-book-search-qa` (below) as a small instance; SWE-smith and SWE-Gym for repositories (bug injection and execution-verified tasks), and the data-synthesis pipeline of Kimi K2 as a large instance (one paragraph, no numbers); difficulty filtering by pass rate (keep tasks with success strictly between 0 and 1 under the current policy).
   - Search agents (Search-R1) and software-engineering agents (SWE-RL: a rule-based reward from patch similarity on mined pull requests, no execution) as the two families, with what their rewards can and cannot see.
7. **10.4.6 Training the search agent** (headline experiment, below).
8. **10.4.7 Reward hacking in agentic settings.** The sloppy-verifier probe (below); examples with test tampering (`repair30` flags it), tool misuse and environment bugs; the mitigations: hidden verifiers, strict matching, tamper checks, monitoring of response length and tool-use statistics, KL to the reference (`sec_regularized`).

#### The question set (`lm-book-search-qa` recipe)

1. Candidates: single R3 chunks (60% of questions, one-hop) and pairs of chunks of the same chapter linked by a `:numref:` cross-reference or sharing a heading ancestor (40%, two-hop).
2. Generation: Qwen3-8B (vLLM, T2) writes a question and a short answer (at most 5 words) that must occur verbatim in the gold chunk(s).
3. Filters: (i) verbatim check; (ii) closed-book filter: drop questions that Qwen3-1.7B answers correctly in at least 2 of 4 samples without retrieval; (iii) retrievability: a gold chunk appears in the top 10 of `BookRetriever` for the question; (iv) near-duplicate removal; (v) manual audit of 100 items, with the error rate reported (to be measured).
4. Split by source section (80/10/10) so that train, dev and test share no gold chunk; `d2l.book_qa(split='test')` is used as the test set so that results are comparable to 2.5; the synthesized items fill `train` and `dev`.
5. Manifest: generator revision, prompts, filter counts per stage.

#### Derivations

**D1. The policy gradient of a multi-turn episode in which only the agent's tokens are actions, and the loss mask.**

- *Statement.* Let an episode be $\tau = (x, y^{(1)}, o^{(1)}, \ldots, o^{(M-1)}, y^{(M)})$ with $h_m = (x, y^{(1)}, o^{(1)}, \ldots, o^{(m-1)})$, $y^{(m)} \sim \pi_\theta(\cdot \mid h_m)$ and $o^{(m)} \sim P_{\textrm{env}}(\cdot \mid h_m, y^{(m)})$, where $P_{\textrm{env}}$ does not depend on $\theta$. Let $R(\tau)$ be a terminal reward and $b$ a baseline. Then
  $$\nabla_\theta J = E\Big[(R(\tau) - b)\sum_{m=1}^{M}\sum_{t=1}^{|y^{(m)}|} \nabla_\theta \log \pi_\theta\big(y^{(m)}_t \mid h_m, y^{(m)}_{<t}\big)\Big],$$
  with no term for the observation tokens (`eq_lm_arl_mt_gradient`). A loss over the concatenated sequence with weight $\kappa_t = \mathbf 1(\text{token } t \text{ was sampled by the policy})$ computes this sum exactly. A loss without the mask adds $E[(R - b)\sum_{m,t}\nabla_\theta \log \pi_\theta(o^{(m)}_t \mid \cdot)]$, which is nonzero in general.
- *Assumptions.* Causal attention; sampled ids are the ids in the training sequence (token-in-token-out); $P_{\textrm{env}}$ independent of $\theta$; $\gamma = 1$; $b$ depends only on $x$ (or on $h_m$ for turn $m$).
- *Steps.*
  1. By the chain rule and `eq_traj_prob`, $P(\tau; \theta) = \mu_0(x)\prod_{m=1}^{M}\pi_\theta(y^{(m)} \mid h_m)\prod_{m=1}^{M-1}P_{\textrm{env}}(o^{(m)} \mid h_m, y^{(m)})$.
  2. Each turn factorizes into token factors $\prod_t \pi_\theta(y^{(m)}_t \mid h_m, y^{(m)}_{<t})$ (`eq_seq_factorization`).
  3. Take $\nabla_\theta \log$: the environment factors and $\mu_0$ do not depend on $\theta$, so only agent-token scores remain.
  4. The score-function identity gives $\nabla_\theta J = E[R(\tau)\,\nabla_\theta \log P(\tau;\theta)]$, which is the displayed form with $b = 0$.
  5. Baselines: for fixed $h_m$, $E_{y^{(m)}}[\nabla_\theta \log \pi_\theta(y^{(m)} \mid h_m)] = 0$, so $b(h_m)$ multiplies a zero-mean score and leaves the expectation unchanged (the zero-mean lemma of `sec_baselines`). A baseline that depends on $o^{(m)}$ or on later tokens is not allowed: $E[f(o^{(m)})\nabla_\theta \log \pi_\theta(y^{(m)} \mid h_m)] = \nabla_\theta E[f(o^{(m)})] \neq 0$ in general (exercise).
  6. Mask equivalence: one forward pass over the concatenated ids yields at position $t$ the distribution $\pi_\theta(\cdot \mid \text{prefix}_t)$ by causality; for an agent token the prefix is $(h_m, y^{(m)}_{<t})$. Weighting each position by $\kappa_t (R - b)$ therefore reproduces step 3's sum.
  7. Unmasked: the observation positions contribute $(R - b)\nabla_\theta\log\pi_\theta(o^{(m)}_t \mid \text{prefix}_t)$. Its expectation does not vanish because $R$ depends on $o$ (a good retrieval raises the reward). Trajectories with positive weight raise the model's own likelihood of the retrieved text, negative ones lower it: the model is trained to predict, and then to hallucinate, observations.
  8. PPO and GRPO versions: the ratio $\rho_t = \pi_\theta/\pi_{\theta_{\textrm{old}}}$ and the clip are applied to positions with $\kappa_t = 1$ only; the group advantage $A = (r_j - \mu)/(\sigma + 10^{-8})$ of `sec_rl_sequences` multiplies every agent token of trajectory $j$. With the length normalization `agg`, the denominators count agent tokens only: `token_mean` divides by $\sum \kappa_t$ over the batch, `seq_mean` by $\sum_t \kappa_t$ per sequence, `const` by a fixed length.
  9. Turn-level rewards $r_m$ with $\gamma = 1$: tokens of turn $m$ are weighted by the reward-to-go $\sum_{m' \ge m} r_{m'} - b(h_m)$; terms with $m' < m$ drop out because $r_{m'}$ is a function of $h_m$ and so has zero covariance with the score of $y^{(m)}$ given $h_m$.
- *Final formulas.* `eq_lm_arl_mt_gradient` above; masked loss $\mathcal L(\theta) = -\frac{1}{N}\sum_{i}\sum_{t}\kappa_{i,t}\, \min\!\big(\rho_{i,t}A_i,\ \textrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)A_i\big)$ with $N = \sum_{i,t}\kappa_{i,t}$ for `token_mean`.
- *Numerical checks.*
  1. Cell `arl-tabular-check` (T0, exact): a two-turn toy with a query softmax $\theta_1$ over 3 queries, a stochastic observation $o\in\{0,1\}$ with a fixed table $P_{\textrm{env}}(o \mid q)$, an answer softmax $\theta_2[o]$, reward $\mathbf 1[a = f(q, o)]$, and a third parameter block $\theta_3$ that gives the model its own distribution over $o$ given $q$ (the "unmasked" score). Enumerate all $(q, o, a)$ to compute the exact $\nabla J$; assert that the masked estimator's expectation equals it to 1e-12 and print the norm of the difference for the unmasked estimator (nonzero).
  2. Cell `arl-transformer-check` (T0): a tiny random `Qwen3Config` model (2 layers, hidden 64, fp32, CPU); one 3-turn trajectory with environment tokens; the gradient of the masked loss on the concatenated sequence equals the sum of per-turn gradients obtained from separate forward passes on `(h_m, y^{(m)})`; assert relative difference below 1e-5; assert that adding environment positions to the mask changes the gradient by more than 1e-3 relative.
  3. Cell `arl-retokenization`: on 200 rollouts count the fraction whose decode-then-encode ids differ from the sampled ids, and compare trainer log-probabilities with engine log-probabilities (to be measured).

#### From-scratch implementation

Budget: about 150 lines.

| Component | Lines | Contract |
|---|---|---|
| `assemble_trajectory` | 25 | concatenation, mask, behavior log-probabilities aligned with `token_ids` |
| `masked_pg_loss` (uses 8.5's `group_advantages`) | 40 | three aggregations; stats: clip fraction, mean ratio, masked-token count |
| `SearchEnv` and observation formatting | 40 | parses `search(query)`, returns formatted top-`k` passages truncated to `obs_tokens`, handles malformed calls as errors (no reward hack by crashing) |
| `search_rollout` | 35 | lockstep batched turns; episodes finish at a final answer or `max_turns` |
| rewards | 10 | `em_reward`, `f1_reward`, `sloppy_reward` |

Tests (seeded):

1. D1 checks 1 and 2 above.
2. Mask test: for a scripted rollout, `loss_mask.sum()` equals the sum of the sampled-token counts; every observation position has mask 0; the first token after an observation block is masked 1.
3. `masked_pg_loss` reduces to the plain GRPO token loss when the mask is all ones (compare with a 10-line reference implementation); with `clip_eps` large it reduces to $-A\sum\log\pi$ (gradient equals the policy gradient at $\theta = \theta_{\textrm{old}}$, as in the book's capstone A).
4. 8.5's `d2l.group_advantages` on a `(G, K)` tensor is zero-mean within each group and zero for degenerate groups (checked on this section's reward tensors).
5. `SearchEnv`: a passage containing the literal text `<|im_end|>` does not produce the special token id in `token_ids` (special-token parsing off).
6. Rollout determinism at temperature 0 (same ids twice).

#### In production

- **verl** (multi-turn rollout with tool interaction in its rollout workers) and **AReaL** (asynchronous agent training) are the two production routes named by v5. Use them for the T3 run if Chapter 8's trainer does not take loss masks; check each library's multi-turn documentation for how it marks environment tokens. The from-scratch `masked_pg_loss` is the oracle: feed the same trajectories to both and compare per-token losses (tolerance 1e-4 in fp32, to be confirmed).
- **Search-R1's code** is the reference implementation of retrieved-token masking and outcome-only rewards; **RAGEN** (StarPO) and **SkyRL** are alternative multi-turn frameworks; **Agent Lightning** decouples agent execution from the trainer by exporting transitions (the interface idea behind `Trajectory`).
- **vLLM** rollouts return per-token log-probabilities; compare them with trainer log-probabilities (D1 check 3).
- Hardware: the T3 run uses A100 40 GB with bf16 and no FP8 (one 8× A100 node, plans fit 8× 40 GB; interconnect as recorded by `nvidia-smi topo -m`); the rollout/training split (for example 2 GPUs for vLLM replicas and 6 for training, or colocated) follows 8.6; the notebook states the choice and prints GPU utilization per phase.

#### Headline experiment

- **Hypothesis.** A search agent trained with GRPO on outcome rewards, with environment tokens masked, answers more `test` questions exactly than (a) Qwen3-1.7B prompted with the same tool and (b) single-turn RAG with the same retriever (2.5); it also learns to issue more than one search on multi-hop questions. Direction only for the size of the gain.
- **Setup (T3).**
  - Model Qwen3-1.7B, non-thinking, append-only prompts; 8× A100 40 GB, bf16 (FSDP2 trainer; the memory arithmetic is in the Budget line).
  - Rollouts: `max_turns = 4`, `k = 3`, `obs_tokens = 200`, `max_new_tokens = 256` per turn, temperature 1.0, 64 prompts per step with $K = 8$ (512 trajectories per step).
  - Optimizer: AdamW, learning rate 1e-6 constant, 10-step warm-up, gradient clipping at 1.0, clip half-width $\epsilon = 0.2$, $\beta = 0$ (a $\beta = 0.001$ variant is an exercise), `loss_agg = 'token_mean'`, one policy update per rollout batch.
  - Steps: as many as fit in the 2-hour cap (the notebook sets `max_wallclock_s`; expected on the order of 100 steps; to be measured).
  - Evaluation: every 10 steps on `dev` (greedy, 300 questions); final on `test` for four conditions: closed-book, prompted agent, single-turn RAG, RL agent.
  - Seeds: one T3 training run; run-to-run spread is estimated from the three T1 runs (0.6B) with different seeds and from bootstrap over test questions.
- **Setup (T1 reduction).** Qwen3-0.6B, 16 prompts × $K = 8$ = 128 trajectories per step, `max_turns = 3`, `obs_tokens = 120`, 40 steps, rollouts with the same engine abstraction (vLLM colocated at a low memory fraction if it fits, otherwise batched HF generation). Same code, flag `--tier t1`.
- **Procedure.** Evaluate base conditions; train; log reward mean, EM on dev, mean searches per episode, fraction without final answer, response length, clip fraction, mean KL to the initial policy, and the mean log-probability of observation tokens under the policy (the monitor for D1); evaluate the four conditions on `test`.
- **Metrics.** EM and F1 on `test`; gold-passage recall in retrieved results; mean searches; summed prompt tokens per question (10.3 ledger); seconds per question.
- **Plots.**
  - `fig_lm_arl_curves`: training reward and dev EM versus step (T3 run, one line) with the three T1 runs as a separate panel and their min–max envelope.
  - `fig_lm_arl_comparison`: bars of test EM for the four conditions with 95% bootstrap intervals, and a second panel of mean searches and prompt tokens per question.
  - `fig_lm_arl_episode`: one real episode as a token timeline, agent tokens in one color and environment tokens in another, with the loss mask drawn as a strip underneath.
  - `fig_lm_arl_throughput`: environment latency (search versus sandbox) and the idle fraction versus environment seconds.
- **Expected outcome.** Direction as above. For scale of expectation only: Search-R1 reports gains of 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over RAG baselines across seven QA datasets under its setup. The book QA task, model and corpus differ, so no magnitude is predicted here. A floor risk exists for the 0.6B run (EM near 0 gives no signal); see Risks.
- **Noise check.** Paired bootstrap over test questions between conditions (cluster = source section); the three T1 seeds give the envelope for training curves; the T3 headline difference counts only if its interval excludes 0 and exceeds the T1 spread in relative terms.
- **Probes (T1, labelled; offline runs; no claims beyond their intervals).**
  1. *Mask ablation*: `loss_mask` on versus an all-ones mask over the same trajectories; 3 seeds each; report reward, dev EM, and the mean log-probability of observation tokens. D1 predicts that the observation log-probability drifts upward under the unmasked loss when advantages correlate with observations; the sign and size are to be measured.
  2. *Verifier hacking*: `em_reward` versus `sloppy_reward` (the answer string occurring anywhere in the response); 2 seeds each; report dev EM (strict), the sloppy score, response length, and the fraction of responses that list three or more candidate answers.
  3. *Aggregation*: `token_mean` versus `seq_mean` versus `const` (exercise measure).
- **Artifacts committed.** `lm-book-search-qa` (splits and manifest), `lm-search-agent-logs` (T3 metrics, 200 sampled rollouts with token ids, masks and rewards, test-set trajectories for the four conditions, the T1 0.6B checkpoint (about 1.2 GB in bf16) and its 3-seed logs). The 1.7B checkpoint (3.4 GB in bf16) goes to the model hub rather than `DATA_HUB` (open question).

#### Figures and tables

- `fig_lm_arl_episode`, `fig_lm_arl_curves`, `fig_lm_arl_comparison`, `fig_lm_arl_throughput`, `fig_lm_arl_mask` (mask ablation panel: reward and observation log-probability versus step), `fig_lm_arl_hack` (strict EM and response length versus step for the two verifiers).
- `tab_lm_arl_notation`, `tab_lm_arl_rewards`, `tab_lm_arl_qa_filters` (counts after each synthesis stage), `tab_lm_arl_conditions` (the four test conditions with EM, F1, searches, tokens).

#### Pitfalls and misconceptions

- "Deterministic tools need no mask." The tokens are still not samples from the policy.
- Re-rendering history with the chat template changes tokens for Qwen3; always concatenate sampled ids.
- Tokenizing retrieved text with special-token parsing on lets a document forge turn boundaries.
- Trainer and rollout-engine log-probabilities differ (kernels, precision); decide which is "old" and say so; large gaps bias the ratio.
- A truncated or malformed episode gets reward 0 and stays in the batch; dropping it rewards giving up.
- `seq_mean` length normalization rewards shorter responses and under-penalizes long failures; multi-turn agents add turns, so the effect shows up in the number of searches.
- With a strict outcome reward and a weak model, most groups have zero advantage; check the degenerate-group fraction before blaming the algorithm.
- Reward curves from a single run are not evidence; the T1 envelope exists for that reason.
- Environment throughput, not GPU speed, can set the training speed (software-engineering environments).

#### Summary (draft bullets)

- A multi-turn episode interleaves tokens the policy samples with tokens the environment writes; only the former are actions.
- The policy gradient has no term for environment tokens, so the loss mask is part of the algorithm, not an engineering detail; the unmasked loss adds a biased term that trains the model to predict observations.
- Rollouts must be token-in-token-out: sampled ids are the training ids, and observation text is tokenized without special-token parsing.
- GRPO needs only trajectory-level outcome rewards; turn-level shaping can speed learning and open new ways to hack the reward.
- Environments set the throughput, and tasks must be verifiable, retrievable and of intermediate difficulty; synthesizing them is part of the training recipe.
- Evidence needs intervals: paired bootstrap over questions and several seeds at the small scale.

#### Exercises

1. **Check.** A transcript has 120 prompt tokens, then agent 30, observation 150, agent 20, observation 150, agent 40. What fraction of non-prompt tokens carry loss? Under `token_mean`, what is the per-token weight when the batch contains this trajectory and one 60-token single-turn trajectory with the same $|A|$? *Sketch:* loss positions 90 of 390 non-prompt tokens (23%); `token_mean` divides by 150 masked tokens for both, so each masked token has weight $|A|/150$ regardless of its trajectory.
2. **Check.** The retriever is deterministic. Is the loss mask still needed? What changes if the tool is a calculator? *Sketch:* yes; observation tokens are not drawn from $\pi_\theta$ and their score term is not part of the objective gradient; the calculator's output is a deterministic function of the call and is also masked.
3. **Derive.** Show that $E[f(o^{(m)})\,\nabla_\theta\log\pi_\theta(y^{(m)}\mid h_m)] = \nabla_\theta E[f(o^{(m)})]$ for $o^{(m)} \sim P_{\textrm{env}}(\cdot\mid h_m, y^{(m)})$, and conclude that a baseline depending on the observation is biased. *Sketch:* write the expectation over $y$ then $o$, $E_y[\nabla\log\pi(y)\,g(y)]=\nabla E_y[g(y)]$ with $g(y) = E_o[f(o)\mid h_m,y]$ since $g$ does not depend on $\theta$.
4. **Derive.** Prove that dropping earlier turn rewards from the weight of turn $m$ does not change the expected gradient. *Sketch:* $r_{m'}$ for $m' < m$ is a function of $h_m$; conditional on $h_m$ the score of $y^{(m)}$ has mean zero.
5. **Measure.** Compare `token_mean`, `seq_mean` and `const` on the T1 setup (3 seeds); predict the effect on mean searches and response length first.
6. **Measure.** Add a retrieval-hit bonus of 0.1 for the first search whose results contain a gold passage; predict whether learning speeds up and whether searches per episode inflate; measure with 3 seeds.
7. **Open.** Train a software-engineering agent on `repair30` with the execution reward of 10.3's grader, using `Sandbox` inside the rollout; report throughput limits and tamper attempts. Or add a learned verifier for answers that are not short strings.

#### Further reading

- Papers: [Search-R1](https://arxiv.org/abs/2503.09516) (retrieved-token masking, outcome reward; the 41% and 20% figures); [SWE-RL](https://arxiv.org/abs/2502.18449) (rule-based reward on mined pull requests); [Kimi K2](https://arxiv.org/abs/2507.20534) (large-scale agentic data synthesis); new: [ReTool](https://arxiv.org/abs/2504.11536) (RL for code-interpreter use); new: [ToolRL](https://arxiv.org/abs/2504.13958) (reward design for tool use); new: [RAGEN](https://arxiv.org/abs/2504.20073) (multi-turn RL and its instabilities); new: [Agent Lightning](https://arxiv.org/abs/2508.03680) (decoupling agents from training); [SWE-smith](https://arxiv.org/abs/2504.21798) (environment synthesis; also in 10.3).
- Tutorials and docs: Berkeley [Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25) (agent RL lectures); the book's `sec_rl_sequences` exercises; the multi-turn sections of the verl and AReaL documentation (check current URLs at authoring).
- Code bases: [verl](https://github.com/verl-project/verl); [AReaL](https://github.com/inclusionAI/AReaL); new: [Search-R1](https://github.com/PeterGriffinJin/Search-R1); new: [RAGEN](https://github.com/RAGEN-AI/RAGEN); new: [SkyRL](https://github.com/NovaSky-AI/SkyRL).

#### Risks and open decisions

- **Chapter 8 trainer interface.** If it cannot take external loss masks, use verl or AReaL for T3 and the section's loss for T1 and the oracle tests; the editor should align the two blueprints.
- **Floor effect at 0.6B.** If base EM with the tool is near 0, the outcome reward gives no signal. Fallbacks: a cold-start SFT on a few hundred execution-verified trajectories (teacher Qwen3-8B, the 10.1 generator style, rejection-sampled by EM), a shaped reward for the first 10 steps, or Qwen3-1.7B at T1 with fewer steps.
- **Question synthesis quality.** If the manual audit error rate exceeds the threshold the editor sets (propose 10%), tighten filter (ii) or add a second-model answer check.
- **vLLM colocation in a 15 GiB slot** may not fit with the trainer (a whole 24 GB GPU gives more room); fallback: batched HF generation and smaller `K`, or sequential rollout then train with weights offloaded.
- **Train–inference mismatch** may need a truncated importance correction (8.6) at T3; the notebook reports the measured mismatch and applies the correction only if the gap is large.
- **Checkpoint size** for `DATA_HUB`; see above.
- **Qwen3 template drift** may lower the append-only versus re-rendered score gap; the Qwen3.5 flag exists to remove it.

### 10.5 Evaluating and securing agents

- **Label:** `sec_lm_agent_eval` · **Tier:** T1 (the reliability analysis of committed trajectories is T0; T2 optional extensions behind `--tier t2`) · **Budget:** T0 analysis cells take seconds on CPU; the fresh BFCL single-turn pass^k run (Qwen3-1.7B, ~300 items × 8 trials, temperature 0.7, batch 64) ≤ ~10 min on T1 (to be measured); the interactive injection cells (100 episodes per arm, 2 arms) ≤ ~15 min; the full injection grid (about 4,000 episodes, design cap 2 GPU-hours) and the 8-trial `repair30` run (240 episodes) are offline and committed; T2 extensions: ≤ a few hours each · **Codex units:** none (v5 `[new]`).
- **Builds on (book):**
  - `sec_mdl-statistics`: bias and variance of estimators, the bootstrap and confidence intervals, comparing two models; this section applies them and does not re-derive them.
  - `sec_rl_sequences` (`subsec_rl_sequences_hacking`): a benchmark is a verifier, and verifiers can be gamed; the security half is the same mechanism with an adversary.
  - `sec_kv-cache`, `sec_perf_model`: only for pricing tokens (one sentence each).
- **Builds on (this part):**
  - 1.4: `d2l.pass_at_k(n, c, k)`; 1.5: `d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None)` (returns `BootstrapResult`, `.mean` and `.ci`) and `d2l.EvalResult` for per-item scores; 9.7: `d2l.pareto_front(points, x='tokens', y='acc') -> DataFrame` (lower x better, higher y better).
  - 10.1: `d2l.load_bfcl`, `d2l.bfcl_match`, `d2l.tool_chat`, `d2l.ToolRegistry`, `d2l.search_book`; 10.3: `d2l.Episode`, `d2l.run_batch`, `d2l.Sandbox` (its `capabilities()` gates the exfiltration experiments), `d2l.load_repair30`, `d2l.grade_repair`, `d2l.cost_report`, the committed `lm-agent-runs-ablation`; 10.4 (optional): `d2l.SearchEnv`, `d2l.load_book_search_qa`, the trained search agent.
  - 2.4: `d2l.BookRetriever` (the corpus that gets poisoned).
- **Exports:**
  - `pass_pow_k(n, c, k) -> float`: unbiased pass^k estimator `C(c, k) / C(n, k)`; accepts arrays of `c`.
  - `agent_report(outcomes: ndarray[tasks, trials], ks=(1, 2, 4, 8), n_boot=10000, seed=0) -> DataFrame`: pass@1, pass@k, pass^k with cluster-bootstrap intervals (tasks resampled; an option resamples trials within tasks as a second level).
  - `beta_binomial_fit(c: ndarray, n: int) -> tuple[float, float]` (maximum likelihood) and `beta_passk(alpha, beta, k, kind='pow') -> float` (`kind in {'pow', 'at'}`).
  - `variance_decomposition(outcomes) -> dict` (between-task and within-task variance components and the implied variance of the mean).
  - `AgentCost` dataclass (`prompt_tokens`, `cached_tokens`, `gen_tokens`, `tool_s`, `model_s`) with `.usd(prices: dict) -> float` (prices supplied by the caller; none are hard-coded). The cost-accuracy frontier is 9.7's `d2l.pareto_front` (imported, not redefined); if the 2-D ndarray mask form is needed, the section defines `agent_pareto(points: ndarray[n, 2]) -> ndarray[bool]` (minimize cost, maximize score) as a thin wrapper.
  - `load_tau_trajectories(name: str, commit: str = TAU_BENCH_COMMIT) -> DataFrame` (`task_id`, `trial`, `reward`, `n_messages`, `user_cost`) for the four historical files of τ-bench.
  - `load_injection_suite() -> dict`; `poison_chunks(chunks, template, goal, seed) -> list[dict]`; `spotlight(text: str, mode: str, key: str) -> str` (`mode in {'delimit', 'datamark'}`); `ToolPolicy(allowed: set[str])` with `.check(call: ToolCall) -> bool`; `attack_success(episode, goal) -> bool`.
  - Data: `DATA_HUB['lm-injection-suite']`, `['lm-agent-eval-trials']`. Licence of `lm-injection-suite`: its poisoned-chunk patches modify R3, the book's own text, so they are CC BY-SA 4.0; the injection templates are book-written prose under the same licence; the injected repair variants are variants of `repair30` and stay MIT-0; the suite's manifest records the licence per file.
- **Models and data:**
  - Agents: `Qwen/Qwen3-1.7B` for BFCL pass^k, `Qwen/Qwen3-4B` for `repair30` and for injection (revisions as in 10.1 and 10.3). T2 extensions: `Qwen/Qwen3-8B` (8.19 B parameters, 16.4 GB in bf16) as agent and user simulator (one served instance playing both roles on one A100 40 GB; two separate 8B instances would need 32.8 GB of weights and do not fit one 40 GB card); `Qwen/Qwen3-30B-A3B` (30.5 B parameters, 61 GB in bf16, KV cache 98,304 bytes per token from its config: 48 layers, 4 KV heads, head dimension 128) for the SWE-bench subset on a T2 pair (two A100 40 GB joined by an NVLink bridge), served by vLLM with tensor-parallel size 2: 30.5 GB of weights per GPU, leaving about 20 GB across the pair for KV cache and activations (at most about 200k cached tokens before activations; to be measured), reached over HTTP from the notebook's environment.
  - τ-bench historical trajectories: `sierra-research/tau-bench` at commit `59a200c6d575d595120f1cb70fea53cef0632f6b` (MIT), files `historical_trajectories/{gpt-4o,sonnet-35-new}-{airline,retail}.json` (4.1, 10.8, 10.9 and 26.5 MB). The blueprint author's check on `gpt-4o-airline.json`: 200 records = 50 tasks × 4 trials; recomputed pass^1…pass^4 = 0.420, 0.273, 0.220, 0.200, equal to the repository's README table. Note the README states that the repository's tasks are outdated and points to the τ²/τ³-bench repository.
  - BFCL v3 subset ids from `lm-bfcl-subset` (10.1; dataset Apache-2.0; `bfcl-eval` Apache-2.0).
  - Optional T2: `sierra-research/tau2-bench` (MIT; Python ≥ 3.12 per its README badge; `text-v1` has 50 frozen tasks per domain for airline, retail and telecom); `SWE-bench/SWE-bench_Verified` (revision `78f471bf655a3137b2e8a75af1501690ec009ec3`, 500 test instances, no license tag on the dataset card, the harness is MIT); AgentDojo (MIT).
- **Packages:** `numpy`, `scipy` (Beta functions), `pandas`; optional T2: `swebench` (PyPI 5.0.2, MIT), `mini-swe-agent` (2.4.6, MIT), `bfcl-eval`, the `tau2` package from its repository, Docker. Check `Sandbox.capabilities()` before any experiment that lets an agent run shell commands on injected content.

#### Narrative arc

1. **The question.** Opening cell (T0): `load_tau_trajectories('gpt-4o-airline')`; print pass@1 = 0.42 and pass^4 = 0.20 computed from the 200 episodes, then pass@4 = 0.72 from the same episodes (the blueprint author's check). Print the histogram of per-task successes among 4 trials (14 tasks at 0, 12 at 1, 10 at 2, 4 at 3, 10 at 4). One trial gives a rate; four trials expose that only 10 of 50 tasks are solved every time. The sentence to land: a demo-quality number and a deployment-quality number come from the same data.
2. **10.5.1 What a score means.**
   - Success checks: database-state comparison with a goal state (τ-bench), hidden tests (SWE-bench, `repair30`), AST matching (BFCL), exact match (10.4). Each is a verifier and has blind spots (`subsec_rl_sequences_hacking`).
   - Validity checklist from the ABC paper, applied to `repair30` in a table `tab_lm_ae_validity`: task validity (solvable, unambiguous), outcome validity (tests detect the fix and nothing else), reporting (seeds, harness, cost). A paragraph on SWE-bench-style harness sensitivity: scaffold, prompt, tool format and step limit can move a score as much as a model change; the 10.3 grid is the in-book evidence (spread across cells versus spread across seeds).
   - Environment reproducibility: pinned images and commits, frozen tasks (τ²-bench `text-v1`), user simulators as part of the test (their model and temperature change scores; the notebook lists what each benchmark pins).
3. **10.5.2 pass@k versus pass^k.** Definitions, unbiased estimators, derivation D1, the Beta model, and `fig_lm_ae_passk` (measured curves from τ-bench trajectories and from `repair30`, with the Beta-binomial prediction).
4. **10.5.3 Variance: tasks, trials, seeds.** Derivation D2; `variance_decomposition` on the τ-bench file and on `repair30`; cluster bootstrap versus naive binomial intervals; how many trials resolve a pass^k difference; paired comparison of two agents on the same tasks.
5. **10.5.4 Cost and latency.** `AgentCost`, the ledger of 10.3 extended to latency percentiles, the Pareto frontier of the 10.3 grid with the "retry" baseline (D3), and a price table supplied by the reader (tokens, GPU-seconds for local models; any currency price must cite a vendor page at authoring time).
6. **10.5.5 Benchmarks you can actually run.** `tab_lm_ae_benchmarks` and the choice of subsets for T1 and T2 (below).
7. **10.5.6 Securing agents.** Threat model; the injection suite; defenses; results; limits.
8. **10.5.7 Putting it together.** A one-page evaluation card for the 10.3 agent: success with interval, pass^k, cost, latency, ASR with and without defense, harness and seed details; this is the deliverable that the book recommends for every agent report (`tab_lm_ae_card`).

#### Benchmarks: availability, license, and what fits which tier

`tab_lm_ae_benchmarks` (all facts checked on 28 Oct 2026 unless marked):

| Benchmark | Measures | Size | License | Needs | Tier and what this section runs |
|---|---|---|---|---|---|
| BFCL v3 (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`, `bfcl-eval`) | function-call correctness by AST match; single-turn, parallel, irrelevance; multi-turn state; v4 adds agentic web search and memory | about 200 to 400 items per single-turn category; multi-turn base about 200 tasks | Apache-2.0 | local model, no network for the single-turn AST categories; web search needs a SerpAPI key | T1: single-turn categories (pass^k with 8 trials). T2 optional: `multi_turn_base` with vLLM. Not run: web search |
| τ-bench historical trajectories (`sierra-research/tau-bench`) | tool-agent-user policy following; DB-state reward; origin of pass^k | 50 airline tasks (checked from the trajectory file) and 115 retail tasks (to be checked) | MIT | none (files only) | T0: recompute and analyze four trajectory files; the section never runs the original benchmark |
| τ²-bench (`sierra-research/tau2-bench`; its README also calls it τ-bench 2.0 and names a τ³-bench lineage) | the same with dual-control telecom, banking knowledge, voice, six languages | `text-v1`: 50 frozen tasks per domain per language | MIT | an agent LLM, a user-simulator LLM, and (README, at fetch time) an API judge for natural-language assertions | T2 optional: `airline`, 50 tasks, agent and user simulator both Qwen3-8B behind local vLLM servers, n ≥ 4 trials; the judge dependency must be resolved first (see Risks) |
| SWE-bench Verified | issue resolution in real repositories | 500 tasks, 12 repositories | code MIT; dataset card has no license tag; underlying repositories carry their own licenses | Docker; harness recommends x86_64, about 120 GB free disk, 16 GB RAM, 8 cores | T2 optional: 30-task subset with mini-swe-agent and Qwen3-30B-A3B; not at T1 |
| Terminal-Bench | terminal tasks in containers | to be checked at authoring | Apache-2.0 (repository `terminal-bench-1`, now under `harbor-framework`) | Docker | not run; discussed |
| WebArena | web tasks on self-hosted sites | 812 tasks (to be checked at authoring) | to be checked | several containers hosting sites | not run; discussed |
| AgentDojo | utility, utility under attack, targeted attack success for tool-using agents | four suites | MIT | a function-calling model endpoint; local-endpoint support to be checked | T2 optional; the section's T1 injection suite plays this role |

SWE-bench Verified subset rule (committed as `swe_verified_t2_subset.json`): from the `test` split, keep instances with `difficulty == '<15 min fix'` (194 instances in the split); keep repositories whose environment images are small enough for the chosen host (to be decided by the build owner after a pull test); sample 30 with a fixed seed stratified by repository; record instance ids, image digests and the harness commit. Reason for the small subset: a 30-task sample has wide intervals (a Wilson half-width near 0.17 at 50% success for one trial) and an 8B to 30B local model is expected to score low, so the exercise is about harness, cost and variance accounting, not about leaderboard numbers.

#### Derivations

**D1. pass@k, pass^k, and what they do as k grows.**

- *Statement.* For a task with per-trial success probability `p` and independent trials, `pass@k = 1 − (1 − p)^k` (at least one success in `k` trials) and `pass^k = p^k` (all `k` succeed). With `n` trials of which `c` succeed, unbiased estimators are `pass@k = 1 − C(n − c, k) / C(n, k)` (`d2l.pass_at_k`) and `pass^k = C(c, k) / C(n, k)` (`pass_pow_k`). Over a task population, benchmark scores are `E[pass@k]` and `E[pass^k]` over tasks. They are mirror images: `pass^k(p) = 1 − pass@k(1 − p)` applied to failure. (i) If every task has the same `p < 1`, `pass^k` decays exponentially and `pass@k` saturates exponentially. (ii) If `p ~ Beta(α, β)` across tasks, `E[p^k] = Π_{j=0}^{k−1} (α + j) / (α + β + j) ~ Γ(α + β) / Γ(α) · k^{−β}` and `1 − E[pass@k] = Π_{j=0}^{k−1} (β + j) / (α + β + j) ~ Γ(α + β) / Γ(β) · k^{−α}` as `k → ∞`: both are power laws, with exponents set by the mass of the success distribution near 1 (`β`) and near 0 (`α`).
- *Assumptions.* Trials are independent given the task (not true if the environment has persistent state or if seeds correlate); `n ≥ k`.
- *Steps.*
  1. The `k` trials are drawn without replacement from the `n` observed: the probability that all of a given `k`-subset succeed is `C(c, k) / C(n, k)` averaged over subsets, which is unbiased for `p^k` because the `k` trials are independent draws (the number of `k`-subsets of successes over all `k`-subsets).
  2. Likewise the probability that a `k`-subset has no success is `C(n − c, k) / C(n, k)`; one minus it is unbiased for `1 − (1 − p)^k`.
  3. Mirror: replace successes by failures.
  4. (i) is immediate from `p^k` and `(1 − p)^k`.
  5. (ii): `E[p^k] = B(α + k, β) / B(α, β)`; expanding the Beta function recursively with `Γ(x + 1) = xΓ(x)` gives the product. The asymptotic follows from `Γ(α + k) / Γ(α + β + k) ~ k^{−β}`.
  6. For the second statement apply the same step to `E[(1 − p)^k]` with the roles of `α` and `β` swapped.
  7. Jensen: `x ↦ x^k` is convex, so `E[p^k] ≥ (E p)^k`; and `x ↦ 1 − (1 − x)^k` is concave, so `E[pass@k] ≤ 1 − (1 − E p)^k`. Heterogeneity across tasks makes `pass^k` decay more slowly than the homogeneous rate and `pass@k` rise more slowly than the homogeneous rate.
- *What follows for the claim "pass^k falls far faster than pass@k rises".* It holds for tasks that are individually reliable but imperfect (homogeneous `p = 0.8`: `pass^8 = 0.168`, `pass@8 = 1 − 0.2^8 = 0.99999744`), the deployment regime. It does not hold in general: for an agent with mass mostly near 0 (the airline file, mean success 0.42) `pass@k` rises more than `pass^k` falls over `k ≤ 4` (+0.30 versus −0.22). The symmetric statement is that each metric is controlled by the tail of the success distribution on its own side. The notebook must show both regimes, one measured and one simulated.
- *Numerical checks* (cells `ae-passk-unbiased`, `ae-beta`): (a) for `n = 6` and every `c`, enumerate all `k`-subsets of a 0/1 vector and compare the average with `pass_pow_k` and `d2l.pass_at_k` (tolerance 1e-12). (b) Monte Carlo (200,000 draws) of `E[p^k]` for `Beta(2, 1)`, `k = 4`: product formula 1/3 versus simulation (blueprint check: 0.3341 against 0.3333). (c) Fit `beta_binomial_fit` to the airline file's `c` values (author's check: `α ≈ 0.65`, `β ≈ 0.85` on a 0.05 grid); predicted `pass^k` for `k = 1..4` (0.433, 0.286, 0.217, 0.176) versus recomputed (0.420, 0.273, 0.220, 0.200); report the differences. (d) On the Beta model, the log–log slope of `E[p^k]` for `k` from 64 to 1024 is within 5% of `−β`.

**D2. Variance across tasks and across trials, and why the naive interval is too narrow.**

- *Statement.* For `N` tasks with success probabilities `p_i` (random draws from a population with mean `μ_p` and variance `σ_t²`) and `n` independent trials per task, the estimator `p̂ = (1 / (N n)) Σ_i Σ_j X_ij` has `Var(p̂) = σ_t² / N + E[p_i (1 − p_i)] / (N n)`. A binomial interval treating all `N n` trials as independent uses `μ_p (1 − μ_p) / (N n)` and is too narrow whenever `σ_t² > 0`, because `μ_p (1 − μ_p) = σ_t² + E[p_i (1 − p_i)]`.
- *Assumptions.* Tasks sampled i.i.d. from a population; trials independent given the task.
- *Steps.*
  1. Law of total variance: `Var(p̂) = Var(E[p̂ | p]) + E[Var(p̂ | p)]`.
  2. `E[p̂ | p] = (1/N) Σ_i p_i`, whose variance is `σ_t² / N`.
  3. `Var(p̂ | p) = (1 / (N² n)) Σ_i p_i (1 − p_i)`, with expectation `E[p_i (1 − p_i)] / (N n)`.
  4. The naive variance is `(σ_t² + E[p_i(1 − p_i)]) / (N n)`: the between-task term is divided by `N n` instead of `N`, so it understates by a factor up to `n`.
  5. For a fixed budget `N n`, the variance is minimized by `n = 1` (all tasks, one trial) when only `pass@1` matters; estimating `pass^k` needs `n ≥ k` and the allocation trades these off.
  6. A paired comparison of agents A and B on the same tasks: `Var(p̂_A − p̂_B)` has the between-task terms reduced by the correlation of `p_{i,A}` and `p_{i,B}`; hence paired intervals are narrower.
- *Final formula.* `Var(p̂) = σ_t²/N + E[p_i(1 − p_i)]/(N n)`; the cluster bootstrap over tasks estimates it without assuming the form.
- *Numerical check* (cell `ae-coverage`): simulate 2,000 experiments with `N = 50`, `n = 4`, `p_i ~ Beta(0.65, 0.85)`; compute the coverage of the nominal 95% intervals from the naive binomial interval, a task-cluster bootstrap, and a nested bootstrap. Expect cluster bootstrap near 95% and naive well below (to be measured). Also compare `variance_decomposition` on the airline file with the formula.

**D3. When does a costlier agent beat a cheap one with retries?**

- *Statement.* If a task has a perfect verifier (tests), agent A with success `p` and cost `c_A` per attempt, retried until verified up to `r = ⌊c_B / c_A⌋` times, succeeds with probability `1 − (1 − p)^r` at cost at most `c_B`. Agent B (success `q`, cost `c_B`) is on the cost–success Pareto frontier against A only if `q > 1 − (1 − p)^r`.
- *Assumptions.* Independent attempts; perfect verifier; cost per attempt constant. Where no verifier exists at deployment (τ-bench), the argument does not apply and the report says so.
- *Steps.* (1) `r` attempts cost `r c_A ≤ c_B`. (2) Success after `r` independent attempts is `1 − (1 − p)^r`. (3) B must exceed this at equal cost.
- *Numerical check* (cell `ae-retry-frontier`): apply to the four cells of the 10.3 grid using their measured per-episode costs in token-equivalents and success rates per task; list which cells are dominated by retries of a cheaper cell (to be measured).

#### From-scratch implementation

Budget: about 150 lines.

| Component | Lines | Contract |
|---|---|---|
| `pass_pow_k`, `agent_report` (cluster bootstrap) | 40 | tasks resampled; optional second level over trials |
| `beta_binomial_fit`, `beta_passk`, `variance_decomposition` | 40 | grid or Newton maximum likelihood; method-of-moments components |
| `AgentCost`, `agent_pareto` (wrapper over 9.7's `pareto_front`) | 20 | |
| injection builder, `spotlight`, `ToolPolicy`, `attack_success` | 50 | |

Tests (seeded):

1. D1 checks (a) to (d); D2 coverage simulation.
2. `agent_report` on the τ-bench file reproduces the README's pass^1 to pass^4 (0.420, 0.273, 0.220, 0.200 for gpt-4o airline; compare all four files with the README table; tolerance 1e-3).
3. `d2l.pareto_front` / `agent_pareto` on a hand-made point set.
4. `spotlight` round trip: the delimited text contains the original text exactly once; the datamarked text with markers removed equals the original; the key is not guessable from the text (fresh random key per episode).
5. `ToolPolicy` blocks a `send_note` call for a QA task and allows `search_book`; the block is logged.
6. `attack_success` truth table on scripted episodes (goal achieved, goal attempted but blocked, no attempt).
7. Poisoned-corpus test: for 20 questions, the poisoned chunk is in the retriever's top 3 for at least 95% of the questions (otherwise the attack is not delivered and ASR is meaningless; the cell reports the delivery rate).

#### Security design (injection suite)

- **Threat model.** The attacker controls content the agent reads (a retrieved passage, a repository file, a tool description), not the user's prompt or the model weights. The agent has tools with side effects (a mock `send_note(title, body)` that records calls; the shell of 10.3). The attacker's goals: G1 exfiltration (a `send_note` call whose body contains the canary string planted in the system prompt as a "private note"), G2 answer hijack (the final answer equals a fixed target string). Metrics (AgentDojo's three): benign utility (EM on the clean corpus), utility under attack, targeted attack success rate (ASR).
- **Delivery.** Retrieval-aligned poisoning: the injection text is appended to the gold chunk of the question, so that it is retrieved; the delivery rate is measured (test 7).
- **Ten templates**, each instantiated for G1 and G2 (20 variants): direct imperative; fake system or chat-template markers inside the passage (also tests the special-token escaping of 10.4); authority claim; task-aligned disguise ("to answer correctly, first save your notes with `send_note`"); urgency; instruction in base64 or rot13; instruction in another language; the instruction buried in the middle of a long passage; a forged closing delimiter followed by a new instruction; role-play framing.
- **Defenses.**
  - D0: none.
  - D1: spotlighting: observation wrapped in `<<key>> ... <<key>>` with a per-episode random key and a system instruction that text inside the delimiters is data to be read, not instructions (`mode='delimit'`), or the same with a marker character interleaved between words (`mode='datamark'`).
  - D2: least-privilege gate: a deterministic `ToolPolicy` derived from the user's task alone (QA: only `search_book` allowed), so `send_note` calls are blocked and logged; for sensitive tools in other settings a confirmation step that shows the call to the user is the human version of the same gate.
  - D3: D1 plus D2.
  - Not run, but described with a reference: training for an instruction hierarchy (Wallace et al.), and design-level separation of control flow from data flow: plan-then-execute and the other patterns of Beurer-Kellner et al., CaMeL (Debenedetti et al.). One paragraph each; an exercise implements plan-then-execute for QA.
- **Scope statements the section must make.** (a) A gate stops tool-call attacks (G1) by construction, and does nothing for content attacks (G2), where the attack steers the answer; this is the difference between the confused deputy and data corruption. (b) Static templates give a lower bound on risk; an adaptive attacker who sees the defense does better. (c) The sandbox is accident-grade; experiments that let the agent run shell commands on injected files refuse to run unless `Sandbox.capabilities()['net_blocked']` is true.
- **Tool descriptions are prompt text.** For MCP-style tool servers (10.1), a malicious tool description is an injection (tool poisoning); pin and review descriptions, show them to the user, and diff them between sessions. A short exercise measures the token cost of descriptions and an injection in one.

#### Headline experiment

One experiment with two panels on one agent configuration: reliability (A) and security (B).

- **Hypotheses.** A: for the 10.3 agent, `pass^k` falls with `k` while `pass@k` rises, and the Beta-binomial model fitted at `n = 8` predicts the curve; the cluster-bootstrap interval of `pass^8` is wide. B: without a defense the ASR is above 0 for at least some templates (to be measured; direction only); the gate reduces G1 ASR to 0 by construction and leaves G2 unchanged; spotlighting reduces ASR by an amount to be measured; every defense has a utility cost to be measured.
- **Setup A.**
  - Agents and data: (A1) the best cell of the 10.3 grid (Qwen3-4B, ACI interface, the policy chosen by the 10.3 results) on `repair30`, 8 trials per task at temperature 0.7 (240 episodes, offline, committed); (A2) Qwen3-1.7B on the BFCL single-turn subset (about 300 items), 8 trials at temperature 0.7, run in the notebook; (A3) the τ-bench historical trajectories as the reference curve.
  - Metrics: pass@1, pass@k and pass^k for `k ∈ {1, 2, 4, 8}` (`k ≤ 4` for A3) with cluster-bootstrap intervals; Beta-binomial fit and prediction; variance components; mean and 95th percentile of latency and tokens.
- **Setup B.**
  - Agent: Qwen3-4B with `search_book` and `send_note` over the poisoned R3 corpus, using `dev` questions from `lm-book-search-qa` (10.4); the 10.4 RL agent is an optional extra arm.
  - Interactive cell: 100 episodes per arm for D0 and D1, templates cycled, seed 0. Offline full grid: 50 questions × 20 variants × 4 defenses (D0 to D3) = 4,000 episodes (design cap 2 GPU-hours; to be measured).
  - Metrics: ASR (G1 and G2 separately), benign utility on the clean corpus, utility under attack, delivery rate, per-template ASR, blocked-call counts.
- **Plots.**
  - `fig_lm_ae_passk`: x = `k`, y = score; series pass@k and pass^k for A1, A2 and A3 (solid) with the Beta-binomial predictions (dashed) and intervals as bands; an inset with the simulated homogeneous case (`p = 0.8`).
  - `fig_lm_ae_variance`: bars of between-task and within-task variance components, and the coverage result of D2.
  - `fig_lm_ae_frontier`: success versus cost (token-equivalents) for the 10.3 grid with the Pareto front and the retry curve of D3.
  - `fig_lm_ae_asr`: left, grouped bars of ASR by defense and goal with intervals; middle, a heatmap of ASR by template and defense; right, ASR against utility under attack (the cost of each defense).
- **Expected outcome.** A: directions from D1 (stated as such). The τ-bench file is the cited magnitude (recomputed in the notebook; the original paper reports that gpt-4o-class agents succeed on under 50% of tasks and fall below 25% in `pass^8` for retail). B: directions only; AgentDojo and the indirect-injection literature report that attacks succeed on tool-using models and that defenses reduce but do not remove them, but no magnitude transfers to a 4B model, so none is stated.
- **Noise check.** Cluster = task for A and = question for B; D2's coverage check justifies the intervals; B's arms are paired on the same `(question, variant)` pairs, so differences use `d2l.paired_bootstrap(..., cluster=question)`; a defense effect counts only if the interval excludes 0. With 4,000 episodes and one sample each, within-episode sampling noise is not estimated separately; state it.
- **T1 reduction.** The notebook's own runs are the A2 and B-interactive cells; A1 and the B grid are committed artifacts.
- **T2 extensions** (behind `--tier t2`, each reported as a separate table with its own caveats): τ²-bench airline, 50 tasks × 4 trials, `pass^k` for `k ≤ 4`; the SWE-bench Verified 30-subset with mini-swe-agent, 1 trial per task plus cost and latency accounting; BFCL `multi_turn_base`.
- **Artifacts.** `lm-agent-eval-trials` (per-task, per-trial outcomes with tokens and seconds for A1 and A2), `lm-injection-suite` (templates, poison patches, canary scheme, the seeds, full-grid outcomes).

#### Figures and tables

- `fig_lm_ae_passk`, `fig_lm_ae_variance`, `fig_lm_ae_frontier`, `fig_lm_ae_asr`; `fig_lm_ae_threat` (diagram: user, agent, tools, retrieved content with the attacker's path and the two gates).
- `tab_lm_ae_benchmarks`, `tab_lm_ae_validity`, `tab_lm_ae_templates` (the ten injection templates with an example each), `tab_lm_ae_card` (the evaluation card of the 10.3 agent).

#### Pitfalls and misconceptions

- A single trial per task reports `pass@1` only; saying "reliable" requires `n ≥ k` trials.
- Treating all trials as independent shrinks intervals (D2); resample tasks.
- `pass@k` with a verifier is a search budget, not reliability; do not mix it with `pass^k` in one headline.
- Harness changes, user-simulator changes and judge changes make scores incomparable with leaderboards; report them.
- A benchmark that you tune on is no longer a test (Goodhart); keep a frozen subset.
- Cost without a stated token convention (cached or not) and without a price source is not a number.
- Zero observed injections is not safety: the interval's upper bound is positive; adaptive attackers are not in the templates.
- A gate with an allow-list is only as good as the task-derived policy: an attack that fits inside the allowed tools still works.
- An injection test where the attack was never delivered (not retrieved) measures nothing; report the delivery rate.
- Running shell-capable agents on injected content without network isolation is unsafe; the notebook refuses.

#### Summary (draft bullets)

- Evaluate an agent with several trials per task: `pass@k` measures reach, `pass^k` measures reliability, and the two are mirror images controlled by opposite tails of the per-task success distribution.
- Between-task variance dominates; intervals must resample tasks, and paired comparisons are narrower than unpaired ones.
- Report cost and latency with the score, with a stated token convention; a more expensive agent must beat a cheaper one with retries (when a verifier exists).
- Benchmarks differ in what they need: BFCL single-turn and trajectory files fit T0 to T1; τ²-bench and SWE-bench need user simulators or Docker and fit T2 at small subsets.
- Indirect prompt injection exploits the fact that tool outputs enter the prompt; spotlighting lowers risk, gates remove tool-call attacks by construction, and neither stops attacks on content.
- A sound evaluation card lists harness, seeds, intervals, cost and the attack results together.

#### Exercises

1. **Check.** An agent has per-task success counts `c = (8, 8, 0, 4)` out of `n = 8`. Compute `pass@1`, `pass^2` and `pass@2`. *Sketch:* `p̂ = (1, 1, 0, 0.5)`, `pass@1 = 0.625`; `pass^2 = (28/28 + 28/28 + 0 + 6/28) / 4 = 0.5536`; `pass@2 = (1 + 1 + 0 + (1 − 6/28)) / 4 = 0.6964`.
2. **Check.** Why can `pass^k` be computed from `n = 8` trials for `k = 8` only with a single value per task, and what does that do to the interval? *Sketch:* `C(c, 8) / C(8, 8)` is 1 only if `c = 8`; the estimator is a 0/1 per task, so the variance is that of a proportion over tasks.
3. **Derive.** Prove `E[p^k] = Π (α + j) / (α + β + j)` for `p ~ Beta(α, β)` and derive the log–log slope `−β`. *Sketch:* D1 steps 5 and 6.
4. **Derive.** Show that the cluster-bootstrap variance is a consistent estimator of D2's `Var(p̂)` for large `N`, and compute the ratio naive/true for `σ_t² = 0.1`, `E[p_i(1 − p_i)] = 0.15`, `n = 4`. *Sketch:* naive `0.25 / (N n)` versus true `0.1/N + 0.15/(N n)`; with `n = 4`: `0.0625/N` versus `0.1375/N`; ratio 0.45, so the naive interval is about 0.67 of the true width.
5. **Measure.** For `repair30`, estimate how many trials per task are needed so that the interval of `pass^4` for two cells is narrower than their difference; predict first, then use the bootstrap on the committed trials.
6. **Measure.** Add a fifth defense (a classifier-free prompt that repeats the user's task after each tool output) and measure ASR and utility; predict the order of D0 to D3 first.
7. **Open.** Design an adaptive attack against D1 (spotlighting) for the QA agent: write the attack, measure it against the same 50 questions, and report whether the defense ranking changes. Or implement plan-then-execute for the QA agent and place it on the ASR–utility plot.

#### Further reading

- Papers: [SWE-bench](https://arxiv.org/abs/2310.06770); [τ-bench](https://arxiv.org/abs/2406.12045) (origin of pass^k); [τ²-bench](https://arxiv.org/abs/2506.07982); [WebArena](https://arxiv.org/abs/2307.13854); [Indirect prompt injection](https://arxiv.org/abs/2302.12173); [AgentDojo](https://arxiv.org/abs/2406.13352); new: [AI Agents That Matter](https://arxiv.org/abs/2407.01502) (cost-controlled evaluation and the retry baseline); new: [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825) (validity checklist); new: [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (clustered standard errors and paired differences); new: [Spotlighting](https://arxiv.org/abs/2403.14720); new: [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208); new: [Defeating Prompt Injections by Design (CaMeL)](https://arxiv.org/abs/2503.18813); new: [Design Patterns for Securing LLM Agents](https://arxiv.org/abs/2506.08837).
- Tutorials and benchmarks sites: [SWE-bench Verified announcement](https://openai.com/index/introducing-swe-bench-verified/); [Terminal-Bench](https://www.tbench.ai/); [τ-bench leaderboard](https://taubench.com) (new; verified 200); [BFCL v3 multi-turn blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) (new).
- Code bases: [tau-bench](https://github.com/sierra-research/tau-bench) (historical trajectories); [tau2-bench](https://github.com/sierra-research/tau2-bench); [SWE-bench](https://github.com/SWE-bench/SWE-bench); [Terminal-Bench](https://github.com/harbor-framework/terminal-bench-1) (new, repository moved); [AgentDojo](https://github.com/ethz-spylab/agentdojo); [Gorilla / BFCL](https://github.com/ShishirPatil/gorilla).

#### Risks and open decisions

- **τ²-bench judge and user simulator.** At fetch time the README names an API model as the natural-language-assertion judge and a hosted model for text runs. If the judge cannot be pointed at a local model, run only the database-state and action checks and label the reward as such, or skip the T2 τ²-bench extension. The repository lineage is also changing (τ-bench to τ²-bench to a τ³-bench name): pin a commit.
- **Original τ-bench tasks are stated to be outdated** by their repository; the notebook uses its historical trajectories only to teach `pass^k` and cites the successor for new work.
- **SWE-bench Verified needs Docker** and large disk. If the T2 host has neither, drop the extension and keep the `repair30` evaluation; do not substitute a different benchmark silently.
- **Licensing of SWE-bench Verified instance text** (no license tag): do not commit instance text into `DATA_HUB`; commit ids and fetch from the pinned revision.
- **Injection results for small models may be near 0 or near 1** (floor and ceiling). Report them as measured; if every arm is at a bound, add harder templates (10.5 exercise 7) rather than changing the model.
- **Judges and humans:** G2 success is a string match; semantic hijacks (a plausible wrong answer) are not caught. State the limit.
- **Shell-capable injection tests** require `net_blocked`; if the host lacks namespaces, run only the search-agent arm.
