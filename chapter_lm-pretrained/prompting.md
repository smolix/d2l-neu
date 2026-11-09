# Prompting and In-Context Learning
:label:`sec_lm_prompting`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.3 Prompting and In-Context Learning · T1").
Blueprint: docs/llm/blueprints/ch01_pretrained.md section 1.3.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** The prompt is tokens, and few-shot accuracy depends on how the demonstrations are laid out.
**Opening:** The reader asks a base checkpoint to classify a news headline and gets a continuation instead of a label. Four solved examples placed before the query lift accuracy by tens of points, and the rendered template shows why: the model continues a pattern of tokens. The section measures what in that pattern matters and closes on what a few-shot score therefore measures.
**Teaches:**
- A chat template as token ids: roles, special tokens, the generation prefix and the thinking switch, printed as the rendered tail so the text cannot drift from the template file; `d2l.fewshot_prompt` renders plain, question-answer and chat layouts.
- Zero- and few-shot formats; scoring by restricted next-token logits over the label words (`d2l.label_logprobs`), the special case that 1.4 generalizes.
- Sensitivity to the number, order and label correctness of demonstrations, with contextual calibration as the one correction worked (`d2l.calibrate_labels`).
- In-context learning against a weight update: the same labelled examples as a probe on frozen features, at equal labelled data and unequal cost.
**Derivation:** none.
**Notebook:**
1. Render the chat template of Qwen3-1.7B for one conversation and print the token ids with the special tokens marked; render the same demonstrations as plain text for the base checkpoint.
2. On 200 stratified test items of a four-class news task, measure accuracy of Qwen3-1.7B-Base and Qwen3-1.7B against the number of demonstrations (0, 1, 2, 4, 8, 16) with 3 demonstration seeds.
3. At 8 demonstrations, measure accuracy over four orders of the same demonstrations per seed; then over a fraction of deliberately wrong labels (0, 0.25, 0.5, 0.75, 1.0) with natural and with abstract label words.
4. Fit a logistic probe on the frozen base model's mean-pooled features with the same labelled examples and add it to the plot.
5. Claim that survives noise: order alone moves accuracy by several points (the spread over orders is comparable to the spread over demonstration sets); shuffled labels cost less than removing the demonstrations altogether for natural label words. Each claim needs a paired bootstrap interval over items that excludes zero and an effect larger than the across-seed standard deviation; 200 items give about plus or minus 7 points unpaired, so the text interprets only paired differences.
6. Reused pieces: `d2l.load_hf`, `d2l.generate` for free-generation spot checks, a 6-line local bootstrap (the shared `d2l.paired_bootstrap` arrives in 1.5 and the text says so).
**Figures:** `fig_lm_chat_template` (the rendered conversation as a token sequence with roles, special tokens and the generation prefix labelled); `fig_lm_icl_sensitivity` (plot: accuracy against demonstration count for base, instruct and the probe; a strip plot of accuracy over orders at k = 8).
**Restate / forward:** `sec_what-attention-computes` (an induction head copies what followed an earlier occurrence of the current token; the repeated-sequence probe is the behavioural signature of in-context copying, and `tab_induction-evidence` says what a behavioural result can claim). Forward: 1.4 (what a restricted-logit score does not measure).
**Exercises:** 5 to 7. Load-bearing: measure the spread of accuracy over 20 orders of the same 8 demonstrations and compare it with the spread over 20 demonstration sets (measure); derive why subtracting the content-free log-probabilities shifts the model's label prior and when it cannot help (derive); open: design a prompt format whose few-shot accuracy for the base model is within the order spread of the instruct model's, and report the cost per query of each.
**Writer notes:**
- Exports: `d2l.fewshot_prompt(demos, query, template='plain', *, label_map=None, system=None)` returning a string or a message list; `d2l.label_logprobs(model, tokenizer, prompts, label_words, *, batch_size=8, chat=False, enable_thinking=False)` returning log-probabilities renormalized over the label set, which asserts the label words' first tokens are distinct and single after the format's separator (the leading-space convention: encode `separator + word` and take the last token of the difference); `d2l.calibrate_labels(logprobs, content_free_logprobs)`.
- Silent breakages: leading-space tokenization of label words (`" Sports"` versus `"Sports"`) when a format changes; chat-templating a base model or omitting the template for an instruct model and attributing the change to ability; Qwen3 post-trained defaults to thinking on and Qwen3.5 to off, read from the pinned template file; test items or near-duplicates leaking into the demonstration pool.
- Data: `fancyzhx/ag_news` (train 120,000 for demonstrations, test 7,600; licence listed as unknown on the Hub, so the notebook downloads it at run time and commits only item indices and predictions); the fallback with a clear licence is ARC-style multiple choice (`allenai/ai2_arc`, CC BY-SA 4.0), with the same code.
- The notebook reduces the blueprint's grid: four orders instead of twelve (twenty orders is the exercise), three seeds, 200 items; the committed per-item grid `lm-fewshot-grid.csv` makes the figure cells seconds at T0.
- What a few-shot score measures: with natural labels much of the gain is format and label-space information rather than the input-label mapping (Min et al.); the section says this and does not call it "learning the task". Restricted-logit accuracy is not free-generation accuracy; 1.4 owns that distinction.
- Cost: few-shot is not free; the probe comparison is at equal labelled data, not equal cost, and the text says which.
- Do not re-teach the induction circuit; restate it in two sentences with the label.
- Decide at authoring, with the editor, whether the clearly licensed ARC fallback is the default task.
**Sources:**
- Papers: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (few-shot as a capability of scale); [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837) (label correctness matters less than format and label space); [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (the mechanistic account restated from `sec_what-attention-computes`); [Calibrate Before Use](https://arxiv.org/abs/2102.09690) (contextual calibration); [Fantastically Ordered Prompts and Where to Find Them](https://arxiv.org/abs/2104.08786) (order sensitivity).
- Code and documentation: Transformers, [chat templating](https://huggingface.co/docs/transformers/chat_templating) (the rendered template and `add_generation_prompt`); [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (its few-shot construction options as the oracle).
- Courses, tutorials and blogs: [CS224N](https://web.stanford.edu/class/cs224n/) (prompting and efficient adaptation lecture); [Hugging Face LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1) (tokenizers and templates).
-->
