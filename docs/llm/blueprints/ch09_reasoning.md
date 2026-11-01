# Chapter 9: Distillation and Reasoning — Blueprint

**Label:** `chap_lm_reasoning` · **Arc:** IV. Post-training (after Chapter 8, before Chapter 10) · **Sections:** 9.1–9.7 (`sec_lm_distillation`, `sec_lm_onpolicy_distillation`, `sec_lm_reasoning_training`, `sec_lm_verifiers`, `sec_lm_parallel_tts`, `sec_lm_sequential_tts`, `sec_lm_efficient_reasoning`) · **Tiers:** T1 (9.1, 9.5, 9.6), T2 (9.2, 9.4, 9.7), T3 with a T2 reduction (9.3). Every T2/T3 section also has a T1 configuration that analyzes committed artifacts.

## Chapter overview

### Draft introduction (for the chapter opener)

Chapter 8 taught a model to maximize a reward. This chapter asks where a capable model comes from when a stronger one already exists, and what to do with compute at the moment of answering. We start with distillation, the oldest way to move capability between models, and show that the choice that matters is not "logits or text" but *whose prefixes the teacher is asked about* and *which direction of KL we minimize*. On-policy distillation (9.2) turns that observation into a dense-reward form of the RL that Chapter 8 built. We then assemble reasoning models from SFT on traces, RL from a base model, and distillation (9.3), and measure what each stage changes and what it leaves alone. Verifiers (9.4) become the shared currency of the second half: with a verifier, spending more compute at test time, in parallel (9.5) or in sequence (9.6), buys accuracy, and the question turns into how much per token. We close by asking how to get the same accuracy with fewer tokens (9.7). Every comparison in the chapter fixes a compute unit first and reports uncertainty from a paired bootstrap.

### Learning objectives (refined from v5; each is measurable in a notebook)

1. Given a teacher, a student and a data budget, choose among sequence-level, full-vocabulary token-level, top-k and cross-tokenizer distillation; state the divergence each minimizes, and compute the storage and FLOP cost of each (9.1).
2. Derive the exact decomposition of sequence-level forward and reverse KL into per-token terms; show on a toy mixture that the forward direction is mode-covering and the reverse direction is mode-seeking, and predict which one a capacity-limited student will pick (9.1).
3. Write on-policy distillation as policy-gradient RL with the dense per-token reward $r_t=\log\pi_{\textrm{tea}}-\log\pi_\theta$; verify numerically that the return-to-go estimator is unbiased for the sequence reverse-KL gradient and that the per-token (discount-zero) estimator differs from it (9.2).
4. Run offline distillation, on-policy distillation and GRPO at matched generated tokens, and report accuracy, teacher/verifier cost and a paired-bootstrap interval for each difference (9.2).
5. Build a reasoning model by SFT on verified traces, by RL from a base model, and by both; evaluate on unseen templates and on a harder-than-training slice, with pass@k, length and behavior statistics (9.3).
6. Train outcome and step-level scorers; state what a Monte Carlo step label measures, derive the ORM/PRM credit-assignment identity, and compare selection accuracy per unit of compute on the same candidate pools (9.4).
7. Derive majority-vote, best-of-N and coverage curves under an independence model, then show how problem-level correlation caps them; plot accuracy against generated tokens for majority vote, verifier reranking and the oracle (9.5).
8. Sweep budget forcing, fit a difficulty-based allocation rule on validation problems, freeze it, and evaluate on test problems; decide when many samples from a small model beat one answer from a larger one at equal FLOPs (9.6).
9. Train a length-penalized variant, build accuracy–token frontiers stratified by difficulty, and identify dominated configurations including the amortized training cost (9.7).

### Prerequisites

Book labels (what each supplies; the chapter does not re-teach any of it):

| Label | Used for |
|---|---|
| `sec_mdl-information_theory` (`subsec_mdl-distillation`, `eq_mdl-distill-loss`) | Temperature-softened soft-target distillation and the $T^2$ gradient scaling. 9.1 recaps this in one paragraph and renames the temperature $\kappa$ because $T$ is the horizon here. |
| `sec_mdl-divergences-distances` (`sec_mdl-fwd-vs-rev-kl`, `sec_mdl-f-divergences`) | Forward vs reverse KL on a Gaussian-fit-to-mixture toy (numbers 0.357 and 1.204 nats), f-divergence template. 9.1 re-does the comparison for *autoregressive* students and reuses its two numbers as a cross-check. |
| `sec_imitation` (`subsec_il_error_bounds`, `subsec_il_dagger`) | The $\varepsilon T^2$ vs $\varepsilon T$ proposition and DAgger. 9.2 extends them to a soft expert. |
| `sec_rl_sequences` (`tab_rl_notation_contract`, `eq_seq_factorization`, `subsec_rl_sequences_group`, `subsec_rl_sequences_hacking`) | Token MDP, notation, the factorization of the response score, group baselines, verifier hacking. |
| `sec_baselines` (`subsec_leave_one_out`, `subsec_pg_loss_divisor`), `sec_ppo` (`eq_ppo_clip`, `subsec_ppo_asymmetric`), `sec_regularized` (`subsec_kl_direction`, "Dense versus Terminal Rewards") | Zero-mean lemma, loss divisors, clipped objective, KL direction, one paragraph on ORM vs PRM. 9.4 turns the last into a derivation and an experiment. |
| `sec_decoding` (`d2l.generate`, `sample_next`) | Sampling strategies; 9.5 uses temperature 0.6 / top-p 0.95 / top-k 20 as the model card prescribes for thinking mode. |
| `sec_scaling-laws` (six-FLOPs rule, `subsec_scaling-laws-checking`) | $2N$ FLOPs per forward token and $6N$ per training token; the compute ledger in 9.1 builds on them. |
| `sec_mdl-statistics` | Bootstrap and paired comparisons behind `d2l.paired_bootstrap`. |
| `sec_mdl-bayesian-computation` | Importance sampling, used for the unbiased sampled-top-k estimator in 9.1 and the stale-sample correction in 9.2. |
| `sec_kv-cache`, `sec_perf_model` | Decode is memory-bound, prefill is compute-bound; why generated tokens and scored tokens have different wall-clock cost. |

Earlier sections of this part:

| Section | Used for |
|---|---|
| 1.1, 1.2 | `d2l.load_hf`, `d2l.load_qwen3`, `d2l.generate`; Qwen3 chat template and `enable_thinking`. |
| 1.3 | Chain-of-thought prompting as the "think before answer" baseline. |
| 1.4 | `d2l.pass_at_k(n, c, k)` for coverage; answer-extraction conventions. |
| 1.5 | `d2l.EvalSuite`, `d2l.paired_bootstrap` for every difference reported here. |
| 7.2 | `d2l.sft_train` (with `render_chat`, `sft_collate`, and the `loss_fn` hook for distillation losses). |
| 7.4 | `d2l.LoRALinear`, used for the T1 configurations of 9.1, 9.2, 9.3, 9.4. |
| 8.1 | `d2l.RewardModel(backbone, hidden_size)` (pattern for the 9.4 scorers) and `d2l.token_logprobs`. |
| 8.5 | `d2l.grpo_train`, `d2l.verify_math`, `d2l.canonical_answer`, `d2l.sample_responses`, and the 8.5 headline checkpoints (the baseline of 9.7). |
| 8.6 | RL systems setup: `d2l.sync_weights`, colocated vs disaggregated layouts. |

Names imported from Chapter 8 and from 7.2 (canonical signatures from `01_interfaces.md` section 1, restated here for convenience; see also each section's **Builds on (this part)** line):

| Name | Section | Signature this chapter needs |
|---|---|---|
| `d2l.grpo_train(policy, tasks, cfg, reward_fn=None) -> list[dict]` | 8.5 | `cfg` fields: `K` (this chapter's $K$), `prompts_per_step`, `steps`, `aggregation` in `{'sample','token','const'}`, `std_norm`, `loo`, `eps_low`, `eps_high` ($\epsilon$, asymmetric), `beta` ($\beta$), `kl_estimator`, `dynamic_sampling`, `overlong`, `num_minibatches`, `engine` in `{'hf','vllm'}`, `seed`. Returns the per-step history (list of dicts). `reward_fn(texts, items) -> list[float]` is batch-level and defaults to `verify_math` per item. There are no `GRPOTrainer` or `GRPOConfig` classes. Chapter 9 has no `ref_reset_every`; the prolonged-RL arm of 9.3 resets the reference by calling `grpo_train` in segments (see 9.3). |
| `d2l.verify_math(text, gold) -> float` | 8.5 | Verifier returning 1.0 or 0.0; composes `d2l.extract_answer(text, kind='boxed')` (1.4) with 8.5's `d2l.canonical_answer`. Replaces the earlier `math_reward`. |
| `d2l.sample_responses(engine, prompts, K, max_new_tokens, temperature=1.0, top_p=1.0, seed=0)` | 8.5 | Returns `{ids, resp_mask, logp_sampler, text, finished}`. `engine` is an HF model with its tokenizer, or a vLLM `LLM`. Weights are pushed with `d2l.sync_weights(trainer, engine, mode) -> (seconds, bytes)` (8.6). There is no `RolloutEngine`. |
| `d2l.sft_train(model, batches, optimizer, scheduler=None, loss_fn=None, grad_accum=1, max_grad_norm=None, log_every=10, callback=None)` | 7.2 | `loss_fn(model, batch, num_items) -> Tensor` is the hook for distillation losses; batches come from `d2l.render_chat` and `d2l.sft_collate`. There is no `SFTTrainer`. |
| `d2l.RewardModel(backbone, hidden_size)` | 8.1 | `forward(input_ids, attention_mask) -> (B,)`: reward at the last non-pad token. 9.4 uses it only as the constructor pattern and defines its own `d2l.ProcessScorer`, which reads several positions. |
| `d2l.token_logprobs(model, input_ids, attention_mask, response_mask=None, temperature=1.0) -> (B, T-1)` | 8.1 | Log-probability of each realized token (position $t$ of the output is token $t+1$). Imported by 9.2. |
| `DATA_HUB['lm-grpo-qwen3-1p7b-base']`, `DATA_HUB['lm-grpo-qwen3-0p6b']` | 8.5 | The 8.5 headline checkpoint on Qwen3-1.7B-Base (T2; the baseline of 9.7) and the Qwen3-0.6B arm A (SFT + LoRA, T1/T2). |

### Section dependency graph

```
 9.1 distillation ──► 9.2 on-policy distillation ──► 9.3 reasoning models ──► 9.7 efficient reasoning
        │                      │                           │                        ▲
        │                      └──────(OPSD, loss code)────┼────────────────────────┤
        └──(ComputeLedger, kd_loss)────────────────────────┤                        │
                                                           ▼                        │
                              9.4 verifiers ──► 9.5 parallel TTS ──► 9.6 sequential/adaptive TTS
                               (scorers, pools)   (pools, votes)        (allocation, budget forcing)
```

9.1 and 9.2 form a distillation track. 9.4 to 9.6 form a test-time-compute track and can be taught without 9.1–9.3 (they need only a sampling model and committed pools). 9.3 feeds both tracks with checkpoints; 9.7 joins them. Course paths from v5 use 9.2, 9.3, 9.5.

### Running examples and shared artifacts

*Used:* Qwen3 dense models (R1-compatible, 1.2); the Qwen3.5 tokenizer as the mismatched-vocabulary example (9.1); the 8.5 checkpoint (9.7).

*Produced and shared inside the chapter:*

| Artifact | `DATA_HUB` key (proposed) | Producer | Consumers | Format |
|---|---|---|---|---|
| Problem generator and splits (`d2l.make_problems`) | code only; splits regenerate from seeds | 9.1 | 9.2, 9.3, 9.4, 9.6, 9.7 | Python; JSONL of `{id, family, template_id, difficulty, question, gold}` |
| Teacher sample sets and top-k logit cache | `lm-kd-qwen3-1p7b-samples-v1` | 9.1 | 9.1, 9.2 | JSONL samples; `.npz` top-k `idx int32 [n_tok,32]`, `logp fp16 [n_tok,32]`, `tail fp16 [n_tok]` |
| Training logs of every run (the matched-compute tables) | `lm-reasoning-logs-v1` | 9.2, 9.3, 9.7 | the same notebooks | JSONL, one record per step with the fields of the `ComputeLedger` |
| SFT/RL checkpoints | `lm-reason-sft-q17b`, `lm-reason-rl-q17b`, `lm-reason-sftrl-q17b`, `lm-len-penalty-q17b` | 9.3, 9.7 | 9.3–9.7 | safetensors (bf16); LoRA adapters for T1 variants |
| MC step labels and scorers | `lm-prm-mc-labels-v1`, `lm-scorers-v1` | 9.4 | 9.4, 9.5 | JSONL step labels; scorer weights |
| Sample pools for test-time compute | `lm-tts-pools-v1` | 9.5 (generated by a T2 job) | 9.5, 9.6, 9.7 | Parquet, one row per (problem, sample): `model, mode, problem_id, sample_id, n_gen_tokens, think_tokens, answer, correct, orm_score, prm_scores, logprob_sum` |
| Allocation policy (frozen) | `lm-alloc-policy-v1` | 9.6 | 9.6 | JSON |

*Compute unit (chapter-wide, defined in 9.1 and used in every comparison).* The unit is **FLOPs counted from tokens by a `d2l.ComputeLedger`**: $2N_{\textrm{ne}}$ per token processed in a forward pass (generated or scored) and $6N_{\textrm{ne}}$ per token trained, with $N_{\textrm{ne}}$ the non-embedding parameter count of the model that processed the token (`sec_scaling-laws`). Attention FLOPs (quadratic in context) are excluded and the exclusion is stated at every use. Because decode is memory-bound and prefill compute-bound (`sec_kv-cache`), every plot that has a FLOP or token axis also reports measured wall-clock for the same run, and the text says when the two orderings disagree. When all models in a comparison are the same size (9.5, most of 9.6), the unit reduces to **generated tokens**, counting *all* generated tokens including discarded samples and thinking tokens, and counting verifier tokens as scored tokens at the verifier's $N_{\textrm{ne}}$. Cost of building an artifact once (pools, teacher traces) is reported separately as "amortized" and also included in an "all-in" column wherever a method's inputs are bought with another model's compute.

*Contamination policy (chapter-wide).*
- Qwen3 (released April 2025) was trained after GSM8K (2021), MATH (2021) and AIME 2024/2025 were public, so absolute accuracies on public benchmarks are not clean measurements. They are used for **relative, paired, within-model comparisons only**, each marked "public, possibly seen" in plots and captions.
- All **training** experiments (9.2, 9.3, 9.7) use procedurally generated problem families with held-out *templates* and a harder-than-training slice, so no train/test overlap exists by construction (the idea of GSM-Symbolic, `arXiv:2410.05229`; the notebook builds its own generator and does not redistribute `apple/GSM-Symbolic`, whose card is CC-BY-NC-ND-4.0).
- Public sets serve as a reality check: GSM8K test and MATH-500 for every section, AIME 2024+2025 (60 problems) for the hardest slice, with 95% intervals that are wide by construction. The perturbation check of GSM-Symbolic and the fresh-split idea of GSM1k (`arXiv:2405.00332`) are cited as the reason for the caution; a model whose GSM8K accuracy drops sharply on resampled numbers is flagged.
- Code: LiveCodeBench problems carry contest dates; any code exercise selects problems *after* the model's stated cutoff when such problems exist, and otherwise labels the set contaminated-risk (9.5 exercises).

### Compute plan

| § | Tier | Reduced T1 configuration | Wall-clock and memory | Committed artifacts |
|---|---|---|---|---|
| 9.1 | T1 | Qwen3-0.6B-Base student with LoRA (`full_ft=False`) on cached teacher top-k logits; full fine-tuning behind `--full-ft` | to be measured; target ≤ ~15 min, ≤ 15 GiB | teacher samples and top-k cache |
| 9.2 | T2 | 0.6B student with LoRA, 1.7B teacher, 40 online steps; plus committed logs for the full run | T2 run: a few hours on one A100 40 GB, target ≤ 38 GiB (1.7B student with LoRA, bf16 weights 3.4 GB, plus 8B teacher 16.4 GB, plus the rollout engine); T1: to be measured | logs of the four arms, final checkpoints (LoRA adapters at both scales) |
| 9.3 | T3 (T2 reduced) | T3: 8× A100 ≤ 2 h; T2 reduced: Qwen3-0.6B-Base, one GPU; T1: LoRA 0.6B, analysis of committed logs | T3 ≤ 2 h; T2 a few hours | traces, logs, checkpoints, eval JSON |
| 9.4 | T2 | Labels from committed `lm-prm-mc-labels-v1`; scorer = Qwen3-0.6B + LoRA | T2: labeling + training a few hours; T1: to be measured | MC labels, scorer weights, reranking tables |
| 9.5 | T1 | Pools committed; live mini-run N=16 on 100 problems | analysis minutes on CPU/T1; pool generation is a T2 job (a few hours) | `lm-tts-pools-v1` |
| 9.6 | T1 | Same pools plus 20 live problems for budget forcing | to be measured; ≤ ~15 min | allocation policy, sweeps |
| 9.7 | T2 | LoRA 0.6B continuation of 8.5 checkpoint, committed frontier logs | T2: a few hours | logs, length-penalized checkpoint |

No section uses T4. GPU-hours are not quoted beyond the tier budgets because they depend on pool sizes and are to be measured on the build hosts; each T2/T3 notebook prints its own ledger.

## Assignment

### Optional mini-assignment: on-policy distillation

**Label:** `sec_lm_a9_mini`

v5 attaches no assignment to Chapter 9: A7 (SFT → DPO → GRPO) spans Chapters 7–8 and is not changed. Decided 2026-10-29 (README item 6): Chapter 9 gets an optional mini-assignment, modeled on `sec_lm_a11_mini`, that continues from A7's SFT checkpoint. Like every assignment it is optional for readers. It uses 9.2's T1 reduction (student Qwen3-0.6B-Base with a LoRA adapter, teacher `Qwen/Qwen3-1.7B`, same tokenizer, vocabulary 151,936) and is graded on correctness of the estimator code and on a written report, **not on a gain threshold**: 40 online steps on a 0.6B model give an accuracy change that noise can swamp, so the autograder never asks for one. Tier T1.

**Starting point.** The student's own A7 part 1 SFT checkpoint (`sec_lm_a7_sft`), or the staff checkpoint `DATA_HUB['lm-sft-qwen3-0p6b']` (7.2's reference run; `lm-sft-qwen3-0p6b-v1` is the curated-data variant) if the student's own fails A7's preflight `check_sft()` (Chapter 8). The checkpoint carries `d2l_chatml_v1`; the starter renders every prompt and response with it (`d2l.render_chat`), for the student and for teacher scoring, so both models see identical token ids. This checkpoint has not been taught 9.2's `#### <integer>` answer format (9.2's own `S0` warm-up does that); the starter's prompts ask for it, and the report states the parsable-answer rate. Nothing below depends on that rate.

**Tasks.**
1. *Estimator.* Implement `opd_loss(logp, old_logp, teacher_logp, mask, mode='sampled', clip=None) -> (loss, stats)` to 9.2's contract (dense reward $r_t=\log\pi_{\textrm{tea}}-\log\pi_{\textrm{old}}$, ratio $\rho_t$, `stats` with mean per-token reverse KL, token count and fraction clipped).
2. *Two derivation checks on the tabular toy* (9.2's cells `opd-kl-identity` and `opd-gradient-check`, as stubs with the toy models supplied): the D1 identity with the exact optimum, and the D2 estimator comparison ($\hat g_{\textrm{full}}$ against $\hat g_0$, with the exact enumerated gradient).
3. *Run.* 40 online steps of `d2l.OnPolicyDistiller` (`config.mode='sampled'`, `lam=1.0`, `teacher_context='none'`) with the student's `opd_loss` bound in place of `d2l.opd_loss` (the starter's runner shows the one-line binding), on the starter's `d2l.make_problems('chain', ...)` train stream, 32 prompts per step, `max_new_tokens=256`, LoRA r=16 with LR $10^{-4}$, seed 0; then a second seed. Evaluate before and after on 100 problems of `test_unseen_templates` and 100 of `test_harder` (greedy), and log `gen_tokens, teacher_tokens, flops, wall_s, rkl, mean_len` per step.
4. *Report* (below).

**Deliverables.** `opd.py` (the estimator and the two toy checks), the two run logs (`opd_seed0.jsonl`, `opd_seed1.jsonl`), the final LoRA adapter, `eval_before.json` and `eval_after.json`, and `a9_report.md`.

**Autograder (hidden tests unless stated).**
1. *Unit tests, float64, CPU, ≤ 2 min* (9.2's own numerical checks, with their tolerances):
   - `opd_loss` fixture: on 8 fixed random tensors with `old_logp = logp.detach()`, the gradient with respect to `logp` equals $-\frac1n\sum_t r_t\nabla\log Q$ (the analytic form) to `atol=1e-10`; with `teacher_logp = logp.detach()` the reward is exactly 0 and the gradient exactly zero; masked (padding) tokens have exactly zero gradient; `stats` (mean reverse KL, token count, clipped fraction) match the fixture to `1e-10`.
   - `opd-kl-identity` (D1; $V=3$, $T=3$, seed 0): the identity $D_{\textrm{KL}}(Q\Vert P)=D_{\textrm{KL}}(Q\Vert\pi_{\textrm{ref}})-\mathbb E_QR$ holds to $<10^{-10}$ for the student's random $\pi_{\textrm{ref}}$, and gradient descent on the regularized objective reaches the teacher within total variation $10^{-6}$.
   - `opd-gradient-check` (D2; same toy): (i) the enumerated $\mathbb E\hat g_{\textrm{full}}$ matches finite differences of the exact objective to max absolute difference $<10^{-8}$; (ii) the enumerated $\mathbb E\hat g_0$ matches the frozen-state per-token KL gradient to $<10^{-8}$; (iii) the student's printed cosine and relative difference between $\mathbb E\hat g_0$ and $\mathbb E\hat g_{\textrm{full}}$ satisfy relative difference $>10^{-3}$ and cosine $>0$ (the values themselves are not graded); (iv) 20,000 Monte Carlo sequences reproduce (i) within 3 standard errors per coordinate (median over coordinates).
   - `d2l.token_logprobs` shift check against `F.cross_entropy(reduction='none')` to $10^{-6}$.
   - Loop sanity (T0): a 2-layer `d2l.GPT` student and a differently seeded 2-layer teacher, vocabulary 64, length-8 responses, 30 steps: the reverse KL on student samples falls by at least 20%.
2. *Integration (GPU, T1, ≤ 15 GiB):* the run completes 40 steps; the tokenizers are asserted equal; the logs contain every field above, and `flops` per step equals the `ComputeLedger` formula ($2N_{\textrm{ne}}$ per generated or scored token, $6N_{\textrm{ne}}$ per trained token, with `d2l.nonembedding_params`) to relative $10^{-6}$; the recomputed trainer log-probabilities of 32 sampled responses agree with the sampler's within a tolerance the staff set from the reference solution (3× the reference run's mean absolute difference, to be measured). No gate is placed on accuracy or on the final KL.
3. *Report check (rubric):* the sections below are present and every number in the report matches the logs.

**Written report (≤ 2 pages; graded by rubric on correct use of the evidence, not on the size of any effect).**
1. Plot the per-token reverse KL to the teacher by response position for your starting checkpoint before and after the run (9.2's opening cell). Where does it change, and does the first-bad-token position move?
2. Over your two seeds report the mean reverse KL on student samples at steps 0 and 40 and the accuracy before and after on each split with a paired-bootstrap interval over problems (`d2l.paired_bootstrap`). State what the intervals can and cannot support at 100 problems and two seeds, and say which claim of 9.2's headline you would *not* make from this run.
3. Using your D2 check, explain in two sentences what the discount-zero estimator drops, and give the cosine you measured on the toy.
4. Report the fraction of responses truncated at 256 tokens and the share of the teacher's scored tokens that fall in truncated responses; which direction does truncation bias the reward (9.2, exercise 2)?
5. Which template did you render with, and what does the teacher's log-probability of the first response tokens look like with and without Qwen's empty think block (the 9.2 cell)? What does that imply for the first-token rewards?

**T1 path.** The whole assignment is T1. The teacher (`Qwen/Qwen3-1.7B`, 2,031,739,904 parameters, 4.06 GB in bf16) is resident beside the 0.6B student with its LoRA adapter and the rollout engine; this fits a 15 GiB build slot only with rollout batch ≤ 32 and response length ≤ 256 (9.2's risk list), which the starter's config enforces. The readers' 24 GB GPU is sufficient. **Estimated student hours:** 6–8, split roughly into 2 for the estimator and toy checks, 2–3 for the runs and evaluation, and 2–3 for the report.

The chapter's "open" exercises (one per section) remain mini-projects.

## Teaching sources

Kept from v5, annotated with what to take:

- Thinking Machines, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/): the per-token reverse-KL reward, the cost comparison against RL, the personalization/continual-learning use. The source for 9.2's design; take its framing, but derive the objective in the notebook.
- Lilian Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/): the map of parallel and sequential test-time compute, PRM/ORM, latent thinking. Use for the vocabulary of 9.4–9.7 and as outside reading; do not copy its structure.
- Sebastian Raschka, [Reasoning from Scratch](https://github.com/rasbt/reasoning-from-scratch): from-scratch inference scaling, self-consistency, GRPO. Code reference for 9.3 and 9.5 and a test oracle for the voting code.
- Jay Alammar, [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1): the picture of the R1 pipeline; the model for the 9.3 pipeline figure.
- [Open R1](https://github.com/huggingface/open-r1): reproducible recipes (SFT on `open-r1/OpenR1-Math-220k`, GRPO), the reference for 9.3 and 9.7.

Added (verified 2026-10-28):

- Qwen team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388): the strong-to-weak distillation stages (off-policy then on-policy) used to build the small Qwen3 models, thinking/non-thinking fusion, and the thinking-budget mechanism. The only place where a released model family documents the distillation this chapter teaches.
- Hugging Face, [Scaling test-time compute with open models](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) and its code [search-and-learn](https://github.com/huggingface/search-and-learn) (Apache-2.0): a worked open reproduction of Snell et al. with PRM-guided search; tutorial and code base for 9.4–9.6.
- [TRL GKD trainer](https://huggingface.co/docs/trl/gkd_trainer) and [verl OPD documentation](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md): production counterparts for 9.1–9.2.

## Sections

### 9.1 Knowledge distillation for language models

- **Label:** `sec_lm_distillation` · **Tier:** T1 (reduced T1 configuration: the T1 run *is* the reduced configuration: student Qwen3-0.6B-Base trained with LoRA rank 64 from a cached teacher top-k logit file; full fine-tuning behind `full_ft=True` for a T2 host) · **Budget:** to be measured; target ≤ ~15 min wall-clock and ≤ 15 GiB peak at T1 (teacher is *not* loaded in the training cell; it is loaded only in the separate scoring cell that builds the cache and in the evaluation cell that scores student samples, each run once, one after the other) · **Codex units:** 10.1, 10.2, 10.3
- **Builds on (book):**
  - `sec_mdl-information_theory` / `subsec_mdl-distillation`: soft targets, temperature, the $T^2$ factor and its gradient derivation. Recap in one paragraph, with the temperature written $\kappa$. Not re-derived.
  - `sec_mdl-divergences-distances` / `sec_mdl-fwd-vs-rev-kl`: the direction of KL, zero-avoiding vs zero-forcing, the 0.7/0.3 mixture and its two reference values ($\log(1/0.7)=0.357$, $\log(1/0.3)=1.204$ nats). Not re-taught; 9.1 re-uses the *same weights* in a discrete autoregressive toy.
  - `sec_rl_sequences` (`tab_rl_notation_contract`, `eq_seq_factorization`): $x$, $y$, $\pi_\theta$, $D_{\textrm{KL}}$. The teacher symbol $\pi_{\textrm{tea}}$ is new to this chapter (the contract has none) and is introduced once in a sentence.
  - `sec_scaling-laws`: $2N$ / $6N$ FLOPs per token.
- **Builds on (this part):** 1.1–1.2 (`d2l.load_hf`, `d2l.load_qwen3`, `d2l.generate`; chat template), 1.4 (`d2l.pass_at_k`), 1.5 (`d2l.EvalSuite`, `d2l.paired_bootstrap`), 7.2 (`d2l.sft_train`, `d2l.render_chat`, `d2l.sft_collate`; used for the hard-label arms and, through `loss_fn`, for the soft-label arms), 7.4 (`d2l.LoRALinear`). 8.x is not needed.
- **Exports** (`#@save`, all new, unique):
  - `d2l.nonembedding_params(config) -> int`: non-embedding parameters of a Qwen3-style dense config (attention q/k/v/o, SwiGLU MLP, norms; excludes token embedding and untied head). Test: within 3% of the model-card values 0.44B, 1.4B, 3.6B, 6.95B for 0.6B, 1.7B, 4B, 8B (checked by hand: 0.6B gives 0.440B and 1.7B gives 1.41B from the configs).
  - `d2l.ComputeLedger()`: `.add_generate(n_ne, tokens, label)`, `.add_score(n_ne, tokens, label)`, `.add_train(n_ne, tokens, label)`, `.flops() -> float` ($2N_{\textrm{ne}}$ for generate and score, $6N_{\textrm{ne}}$ for train), `.tokens(kind) -> int`, `.table() -> DataFrame`, `.to_jsonl(path)`. Shared by every later section.
  - `d2l.make_problems(family, n, seed, split, difficulty=None) -> list[dict]`: procedural problem families with a verifiable integer answer (specified in "From scratch" below). `split` in `{'train', 'test_seen_templates', 'test_unseen_templates', 'test_harder'}`. Relationship to 8.5's `d2l.ArithmeticTasks`: 8.5's class is a flat arithmetic task source for the GRPO tutorial; `make_problems` adds template-level splits and difficulty control, and each item also carries `prompt` (one user chat message built from `question`) and `answer` (`str(gold)`), the two fields `d2l.grpo_train` and its default reward read, so the outputs pass to `grpo_train` as `tasks` without an adapter.
  - `d2l.kd_loss(student_logits, teacher_logits, mask, kind='fkl', alpha=0.5, kappa=1.0, chunk=4096) -> Tensor`: mean per-token divergence over `mask`; `kind` in `{'fkl', 'rkl', 'jsd'}`; vocabulary processed in chunks of positions to bound memory; computed in fp32.
  - `d2l.topk_compress(logits, k=32) -> (idx, logp, tail)`: top-$k$ log-probabilities and the log of the remaining mass.
  - `d2l.kd_topk_loss(student_logits, idx, logp, tail, mask, mode='tail_bucket') -> Tensor`: forward KL against the compressed teacher, with `mode` in `{'tail_bucket', 'renorm'}`.
  - `d2l.uld_loss(student_logits, teacher_logits, mask) -> Tensor`: vocabulary-agnostic sorted-probability loss (cross-tokenizer case).
  - Data: `DATA_HUB['lm-kd-qwen3-1p7b-samples-v1']`: teacher samples on 2,000 GSM8K-train prompts and 1,000 generated-family prompts (JSONL: `prompt, response, n_tokens, teacher_logp_sum`), and `lm-kd-qwen3-1p7b-topk32-v1` (`.npz` as in the chapter table).
- **Models and data:**
  - Teacher `Qwen/Qwen3-1.7B` (apache-2.0, 2.03B parameters incl. embeddings; vocab 151,936; 28 layers, hidden 2048, tied embeddings) used in non-thinking mode (`enable_thinking=False`, sampling temperature 0.7, top-p 0.8, top-k 20 as the model card recommends).
  - Student `Qwen/Qwen3-0.6B-Base` (apache-2.0, 0.596B parameters; vocab 151,936; 28 layers, hidden 1024). Same tokenizer as the teacher, so logits align position by position; the notebook asserts `len(tok)` equality and masks the padded embedding rows (`vocab_size` 151,936 exceeds the tokenizer length by the padding rows; the notebook prints the exact numbers).
  - Cross-tokenizer example only: `Qwen/Qwen3.5-2B` (apache-2.0, vocab 248,320; hybrid architecture) as a teacher whose vocabulary does *not* match.
  - Data: `openai/gsm8k` (MIT; 7,473 train, 1,319 test) for prompts and a public check; generated families from `make_problems` for held-out templates. Public numbers are marked "public, possibly seen" (contamination policy, chapter overview).
- **Packages:** `transformers>=4.51` (Qwen3 support; pin the actual version in the notebook), `torch>=2.4`, `peft` only in the production cell, `datasets`, `vllm` optional (teacher sampling when building the sample file offline).

#### Narrative arc

1. **The question.** Opening cell: load Qwen3-1.7B and Qwen3-0.6B-Base, sample one solution to each of 200 GSM8K training prompts from the teacher, and print (a) teacher accuracy, student accuracy (few-shot), (b) the teacher's mean next-token entropy on its own samples vs the student's, and (c) the bytes needed to store the teacher's full next-token distribution for one million tokens: $151{,}936\times2$ B $=303{,}872$ B per token in bf16, i.e. about 304 GB per million tokens. The symptom: the student is much worse, the teacher knows more than the text of its answers says, and storing what it knows is expensive. "What should we transfer, and from which prefixes?"
2. **A taxonomy by three decisions** (a table, `tab_lm_kd_design`): (i) *target*: hard token (sequence-level), full distribution, top-$k$ distribution, sampled tokens; (ii) *prefix source*: human text, teacher samples, student samples (deferred to 9.2); (iii) *divergence*: forward KL, reverse KL, generalized JSD. Place Hinton et al. (soft targets on fixed inputs), Kim and Rush (sequence-level: train on the teacher's output), MiniLLM and GKD (student prefixes, reverse/JSD) in the table. One paragraph recaps `eq_mdl-distill-loss`, the $T^2$ scaling, and says that here $\kappa$ is the temperature.
3. **Sequence-level distillation.** Teacher samples as hard labels; show that this is a Monte Carlo estimate of the forward-KL gradient (derivation D1 step 3). Cells: build the sample file, filter variants (raw vs verifier-filtered, defined in the notebook and compared in exercise 5), SFT via `d2l.sft_train`. Cost: teacher decode tokens.
4. **Token-level distillation with full logits.** `d2l.kd_loss(kind='fkl')` on the same teacher samples. Memory: logits tensor $b\times L\times V$ in fp32 is $8\times512\times151{,}936\times4$ B $=2.49$ GB, several copies in softmax and backward, hence the chunked loss. Cell: peak-memory printout with and without chunking.
5. **Direction of KL (derivation D1).** Toy mixture check first (CPU, seconds), then forward vs reverse vs JSD on the same teacher samples (reverse and JSD on *teacher* prefixes are still off-policy; 9.2 adds student prefixes). Figure `fig_lm_kd_modes`.
6. **Compressing the teacher: top-$k$.** The cache and its size; bias of renormalized top-$k$ vs tail-bucket; the sampled (importance-sampled) estimator as the unbiased alternative; plot KL captured vs $k$ on teacher samples (`fig_lm_kd_topk`).
7. **Teachers with another tokenizer.** Show vocabulary overlap between Qwen3 and Qwen3.5 (exact token-string match fraction, to be measured), why position-wise KL is undefined, and two remedies: text-only sequence-level distillation (works with any teacher including an API) and `uld_loss`. One cell trains nothing: it evaluates `uld_loss` on the two tokenizations of the same text and shows the loss is invariant to token identity (the weakness).
8. **Matched-compute comparison (headline experiment).** The ledger table, the accuracy table with intervals, the cost-per-accuracy-point plot.

#### Derivations

**D1. Forward vs reverse KL for autoregressive students (mode-covering vs mode-seeking), with the chain rule.**

*Statement.* Let $P=\pi_{\textrm{tea}}(\cdot\mid x)$ and $Q=\pi_\theta(\cdot\mid x)$ be distributions over responses $y=(y_1,\dots,y_T)$. (a) $D_{\textrm{KL}}(P\Vert Q)=\sum_{t=1}^{T}\mathbb{E}_{y_{<t}\sim P}\,D_{\textrm{KL}}\big(P(\cdot\mid y_{<t})\Vert Q(\cdot\mid y_{<t})\big)$ and the same with the roles exchanged for $D_{\textrm{KL}}(Q\Vert P)$, where the expectation is under $Q$. (b) $\nabla_\theta D_{\textrm{KL}}(P\Vert Q)=-\mathbb{E}_{y\sim P}\sum_t\nabla_\theta\log Q(y_t\mid y_{<t})$, so SFT on teacher samples is an unbiased estimator of the sequence-level forward-KL gradient, and token-level full-vocabulary forward KL on teacher-sampled prefixes is its Rao–Blackwellization over the last token. (c) $\nabla_\theta D_{\textrm{KL}}(Q\Vert P)=\mathbb{E}_{y\sim Q}\big[\sum_t\nabla_\theta\log Q(y_t\mid y_{<t})\,(\log Q(y\mid x)-\log P(y\mid x))\big]$ needs samples from $Q$. (d) For a family that cannot represent $P$, the minimizers differ: forward KL spreads mass over every mode of $P$; reverse KL concentrates on one.

*Assumptions.* Finite vocabulary, fixed length $T$ (or EOS absorbed as a token), $P$ and $Q$ strictly positive for (a)–(c); the toy for (d) has a factorized student by construction.

*Steps.*
1. Factorize $P(y)=\prod_tP(y_t\mid y_{<t})$ and $Q$ likewise (`eq_seq_factorization`). Then $\log\frac{P(y)}{Q(y)}=\sum_t\log\frac{P(y_t\mid y_{<t})}{Q(y_t\mid y_{<t})}$.
2. Take $\mathbb{E}_{y\sim P}$. For term $t$ the conditional expectation given $y_{<t}$ is $D_{\textrm{KL}}(P(\cdot\mid y_{<t})\Vert Q(\cdot\mid y_{<t}))$ by the tower property. Summing gives (a); exchanging $P$ and $Q$ gives the reverse statement. This is the identity behind "per-token KL on prefixes drawn from the first argument" (eq `eq_lm_kd_chain`).
3. Gradient of the forward direction: only $-\mathbb{E}_P\log Q(y)$ depends on $\theta$, so $\nabla_\theta D_{\textrm{KL}}(P\Vert Q)=-\mathbb{E}_{y\sim P}\nabla_\theta\log Q(y)=-\mathbb{E}_{y\sim P}\sum_t\nabla_\theta\log Q(y_t\mid y_{<t})$, by `eq_seq_factorization`. A sample $y\sim P$ gives an unbiased estimate: sequence-level distillation. For the last token replace the sampled $y_T$ by the exact expectation $\sum_a P(a\mid y_{<T})\nabla\log Q(a\mid y_{<T})$; this has the same mean and no larger variance (Rao–Blackwell). Doing this at every position (the prefix is sampled, the next-token distribution is exact) is full-vocabulary token-level distillation with `kind='fkl'`; its gradient with respect to the logits $z$ at each position is $(q-p)$ (at $\kappa=1$) by the book's softmax-KL derivation.
4. Reverse direction: $\nabla_\theta\sum_yQ(y)\log\frac{Q(y)}{P(y)}=\sum_y\nabla Q(y)\big(\log\frac{Q(y)}{P(y)}+1\big)=\mathbb{E}_Q\big[\nabla\log Q(y)\,\log\frac{Q(y)}{P(y)}\big]$, because $\sum_y\nabla Q(y)=0$. This is a score-function (REINFORCE) estimator with "reward" $-\log\frac{Q}{P}$; the prefix distribution depends on $\theta$, which is why it needs student samples (continues in D2 of 9.2).
5. Toy for (d). Vocabulary $\{A,B\}$, $T=2$, teacher $P(AA)=0.7$, $P(BB)=0.3$ (the 0.7/0.3 weights of `sec_mdl-fwd-vs-rev-kl`; the two modes are the coherent sequences). Student family: independent positions, $Q(y_1,y_2)=q_1(y_1)q_2(y_2)$ with $q_1=q_2=(a,1-a)$. Forward KL: $D_{\textrm{KL}}(P\Vert Q)=-H(P)-0.7\log a^2-0.3\log(1-a)^2$ with $H(P)=-0.7\log0.7-0.3\log0.3$; minimizing gives $a=0.7$ (the marginals; by the moment-matching argument of the book's proposition). The student then puts $2\cdot0.7\cdot0.3=0.42$ of its mass on the incoherent sequences $AB$, $BA$, where $P=0$. Reverse KL is finite only if $Q(AB)=Q(BA)=0$, which forces $a\in\{0,1\}$: $a=1$ gives $\log(1/0.7)=0.357$ nats, $a=0$ gives $\log(1/0.3)=1.204$ nats, the same two local optima the book finds for the Gaussian fit.
6. Interpretation for language: the incoherent mass of the forward optimum is a model of "a fluent blend of two answers", and the reverse optimum is a model that commits to one answer and loses the other mode's coverage. With a student that can represent $P$, both directions agree, which is why the choice matters only for the small-capacity students we distill into.

*Numerical check (cell `kd-toy-direction`, CPU, seconds).* Enumerate the four sequences; minimize both divergences over $a\in(0,1)$ on a 2001-point grid, and also with gradient descent on a logit. Assertions: forward optimum $a\in[0.699,0.701]$; forward KL at the optimum $=0.6109$ nats (value from the author's check of the same formula) and off-diagonal mass $=0.42\pm10^{-3}$; reverse KL at $a=1$ is $0.3567\pm10^{-4}$ and at $a=0$ is $1.2040\pm10^{-4}$; these two agree with the `sec_mdl-fwd-vs-rev-kl` numbers. A second assertion verifies step 2 on a random $3$-symbol, length-3 tabular model by enumerating all 27 sequences: $\big|D_{\textrm{KL}}(P\Vert Q)-\sum_t\mathbb{E}\,D_t\big|<10^{-10}$.

**D2 (supporting): top-$k$ compression and its bias.** *Statement.* Let $\tilde P$ be $P$ coarsened to $k+1$ cells: the top-$k$ tokens and one tail cell with mass $1-\sum_{i\le k}p_i$; define $\tilde Q$ likewise with the *same* cells. Then $D_{\textrm{KL}}(\tilde P\Vert\tilde Q)\le D_{\textrm{KL}}(P\Vert Q)$, with equality when $P/Q$ is constant on the tail, and the inequality tightens monotonically as $k$ grows (nested partitions). Renormalizing the top-$k$ instead (dropping the tail) is *not* a coarsening and can over- or under-estimate. *Steps.* (1) Data-processing inequality for the map "merge tail tokens". (2) Nested partitions refine, so the bound is monotone in $k$. (3) The gradient of the tail-bucket loss with respect to the student's logits for a tail token $j$ is $q_j\cdot(\tilde q_{\textrm{tail}}-\tilde p_{\textrm{tail}})/\tilde q_{\textrm{tail}}$, i.e. the student is told the *total* tail mass but not how it is spread. (4) The unbiased alternative: draw $m$ tokens $a_j\sim P$ per position and use $-\frac1m\sum_j\nabla\log Q(a_j)$, unbiased for $\nabla D_{\textrm{KL}}(P\Vert Q)$ at that position; its variance falls as $1/m$ (the "random sampling" idea of the sparse-logit paper cited below). *Numerical check (cell `kd-topk-bias`).* Random logits with a Zipf-like tail, $V=1000$, $k\in\{1,4,16,64\}$: tail-bucket KL is nondecreasing in $k$ and $\le$ the full KL to $10^{-9}$; the average of 20,000 sampled estimates of the gradient matches the full gradient within 3 Monte Carlo standard errors; renormalized top-$k$ gradient differs by an amount that the cell prints.

#### From-scratch implementation

Budget ≤ ~150 lines in total; every piece tested.

- `kd_loss` (≈45 lines). Per position: `lp = log_softmax(teacher/kappa)`, `lq = log_softmax(student/kappa)`; `fkl = (lp.exp()*(lp-lq)).sum(-1)`; `rkl = (lq.exp()*(lq-lp)).sum(-1)`; `jsd` with `m = alpha*p + (1-alpha)*q` computed in log space; multiply by `kappa**2` when `kappa != 1`; process positions in chunks; mask and average over response tokens only. Tests: (i) equals `torch.nn.functional.kl_div(lq, lp, log_target=True, reduction='none').sum(-1)` for `fkl` to $10^{-6}$, with a comment that `F.kl_div(input, target)` computes $D_{\textrm{KL}}(\text{target}\Vert\text{input})$, the common direction mistake; (ii) autograd gradient of `fkl` at $\kappa=1$ equals $q-p$ to $10^{-6}$ (seed 0, $V=50$); (iii) `jsd` at `alpha=1e-3` divided by `alpha` is within 1% of `fkl`, and at `alpha=1-1e-3` divided by `1-alpha` is within 1% of `rkl` (the GKD interpolation, argument order teacher first); (iv) the $\kappa^2$ scaling keeps the gradient norm within a factor 2 across $\kappa\in\{1,2,4\}$ on logits with scale 3 (to be measured, the claim is only "no vanishing", as in the book).
- `topk_compress` and `kd_topk_loss` (≈35 lines): `torch.topk` on log-softmax; tail mass via `log1p(-exp(logsumexp(top)))` with a clamp; student side gathers the same indices and forms the same coarsening. Tests: with `k=V` the loss equals `kd_loss('fkl')` to $10^{-6}$; tail-bucket loss $\le$ full loss on random inputs; storage arithmetic asserted: $194$ B per token at $k=32$ (32 int32 + 32 fp16 + one fp16) vs $303{,}872$ B, ratio $\approx1566$.
- `ComputeLedger` and `nonembedding_params` (≈30 lines). Tests as stated under Exports. A ledger test: generating 1,000 tokens with the 1.7B teacher records $2\times1.4\times10^9\times10^3=2.8\times10^{12}$ FLOPs; training the 0.6B student on the same tokens records $6\times0.44\times10^9\times10^3=2.64\times10^{12}$. The notebook points out that, per token, a teacher forward is about as expensive as a student training step; it is the *memory-bound* decode wall-clock that makes teacher generation the bottleneck.
- `make_problems` (≈50 lines). Family `'chain'`: multi-step word problems assembled from $M=24$ story templates (e.g. shopping, trains, recipes), each with $k\in\{2,\dots,6\}$ arithmetic steps (`difficulty=k`), integer answers, numbers drawn from a seeded RNG. Splits: `train` = templates $0$–$15$ with $k\in\{2,\dots,6\}$; `test_seen_templates` = the same templates with fresh numbers (a different seed stream, no shared number tuple); `test_unseen_templates` = templates $16$–$23$; `test_harder` = templates $0$–$15$ with $k\in\{7,8\}$. Tests: determinism by seed; the stated answer equals the answer of an independent evaluator of the underlying expression for 1,000 problems; no `(template, numbers)` pair shared between train and any test split.
- `uld_loss` (≈15 lines): sort probabilities descending in each vocabulary, zero-pad to the longer, L1 distance; tested to be invariant to a permutation of the vocabulary.

#### In production

- TRL's `GKDTrainer` (`lmbda` = on-policy fraction, `beta` = JSD interpolation, `temperature`) implements the same family with student-generated prefixes; the from-scratch `kd_loss` is the oracle: feed identical logits to `GKDTrainer`'s loss function (`generalized_jsd_loss`) and to `d2l.kd_loss(kind='jsd')` and compare to $10^{-5}$ in fp32. The notebook runs this comparison and one `GKDTrainer` step on 64 examples at `lmbda=0` (all off-policy) to show the plumbing; GKD's `beta` is called $\alpha$ here because $\beta$ is the KL coefficient in this part. Pin the TRL version and check the loss-function name in the pinned release (the name above is the one in the documentation at the time of writing; mark "verify in the pinned release").
- Offline logit caches: vLLM can return top-$k$ log-probabilities per generated token (`logprobs=k`); the cache builder in the notebook uses the HF forward for exactness (teacher-forcing the sampled text) and the exercise compares the two.
- Sequence-level distillation at scale is a data pipeline (3.5) plus the SFT of 7.2; for hosted teachers that return text only, sequence-level distillation is the only option, which links to the cross-tokenizer case.

#### Headline experiment

*Hypothesis.* At a fixed number of student training tokens, distilling from teacher distributions beats training on hard labels from the same text, and the ranking among divergences on *teacher* prefixes is small relative to the gap to on-policy training (9.2). Each arm is a *different bet*; no ranking is promised before the run.

*Setup.* Student `Qwen3-0.6B-Base` + LoRA (rank 64, all linear layers, `d2l.LoRALinear`), AdamW, constant LR after warmup, learning rate chosen once on the SFT-gold arm from $\{5\cdot10^{-5},10^{-4},2\cdot10^{-4}\}$ by validation loss and reused for every arm (the choice is reported, not tuned per arm), sequence length 512, 2,000 GSM8K-train prompts plus 1,000 generated prompts, 2 epochs over the sample file (student training tokens identical in all arms: equals the token count of the teacher sample file times 2), 3 seeds. Teacher sampled once offline (the sample file); logits scored once (cache for top-$k$; full logits are recomputed on the fly only in the `full_ft=True` T2 variant).

| Arm | Prefix source | Target | Student-side loss |
|---|---|---|---|
| A. SFT-gold | human GSM8K solutions | hard token | cross-entropy |
| B. SeqKD | teacher samples | hard token | cross-entropy |
| C. fKL-full (T2 variant only) | teacher samples | full distribution | forward KL |
| D. fKL-top32 | teacher samples | top-32 + tail | `kd_topk_loss` |
| E. rKL-top32 | teacher samples | top-32 + tail | reverse KL against the coarsened pair |
| F. JSD($\alpha=0.5$)-top32 | teacher samples | top-32 + tail | generalized JSD |

*Procedure.* Train each arm (3 seeds), then evaluate: (1) GSM8K test accuracy, greedy, 1,319 problems ("public, possibly seen"); (2) `test_unseen_templates` and `test_harder` accuracy (generated, clean), 500 problems each; (3) teacher-forced held-out loss on 200 teacher samples; (4) a *student-sample* reverse-KL estimate $\frac1{|y|}\sum_t(\log\pi_\theta-\log\pi_{\textrm{tea}})(y_t)$ over 200 student samples (this needs the teacher forward, run once per final checkpoint: 18 scorings); (5) pass@8 for coverage via `d2l.pass_at_k` ($n=8$ samples at temperature 0.7); (6) the ledger.

*Metrics and plots.* `fig_lm_kd_cost`: accuracy (y, with 95% paired-bootstrap interval over problems clustered by prompt; seed spread as separate dots) against all-in FLOPs from the ledger (x, log), one point per arm, so SeqKD's teacher decoding cost is visible; for the cached arms the one-time teacher scoring cost is stacked in a lighter bar. `tab_lm_kd_results`: the accuracy and loss numbers per arm. `fig_lm_kd_modes`: for arms D/E/F, histogram of per-sample student log-likelihood under the teacher (mode-seeking arms should show a narrower, higher-likelihood histogram) next to pass@8 (mode-covering arms should retain diversity); the figure states direction predictions as hypotheses, not outcomes.

*Expected outcome.* Directions only: arms using the teacher should outperform A on the generated unseen-template set if the teacher is better than the human text distribution for the student; reverse/JSD arms on teacher prefixes should lower the student-sample reverse KL relative to forward KL but may reduce pass@8. The sources for the directions are the papers: Kim and Rush report sequence-level KD improving over word-level KD in translation; MiniLLM reports that reverse KL on student samples yields more precise and less overconfident-hallucinating students than forward KL; GKD reports gains from student-generated prefixes. All magnitudes: to be measured.

*Noise check.* 3 seeds per arm; a difference is reported as real only if the paired-bootstrap 95% interval over problems excludes 0 *and* the seed-to-seed range of each arm is smaller than the gap. Pairing is by problem across arms. If most pairs fail the criterion, the notebook says so and the table is labeled "not resolved".

*T1 reduction.* This section is T1: LoRA student, cached top-$k$ teacher, teacher not resident during training. T2 host: `full_ft=True`, online full-vocab teacher scoring (arm C), 3 seeds.

*Artifacts committed.* Sample file, top-$k$ cache, ledger JSONL per arm, final LoRA adapters, `tab_lm_kd_results` as JSON.

#### Figures and tables

- `tab_lm_kd_design`: the three-decision taxonomy with cost columns (teacher decode tokens, teacher forward tokens, storage per token); made from the ledger formulas, not typed.
- `fig_lm_kd_modes` (toy): density panel of the 4-sequence teacher, forward-fit student, reverse-fit student (bars over AA, AB, BA, BB); second panel: the same on teacher-vs-student samples from the notebook. Cell `kd-toy-direction` and the headline cell.
- `fig_lm_kd_topk`: x = $k\in\{1,2,4,8,16,32,64,128\}$, y = fraction of full forward KL captured by the tail-bucket coarsening, on 20,000 teacher tokens; series: tail-bucket, renormalized; plus the storage line on a second axis.
- `tab_lm_kd_cost`: bytes per token and per million tokens for full bf16, full fp32, top-$k$ at $k\in\{8,32,128\}$; formulas in the cell.
- `fig_lm_kd_cost`, `tab_lm_kd_results` as above.
- `fig_lm_kd_tokenizer`: bar chart of the fraction of Qwen3 vocabulary strings present in the Qwen3.5 vocabulary, to be measured.

#### Pitfalls and misconceptions

- "Distillation means matching logits." Sequence-level distillation never touches logits, is the only option for API teachers, and is forward KL estimated by one sample.
- `F.kl_div` argument order (see test (i)); wrong direction silently gives reverse KL.
- Padding rows of the embedding matrix and unused special tokens leak probability mass; mask to the tokenizer length before the softmax.
- Applying temperature to the teacher but not the student (or the reverse); the $\kappa^2$ factor only keeps gradients comparable, it does not make different $\kappa$ equivalent.
- Chat template mismatch: Qwen3's own non-thinking template inserts an empty `<think></think>` block, which `d2l_chatml_v1` does not; teacher samples must be re-rendered with `d2l_chatml_v1` for the student (and for teacher scoring) or the first tokens carry a large spurious loss.
- Teacher greedy outputs are low-diversity; the teacher sampling temperature decides how much of $P$ the student ever sees.
- Comparing arms at equal steps instead of equal tokens or FLOPs; counting only student cost while the teacher's generation dominates.
- Renormalized top-$k$ is not an unbiased truncation of the teacher.

#### Summary (draft bullets)

- Distillation minimizes a divergence between teacher and student next-token distributions; the three decisions are target, prefix source and divergence.
- Sequence-level distillation is a one-sample Monte Carlo estimate of forward KL; full-vocabulary token-level distillation is its Rao–Blackwellized version on the same prefixes.
- Forward KL covers modes and can place mass on blends; reverse KL commits to a mode; capacity decides how much the choice matters.
- Top-$k$ caches shrink storage by three orders of magnitude; a tail bucket makes the loss a lower bound on the true KL, while renormalizing is biased.
- A different tokenizer breaks position-wise matching; sequence-level or sorted-distribution losses are the remedies.
- Compare methods at matched FLOPs from a ledger that includes the teacher.

#### Exercises

1. *(check)* The teacher puts 0.5 on token $a$, 0.5 on token $b$; a student can only output one of the two with probability 1 or a uniform mix over both. Which divergence prefers which, and what happens to coverage? *Sketch:* forward KL: uniform mix (KL 0); with the restricted family $\{\delta_a,\delta_b,\mathrm{mix}\}$ the mix achieves 0 and delta has infinite forward KL; reverse KL: both deltas give $\log 2$, the mix gives 0 as well; with only deltas available forward KL is infinite while reverse KL is $\log2$, so reverse picks a delta.
2. *(check)* Explain why `F.kl_div(log_q, p)` is $D_{\textrm{KL}}(p\Vert q)$ and which argument must hold the teacher for forward KL. Predict what changes if the arguments are swapped on a student that is nearly uniform. *Sketch:* PyTorch computes $\sum t(\log t-\textrm{input})$ with `target=t`; swapping yields reverse KL, which is mode-seeking and has smaller gradients where the student mass is small.
3. *(derive)* Show that for $\kappa>1$ the gradient of $\kappa^2D_{\textrm{KL}}(p^{(\kappa)}\Vert q^{(\kappa)})$ with respect to the student logits tends, as $\kappa\to\infty$, to $\tfrac{1}{V}\big[(z^{\textrm{stu}}_j-\bar z^{\textrm{stu}})-(z^{\textrm{tea}}_j-\bar z^{\textrm{tea}})\big]$ (logit matching), citing the book's derivation. *Sketch:* as in `subsec_mdl-distillation`; the result is the $\kappa\to\infty$ limit of the $T^2$ analysis, so high-temperature distillation is mean-centered logit regression.
4. *(derive)* Define the renormalized top-$k$ loss as $D_{\textrm{KL}}(\bar P\Vert\bar Q)$, where $\bar P,\bar Q$ are $P,Q$ restricted to the teacher's top-$k$ set and renormalized separately. Show it is not bounded above by the full KL by giving an example. *Sketch:* restriction by conditioning is not a data-processing map. Take $P=(0.5,0.3,0.2)$, $Q=(0.3,0.5,0.2)$, $k=2$: $\bar P=(0.625,0.375)$, $\bar Q=(0.375,0.625)$, so the renormalized loss is $0.25\ln(5/3)=0.128$ nats while the full KL is $0.2\ln(5/3)=0.102$ nats. (The tail bucket instead gives $0.102$ here because the tail cell has equal mass in both.)
5. *(measure)* Add arm G: SeqKD on verifier-filtered teacher samples only. Predict whether it beats B on `test_unseen_templates`, run, and report the paired interval. Say what fraction of the teacher's tokens are discarded and what that does to the ledger.
6. *(measure)* Vary $k\in\{4,32,128\}$ and plot accuracy of arm D against storage per token. Predict where the curve saturates using `fig_lm_kd_topk`; run three seeds at each $k$.
7. *(open)* Distill Qwen3.5-2B (different vocabulary) into Qwen3-0.6B-Base with `uld_loss` and with text-only sequence-level distillation on the same prompts. Design a fair comparison; defend how you count the cost of the teacher's tokenizer mismatch. There is no reference answer.

#### Further reading

- Hinton, Vinyals and Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531): soft targets and temperature.
- Kim and Rush, [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947): training on the teacher's outputs as an approximation of the sequence-level objective.
- Gu et al., [MiniLLM](https://arxiv.org/abs/2306.08543): reverse KL for generative students, policy-gradient optimization; the white-box setting.
- Agarwal et al., [GKD](https://arxiv.org/abs/2306.13649): generalized JSD and student-generated prefixes; read it together with 9.2.
- [DistiLLM](https://arxiv.org/abs/2402.03898) (new): skew KL divergences and an adaptive off-policy schedule, a practical middle path between the two directions.
- [Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss](https://arxiv.org/abs/2402.12030) (new): cross-tokenizer distillation; the source of `uld_loss`.
- [Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](https://arxiv.org/abs/2503.16870) (new): biased top-$k$ and an unbiased sampled alternative in LLM distillation.
- Qwen team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (new): how the small Qwen3 models were distilled from large ones.
- Code: [TRL GKD trainer](https://huggingface.co/docs/trl/gkd_trainer); [MiniLLM reference code in LMOps](https://github.com/microsoft/LMOps/tree/main/minillm).
- Course/blog: Thinking Machines, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (read after 9.2); Stanford CS336 and CMU Advanced NLP lecture notes on distillation, from the part's course list (specific lecture links to be added by the editor).

#### Risks and open decisions

- *Student must see the same chat template as the teacher.* The part's template is `d2l_chatml_v1` (7.2; ChatML, no think blocks): samples are stored as response text and re-rendered with `d2l.render_chat` for the student and for any teacher scoring, so both see the same token ids. Qwen's own template (`enable_thinking=False`, which renders an empty think block) is used only where the teacher generates. Fallback: build both from the same Jinja path and assert equality of the first 20 tokens.
- *T1 memory with a resident teacher.* The training cell never loads the teacher. Fallback if the cache is too large: reduce to $k=16$ or 1,000 prompts; the notebook prints the cache size before training.
- *Qwen3.5-2B tokenizer and chat template under the pinned `transformers` version.* If unsupported, the cross-tokenizer cell uses text-only samples committed in the artifact and skips the live tokenizer comparison.
- *TRL API drift for `GKDTrainer`.* The production cell pins a version and tests the loss function by name; if renamed, compare only the final losses via a one-step trainer run.
- *GSM8K contamination.* Handled by the chapter policy; if the student's GSM8K accuracy exceeds its accuracy on `test_seen_templates` by a margin the notebook sets after measurement, print a contamination warning.
- *Custom losses in `d2l.sft_train` (7.2).* Arms D–F need a distillation loss; they pass `loss_fn(model, batch, num_items)` to `d2l.sft_train` (the hook exists in 7.2's signature). If a loss needs per-batch teacher tensors that `sft_collate` does not carry, the notebook wraps the batches in a generator that attaches them.

### 9.2 On-policy distillation

- **Label:** `sec_lm_onpolicy_distillation` · **Tier:** T2 (reduced T1 configuration: student Qwen3-0.6B-Base + LoRA, teacher Qwen3-1.7B, 40 online steps, plus the committed logs of the T2 run for every analysis cell) · **Budget:** T2: a few hours on 1× A100 40 GB; the arms train a LoRA adapter (r=16, all linear layers, 17,432,576 parameters on Qwen3-1.7B: 0.28 GB at 16 B per parameter) on the frozen bf16 student (1,720,574,976 × 2 B = 3.44 GB), beside a second copy of the student in the vLLM rollout engine (3.44 GB plus KV cache) and the 8B teacher in bf16 (8,190,735,360 × 2 B = 16.38 GB): 23.3 GB of resident weights, leaving about 17 GB of the card's ~38 GiB for KV cache, activations and teacher logits (teacher scoring in micro-batches of at most 8 sequences with a chunked log-softmax: one 8 × 512 × 151,936 fp32 logits tensor is already 2.49 GB, the 9.1 figure); peak memory to be measured, target ≤ 38 GiB. The earlier plan (full fine-tuning of the 1.7B student, AdamW state 27.5 GB, beside the 16.4 GB teacher: 43.9 GB before any activation) does not fit a 40 GB card, so the T2 arms use LoRA as the T1 configuration does; the S0 SFT warm-up (below) runs alone and may be full fine-tuning, because the teacher is not loaded then. T1: to be measured, target ≤ ~15 min for the live 40-step run and ≤ 15 GiB (teacher resident, student LoRA) · **Codex units:** 10.4, 10.5, 10.6, 10.7
- **Builds on (book):**
  - `sec_imitation` / `subsec_il_error_bounds`, `subsec_il_dagger`: the $\varepsilon T^2$ vs $\varepsilon T$ proposition and DAgger. 9.2 uses them as the frame; it does not re-prove either.
  - `sec_rl_sequences` (`tab_rl_notation_contract`, `eq_seq_factorization`): token MDP, $\hat G_t$, $K$, $\rho_t$, $\beta$. Notation used verbatim; $\pi_{\textrm{tea}}$ as in 9.1.
  - `sec_baselines` (zero-mean lemma, reward-to-go), `sec_ppo` (`eq_ppo_clip`, ratio $\rho_t$), `sec_regularized` (`eq_kl_objective`, `eq_kl_optimum`, `eq_dpo_reward`, `subsec_kl_direction`): the derivation D2 recasts distillation as the KL-regularized problem of `sec_regularized`.
- **Builds on (this part):** 9.1 (`d2l.kd_loss`, `d2l.topk_compress`, `d2l.ComputeLedger`, `d2l.make_problems`, teacher samples), 8.1 (`d2l.token_logprobs`), 8.5 (`d2l.grpo_train`, `d2l.verify_math`, `d2l.sample_responses`), 8.6 (`d2l.sync_weights`), 7.2 (`d2l.sft_train`, for `loss_fn`-style SFT updates only if reused), 1.5 (`d2l.paired_bootstrap`), 7.4 (`d2l.LoRALinear`).
- **Exports** (`#@save`, new):
  - `d2l.opd_loss(logp, old_logp, teacher_logp, mask, mode='sampled', clip=None) -> (loss, stats)`: `mode='sampled'` uses the dense reward $r_t=\log\pi_{\textrm{tea}}-\log\pi_{\textrm{old}}$ with ratio $\rho_t$; `mode='full'` is delegated to `kd_loss(kind='rkl')` on logits and takes `logits` instead; `stats` has mean per-token reverse KL, token count, fraction clipped.
  - `d2l.OnPolicyDistiller(student, tokenizer, teacher, engine, config)`: `.step(problems) -> dict` (rollout, teacher scoring, update, optional weight sync), `.train(problems, gen_token_budget) -> history`; the history records `gen_tokens, teacher_tokens, flops, wall_s, rkl, mean_len` per step; `config.mode in {'sampled', 'full', 'jsd'}`, `config.lam` ($\lambda$, on-policy fraction, default 1.0), `config.teacher_context` in `{'none', 'privileged'}` (OPSD).
  - `d2l.privileged_prompt(problem, level='solution') -> str`: builds the OPSD teacher context from the gold solution (`level='answer'` or `'solution'`) and asserts the student prompt never contains it.
  - Data: `DATA_HUB['lm-reasoning-logs-v1']` (JSONL logs of the four arms; shared with 9.3 and 9.7), `lm-opd-student-sft0` (the SFT-initialized student shared by all arms).
- **Models and data:**
  - T2: student `Qwen/Qwen3-1.7B-Base` (apache-2.0; 1.72B parameters; 28 layers, hidden 2048), teacher `Qwen/Qwen3-8B` (apache-2.0; 8.19B; hidden 4096, 36 layers, untied head), non-thinking mode (`enable_thinking=False`; recommended sampling temperature 0.7, top-p 0.8, top-k 20). Teacher and student share the tokenizer (vocab 151,936), the notebook asserts it. Chat template: prompts and responses are rendered with the part's `d2l_chatml_v1` (7.2) for the student and for teacher scoring alike, so both models see identical token ids; Qwen's own `enable_thinking=False` template is used only when the teacher *generates* the offline arm's samples, and the stored sample is the response text without the empty think block. The teacher was post-trained with that empty block in front of its answers, so scoring text without it is slightly off its training distribution at the first response tokens: a cell prints the teacher's log-probability of the first 8 response tokens with and without the block, and the headline uses the block-free render for both models.
  - T1: student `Qwen/Qwen3-0.6B-Base`, teacher `Qwen/Qwen3-1.7B`.
  - All arms start from the *same* SFT-initialized student `S0`: 500 teacher samples on `chain` train problems, 1 epoch of SFT (7.2 trainer), to teach the answer format `#### <integer>` so that early rollouts are verifiable. A raw base model would give GRPO almost no reward at step 0 and OPD off-format rollouts.
  - Problems: `d2l.make_problems('chain', ...)` from 9.1, splits `train`, `test_unseen_templates`, `test_harder`; public check: `openai/gsm8k` test (1,319; MIT; "public, possibly seen").
- **Packages:** `vllm>=0.10` (rollouts; HF `generate` fallback), `transformers`, `peft`, `torch>=2.4`; the pinned versions go in the notebook header.

#### Narrative arc

1. **The question.** Opening cell: take $S_0$, the offline-distilled student of 9.1 (arm D), and measure what the offline objective did not see: sample 200 solutions from the student, score them under the teacher, and print the per-token reverse KL as a function of position $t$ in the response. Symptom: the reverse KL grows with position, and the first token at which the student's log-probability under the teacher drops below $\log(10^{-3})$ is typically followed by a long tail of high-loss tokens: the student wanders to prefixes the teacher was never asked about. (The numbers are to be measured; the cell prints a median first-bad-token position and the share of student responses that contain one.)
2. **Why offline distillation trains on the wrong prefixes.** One paragraph tying `subsec_il_dagger` to this setting. A table `tab_lm_opd_prefix` of *who generates the prefix* and *who labels the next token* for BC/SFT (human, human), SeqKD (teacher, teacher), fKL on teacher prefixes (teacher, teacher distribution), GKD with $\lambda$ (student mixture, teacher distribution), OPD (student, teacher distribution), GRPO (student, verifier on the final answer), OPSD (student, same model with privileged context).
3. **The estimator in code.** A 15-line sampled-token OPD loss and a 10-line full-distribution version; tests against enumerated gradients on the tabular toy (D2 below). Show that the loss needs only the *student's* sampled tokens, so any teacher that can score a text can be used.
4. **Derivation D1: OPD is KL-regularized RL with the teacher's implicit reward.** Result and numerical check (below). This is the part that makes OPD "dense-reward RL" rather than a new idea.
5. **Derivation D2: gradient estimators, and what "per-token" drops.** Return-to-go vs discount-zero, with the exact-enumeration check.
6. **GKD and the mixing knob.** $\lambda$ interpolates offline and on-policy prefixes; JSD($\alpha$) interpolates directions (9.1 `kd_loss`). Table of `GKDTrainer` options mapped to `OnPolicyDistiller` config.
7. **Self-distillation with privileged context (OPSD).** Same code, teacher context differs; leakage audit cell; failure modes (answer-first shortcuts, teacher copying the hint).
8. **Cost accounting.** A ledger-based cost model per update for GRPO ($K$ samples per prompt, one scalar reward per sample) and OPD (one sample per prompt, one teacher score per token); decode vs prefill wall-clock.
9. **Headline experiment** (below): four arms at matched generated tokens.

#### Derivations

**D1. On-policy distillation as KL-regularized RL with a dense per-token reward.**

*Statement.* Let $P=\pi_{\textrm{tea}}(\cdot\mid x)$, $Q=\pi_\theta(\cdot\mid x)$ and let $\pi_{\textrm{ref}}$ be any strictly positive reference (the SFT-initialized student $S_0$ in practice). Define the sequence reward $R(x,y)=\log P(y\mid x)-\log\pi_{\textrm{ref}}(y\mid x)$, the teacher's *implicit reward* (`eq_dpo_reward` with $\beta=1$). Then (a) $D_{\textrm{KL}}(Q\Vert P)=D_{\textrm{KL}}(Q\Vert\pi_{\textrm{ref}})-\mathbb{E}_{y\sim Q}R(x,y)$, so minimizing the sequence reverse KL is exactly the KL-regularized objective `eq_kl_objective` with $\beta=1$ and reward $R$; (b) its unique optimum is $Q^\star\propto\pi_{\textrm{ref}}e^{R}=P$ (`eq_kl_optimum`); (c) $R$ decomposes over tokens, $R=\sum_t\big[\log P(y_t\mid s_t)-\log\pi_{\textrm{ref}}(y_t\mid s_t)\big]$, so the per-token reward $r_t=\log P(y_t\mid s_t)-\log Q(y_t\mid s_t)$ (the $\pi_{\textrm{ref}}$ part moved into the KL penalty) assigns each token its own share of the credit; (d) the sequence-level gradient is $\nabla_\theta D_{\textrm{KL}}(Q\Vert P)=-\mathbb{E}\big[\sum_t\nabla\log Q(y_t\mid s_t)\,\hat G_t\big]$ with $\hat G_t=\sum_{t'\ge t}r_{t'}$.

*Assumptions.* Finite vocabulary and length; $Q\ll P$ (or both positive); $P$ is evaluated exactly by the teacher forward pass.

*Steps.*
1. Expand $D_{\textrm{KL}}(Q\Vert P)=\mathbb{E}_Q[\log Q-\log P]=\mathbb{E}_Q[\log Q-\log\pi_{\textrm{ref}}]-\mathbb{E}_Q[\log P-\log\pi_{\textrm{ref}}]$. This is (a).
2. By `eq_kl_optimum` the maximizer of $\mathbb{E}_QR-D_{\textrm{KL}}(Q\Vert\pi_{\textrm{ref}})$ is $\pi_{\textrm{ref}}e^{R}/Z$; here $e^{R}\pi_{\textrm{ref}}=P$ is already normalized, so $Z=1$ and $Q^\star=P$ (b).
3. Factorize $P$ and $\pi_{\textrm{ref}}$ with `eq_seq_factorization` to get (c). The token-MDP of `sec_rl_sequences` has deterministic transitions, so $r_t$ depends on $(s_t,y_t)$ only; reward arrives at *every* step, unlike the terminal verifier of 8.5. This is the sense of "dense".
4. Write $J(\theta)=\mathbb{E}_{y\sim Q}\sum_tc_t$ with $c_t=\log Q(y_t\mid s_t)-\log P(y_t\mid s_t)=-r_t$. The score-function identity gives $\nabla J=\mathbb{E}\big[\sum_t\nabla\log Q(y_t\mid s_t)\sum_{t'}c_{t'}\big]+\mathbb{E}\big[\sum_t\nabla c_t\big]$. The second term is $\mathbb{E}\sum_t\nabla\log Q(y_t\mid s_t)=0$ by the zero-mean lemma (`sec_baselines`) applied conditionally on $s_t$. Causality (`sec_baselines`, reward-to-go) removes $c_{t'}$ for $t'<t$ from the weight of $\nabla\log Q(y_t\mid s_t)$. Hence (d).
5. Interpretation: with $\hat G_t$ replaced by the terminal verifier reward and a group baseline one obtains GRPO (`sec_rl_sequences`); OPD replaces the sparse reward by the teacher's dense log-ratio. The learned-from-teacher reward needs no preference data and no reward model, and it cannot be hacked in the sense of `subsec_rl_sequences_hacking` *relative to the teacher*, though it inherits the teacher's errors.

*Numerical check (cell `opd-kl-identity`, CPU, seconds).* Random tabular autoregressive models with $V=3$, $T=3$ (13 prefixes, logits $\sim\mathcal N(0,1)$ student, $\mathcal N(0,1.5^2)$ teacher, seed 0): enumerate all 27 sequences; assert (a) $|D_{\textrm{KL}}(Q\Vert P)-[D_{\textrm{KL}}(Q\Vert\pi_{\textrm{ref}})-\mathbb{E}_QR]|<10^{-10}$ for a random $\pi_{\textrm{ref}}$; (b) gradient descent on the regularized objective converges to $P$ within $10^{-6}$ total variation.

**D2. Gradient estimators for OPD: return-to-go, per-token, and what each drops.**

*Statement.* (a) $\hat g_{\textrm{full}}=-\sum_t\nabla\log Q(y_t\mid s_t)\hat G_t$ is an unbiased estimator of $\nabla D_{\textrm{KL}}(Q\Vert P)$. (b) $\hat g_0=-\sum_t\nabla\log Q(y_t\mid s_t)\,r_t$ (discount zero) is unbiased for $\sum_t\mathbb{E}_{s_t\sim Q_{\theta_0}}\nabla_\theta D_{\textrm{KL}}\big(Q(\cdot\mid s_t)\Vert P(\cdot\mid s_t)\big)\big|_{\theta=\theta_0}$, the gradient of the per-token reverse KL with the visited-state distribution *held fixed*. (c) $\mathbb{E}\hat g_0-\mathbb{E}\hat g_{\textrm{full}}=\mathbb{E}\big[\sum_t\nabla\log Q(y_t\mid s_t)\sum_{t'>t}r_{t'}\big]$: the effect of today's token on later states, which discount zero ignores. (d) With full-vocabulary evaluation at each visited state the first-token sampling noise disappears: $\hat g_{\textrm{vocab}}=\sum_t\nabla D_{\textrm{KL}}(Q(\cdot\mid s_t)\Vert P(\cdot\mid s_t))$ has the same mean as $\hat g_0$.

*Steps.*
1. (a) is D1(d).
2. (b): conditional on $s_t$, $\mathbb{E}_{y_t}[\nabla\log Q(y_t\mid s_t)\,r_t]=-\sum_a\nabla Q(a\mid s_t)\log\frac{Q(a\mid s_t)}{P(a\mid s_t)}=-\nabla D_{\textrm{KL}}(Q(\cdot\mid s_t)\Vert P(\cdot\mid s_t))$ (the same algebra as D1 of 9.1 step 4, per state). Taking $\mathbb{E}_{s_t\sim Q}$ gives (b).
3. (c) subtract (a) from (b).
4. (d) is step 2 without sampling $y_t$.
5. Where the omitted term matters: it is the "state-distribution term" that DAgger and GKD also drop (they differentiate a per-state loss on learner states); it is exactly the term a long-horizon return would add. Thinking Machines argues discount zero works because the reward is dense; the check below measures how much it differs on a small model.
6. Relation to `subsec_il_error_bounds`: $J(Q)=\sum_t\mathbb{E}_{s_t\sim Q}c_t$ is *the* quantity OPD minimizes, measured on the student's own states, which is the "(ii)" condition of the proposition (error controlled under $\hat\pi$'s own state distribution). Offline distillation minimizes $\sum_t\mathbb{E}_{s_t\sim P}D_{\textrm{KL}}(P\Vert Q)=D_{\textrm{KL}}(P\Vert Q)$ (9.1 D1(a)), the "(i)" condition. For a terminal reward $R\in[0,1]$ the population-level offline bound is already linear in $T$: $|\mathbb{E}_QR-\mathbb{E}_PR|\le\mathrm{TV}(P,Q)\le\sqrt{D_{\textrm{KL}}(P\Vert Q)/2}$ (Pinsker), and the hybrid-argument bound $\mathrm{TV}\le\sum_t\mathbb{E}_{s_t\sim P}\mathrm{TV}_t$ also holds. The $T^2$ of the proposition belongs to the model where one deviation irrecoverably loses all remaining reward. *What on-policy training removes* is therefore the gap between the state distribution on which the per-token loss is *measured and trained* (empirical teacher samples, finite $N$) and the one on which the student *acts* (its own samples): generalization to unvisited prefixes is not controlled by teacher-state training, which is the cloned-imitator mechanism of the book's chain example. Phrase this precisely in the text; do not claim that offline distillation compounds quadratically for a soft teacher.

*Numerical check (cell `opd-gradient-check`, CPU).* Same tabular model, seed 0. (i) Exact $\nabla J$ by finite differences vs the enumerated expectation of $\hat g_{\textrm{full}}$: max absolute difference $<10^{-8}$ (the author's check gives $3.5\times10^{-10}$). (ii) Enumerated $\mathbb{E}\hat g_0$ vs $\sum_t\mathbb{E}_{s_t}\nabla D_{\textrm{KL}}(Q(\cdot\mid s_t)\Vert P(\cdot\mid s_t))$ computed with autograd and frozen state weights: difference $<10^{-8}$. (iii) $\mathbb{E}\hat g_0$ vs $\mathbb{E}\hat g_{\textrm{full}}$: the cell prints the cosine and relative difference; the author's run gives cosine $0.984$ and relative difference $0.30$ for this seed, to be re-measured (values depend on the seed; the assertion is only that they differ by more than $10^{-3}$ and that cosine $>0$). (iv) 20,000 Monte Carlo sequences reproduce (i) within 3 standard errors per coordinate (median over coordinates).

*Toy offline-vs-online check (cell `opd-toy-offline-online`, CPU).* $V=3$, $T=4$ (40 prefixes), teacher logits $\mathcal N(0,2^2)$, student tabular logits starting uniform. Offline: 40 teacher sequences, 300 full-vocabulary forward-KL sweeps over visited prefixes. Online: 10 rounds of 4 student sequences, 30 full-vocabulary reverse-KL sweeps over the visited prefixes of each round (40 sampled sequences in total). Report exact $D_{\textrm{KL}}(Q\Vert P)$ and $D_{\textrm{KL}}(P\Vert Q)$ over 20 seeds. The author's run shows the online reverse KL lower in 16 of 20 seeds (mean $0.084$ vs $0.138$) and lower forward KL as well (mean $0.049$ vs $0.073$), because in this toy the offline student never updates unvisited prefixes. The cell must report both directions, state that the toy confounds sample diversity with training direction, and make no claim beyond the paired counts over seeds.

#### From-scratch implementation

Budget ≤ ~150 lines.

- `opd_loss(mode='sampled')` (≈25 lines): `r = teacher_logp - old_logp` (detached); `rho = exp(logp - old_logp)`; `loss = -(rho * r * mask).sum() / mask.sum()`; optional `clip` on `r` (e.g. to $[-c,c]$) and on $\rho$ following `eq_ppo_clip` with `clip_low/high`; `stats` as above. Test: with `old_logp = logp.detach()` the gradient of `loss` equals $-\frac1n\sum_t r_t\nabla\log Q$; on the tabular toy its Monte Carlo average matches $\mathbb{E}\hat g_0$ within 3 SE. A `mode='sampled'` test where teacher = student gives $r_t=0$ and zero gradient.
- `token_logprobs` is imported from 8.1 (`(model, input_ids, attention_mask, response_mask=None, temperature=1.0) -> (B, T-1)`), not redefined. This section adds only a test cell: compare against `torch.nn.functional.cross_entropy(reduction='none')` to $10^{-6}$ and check shift handling (output position $t$ scores token $t+1$).
- `OnPolicyDistiller.step` (≈70 lines): sample $B$ prompts; `d2l.sample_responses(engine, prompts, K=1, max_new_tokens=L_max, temperature=...)` returns `ids, resp_mask, logp_sampler` (8.5; use `logp_sampler` as `old_logp`, or recompute with `d2l.token_logprobs` under `no_grad`); teacher forward on prompt+response with the teacher context if privileged; compute $r_t$ and loss; optimizer step; ledger entries `add_generate(N_student, tokens)`, `add_score(N_teacher, tokens)`, `add_train(N_student, tokens)`; every `sync_every` steps call `d2l.sync_weights(trainer, engine, mode)` (8.6). For $\lambda<1$ a fraction $1-\lambda$ of the batch is drawn from the stored teacher samples (9.1 file) and uses `kd_loss(kind='fkl')` (GKD's mixing); `lam=1` is pure OPD. Termination: a response is cut at `L_max` and, if no EOS, marked truncated; truncated tokens are still scored (document the bias in Pitfalls).
- `privileged_prompt` (≈15 lines) with the leakage assertion. Test: student prompt contains no substring of the gold solution beyond the question text, checked on 500 problems.
- T0 test of the whole loop (CPU, seconds): a 2-layer `d2l.GPT` student, a 2-layer teacher with different random weights, vocabulary 64, length-8 responses, 30 steps; the reverse KL on student samples decreases by at least 20% (loop sanity, not a result).

#### In production

- TRL `GKDTrainer` (on-policy fraction `lmbda`, `beta` interpolation, `seq_kd` flag) and verl's OPD recipe (documentation linked below) run the same objective with sharded teachers and vLLM rollouts. The from-scratch loss is the oracle: compare `d2l.opd_loss(mode='full')` against `GKDTrainer`'s loss at `beta=1` (reverse-KL limit) on identical logits to $10^{-5}$ in fp32 (the exact loss function name in the pinned TRL release is to be verified).
- System notes for the text (no new derivation): the teacher forward is a prefill of the student's tokens, so it is compute-bound and parallelizes across the batch; the student decode is memory-bound; teacher and student can live on different GPUs (the 8.6 colocated vs split layouts apply unchanged); the sampler's log-probs differ slightly from the trainer's (different kernels), the standard mismatch that 8.6 treats, so the importance ratio $\rho_t$ uses trainer log-probs for both.
- Multi-teacher on-policy distillation in a frontier setting is mentioned in one sentence with a pointer to the DeepSeek-V4.1-Flash report (arXiv 2609.19969, title verified; the Codex plan places the discussion in its section 5.2.4, *unverified*), per the part rule that technical reports are analyzed only in Chapter 14.

#### Headline experiment

*Hypothesis.* At a fixed number of tokens *generated*, OPD reaches a higher accuracy than GRPO and than offline distillation on `test_unseen_templates` and `test_harder`, and uses fewer student rollouts per unit of improvement because every token carries a reward; the all-in FLOPs ranking may differ because the teacher is larger than the student.

*Setup.* Student $S_0$ (1.7B-Base after SFT warm-up) for all arms; same train problems stream; same generated-token budget $G$ (the cumulative tokens produced by the sampling model: the student for OPD/GRPO/OPSD, the teacher for the offline arm); $G$ set by the config `gen_token_budget`, calibrated after timing so that the GRPO arm takes about one hour on 1× A100 (value to be set from the measurement, not asserted here). Evaluate at 8 evenly spaced token counts. 3 seeds per arm. Common: AdamW, LoRA r=16 on all linear layers with LR $10^{-4}$ at both scales (S0's SFT warm-up excepted; the rate is confirmed on a short dev stream, as 8.5 does), max response 512 tokens, temperature 1.0 for rollouts and 0.0 for evaluation (greedy), 500 problems per eval split.

| Arm | Sampler of tokens | Supervision per update | Code path |
|---|---|---|---|
| Offline | teacher | teacher top-32 distribution on teacher prefixes (forward KL) | 9.1 `kd_topk_loss` |
| OPD | student ($K=1$ per prompt, 64 prompts per step) | teacher log-prob of each sampled token | `OnPolicyDistiller(mode='sampled')` |
| GRPO | student ($K=8$ per prompt, 16 prompts per step) | verifier on the final answer | 8.5 `d2l.grpo_train` (`cfg.K=8`, `prompts_per_step=16`) with the same `max_new_tokens` |
| OPSD | student ($K=1$) | the student's own weights with the gold solution in context | `OnPolicyDistiller(teacher_context='privileged')` |

*Procedure.* Run each arm to $G$, logging the ledger at every update. At each evaluation point, record accuracy (3 splits + GSM8K test), mean response length, per-token reverse KL to the 8B teacher on student samples (a diagnostic for all arms, not only OPD), fraction of responses without a parsable answer, and cumulative `gen_tokens`, `teacher_tokens`, FLOPs, wall-clock.

*Plots.* `fig_lm_opd_tokens`: accuracy (y; panels: `test_unseen_templates`, `test_harder`) against cumulative generated tokens (x, log), one line per arm, thin lines per seed, band = paired-bootstrap interval over problems (cluster = template for the unseen split; with only 8 unseen templates the clustered band is wide and the caption says so). `fig_lm_opd_flops`: the same against all-in FLOPs (log), teacher scoring and teacher generation included. `fig_lm_opd_kl`: per-token reverse KL to the teacher vs tokens. `tab_lm_opd_cost`: tokens to reach 80% of the best arm's final accuracy (or "not reached"), with seeds as a range.

*Expected outcome.* Direction only. The Thinking Machines post reports that on-policy distillation reaches RL-level (and better) performance at a fraction of the compute of RL; take its numbers from the post when stating the claim and compare with the measured ratio. GKD reports gains from student-generated prefixes over offline distillation. OPSD: take the claims about improvement without a larger teacher from the source paper; whether a 1.7B model with a hint teaches itself on this task is to be measured, and the notebook is allowed to report that it does not.

*Noise check.* 3 seeds; paired bootstrap over problems (same problems for all arms and checkpoints); claim a ranking at an evaluation point only if intervals for the difference exclude 0 for at least 2 of the 3 seeds' bootstrap and the seed ranges do not overlap; otherwise the table says "unresolved". Curves are averaged over seeds only after verifying that seed curves are ordered consistently.

*T1 reduction.* `scale='t1'`: 0.6B LoRA student, 1.7B teacher, GRPO arm 12 steps, OPD arm 40 steps, evaluation on 100 problems per split (intervals wide, stated); the notebook then loads the committed T2 logs for all plots. T1 live run is a smoke test of the code path and the ledger.

*Artifacts committed for T2.* `lm-reasoning-logs-v1` (JSONL of ledgers and eval records for all arms and seeds), `lm-opd-student-sft0`, final LoRA adapters per arm (safetensors; the merged bf16 weights are not committed), the eval JSON behind every table.

#### Figures and tables

- `tab_lm_opd_prefix`: who generates prefixes and who labels next tokens, per method (static table, with the matching `d2l` call).
- `fig_lm_opd_position_kl`: opening-cell figure, per-position reverse KL of $S_0$ samples under the teacher with a bootstrap band over responses.
- `fig_lm_opd_toy`: toy offline-vs-online paired scatter over 20 seeds (cell `opd-toy-offline-online`).
- `fig_lm_opd_tokens`, `fig_lm_opd_flops`, `fig_lm_opd_kl`, `tab_lm_opd_cost` (headline).
- `tab_lm_opd_estimators`: the D2 check as a table: cosine and relative difference between $\hat g_{\textrm{full}}$, $\hat g_0$ and $\hat g_{\textrm{vocab}}$ expectations.

#### Pitfalls and misconceptions

- "On-policy" means the prefixes come from the *current* student; stale rollouts (reused for many updates) are partly off-policy and need the ratio $\rho_t$ or a refresh.
- Reverse KL measured on a *sampled token* is an unbiased but high-variance estimate of the per-token KL; the full-distribution variant removes the sampling noise at the cost of teacher logits for the whole vocabulary (or top-$k$).
- Truncated responses: tokens beyond `L_max` are never generated, so the student is not trained about ending; an unfinished response gets no EOS reward. Document and count truncation.
- OPD copies teacher errors with high fidelity; the verifier and teacher can disagree. Report accuracy under the verifier, not only KL.
- Privileged-context leakage: the student prompt must not contain the hint; evaluation is question-only; the teacher in context may copy the answer and assign high probability to answer-first text, rewarding shortcut behavior.
- Comparing at equal *steps*: an OPD step with 64 samples and a GRPO step with 128 samples are not equal work; use generated tokens and FLOPs.
- The teacher forward is cheaper per token than decode in wall-clock but not in FLOPs.
- "Removes compounding error" is too strong; see D2 step 6 for the precise statement.

#### Summary (draft bullets)

- Offline distillation trains on the teacher's prefixes; the student acts on its own, and the per-token loss is never measured there.
- Sequence reverse KL is a sum of per-token reverse KLs on the student's own prefixes; with the teacher's log-ratio as per-token reward, OPD is policy-gradient RL with a dense reward.
- OPD is exactly KL-regularized RL with the teacher's implicit reward $\log\pi_{\textrm{tea}}-\log\pi_{\textrm{ref}}$ and $\beta=1$; its optimum is the teacher.
- The per-token (discount-zero) estimator drops the effect of a token on later states; the cost of that approximation is measurable.
- OPSD replaces a larger teacher by the same model with privileged context; it needs a leakage audit and can fail.
- Compare OPD, offline distillation and GRPO at matched generated tokens and again at matched FLOPs; the two orderings can differ.

#### Exercises

1. *(check)* For which teacher/student pairs is the sampled-token reward $r_t$ exactly zero in expectation at every state? What is $\mathbb{E}_{y_t}[r_t\mid s_t]$ in general? *Sketch:* $\mathbb{E}_{y_t\sim Q}[\log P-\log Q]=-D_{\textrm{KL}}(Q\Vert P)(s_t)\le0$, zero iff $Q(\cdot\mid s_t)=P(\cdot\mid s_t)$; reward is non-positive in expectation, so the optimization is "increase a negative number to zero".
2. *(check)* A student response is cut at the length limit before EOS. Explain which terms of $\hat G_t$ are missing, and which direction the bias takes if the teacher would have continued with high-loss tokens. *Sketch:* the tail $\sum_{t'>L}r_{t'}$ is dropped; if it is negative (student mismatches the teacher later) truncation hides it and the student is under-penalized for long rambling responses.
3. *(derive)* Show that for a baseline $b(s_t)$ that depends only on the prefix, $\hat g_0$ with $r_t-b(s_t)$ remains unbiased, and give the baseline that centers $r_t$ at each state and say what it changes. *Sketch:* the zero-mean lemma; $b(s_t)=\mathbb{E}[r_t\mid s_t]=-D_{\textrm{KL}}(Q\Vert P)(s_t)$ lowers the variance and leaves the mean unchanged, so the expectation still equals the full-vocabulary gradient.
4. *(derive)* Using D1(a)–(b), show that if the reference $\pi_{\textrm{ref}}$ is replaced by $S_0$ and the objective is run with a coefficient $\beta\ne1$ on the KL to $\pi_{\textrm{ref}}$, the optimum is $Q^\star\propto\pi_{\textrm{ref}}(P/\pi_{\textrm{ref}})^{1/\beta}$, i.e. a *tempered interpolation* between $\pi_{\textrm{ref}}$ and the teacher. *Sketch:* reward $R/\beta$ in `eq_kl_optimum` gives $Q^\star\propto\pi_{\textrm{ref}}e^{R/\beta}=\pi_{\textrm{ref}}^{1-1/\beta}P^{1/\beta}$; $\beta=1$ recovers $P$; $\beta\to\infty$ recovers $\pi_{\textrm{ref}}$.
5. *(measure)* Vary the on-policy fraction $\lambda\in\{0,0.25,0.5,1\}$ (GKD mixing) at fixed $G$ on the T1 configuration, 3 seeds. Predict the shape of accuracy vs $\lambda$ and say whether any interval excludes $\lambda=1$ vs $\lambda=0.5$.
6. *(measure)* Compare `mode='sampled'` and `mode='full'` (top-32) at equal $G$: report the variance of the per-step gradient norm over 100 steps and the final accuracy. Predict which has the lower variance and why (*sketch for the prediction:* full removes the sampling noise of $y_t$).
7. *(open)* Use the OPSD code with a *concision-conditioned* teacher (same model, prompted to answer briefly) to shorten the student's responses at equal accuracy; this previews 9.7. Define what you will count as "equal accuracy" and how you will keep the concision instruction out of evaluation.

#### Further reading

- Agarwal et al., [GKD](https://arxiv.org/abs/2306.13649): on-policy distillation with generalized divergences; the formulation this section generalizes.
- Thinking Machines, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/): the per-token reverse-KL reward, cost comparisons with RL.
- [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) and the official [OPSD repository](https://github.com/siyan-zhao/OPSD) (new, verified): the OPSD formulation, sampling, masking and teacher-update choices to inspect.
- [On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275) (new, title verified): transferring in-context assistance through student rollouts; the bridge to 9.7.
- Ross, Gordon and Bagnell, DAgger (cited in `subsec_il_dagger`): the reduction behind the argument.
- Code: [verl OPD documentation](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md); [TRL GKD trainer](https://huggingface.co/docs/trl/gkd_trainer).
- Blog: Lilian Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/), for the broader frame in which distillation of reasoning sits.

#### Risks and open decisions

- *8B teacher plus LoRA student plus rollout engine on one A100 40 GB.* The weights alone are 23.3 GB; if the peak exceeds 38 GiB at the rollout and scoring batch sizes, first shrink the scoring micro-batch and `gpu_memory_utilization`; then the T2 configuration swaps the teacher to `Qwen/Qwen3-4B` (4,022,468,096 parameters, 8.04 GB in bf16, 3.6B non-embedding; same tokenizer), the code path is identical. A second fallback is the *T2 pair*: teacher and engine on one GPU, trainer on the other, with `d2l.sync_weights` over the NVLink bridge. The reference to "the 8B teacher" in the metrics follows whichever teacher was used.
- *vLLM sampler log-probs vs trainer log-probs.* Fallback: recompute behavior-policy log-probs with the trainer under `no_grad` (costs one extra student forward per token; the ledger counts it).
- *8.5's trainer API.* `d2l.grpo_train(policy, tasks, cfg)` takes `S_0` as `policy` and the `make_problems` stream as `tasks`. If it cannot start from a LoRA-wrapped $S_0$ with a shared optimizer state, the GRPO arm uses a 40-line GRPO loop written in the notebook from `d2l.group_advantages` (8.5) and `sec_rl_sequences`'s group baseline.
- *OPSD may not improve on this family.* Listed as an arm with a pre-registered possibility of null result; if it fails it becomes the section's cautionary example and exercise 7's baseline.
- *Problem generator duplication with 8.5.* 8.5's `d2l.ArithmeticTasks` is a single-family arithmetic task source for the GRPO tutorial. `make_problems` differs on purpose: it has multi-step `chain` word problems with template-level splits (`test_unseen_templates`, `test_harder`) that this chapter needs for generalization claims. Its problems are passed to `d2l.grpo_train` as `tasks` items (`{question, gold}`) and 9.1 notes the relationship once; no re-pointing is planned.
- *Teacher scoring cost on T1.* The 1.7B teacher resident beside a LoRA student fits in 15 GiB only with rollouts at batch size $\le 32$ and response length $\le 256$; the config enforces it.

### 9.3 Training reasoning models

- **Label:** `sec_lm_reasoning_training` · **Tier:** T3 full (one 8× A100 node, plans fit 8× 40 GB, limited access so each run happens once; student Qwen3-1.7B-Base; the three T3 arms (SFT-traces, RL-zero, SFT→RL) run concurrently on 2 GPUs each, one GPU for rollouts and one for training, so the notebook's total is ≤ ~2 h; the two remaining GPUs generate the teacher traces beforehand with vLLM, TP=1); T2 reduced (1× A100, student Qwen3-0.6B-Base, 3 seeds, a few hours for the whole table); T1 (reduced T1 configuration: committed logs and eval JSON for every table and plot, plus a live 100-problem analysis of released reasoning models; LoRA 0.6B smoke run of the SFT arm) · **Budget:** T3 ≤ ~2 h wall-clock for the three arms together (full fine-tuning state of the 1.7B student is 1,720,574,976 × 16 B = 27.5 GB, so each training GPU holds the whole state with about 10 GB of its ~38 GiB left for activations, which means gradient checkpointing, micro-batches of a few thousand tokens and a chunked loss; each rollout GPU holds a 3.44 GB bf16 engine plus KV cache; the one-off trace generator holds Qwen3-8B, 16.38 GB, plus KV; peaks to be measured, target ≤ 38 GiB per GPU, and the calibration run checks the training GPU's peak before the arms start); T2 reduced a few hours; T1 analysis ≤ ~15 min / ≤ 15 GiB (loads DeepSeek-R1-Distill-Qwen-1.5B and Qwen3-1.7B one at a time) · **Codex units:** 10.2, 11.2
- **Builds on (book):**
  - `sec_rl_sequences` (`subsec_rl_sequences_group`, GRPO components, `tab_rl_notation_contract`): $K$, group baseline, notation. Not re-derived.
  - `sec_baselines` (`subsec_leave_one_out`, `subsec_pg_loss_divisor`), `sec_ppo` (`subsec_ppo_asymmetric`): loss divisors and asymmetric clipping, assumed known from 8.5.
  - `sec_imitation` / `subsec_il_sft`: SFT as behavior cloning, so SFT on traces inherits the distribution-shift caveat; the section refers to it once.
  - `sec_scaling-laws`: the FLOP accounting.
- **Builds on (this part):** 9.1 (`d2l.ComputeLedger`, `d2l.make_problems`), 9.2 (`OnPolicyDistiller` only in the optional exercise), 8.5 (`d2l.grpo_train`, `d2l.verify_math`, `d2l.sample_responses`), 1.4 (`d2l.extract_answer(text, kind='boxed')`), 8.6 (`d2l.sync_weights`), 7.2 (`d2l.sft_train`), 6.1 (`d2l.launch(script, nodes=1, gpus_per_node=8, **env)`, reserved name, used for the T3 jobs), 1.4 (`d2l.pass_at_k`), 1.5 (`d2l.paired_bootstrap`, `d2l.EvalSuite`).
- **Exports** (`#@save`, new):
  - `d2l.format_reward(completion: str) -> float`: 1.0 if the completion has exactly one `<think>…</think>` block followed by `#### <integer>`, else 0.0; used by the R1-Zero-style arm (reward = verifier + 0.1·format is the default, weight set in the config).
  - `d2l.grpo_train_with_reset(policy, tasks, cfg, reset_every, reward_fn=None) -> list[dict]`: Chapter 9 wrapper that calls 8.5's `d2l.grpo_train` in segments of `reset_every` steps and reloads the reference policy between them (replaces the former `ref_reset_every`). Its name in the 9.3 prose above is `grpo_train_with_reset`.
  - `d2l.rejection_sample(engine, problems, n, keep, reward_fn, max_new_tokens, ledger=None) -> list[dict]`: sample `n` completions per problem, keep up to `keep` verified ones per problem (shortest-first optional), record the ledger entries for *all* generated tokens (kept or not).
  - `d2l.behavior_stats(text: str) -> dict`: counts of measurable behaviors per response: length in tokens, number of reflection markers from a fixed list (`wait`, `let me check`, `verify`, `double-check`, `alternatively`, `re-evaluate`), number of distinct intermediate answers stated, presence of think block; documented as a *marker count*, not a claim about cognition.
  - `d2l.eval_families(model_or_engine, problems_by_split, n_samples=1, k_list=(1,8,64), ledger=None) -> DataFrame`: per-split `acc@1`, `pass@k` via `d2l.pass_at_k`, mean tokens, truncation rate, answer-missing rate; returns per-problem rows so `d2l.paired_bootstrap` can be applied.
  - `d2l.compute_matched_grid(logs, flops_grid) -> DataFrame`: interpolates each run's eval records onto a common FLOPs grid (log-linear), never extrapolates; used by every matched-compute plot of the chapter.
  - Data: `DATA_HUB['lm-reason-sft-q17b']`, `lm-reason-rl-q17b`, `lm-reason-sftrl-q17b` (T3 checkpoints, bf16 safetensors; the RFT arm has no 1.7B checkpoint, see the cut below); `lm-reason-traces-q8b-v1` (the verified teacher traces, JSONL `{problem_id, prompt, response, n_tokens, correct}`); logs in `lm-reasoning-logs-v1`.
- **Models and data:**
  - Student base: `Qwen/Qwen3-1.7B-Base` (apache-2.0; T3) and `Qwen/Qwen3-0.6B-Base` (apache-2.0; T2 reduced and T1).
  - Teacher for traces: `Qwen/Qwen3-8B` in thinking mode (`enable_thinking=True`; temperature 0.6, top-p 0.95, top-k 20, min-p 0, never greedy; 8.19B parameters, 6.95B non-embedding).
  - Reference released reasoning models, analysis only (T1): `Qwen/Qwen3-1.7B` in thinking and non-thinking modes, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (MIT license; 1.78B parameters), `open-r1/OpenR1-Distill-7B` (apache-2.0; 7.6B; analysis from the model card and a 50-problem sample only if memory allows, otherwise skipped).
  - Data: the `chain` family (9.1) with splits `train`, `test_seen_templates`, `test_unseen_templates`, `test_harder`; `openai/gsm8k` test (public, possibly seen); optional exercise data `open-r1/OpenR1-Math-220k` (apache-2.0; traces from DeepSeek-R1, provenance to inspect).
- **Packages:** `vllm>=0.10`, `transformers`, `torch>=2.4`, `peft` (T1 only), `math-verify` (through 8.5's `d2l.verify_math`).

#### Narrative arc

1. **The question.** Opening cell (T1, live): sample 100 GSM8K-test problems from Qwen3-1.7B with thinking off, Qwen3-1.7B with thinking on, and DeepSeek-R1-Distill-Qwen-1.5B; print accuracy, mean generated tokens, and the reflection-marker rate per response (`behavior_stats`). The measured symptom is that the three differ greatly in length and in marker rate at similar sizes; accuracy numbers are public-benchmark numbers and are flagged. Question: where do such models come from, and which ingredient produced which difference?
2. **Pipelines.** A figure of four published pipelines drawn from sources only (DeepSeek-R1: cold-start SFT, reasoning RL, rejection sampling + SFT, all-scenario RL; R1-Zero: RL only from the base; the Qwen3 report: long-CoT cold start, reasoning RL, thinking-mode fusion, general RL; Open R1: SFT on distilled traces, then GRPO). Table `tab_lm_reason_pipelines` lists each stage's data source, training signal and size *as stated in the papers* (no numbers beyond what the sources state; cite).
3. **Four ways to make a reasoning model (code).** `rejection_sample` for traces; SFT on traces (7.2 trainer); R1-Zero-style GRPO from the base with the format reward; cold-start SFT then GRPO; self-generated rejection-sampling fine-tuning (RFT). Same prompts, same verifier. The chapter's `chain` family is the problem family; the notebook shows five train examples and five unseen-template examples so that readers see what "unseen template" means.
4. **Why RL from a base can fail to start (derivation D1).** The probability that a group has any learning signal; RFT as REINFORCE without a baseline. Numerical check.
5. **Compute-matched comparison (headline).** Ledger definition; all-in vs student-only accounting; accuracy vs FLOPs.
6. **What changes and what does not.** pass@1 vs pass@64, length, marker rates, entropy, the gap between seen and unseen templates, behavior on the harder slice. One short subsection on *prolonged RL* (reference reset, more steps): does the pass@$k$ curve of the RL model rise above the base model's at large $k$? Report measured, cite both directions (ProRL; Yue et al.).
7. **Reading behavior responsibly.** Marker counts as measurements; sample three RL-arm responses at steps 0, mid, end, show text; state that a marker's rise is not evidence of a new internal algorithm. Cite the R1-Zero-like training paper (Dr. GRPO) for the observation that base models already show such markers; show the base's marker rate at step 0.

#### Derivations

**D1. When does RL from a base model get a learning signal, and what is rejection-sampling fine-tuning in RL terms?**

*Statement.* (a) For a prompt on which the policy succeeds with probability $p$ and a group of $K$ independent samples with 0/1 verifier rewards, the group-standardized advantages (`sec_rl_sequences`) are nonzero iff the group is mixed; this happens with probability $1-p^K-(1-p)^K$. For $p\to0$ it is $\approx Kp$. (b) Fine-tuning on the correct samples of the current policy, with a cross-entropy loss and no negative samples, has expected gradient $\mathbb{E}_{y\sim\pi_\theta}[R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)]=\nabla_\theta\,\mathbb{E}R$, that is, REINFORCE *without a baseline* (`sec_baselines`). (c) GRPO's group-standardized update differs from (b) by (i) subtracting the group mean, which adds a push-down on failures and removes the mean-reward component, (ii) dividing by the group standard deviation, a per-prompt rescaling that upweights prompts with $p$ near 0 or 1 (`sec_rl_sequences` notes it is not a baseline term), and (iii) the $(K-1)/K$ self-inclusion shrinkage of `subsec_rl_sequences_group`.

*Assumptions.* Binary reward, independent samples within a group, the policy $\pi_\theta$ is the sampling policy (on-policy), no length normalization (so token weights are equal; the effect of the divisor is 8.5's topic).

*Steps.*
1. (a): all $K$ rewards equal iff all succeed ($p^K$) or all fail ($(1-p)^K$). In those groups the mean equals every reward, the centered reward is zero, and the update is zero (the book's $K=1$ case generalizes). Expand $1-(1-p)^K-p^K$ for small $p$ to first order to get $Kp$.
2. Table of values: for $K=8$, signal probability is $0.149$ at $p=0.02$, $0.337$ at $p=0.05$, $0.832$ at $p=0.2$, $0.992$ at $p=0.5$ (author's check; formula above). A base model with $p\approx0.02$ on the hard slice leaves 85% of groups without signal, which is why R1-Zero-style training depends on a base with non-trivial pass rate and why cold-start SFT, curriculum and dynamic sampling (8.5, DAPO) exist.
3. (b): by the score-function identity (`sec_baselines`), $\nabla\mathbb{E}R=\mathbb{E}[R\,\nabla\log\pi]$; with $R\in\{0,1\}$ the sample average over $n$ samples of $R_i\nabla\log\pi(y_i)$ is a sum over the correct samples only, which is cross-entropy training on them.
4. (c): write the GRPO weight $A_j=(R_j-\bar R)/(\sigma+10^{-8})$; compare weights: RFT puts weight $1$ on successes and $0$ on failures, GRPO puts $(1-\bar R)/\sigma$ on successes and $-\bar R/\sigma$ on failures; hence GRPO uses failures as negative evidence and RFT cannot. Since $\sum_j A_j=0$, in expectation GRPO equals RFT's gradient minus a baseline term whose expectation is $\bar R\cdot\mathbb{E}\nabla\log\pi=0$ up to the self-inclusion effect.
5. Consequence for the experiment: the RFT arm and the RL arm optimize the same $\mathbb{E}R$ with different variance and different use of negative samples; difference in outcomes is about estimator variance and compute, not about objective.

*Numerical check (cell `reason-signal`, CPU).* (i) Simulate 200,000 groups for $p\in\{0.02,0.05,0.2,0.5\}$, $K\in\{4,8,16\}$ and compare the mixed-group frequency with $1-p^K-(1-p)^K$ within 3 standard errors. (ii) On the four-prompt verifier toy of `sec_rl_sequences` (the book's `policy`, `verify`): assert that the exact expected RFT gradient equals the exact policy gradient $g$ (the book's einsum) to $10^{-12}$, and that the expected GRPO-centered update equals $(K-1)/K$ times $g$ at $K=2$ (as the book shows), so RFT is the baseline-free relative of the same estimator.

#### From-scratch implementation

Budget ≤ ~150 lines.

- `rejection_sample` (≈40 lines): batched sampling through the engine, verifier through `reward_fn`, `keep` selection, ledger entries for all generated tokens (`add_generate(N_ne, tokens)`), JSONL writer. Tests: every kept sample verifies (`reward_fn == 1`); with `keep=n` and a perfect verifier the kept count equals the number of correct samples; token accounting equals the sum of response lengths.
- `format_reward` (≈15 lines): a regular expression with a state check (exactly one block). Tests: 12 hand-written strings (missing close tag, two blocks, answer inside the block, trailing text) with expected 0/1.
- `behavior_stats` (≈25 lines): lowercase, regex counts, distinct stated answers via `d2l.extract_answer` over sentences containing `=` or "answer". Tests: hand-labeled 10 snippets.
- `eval_families` (≈45 lines): batched sampling `n_samples`, per-problem correct count $c$, `d2l.pass_at_k(n, c, k)`; per-problem DataFrame. Tests: with a stub engine that answers correctly with probability $0.3$, `pass@8` matches $1-0.7^8$ within sampling error over 2,000 problems and equals the unbiased estimator on a hand-built count table.
- `compute_matched_grid` (≈20 lines): log-linear interpolation of `(flops, metric)` per run; test that it reproduces the run's own points and returns NaN outside its range.
- The R1-Zero prompt template (≈10 lines): "…Show your reasoning in `<think>` tags, then write `#### answer`." for the base model, as 8.5's plain-text prompt with no chat template (the RL-zero arm has no SFT stage that taught one, and it continues 8.5's setup).
- *Template and `<think>` format.* The `<think>…</think>` block is a convention of the response text, not of a template. Wherever a chat template is used (the SFT-traces, SFT→RL and RFT arms, and every arm's evaluation prompt that follows its training format), it is the part's `d2l_chatml_v1`: the trace sits inside the assistant content, rendered `<|im_start|>assistant\n<think>…</think>…<|im_end|>`, which is why 7.2 keeps think blocks out of that template ("reasoning formats are Chapter 9's"); `format_reward` and `behavior_stats` read the assistant content after the template is stripped. The teacher generates its traces with Qwen's own template (`enable_thinking=True`, no empty block), and the stored trace text is re-rendered through `d2l_chatml_v1` for the student; the SFT→RL arm keeps the template of its SFT stage through its RL stage. Qwen's own template appears only for the released reference models of the opening cell.

#### In production

- Open R1 and TRL provide the SFT and GRPO scripts used at scale (`open-r1` `sft.py` and `grpo.py`, TRL `GRPOTrainer`); the notebook shows the corresponding config of the same recipe (model, dataset, loss, learning rate) and states which numbers are copied from the repository and which are the notebook's own. The from-scratch `rejection_sample`, `format_reward` and `eval_families` are the oracles: run a 64-problem batch through the notebook path and through the library path and compare accuracy to within the binomial interval.
- Released checkpoints: DeepSeek-R1-Distill-Qwen-1.5B (distilled from R1 traces) is evaluated with the same `eval_families` call; the T1 cell prints its marker rate next to the RL arm's.
- Model-level mixing of thinking and non-thinking behavior (Qwen3's `enable_thinking`) is shown only as a switch in the chat template; the training recipe is cited from the technical report, not reproduced.

#### Headline experiment

*Hypothesis.* On a verifiable problem family, at equal compute, (H1) SFT on teacher traces improves unseen-template accuracy faster per FLOP of the student than RL from the base, but the teacher's cost makes the all-in ranking depend on how many times the traces are reused; (H2) RL from the base improves pass@1 more than it improves pass@64; (H3) cold-start SFT followed by RL dominates RL from the base in early compute. The notebook reports which of these hold; none is assumed.

*Setup.* Student `Qwen3-1.7B-Base` (T3). Common: the `chain` train split streamed (prompts never repeated within a run), evaluation every $C/8$ FLOPs on 500 problems per split, maximum response length 1,024 tokens (2,048 for the teacher), AdamW, bf16 with fp32 optimizer state, seeds fixed.

| Arm | Data | Update | Notes |
|---|---|---|---|
| SFT-traces | verified traces of Qwen3-8B (thinking), up to 4 per problem | SFT, 2 epochs, LR $10^{-5}$ | all teacher-generated tokens counted in all-in FLOPs (kept or not) |
| RL-zero | verifier + 0.1·format reward | GRPO, $K=8$, 8.5's headline config | base model, no SFT |
| SFT→RL | 500 traces, 1 epoch of SFT, then RL as above | cold-start | splits the budget 10/90 by FLOPs (a config) |
| RFT | self-generated, verified samples of the base | 2 rounds of rejection sampling + SFT | no teacher; baseline-free estimator of D1; **T2 reduced only** (0.6B, 3 seeds): no stated hypothesis H1–H3 needs it at 1.7B, so the one-shot T3 run does not spend a GPU pair on it; its column in `tab_lm_reason_changes` comes from T2 reduced and says so |

The compute grid $C$ is set in FLOPs from the 8.5 RL run's measured ledger (value to be measured). For each arm the number of steps or tokens is chosen from its ledger to land on each grid point. T2 reduced: Qwen3-0.6B-Base, teacher Qwen3-1.7B (thinking), 3 seeds per arm, the whole table in a few hours on one A100.

*Procedure.* Build teacher traces once and commit them; for each arm train to the largest $C$ logging the ledger; evaluate at each grid point for accuracy (`acc@1` greedy and `acc@1` sampled), `pass@8/64` on 200 problems with $n=64$, mean tokens, truncation rate, marker rate, entropy of the policy on its own samples (from the training logs for RL arms, recomputed for others). After the last point, extend the RL-zero arm 3× on its own GPU pair (prolonged RL, no reset; this is what H2's pass@64 question needs at T3) and re-evaluate pass@64 on the harder slice. The reset variant (a reference reset every $C/4$, implemented as a 9.3 wrapper `grpo_train_with_reset` that runs `d2l.grpo_train` in segments and reloads the reference policy from the latest checkpoint between them; 8.5's `cfg` has no reset option) is run at T2 reduced only (0.6B, 3 seeds), where the ProRL direction is testable with seeds. *Cut at T3 (8× 40 GB node, one run each):* the RFT arm and the reset variant, as stated above.

*Plots.* `fig_lm_reason_flops`: accuracy (y) on `test_unseen_templates` and `test_harder` against cumulative FLOPs (x, log); the SFT arm is drawn twice, student-only cost and all-in cost (dashed), so the cost of the teacher is a visible shift. `fig_lm_reason_passk`: pass@$k$ against $k$ (log) at the final compute point for the base model, SFT, RL, SFT→RL, with intervals. `fig_lm_reason_behavior`: mean tokens and marker rate against compute. `tab_lm_reason_changes`: rows = measures (acc@1 seen/unseen/harder/GSM8K, pass@64, tokens, markers, entropy, truncation rate), columns = base, SFT, RL, SFT→RL, RFT.

*Expected outcome.* Directions only, with sources. The DeepSeek-R1 paper states that distilling a stronger model into small models outperforms applying large-scale RL to small models and that the latter is far more compute-hungry; this is the motivation for H1, not a result to be assumed here. Yue et al. report that RL-trained models gain pass@1 over their base but that the base can match or exceed them at large $k$; ProRL reports that prolonged RL with entropy control and reference resets can expand what the model solves. The notebook tests the direction on this family. All magnitudes are to be measured; the committed logs carry them.

*Noise check.* At T2 reduced: 3 seeds per arm; the seed spread of each arm at each grid point is reported; paired-bootstrap intervals over problems for each arm-pair difference (same problems for every arm); clustered by template on the unseen split. At T3: single seeded runs by the part's convention; a T3 comparison is stated only if the difference exceeds twice the maximum seed spread measured at T2 reduced *and* the paired interval excludes 0; otherwise the table says "not resolved at one seed".

*T1 reduction.* The committed eval JSON and logs rebuild every figure and table; the T1 notebook also runs the opening cell live and, optionally, the SFT arm with LoRA on 200 traces for 30 steps as a smoke test (no result claimed).

*Artifacts committed for T2/T3.* Verified traces (`lm-reason-traces-q8b-v1`), per-arm logs and eval JSON, final checkpoints, the pass@$k$ count tables (so `pass_at_k` runs on committed counts).

#### Figures and tables

- `fig_lm_reason_pipeline`: the four pipelines as boxes (data, signal, stage), drawn with matplotlib in the notebook from the table `tab_lm_reason_pipelines`.
- `tab_lm_reason_pipelines`: stage, data source, signal, as stated by each source (R1, R1-Zero, Qwen3, Open R1).
- `tab_lm_reason_signal`: signal probability $1-p^K-(1-p)^K$ for the grid of D1 (the cell `reason-signal`).
- `fig_lm_reason_flops`, `fig_lm_reason_passk`, `fig_lm_reason_behavior`, `tab_lm_reason_changes` (headline).
- `tab_lm_reason_released`: the opening cell's output for the three released models (flagged public).

#### Pitfalls and misconceptions

- "RL teaches reasoning": the measured effects are changes in length, format, marker rate and pass@1; whether new capabilities appear is what pass@$k$ at large $k$ tests.
- Comparing SFT and RL at equal *steps* or equal *student* compute hides the teacher; use all-in FLOPs and also show student-only.
- Reward hacking through format: the format reward is cheap to satisfy; check that format compliance saturates before accuracy moves.
- Truncation: long responses cut at the length cap look like failures; report the truncation rate per arm.
- Training and test templates share vocabulary and numbers; the generator asserts disjointness of `(template, numbers)` tuples, and the notebook prints the assertion result.
- Public-benchmark numbers (GSM8K test) are contaminated-risk; they appear only as paired relative comparisons and are labeled in captions.
- A rising marker count is not evidence of a new internal algorithm; inspect responses.
- A base model with $p\approx0$ on a family produces all-zero groups; check the signal probability before blaming the optimizer.

#### Summary (draft bullets)

- Reasoning models are built from four ingredients in various orders: SFT on verified traces, RL against a verifier, rejection sampling, and distillation from a stronger model.
- RL needs a base with a non-trivial pass rate: a group has a signal with probability $1-p^K-(1-p)^K$.
- Rejection-sampling fine-tuning is REINFORCE without a baseline; GRPO adds negative evidence and per-prompt rescaling.
- Compare pipelines at matched FLOPs, student-only and all-in; the teacher's cost can reverse a ranking.
- Measure what changes: pass@1, pass@$k$, length, behavior markers, entropy, and generalization to unseen templates.
- Behavior counts are measurements, not explanations.

#### Exercises

1. *(check)* A base model solves a prompt family with $p=0.1$. With $K=8$ and $K=32$, what fraction of groups have no GRPO signal? *Sketch:* $0.9^8+0.1^8\approx0.430$ and $0.9^{32}\approx0.034$; so $0.570$ and $0.966$ of groups carry signal.
2. *(check)* Why can the same verifier give an SFT-on-traces model a higher unseen-template accuracy but a lower pass@64 than the base? Name two mechanisms. *Sketch:* distillation narrows the output distribution toward the teacher's style (lower entropy) and the base can still reach rare correct paths when sampled 64 times; also truncation if teacher traces are long.
3. *(derive)* For a one-parameter Bernoulli policy $\pi_\theta(1)=\sigma(\theta)=p$ with reward $R=y$, compute the mean and variance of the single-sample RFT gradient $R\,\nabla_\theta\log\pi_\theta(y)$, and show that for $n$ samples centering by the sample mean multiplies the expectation by $(n-1)/n$ (the factor of `subsec_rl_sequences_group`). *Sketch:* the score is $y-p$, so $R\cdot\textrm{score}=y(1-p)$; the mean is $p(1-p)=\nabla_\theta p$ and the variance is $p(1-p)^3$. For the centered estimator $\frac1n\sum_i(y_i-\bar y)(y_i-p)$ the expectation is $\frac{n-1}{n}p(1-p)$ by the same self-inclusion calculation as in `sec_baselines`; verify both numerically with $10^6$ draws.
4. *(derive)* Derive the probability that a group of $K$ has *exactly* $m$ successes and the expected number of positive and negative samples in a mixed group; use it to argue how many failure samples GRPO can learn from at $p=0.05$, $K=8$. *Sketch:* binomial; conditional on mixed, $\mathbb{E}[\#\textrm{success}\mid\textrm{mixed}]=(Kp-Kp^K)/(1-p^K-(1-p)^K)$, which at $p=0.05$, $K=8$ is about $0.4/0.3366\approx1.19$, so a mixed group offers about $6.8$ failures and $1.2$ successes.
5. *(measure)* Change the SFT-traces arm to use only the *shortest* correct trace per problem; predict the effect on mean tokens and on `test_harder`, and run on the T2 reduced configuration with 3 seeds.
6. *(measure)* Remove the format reward from the RL-zero arm (T2 reduced). Predict whether the arm still learns to produce parsable answers and how the unseen-template accuracy changes; report intervals and the format-compliance curve.
7. *(open)* Take 5,000 traces from `open-r1/OpenR1-Math-220k`, inspect their provenance and verification fields, and build an SFT subset for Qwen3-0.6B-Base. Define a contamination policy for the eval and defend a selection rule; there is no reference answer.

#### Further reading

- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948): R1-Zero, cold start, rejection sampling, distillation to small models.
- Liu et al., [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) (from 8.5): base-model behavior at step zero, the length bias in the loss.
- Liu et al., [ProRL](https://arxiv.org/abs/2505.24864): prolonged RL with reference resets; the counterpoint to the next entry.
- Yue et al., [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) (new): pass@$k$ at large $k$ as a test of capability expansion.
- Mirzadeh et al., [GSM-Symbolic](https://arxiv.org/abs/2410.05229): why templates and distribution shift matter in reasoning evaluation.
- Qwen team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (new): a four-stage reasoning pipeline with thinking-mode fusion.
- Code/courses: [Open R1](https://github.com/huggingface/open-r1); Raschka, [Reasoning from Scratch](https://github.com/rasbt/reasoning-from-scratch); Alammar, [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1); Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/).

#### Risks and open decisions

- *T3 cost.* Three concurrent arms at 1.7B on 8× A100 40 GB may not finish in ~2 h; fallback: run T3 for two arms (SFT-traces, RL-zero) with 4 GPUs each and T2 reduced for the rest; the committed logs record which configuration produced each number.
- *1.7B full fine-tuning state on a 40 GB GPU.* 27.5 GB of state leaves about 10 GB; if the calibration run's peak exceeds 38 GiB, shorten the maximum response length from 1,024 to 768 tokens before anything else, then fall back to the T2 reduced configuration (0.6B) for the arm that does not fit; never to LoRA for one arm only, which would confound the comparison.
- *Base model with $p\approx0$ on `test_harder`.* Expected; the plots show it as a floor. Fallback: select the family's difficulty range by the base model's measured pass rate (a one-off pilot, committed), not by the experimenter's taste.
- *Teacher traces too long for the student's context.* Cap at 1,024 student tokens; record the dropped fraction and put it in the all-in ledger.
- *Reference resets.* 8.5's `cfg` has no `ref_reset_every`; 9.3's segmented wrapper (above) is the implementation, so no change to 8.5 is needed.
- *Qwen3-8B thinking-mode throughput.* If the T2 reduced teacher (1.7B thinking) is too weak (low verified-trace yield), switch to Qwen3-4B; the code path is the same.
- *Licenses.* Qwen3 and OpenR1 are apache-2.0; DeepSeek-R1-Distill-Qwen-1.5B is MIT (the model card also discusses the licenses of the base models the distilled variants start from; verify before redistributing derived weights; the notebook does not redistribute them).

### 9.4 Verifiers and process supervision

- **Label:** `sec_lm_verifiers` · **Tier:** T2 (reduced T1 configuration: committed MC step labels `lm-prm-mc-labels-v1` and candidate pools; a Qwen3-0.6B + LoRA scorer trained on 2,000 solutions, one seed live, three seeds from committed results) · **Budget:** T2: step labeling plus scorer training, a few hours on 1× A100 (labeling dominates; the notebook prints the FLOPs from the ledger); peak memory to be measured (scorer 0.6B + LoRA with 1,024-token sequences fits well within one A100 40 GB: 1.19 GB of bf16 weights, adapter state and activations; the generator for labeling runs under the rollout engine, not concurrently). T1: to be measured, target ≤ ~15 min and ≤ 15 GiB · **Codex units:** 11.3
- **Builds on (book):**
  - `sec_regularized`, "Dense versus Terminal Rewards": one paragraph defines ORM vs PRM, calls potential-based shaping the safe densification and warns about misspecification. 9.4 turns this into a derivation and an experiment and does not repeat the paragraph.
  - `sec_rl_sequences` (`subsec_rl_sequences_hacking`, "Checked Rewards from Verifiers"): programmatic verifiers vs learned rewards; the scorers here are *learned* verifiers used at inference time.
  - `sec_baselines` (zero-mean lemma, reward-to-go, state-dependent baselines): the credit-assignment identity in D1.
  - `sec_mdl-statistics` (calibration is not covered there; AUROC and ECE are defined in this section, in at most one short paragraph each).
- **Builds on (this part):** 1.4 (`d2l.score_choices(model, tokenizer, prompts, choices, norm='token')` for the generative verifier; `d2l.pass_at_k`), 1.5 (`d2l.paired_bootstrap`), 8.1 (`d2l.RewardModel(backbone, hidden_size)`, whose `forward` returns `(B,)` at the last non-pad token; used only as the pattern for a reward head, and 9.4 defines its own multi-position reader `d2l.ProcessScorer`), 8.5 (`d2l.verify_math`, `d2l.sample_responses`), 1.4 (`d2l.extract_answer`), 9.1 (`ComputeLedger`, `make_problems`), 7.4 (`d2l.LoRALinear`).
- **Exports** (`#@save`, new):
  - `d2l.split_steps(text: str) -> list[tuple[int, int]]`: character spans of steps (a step is a line starting with `Step k:` or a blank-line-delimited paragraph; the generator prompt requests `Step k:` lines, the splitter is the fallback).
  - `d2l.mc_step_labels(engine, problem, steps, K=8, reward_fn=None, max_new_tokens=256, ledger=None) -> dict`: for each step prefix, sample `K` continuations, return `soft` (fraction correct), `hard` (any correct), and the number of generated tokens; the last step's label is the verifier's outcome (no sampling).
  - `d2l.ProcessScorer(model_id, mode='orm'|'prm', lora_rank=32)`: backbone plus one linear head; `.forward(input_ids, attention_mask, read_positions) -> Tensor[B, S]` scores at the given token positions (final token for `orm`, the last token of each step for `prm`); `.score_solution(problem, solution) -> dict(step_scores, final_score)`.
  - `d2l.scorer_loss(logits, targets, mask, soft=True) -> Tensor`: binary cross-entropy with soft or hard targets.
  - `d2l.aggregate_prm(step_scores, how='min'|'prod'|'last'|'mean') -> Tensor`.
  - `d2l.ece(probs, labels, bins=15) -> float`, `d2l.auroc(scores, labels) -> float`.
  - `d2l.select_by_score(scores, correct, n, n_trials=200, seed=0) -> Tensor[num_problems]`: for each problem draws `n` candidates without replacement from its pool, picks the highest score, returns the mean correctness over trials. 9.5 reuses it.
  - Data: `DATA_HUB['lm-prm-mc-labels-v1']` (JSONL: `problem_id, generator, step_spans, soft, hard, n_gen_tokens`), `lm-scorers-v1` (LoRA scorer weights per mode and seed).
- **Models and data:**
  - Generators: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` (apache-2.0), non-thinking, temperature 0.7, top-p 0.8, top-k 20, prompted to answer with `Step k:` lines and a final `#### answer`. The scorer is trained on solutions of the 1.7B generator only; the other two generators are the calibration test (in 9.4 and again in 9.5).
  - Scorer backbones: `Qwen/Qwen3-0.6B` (T1, T2) and `Qwen/Qwen3-1.7B` (T2 extension).
  - Problems: `openai/gsm8k` train (7,473; MIT) for scorer training and labels (3,000 problems), the generated `chain` train split (3,000 problems); test: `chain` `test_unseen_templates` and `test_harder` (clean) and GSM8K test (public, possibly seen).
  - Optional (exercises): `peiyi9979/Math-Shepherd` (MC labels on GSM8K/MATH solutions from a 7B generator; the dataset metadata declares no license, so it is a download-only, analysis-only exercise: the notebook downloads it at run time and nothing from it, nor anything derived from it, is committed or redistributed), PRM800K labels (`github.com/openai/prm800k`, MIT license; human step labels on MATH solutions), `Qwen/ProcessBench` (apache-2.0; error-step identification).
- **Packages:** `transformers`, `peft`, `vllm` (labeling), `torch>=2.4`.

#### Narrative arc

1. **The question.** Opening cell (CPU after loading pools): on one candidate pool (GSM8K test, Qwen3-1.7B, 8 samples per problem) compute majority-vote accuracy and the oracle (pass@8); the gap between them is the headroom a scorer could close. Print the headroom in points. Question: what is the cheapest supervision that closes it?
2. **Three kinds of scorer.** A table of what each reads and what it needs: ORM (one score per complete solution, outcome labels), PRM (a score per step, step labels), generative verifier (the model writes or scores a judgment; zero-shot or fine-tuned). Show the per-token view: all three are the same backbone with different read positions and labels.
3. **Where step labels come from.** Human labels (PRM800K description, no download needed in the notebook), LLM-judge labels, and Monte Carlo rollouts (Math-Shepherd). Implement `mc_step_labels`; plot soft labels along the steps of three example solutions (an all-correct one, one with an early error, one with a late error).
4. **What an MC label measures (derivation D1).** The value function of the generator at that prefix, not whether the step is *correct*; hard vs soft labels; the label-cost ratio.
5. **ORM vs PRM credit assignment (derivation D1 continued).** The martingale identity, local credit vs outcome weight, the toy chain.
6. **Train the scorers.** ORM, PRM-hard, PRM-soft (same backbone, same solutions, different read positions and targets), plus the zero-shot generative verifier via `score_choices`. Training loop (≈25 lines), learning curves.
7. **Rerank the same pools.** Best-of-$N$ by each scorer, with PRM aggregation (min, product, last); weighted voting is deferred to 9.5. Calibration across generators (ECE, reliability diagram), AUROC.
8. **Labeling cost vs benefit (headline).** Selection accuracy at $N=8$ against labeling FLOPs for each scorer.
9. **Limits.** Scorers as reward models in RL (reward hacking via `subsec_rl_sequences_hacking`), step-boundary ambiguity, scorers that never saw the generator they are applied to.

#### Derivations

**D1. ORM vs PRM credit assignment.**

*Statement.* Let a solution consist of $S$ steps, with prefixes $s_0=x,\ s_j=(x,\textrm{step}_1,\dots,\textrm{step}_j)$, and let $R\in\{0,1\}$ be the terminal correctness under the generator $\pi$. Define $V(s_j)=\Pr_\pi(R=1\mid s_j)$. Then:
(a) $V(s_j)=\mathbb{E}[V(s_{j+1})\mid s_j]$, $V(s_S)=R$ (a martingale).
(b) With $\Delta_j=V(s_j)-V(s_{j-1})$: $R-V(s_0)=\sum_{j=1}^S\Delta_j$, $\mathbb{E}[\Delta_i\Delta_j]=0$ for $i\ne j$, and $\mathrm{Var}(R)=V(s_0)(1-V(s_0))=\sum_j\mathbb{E}[\Delta_j^2]$.
(c) In the token MDP of `sec_rl_sequences` (deterministic transitions), $\Delta_j$ is the advantage $A(s_{j-1},\textrm{step}_j)=Q-V$ of step $j$. For the policy-gradient term of step $j$, $\mathbb{E}[\nabla\log\pi(\textrm{step}_j\mid s_{j-1})\,(R-b)]$ with any baseline $b=b(s_{j-1})$ equals $\mathbb{E}[\nabla\log\pi(\textrm{step}_j\mid s_{j-1})\,\Delta_j]$: *local credit from exact values is unbiased*, and dropping the later increments $\Delta_{j'},\,j'>j$ loses nothing in expectation.
(d) A Monte Carlo step label from $K$ rollouts has soft mean $V(s_j)$ and variance $V(1-V)/K$; the hard label $1[\text{any success}]$ has mean $1-(1-V)^K$, biased upward.
(e) Label cost: for $S$ equal-length steps of $L/S$ tokens, labeling all prefixes costs $K\,L\,\frac{S-1}{2}$ generated tokens, so the ratio to generating the solution itself is $1+K(S-1)/2$.

*Assumptions.* Terminal reward only; transitions deterministic; the generator $\pi$ is fixed and is the policy whose values are estimated; $V$ exact in (c); steps independent in the toy chain used for the check (not needed for (a)–(e)).

*Steps.*
1. (a): tower property $V(s_j)=\mathbb{E}[\mathbb{E}[R\mid s_{j+1}]\mid s_j]$.
2. (b): telescope; orthogonality because for $i<j$, $\mathbb{E}[\Delta_i\Delta_j]=\mathbb{E}[\Delta_i\,\mathbb{E}[\Delta_j\mid s_{j-1}]]=0$ by (a); take $\mathbb{E}$ of $(R-V(s_0))^2$ for the variance identity.
3. (c): the policy-gradient term of step $j$ uses $\hat G_j=R$ (terminal reward, `sec_baselines`); write $R-b=\sum_{j'\ge j}\Delta_{j'}+(V(s_{j-1})-b)$. For $j'>j$, $\mathbb{E}[\nabla\log\pi_j\,\Delta_{j'}]=\mathbb{E}[\nabla\log\pi_j\,\mathbb{E}[\Delta_{j'}\mid s_{j'-1}]]=0$ because $\nabla\log\pi_j$ is $s_{j'-1}$-measurable. The $b$ term vanishes by the zero-mean lemma. What remains is $\mathbb{E}[\nabla\log\pi_j\,\Delta_j]$.
4. Corollary: an ORM-weighted update gives every step the same weight $R-b$ (`sec_rl_sequences`, "every token receives the same weight"), while exact process values give step $j$ the weight $\Delta_j$ alone. Both have the same mean. **The variance of the two is not ordered in general:** $\mathrm{Var}$ of the ORM-weighted term includes $\sum_{j'\ne j}\mathbb{E}\Delta_{j'}^2$-type contributions, but the cross term $\mathbb{E}[\textrm{score}_j^2\,\Delta_j\,(\ldots)]$ can be positive or negative. The toy check below reports a ratio that crosses 1; state this explicitly and do not claim a variance reduction without measuring it. What a PRM *does* provide is localization: $\arg\min_jV(s_j)-V(s_{j-1})$ identifies the first step where success probability collapses, which an ORM cannot supply.
5. (d): $\hat y_j\sim\textrm{Binomial}(K,V)/K$ gives mean $V$ and variance $V(1-V)/K$; the hard label is Bernoulli with success probability $1-(1-V)^K$. At $K=8$: $V=0.05\Rightarrow0.337$, $V=0.3\Rightarrow0.942$. So hard labels are *optimistic*: a wrong step is labeled good whenever any rollout recovers; soft labels are unbiased but noisy at small $K$. A step label estimates "can this prefix still reach the right answer under $\pi$", not "is this step correct".
6. (e): prefix $j$ ($1\le j\le S-1$) needs $K$ continuations of remaining length $L(S-j)/S$; sum over $j$ gives $KL\sum_{j=1}^{S-1}(S-j)/S=KL(S-1)/2$. At $S=6$, $K=8$, $L$ tokens: ratio $1+8\cdot5/2=21$. (Prefix sharing across rollouts can cut this; the ledger counts what the notebook actually generates.)
7. Policy dependence: $V^\pi$ depends on $\pi$. If $\pi'$ continues at least as well as $\pi$ from every prefix then $V^{\pi'}\ge V^{\pi}$ pointwise, so a scorer trained on $\pi$'s labels *under*-estimates success for a stronger generator and over-estimates for a weaker one. This predicts the direction of miscalibration across generators in the experiment (tested, not assumed).

*Numerical check (cell `verifier-credit`, CPU, seconds).* Chain of $T=4$ independent steps, each correct with probability $0.8$, $R=\prod$ of outcomes; enumerate all 16 outcomes. Assert (a) and (b) to $10^{-12}$ ($\mathrm{Var}(R)=\sum_j\mathbb{E}\Delta_j^2=0.2418$ in the author's check). Compute for each step $j$ the mean of $\textrm{score}_j\cdot(R-V_0)$ (ORM weight with constant baseline), $\textrm{score}_j\cdot(R-V(s_{j-1}))$ (value baseline), and $\textrm{score}_j\cdot\Delta_j$ (local credit): all three have the same mean ($0.0819$ for every $j$ in the author's check). Print the variance table: ORM/local variance ratios of $1.53,1.12,0.84,0.64$ across the four steps in the author's check (to be re-measured; this is the evidence for "not ordered"). Simulate $10^6$ MC labels at $V\in\{0.05,0.3\}$, $K\in\{4,8,16\}$ and compare with the soft variance $V(1-V)/K$ and hard mean $1-(1-V)^K$ within 3 standard errors.

#### From-scratch implementation

Budget ≤ ~150 lines.

- `split_steps` (≈15 lines) and `mc_step_labels` (≈35 lines). Test with a stub engine whose continuation succeeds with a known probability per prefix (a table): the mean soft label equals $V$ within 3 SE over 2,000 labelings, the mean hard label equals $1-(1-V)^K$, and ledger tokens equal the sum of generated lengths.
- `ProcessScorer` (≈40 lines): `AutoModel` backbone with LoRA on all linear layers (7.4's `LoRALinear`), `nn.Linear(hidden, 1)` head, gather hidden states at `read_positions`; mixed precision; `read_positions` from the tokenizer's offset mapping so a step's score is read at the last token of the step. Tests: reading at the last token of a solution equals reading with `mode='orm'`; changing a *later* step does not change an earlier step's score (causal mask), asserted on random text to $10^{-5}$ in fp32.
- `scorer_loss`, `aggregate_prm`, `ece`, `auroc`, `select_by_score` (≈45 lines total). Tests: `ece` of 20,000 draws with probabilities $p_i$ and labels $\sim\mathrm{Bernoulli}(p_i)$ is below $0.02$; `auroc` equals a brute-force pair count on 200 random pairs-of-scores exactly; `aggregate_prm('min')` equals `scores.min(-1)`; `select_by_score` with a perfect scorer returns `pass@n`-style coverage `1-C(m-c,n)/C(m,n)` (compare with `d2l.pass_at_k` to $10^{-9}$) and with random scores returns the single-sample accuracy within sampling error.
- Training loop (≈25 lines, no new export): AdamW, LR $10^{-4}$ for LoRA and head, 1 epoch over the solutions, batch by tokens; `bf16` autocast.

#### In production

- Library path for the scorer: TRL's `RewardTrainer`/`PRMTrainer` (token-classification head over step separators) and `search-and-learn` (Hugging Face) which runs best-of-$N$, beam search and DVTS with open PRMs; the notebook runs `search-and-learn`'s best-of-$N$ selection on one committed pool with the notebook's scorer and checks that the selected-answer accuracy equals `d2l.select_by_score` at $n=N$ (the library randomizes tie-breaks differently, so equality is checked on the deterministic argmax with no ties). Pin versions; verify that `PRMTrainer` exists in the pinned TRL (to be verified).
- The *released* process reward models (e.g. Math-Shepherd's, Qwen's math PRMs) are mentioned as options and used only if a model ID is verified at authoring time (to be verified); the notebook never depends on them.
- Generative verifiers: `d2l.score_choices` with the prompt template of Zhang et al. (the verifier writes "Yes"/"No"); a fine-tuned variant is an exercise.

#### Headline experiment

*Hypothesis.* With the same solutions and the same backbone, step-level supervision from MC labels selects better candidates than outcome supervision per labeled *solution*, but the labeling compute is about $1+K(S-1)/2$ times larger, so per FLOP the ORM may win; and all scorers lose calibration on generators other than the one they were trained on, in the direction predicted by D1 step 7.

*Setup.* Train on solutions of Qwen3-1.7B for $n_{\textrm{train}}\in\{500,2000,8000\}$ problems (4 solutions each; GSM8K train and `chain` train mixed 1:1); labels: terminal correctness (ORM), MC hard (PRM-hard) and soft (PRM-soft) with $K=8$; backbone Qwen3-0.6B + LoRA rank 32; 1 epoch; 3 seeds per configuration. Evaluate on pools of 16 samples per problem from each of the three generators for 500 problems per test split (GSM8K test, `chain` unseen templates, `chain` harder). Rerank with $n\in\{1,2,4,8,16\}$ drawn without replacement from the pool (`select_by_score`, 200 trials).

*Procedure.* Metrics per scorer, generator and split: best-of-$n$ accuracy; AUROC and ECE of the solution-level score (for PRM: the aggregate; for the generative verifier: the Yes-probability); the three PRM aggregations. Baselines: random choice (= single-sample accuracy), majority vote (computed in 9.5; show here only the number), oracle (pass@$n$ by `pass_at_k`). Cost accounting: labeling FLOPs (ledger) + training FLOPs ($6N$ per training token) + inference FLOPs ($2N$ per scored token, scoring all $n$ candidates).

*Plots.* `fig_lm_verifier_bon`: best-of-$n$ accuracy (y) vs $n$ (x, log) for ORM, PRM-min, PRM-prod, GenRM-zero-shot, majority, oracle, on the 1.7B pool of `chain` unseen templates (a second panel: GSM8K test, flagged public); bands from the paired bootstrap over problems and seed spread as markers. `fig_lm_verifier_cost`: accuracy at $n=8$ (y) vs total labeling+training FLOPs (x, log) at the three $n_{\textrm{train}}$ values for ORM, PRM-hard, PRM-soft. `fig_lm_verifier_calibration`: reliability diagrams (15 bins) for the ORM and PRM-prod on the three generators; ECE in the legend. `fig_lm_verifier_labels`: the step-label curves of the three example solutions (soft and hard).

*Expected outcome.* Directions with sources: Lightman et al. report, with human step labels on MATH, that process supervision beats outcome supervision for best-of-$N$ reranking; Uesato et al. report similar final-answer error for the two on GSM8K with lower trace error for process supervision (confirm in the abstract); Math-Shepherd reports gains from MC-labeled PRMs for verification and RL. Whether MC labels at small $K$ with a 0.6B backbone reproduce this is to be measured; the notebook is allowed to find the ORM competitive per FLOP. Calibration across generators: sign predicted by D1 step 7.

*Noise check.* Three scorer seeds; paired bootstrap over problems (same problems and same candidate draws for every scorer; `select_by_score` uses a common random seed across scorers); differences between scorers claimed only when the bootstrap interval excludes zero and the sign agrees in all three seeds. The oracle-minus-majority headroom of the opening cell is reported with its own interval.

*T1 reduction.* Committed labels and pools; live training of one ORM and one PRM-soft on 2,000 solutions with one seed; remaining seeds read from `lm-scorers-v1`; labeling at T1 only for a demo of 20 solutions.

*Artifacts committed for T2.* `lm-prm-mc-labels-v1`, scorer weights per seed, per-candidate score tables (so 9.5 can use `orm_score` and `prm_scores` without rerunning scorers), the reliability-diagram data.

#### Figures and tables

- `tab_lm_verifier_kinds`: ORM / PRM / generative verifier: reads, labels, inference cost, typical failure.
- `fig_lm_verifier_labels`, `fig_lm_verifier_bon`, `fig_lm_verifier_cost`, `fig_lm_verifier_calibration` (headline).
- `tab_lm_verifier_credit`: the variance table of D1's numerical check (per step: mean of the three weights, variance of ORM and local, ratio).
- `tab_lm_verifier_labelcost`: ratio $1+K(S-1)/2$ for $K\in\{4,8,16\}$ and $S\in\{3,6,10\}$, with measured tokens from the ledger beside it.

#### Pitfalls and misconceptions

- A PRM trained on MC labels learns *value under the labeling generator*, not step correctness; it can reward a wrong step that is later repaired.
- Hard MC labels are optimistic and soft labels are noisy: report both.
- Step boundaries: a different splitter changes labels and scores; evaluate with the same splitter used in training.
- Scorers trained on one generator are not calibrated for another (D1 step 7); ranking within a generator is more stable than thresholds across generators.
- Selection bias: the scorer sees only candidates from one distribution, so it may be best at *distinguishing the generator's typical errors*, not at checking mathematics.
- Using the same problems for labeling, training and evaluation; the splits come from different problem sets, and the generator for training never sees test problems.
- Reporting best-of-$N$ without charging the verifier's inference tokens.
- Using a scorer as an RL reward without a guard invites the optimization failure of `subsec_rl_sequences_hacking`: the scorer's errors become the policy's targets.

#### Summary (draft bullets)

- ORM, PRM and generative verifiers are one backbone with different read positions and labels.
- With deterministic transitions, the value function of the generator is a martingale; step advantages are its increments, and exact local credit has the same mean as the terminal-reward weight.
- Variance is not ordered in general; what a PRM adds is localization of the failing step and denser supervision per solution.
- MC step labels estimate a prefix's success probability under the generator; hard labels are optimistic by $1-(1-V)^K-V$, soft labels are unbiased with variance $V(1-V)/K$.
- Labeling costs $1+K(S-1)/2$ times the solution; weigh accuracy against total labeling and inference FLOPs.
- Scorers lose calibration on other generators in a predictable direction.

#### Exercises

1. *(check)* Hard MC labels with $K=16$ at a prefix whose true success probability is $V=0.1$: what is the expected label? What does this do to a PRM trained on it? *Sketch:* $1-0.9^{16}=0.815$; the PRM is told that this prefix is mostly good; it will accept steps that the generator rarely recovers from.
2. *(check)* Why does an ORM trained on complete solutions and applied to a prefix not estimate $V(s_j)$? *Sketch:* it never saw prefixes without a final answer; the read position and the distribution of inputs differ; training on prefixes with the *final* label (Cobbe et al.'s token-level loss) fixes the input distribution but gives noisy labels with variance $V(1-V)$ instead of $V(1-V)/K$.
3. *(derive)* Prove D1(b): $\mathrm{Var}(R)=\sum_j\mathbb{E}\Delta_j^2$. *Sketch:* square the telescoped sum and use orthogonality of martingale increments; $\mathrm{Var}(R)=V_0(1-V_0)$ for $R\in\{0,1\}$.
4. *(derive)* In the independent-step chain with per-step success $c$ and $S$ steps and no recovery (an error makes $R=0$), compute $V(s_j)$ given all steps correct so far, the hard-label bias at prefix $j$ for $K$ rollouts, and the prefix at which the bias is largest. *Sketch:* $V(s_j)=c^{S-j}$; hard mean $1-(1-c^{S-j})^K$; the bias $1-(1-V)^K-V$ is maximized at $V=1-K^{-1/(K-1)}$ (derivative of $-(1-V)^K+\ldots$); for $K=8$ this is $V\approx0.26$.
5. *(measure)* Replace the three PRM aggregations by a learned linear combination of (min, product, last, length) fitted on a validation pool; predict whether it beats the best fixed aggregation and report an interval.
6. *(measure)* Train the ORM on $n_{\textrm{train}}\in\{500,2000,8000\}$ solutions and the PRM-soft with the same *label budget* in FLOPs (fewer problems). Predict which wins at $n=8$ and plot accuracy against labeling FLOPs.
7. *(open)* Build a verifier for code: define a step (line or block), a label (unit-test pass rate of a rollout from the prefix), and a rerank protocol on a small LiveCodeBench subset *dated after the model's cutoff*, if one exists. No reference answer; state the contamination status of your subset.

#### Further reading

- Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) and [PRM800K](https://github.com/openai/prm800k): human step labels, the process vs outcome comparison.
- Wang et al., [Math-Shepherd](https://arxiv.org/abs/2312.08935): MC step labels without humans.
- [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275) (new, verified): the original controlled comparison of the two supervision types and the trace-error metric.
- Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (new, verified): the GSM8K verifier and best-of-$N$.
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240) (new, verified): the generative verifier formulation behind the zero-shot arm.
- [ProcessBench](https://arxiv.org/abs/2412.06559) (new, verified): benchmark for identifying the first erroneous step; the standard evaluation of step-level verifiers beyond reranking.
- Code: Hugging Face [search-and-learn](https://github.com/huggingface/search-and-learn) (Apache-2.0, verified) with the [Scaling test-time compute with open models](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) write-up; Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/).

#### Risks and open decisions

- *Labeling cost at T2.* If the full 3,000+3,000 problem labeling exceeds budget, lower to $K=4$ and 2,000 problems and report the ledger; the committed labels fix what the notebook was run with.
- *Step splitting for Qwen3 non-thinking outputs.* The model may ignore `Step k:`; fallback: blank-line splitter, and drop solutions with $S<2$ or $S>12$ from PRM training (report the dropped fraction).
- *`PRMTrainer`/`RewardTrainer` API in the pinned TRL.* To be verified; the production cell is optional and falls back to `search-and-learn` alone.
- *Released PRMs.* Not depended upon; verify a model ID at authoring time or drop the sentence.
- *MC labels and verifier errors.* The verifier (8.5's `d2l.verify_math`) is imperfect; the notebook reports its agreement with exact integer match on the `chain` family, where the gold answer is exact.
- *Math-Shepherd license.* Not declared in the dataset metadata (checked 2026-10-29); download-only and analysis-only, nothing derived committed, as stated; `lm-prm-mc-labels-v1` is the book's own labeling of its own generators' solutions and does not contain Math-Shepherd rows.

### 9.5 Parallel test-time compute

- **Label:** `sec_lm_parallel_tts` · **Tier:** T1 (reduced T1 configuration: the T1 run is the reduced configuration: analysis of the committed sample pools on CPU, plus a live mini-run of Qwen3-0.6B non-thinking, $N=16$, 100 MATH-500 problems, at most 768 new tokens; the full pools are produced once by a T2 job of a few hours that is not part of the notebook) · **Budget:** analysis cells to be measured, CPU-only, minutes; live mini-run to be measured, target ≤ ~15 min and ≤ 15 GiB (0.6B bf16 plus KV cache; no scorer resident at the same time) · **Codex units:** 11.4
- **Builds on (book):**
  - `sec_rl_sequences` (`subsec_rl_sequences_group`): a group of $K$ samples used as a baseline in training; here a group of $N$ samples is used at inference. The symbol $N$ (test-time samples) is kept distinct from the training group size $K$.
  - `sec_decoding` (`sample_next`, temperature, top-$p$): sampling parameters; samples are only diverse if the decoder is stochastic.
  - `sec_mdl-statistics`: bootstrap, clustered resampling.
  - `sec_scaling-laws`: the FLOP accounting that defines the compute unit.
  - `sec_kv-cache`: decode is memory-bound, so $N$ samples batch well and wall-clock grows sublinearly in $N$ up to the memory limit; stated in one paragraph with a measurement.
- **Builds on (this part):** 1.4 (`d2l.pass_at_k(n, c, k)` unbiased estimator), 1.5 (`d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None)` returns a `BootstrapResult`; read `r.mean` and `r.ci`), 9.1 (`ComputeLedger`, `nonembedding_params`, `make_problems`), 9.4 (`d2l.select_by_score`, `d2l.ProcessScorer` score columns, `d2l.ece`, `d2l.auroc`), 8.5 (`d2l.verify_math`, `d2l.sample_responses` for the live mini-run), 1.4 (`d2l.extract_answer`), 1.3 (chain-of-thought prompting).
- **Exports** (`#@save`, new):
  - `d2l.load_tts_pool(dataset, model, mode) -> DataFrame`: loads a pool from `DATA_HUB['lm-tts-pools-v1']` with columns `problem_id, sample_id, n_gen_tokens, think_tokens, answer, correct, logprob_sum, orm_score, prm_min, prm_prod, genrm_score` (scores present only where 9.4 scorers were trained: `chain` and GSM8K; `NaN` elsewhere).
  - `d2l.answer_classes(answers: list[str]) -> list[int]`: groups final answers into equivalence classes (Math-Verify equivalence through 8.5's verifier for math; exact string for integers); the unit of voting.
  - `d2l.majority_vote(classes, weights=None, rng=None) -> int`: plurality over classes with optional weights, ties broken uniformly with `rng`.
  - `d2l.accuracy_vs_cost(pool, method, n_list, n_trials=200, seed=0, cost='tokens'|'gte') -> DataFrame`: for each problem and $n$, draws `n_trials` subsets without replacement, applies `method` in `{'single','majority','weighted_orm','bon_orm','bon_prm_min','bon_genrm','oracle'}`, returns per-problem accuracy and the mean cost of the drawn subset.
  - `d2l.maj_accuracy_indep(p, q, N, exact_max=15, n_mc=200000, seed=0) -> float`: plurality accuracy under the independence model: exact enumeration for $N\le$ `exact_max` and $\le3$ wrong classes, Monte Carlo otherwise.
  - `d2l.maj_error_bound(p, q, N) -> float`: $\sum_j[1-(\sqrt p-\sqrt{q_j})^2]^N$.
  - `d2l.coverage_beta(a, b, N) -> float`, `d2l.fit_beta_mle(correct_counts, n) -> (a, b)`: Beta–binomial coverage and its maximum-likelihood fit (Newton or `torch.optim.LBFGS`, ≤ 25 lines).
  - Data: `DATA_HUB['lm-tts-pools-v1']` (Parquet pools, described in the chapter table; the pool-generation script is also committed as `gen_pools.py` through `%%writefile`).
- **Models and data:**
  - Generators: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` in non-thinking mode (temperature 0.7, top-$p$ 0.8, top-$k$ 20; $N_{\max}=64$, at most 1,024 new tokens) and `Qwen/Qwen3-1.7B` in thinking mode (temperature 0.6, top-$p$ 0.95, top-$k$ 20, min-$p$ 0; $N_{\max}=16$, at most 6,144 new tokens, never greedy, as the model card requires). All apache-2.0.
  - Problem sets: `chain` `test_unseen_templates` and `test_harder` (500 each; clean), `openai/gsm8k` test (500-problem seeded subset of 1,319; MIT; public, possibly seen), `HuggingFaceH4/MATH-500` (500; the Hub entry carries no license tag, its card says it is the 500-problem subset of MATH that OpenAI chose for *Let's Verify Step by Step*; record "MIT upstream": the subset comes from OpenAI's PRM800K repository, MIT, <https://github.com/openai/prm800k>, and MATH is MIT, <https://github.com/hendrycks/math>; downloaded at run time from the Hub, with attribution to both upstreams), AIME 2024 `Maxwell-Jia/AIME_2024` (MIT) plus AIME 2025 `math-ai/aime25` (apache-2.0), 60 problems, thinking mode only, public and wide intervals (the problems are MAA material; the notebook does not redistribute them beyond the dataset hosts).
  - Code (exercise only): `livecodebench/code_generation_lite` (the Hub tag is the ambiguous `cc`, checked 2026-10-29, so it is downloaded at run time only and problem text is never committed; the exact variant is to be verified), problems dated after the generator's cutoff when they exist; benchmark description in [LiveCodeBench](https://arxiv.org/abs/2403.07974) (new, verified) and its [repository](https://github.com/LiveCodeBench/LiveCodeBench) (HTTP 200).
  - Contamination: chapter policy; every public-set panel is labeled "public, possibly seen"; the main figure uses the clean `chain` splits.
- **Packages:** `numpy`, `pandas`, `pyarrow`; `vllm` or the 8.6 engine for the live mini-run only; `math-verify` through 8.5's `d2l.verify_math`.

#### Narrative arc

1. **The question.** Opening cell: on the MATH-500 mini-run (100 problems, $N=16$): print pass@1 (mean over samples), majority@16, pass@16 (the oracle) and the mean tokens spent per problem for each. The symptom: pass@16 is far above majority@16, and majority@16 is above pass@1, while the cost is 16×. "Which of these numbers can a deployed system actually have, and what does each cost?" (Numbers are to be measured.)
2. **Terms and the compute unit.** Definitions of pool, sample, answer class, selector. Compute unit (chapter overview): generated tokens; verifier tokens converted by $N_{\textrm{ne,verifier}}/N_{\textrm{ne,generator}}$ into *generator-token equivalents* (GTE). One paragraph on wall-clock: $N$ samples decode as a batch, so wall-clock is sublinear in $N$ until memory binds (`sec_kv-cache`); a cell times $N\in\{1,4,16\}$ on the mini-run and reports tokens/s.
3. **Pass@k is not a deployable number.** `pass@k` (1.4's estimator) vs a selector; best-of-$N$ with the oracle is a ceiling, drawn as a labeled diagnostic in every figure.
4. **Majority vote and self-consistency (derivation D1).** Exact curves, the threshold $p$ vs $\max_jq_j$, the bound; a cell shows a case where voting makes accuracy go *down* ($p=0.3$, wrong mode $0.5$).
5. **Correlated samples (derivation D2).** Per-problem difficulty as the source of correlation; the Beta–binomial ceiling; intraclass correlation and effective sample size; the three-curve check on a real pool (single-$\bar p$ independence, per-problem independence from held-out samples, measured).
6. **Coverage scaling (derivation D3).** $1-\textrm{pass@}N$ as a power law in $N$ with exponent set by the difficulty distribution's lower tail; fit $(a,b)$ per pool; extrapolation check by fitting on $N\le16$ and predicting $N=64$.
7. **Selecting with a scorer.** Weighted voting and best-of-$N$ with ORM/PRM/generative-verifier columns from 9.4; the plateau of imperfect verifiers (D3); calibration changes with generator. No new training here.
8. **Compute-matched comparison (headline).** Accuracy vs generated tokens for all selectors and for thinking-mode single and voted samples.
9. **Beyond math.** One short cell for code: votes over behavior classes (outputs on probe inputs), with the clustering cost stated; the experiment itself is an exercise.

#### Derivations

**D1. Majority vote (self-consistency) under independence, and best-of-$N$ with a perfect verifier.**

*Statement.* Samples are i.i.d. given the problem; each yields the correct answer with probability $p$ and wrong answer class $j$ with probability $q_j$, $\sum_jq_j=1-p$. Let the selector be plurality over answer classes with uniformly random tie-breaking. (a) For one wrong class ($q=1-p$) and odd $N$, accuracy is $\sum_{k>N/2}\binom Nkp^k(1-p)^{N-k}$. (b) As $N\to\infty$ accuracy tends to $1$ if $p>\max_jq_j$, to $0$ if $p<\max_jq_j$, and (with ties) to $1/(\#\textrm{modal classes})$ if $p=\max_jq_j$. (c) The error satisfies $\Pr[\textrm{error}]\le\sum_j\big[1-(\sqrt p-\sqrt{q_j})^2\big]^N$. (d) With a perfect verifier, best-of-$N$ accuracy is $1-(1-p)^N$ (coverage, "pass@$N$").

*Assumptions.* Independence given the problem; fixed answer-class probabilities; classes are the equivalence classes of `answer_classes`.

*Steps.*
1. (a) The correct class wins iff it has more than half of the votes; sum the binomial tail.
2. (b) By the strong law the empirical class frequencies converge to $(p,q_1,\dots)$; the plurality of the limit is the argmax; ties at the limit are broken uniformly.
3. (c) For wrong class $j$ the plurality can fail only if $X_j\ge X_c$ where $X_c,X_j$ are the counts. For $\lambda>0$, $\Pr[X_j-X_c\ge0]\le\mathbb{E}e^{\lambda(X_j-X_c)}=(1-p-q_j+pe^{-\lambda}+q_je^{\lambda})^N$ (Chernoff). The minimum over $\lambda$ is at $e^\lambda=\sqrt{p/q_j}$, giving $(1-p-q_j+2\sqrt{pq_j})^N=(1-(\sqrt p-\sqrt{q_j})^2)^N$ since $p+q_j-2\sqrt{pq_j}=(\sqrt p-\sqrt{q_j})^2$. A union bound over $j$ finishes. In the binary case the base equals $2\sqrt{p(1-p)}$, the exact asymptotic rate of the binomial tail, so the bound is tight in the exponent and loose by a polynomial factor at small $N$.
4. (d) A perfect verifier accepts iff at least one of the $N$ samples is correct: $1-(1-p)^N$.
5. Reading: majority vote helps only when the right answer is the *modal* answer; it converts a better-than-tied margin into reliability at the rate $(\sqrt p-\sqrt q)^2$ per sample. "Wisdom of the crowd" fails when errors agree. Self-consistency (Wang et al.) is this rule applied to final answers after marginalizing over reasoning paths; answer normalization (equivalence classes) is part of the algorithm.

*Numerical check (cell `tts-majority`, CPU, seconds).* (i) Binary $p=0.6$: exact accuracy at $N=1,3,5,9,15$ equals $0.6,0.648,0.6826,0.7334,0.7869$ (values from the author's enumeration; the cell recomputes and compares with `maj_accuracy_indep` to $10^{-10}$). (ii) Three-way case $p=0.4$, $q=(0.3,0.3)$: $0.4,0.424,0.447,0.4887,0.5341$ at the same $N$ (random tie-breaking). (iii) The *failure* case $p=0.3$, $q=(0.5,0.1,0.1)$: $0.3,0.282,0.2738,0.2362,0.1871$, i.e. voting decreases accuracy because the wrong answer is the mode (the exact value at $N=1$ is $p$). (iv) The bound of (c) is $\ge$ the exact error for all $N\le33$; the slope of $\log(\textrm{error})$ between $N=257$ and $513$ at $p=0.6$ is within 10% of $\log(2\sqrt{0.6\cdot0.4})$ (the author's check gives $-0.0216$ vs $-0.0204$). (v) Monte Carlo with $10^6$ draws agrees with the exact values within 3 standard errors.

**D2. Correlated samples: a ceiling for voting and an effective sample size.**

*Statement.* Let the per-problem success probability $p_i$ vary across problems with distribution $F$, samples i.i.d. given the problem. Unconditionally the samples for one problem are exchangeable and positively correlated. (a) Binary case: majority accuracy at $N\to\infty$ is $\Pr_F[p_i>1/2]$, not $\bar p$-dependent through $N$: more samples cannot help problems with $p_i<1/2$. For $F=\textrm{Beta}(a,b)$ and odd $N$, accuracy is a Beta–binomial tail. (b) The variance of the empirical success rate of $N$ samples on one problem drawn from the population is $\bar p(1-\bar p)\,[1+(N-1)\rho]/N$ with $\rho=1/(a+b+1)$ for Beta$(a,b)$, so $N_{\textrm{eff}}=N/(1+(N-1)\rho)\to1/\rho=a+b+1$. (c) An independence model that uses the *population* mean $\bar p$ for every problem overstates majority accuracy; one that uses per-problem $p_i$ is exact if samples are conditionally i.i.d.

*Assumptions.* Binary outcomes for (a), (b); conditional independence given the problem; $F$ Beta for closed forms.

*Steps.*
1. (a) Conditional on $p_i$, accuracy at $N$ is the binomial tail $T_N(p_i)$, which increases to $1[p_i>1/2]$ for $N\to\infty$ (D1(b)). Average over $F$.
2. For Beta$(a,b)$: number of successes among $N$ samples has the Beta–binomial pmf $\binom Nk\frac{B(k+a,N-k+b)}{B(a,b)}$; sum $k>N/2$.
3. (b) $\textrm{Cov}(y_s,y_t)=\textrm{Var}(p_i)=\frac{ab}{(a+b)^2(a+b+1)}$ for $s\ne t$; $\textrm{Var}(\bar y_N)=\frac{\bar p(1-\bar p)}{N}\big[1+(N-1)\rho\big]$ with $\rho=\textrm{Var}(p_i)/(\bar p(1-\bar p))=1/(a+b+1)$.
4. Example: $a=2,b=1$ ($\bar p=2/3$, $\rho=0.25$, $N_{\textrm{eff}}\to4$). Independence at $p=2/3$: $0.6667,0.7407,0.8552,0.9765,1.0$ at $N=1,3,9,33,129$. Beta–binomial: $0.6667,0.7000,0.7273,0.7429,0.7481$ and limit $\Pr[p_i>1/2]=1-(1/2)^2=0.75$ (author's check). The population-mean independence model predicts $1.0$ where the truth is $0.75$.
5. Where correlation comes from in LLMs: problem difficulty, a shared misconception (the same wrong answer appears in many samples), low temperature or duplicated prefixes (samples that are not conditionally independent), correlated errors across *models* if two models are mixed. The remedy is more *diverse* samples (temperature, prompt variants, different models), not more of the same.

*Numerical check (cell `tts-correlation`, CPU).* Compute the Beta–binomial accuracy for $(a,b)=(2,1)$ at $N\in\{1,3,9,33,129,1025\}$ and assert it increases to within $10^{-3}$ of $0.75$; simulate $10^6$ problems with $p_i\sim\textrm{Beta}(2,1)$, estimate the variance of $\bar y_N$ and compare with $\bar p(1-\bar p)[1+(N-1)\rho]/N$ within 3 SE for $N=4,16,64$; on a committed pool, fit the three predictions of (c) as the real-data version (see headline experiment).

**D3. Coverage scaling and the plateau of imperfect verifiers.**

*Statement.* (a) For one problem with success probability $p$, $\textrm{pass@}N=1-(1-p)^N$; `d2l.pass_at_k` (1.4) estimates it from $n\ge N$ samples. (b) If $p\sim\textrm{Beta}(a,b)$ across problems, population coverage is $c(N)=1-B(a,b+N)/B(a,b)$, and the uncovered fraction decays as $B(a,b+N)/B(a,b)\sim\frac{\Gamma(a+b)}{\Gamma(b)}N^{-a}$ as $N\to\infty$: a power law whose exponent $a$ is the lower-tail exponent of the difficulty density ($f(p)\propto p^{a-1}$ near $0$). (c) With a binary verifier that accepts a correct sample with probability $\alpha$ and a wrong one with probability $\beta$, and selects uniformly among accepted candidates (uniformly among all if none is accepted), accuracy tends as $N\to\infty$ to $\frac{\alpha p}{\alpha p+\beta(1-p)}$, which is $<1$ unless $\beta=0$. (d) With a continuous scorer, best-of-$N$ selection moves the output distribution away from the generator's; the commonly quoted $\log N-(N-1)/N$ for the KL cost of best-of-$N$ is an *upper bound*, not an exact value (Beirami et al.).

*Steps.*
1. (a) Independence; the unbiased estimator is in 1.4.
2. (b) $\mathbb{E}[(1-p)^N]=\int_0^1(1-p)^N\frac{p^{a-1}(1-p)^{b-1}}{B(a,b)}dp=\frac{B(a,b+N)}{B(a,b)}$ and $B(a,b+N)=\frac{\Gamma(a)\Gamma(b+N)}{\Gamma(a+b+N)}\sim\Gamma(a)N^{-a}$; so the ratio is $\frac{\Gamma(a+b)}{\Gamma(b)}N^{-a}$. On log–log axes of $1-c(N)$ against $N$ the slope is $-a$.
3. (c) With $N$ large and $p>0$, about $Np$ correct and $N(1-p)$ wrong samples exist; the accepted correct count is $\approx N\alpha p$ and the accepted wrong count $\approx N\beta(1-p)$; uniform selection among accepted picks a correct one with the ratio above. Example: $\alpha=0.9$, $\beta=0.1$, $p=0.3$ gives $0.794$.
4. Reading: coverage keeps growing with $N$ (empirically over several orders of magnitude, as Large Language Monkeys reports for problems with an automatic verifier) but deployable accuracy is capped by the selector's false-positive rate; verifier quality, not $N$, sets the plateau. This is why 9.4 and 9.5 are one story.

*Numerical check (cell `tts-coverage`, CPU).* For $(a,b)=(0.3,2)$: exact coverage at $N=1,10,100,1000$ is $0.1304,0.4415,0.7084,0.8532$; the asymptotic $1-\frac{\Gamma(2.3)}{\Gamma(2)}N^{-0.3}$ gives $0.4153,0.7069,0.8531$ at $N=10,100,1000$ (author's check; agreement within $10^{-3}$ at $N\ge100$). Simulate 200,000 problems with $p_i\sim$ Beta and a verifier with $(\alpha,\beta)=(0.9,0.1)$ at $N=256$ and compare with $\alpha p/(\alpha p+\beta(1-p))$ at the *per-problem* level within 3 SE. Fit $(a,b)$ by `fit_beta_mle` to the counts of a committed pool and check that fitting on $N\le16$ samples per problem predicts coverage at $N=64$ within the bootstrap interval of the measured value (a real-data test, reported whether it passes or not).

#### From-scratch implementation

Budget ≤ ~150 lines.

- `answer_classes`, `majority_vote`, `weighted_vote` (≈30 lines). Tests: uniform random tie-breaking on `[A,A,B,B]` (A and B each within 3 SE of 0.5 over 20,000 draws); weights equal to one reproduce plurality; equivalence classes merge `0.5`, `1/2`, `\frac12` through the verifier.
- `maj_accuracy_indep` and `maj_error_bound` (≈35 lines); `coverage_beta`, `fit_beta_mle` (≈30 lines). Tests: D1/D3 numerical checks; `fit_beta_mle` recovers $(a,b)=(0.5,1.5)$ from 5,000 simulated problems at $n=64$ within 15%.
- `accuracy_vs_cost` (≈45 lines): vectorized subset draws with a common RNG; cost = mean over trials of the sum of `n_gen_tokens` of the drawn subset, plus the verifier tokens in `gte` mode (`scored_tokens × N_ne,ver/N_ne,gen`; the scorer's solution lengths are in the pool). Tests: `method='oracle'` equals the mean of `1-C(m-c,n)/C(m,n)` over problems (compare to `d2l.pass_at_k` to $10^{-9}$); `method='single'` equals the mean single-sample accuracy within sampling error; with `n=m` all methods are deterministic (one draw).
- Cross-check: Raschka's `reasoning-from-scratch` implements self-consistency; run the repository's voting function on 50 problems of the mini-run pool and compare with `majority_vote` (the repo API is pinned by commit in the notebook; mark "verify the call signature in the pinned commit").

#### In production

- vLLM/SGLang `n` and `best_of` parameters sample $N$ completions with shared prefill and prefix caching (`sec_kv-cache`, 13.3), which makes parallel sampling cheaper than $N$ independent requests: the notebook measures tokens/s for $N\in\{1,4,16\}$ on the mini-run and states the prefill share. The from-scratch voting code is the oracle for any library's majority or weighted-vote implementation; `search-and-learn` implements best-of-$N$, weighted best-of-$N$ and beam search over PRMs (Apache-2.0), and the notebook checks its selected answers equal `accuracy_vs_cost(method='bon_prm_min')` on a 20-problem pool without ties.
- Hosted models: parallel sampling multiplies the bill by $N$; the ledger's GTE column is the unit for that conversation.

#### Headline experiment

*Hypothesis.* At matched generated tokens, (H1) majority vote over non-thinking samples beats a single sample on the clean `chain` splits until correlation saturates it, (H2) a trained scorer (9.4) beats majority at small $N$ and saturates earlier because of its false-positive rate, (H3) one thinking-mode answer is competitive with majority over non-thinking samples at the same token cost on hard problems and not on easy ones. These are predictions to test; none is assumed to hold.

*Setup.* Pools of `lm-tts-pools-v1`; for each problem and each $n\in\{1,2,4,8,16,32,64\}$ ($\le16$ for thinking mode), 200 random subsets without replacement; methods: single sample, majority, weighted vote with ORM score, best-of-$n$ with ORM, with PRM-min, with the generative verifier, oracle. Generator Qwen3-1.7B (headline); Qwen3-0.6B and Qwen3-4B as secondary panels. Scorers from 9.4 are used where they exist (`chain`, GSM8K).

*Procedure.* Per method and dataset, accuracy at each $n$ with its mean generated tokens and mean GTE. Fit $(a,b)$ per dataset. Compute the three predicted majority curves of D2(c) (single-$\bar p$ independence; per-problem independence, with $p_i$ estimated from the *held-out half* of the pool, i.e. samples $33$–$64$, and the vote simulated on samples $1$–$32$; measured). Report the gap between oracle and best selector as headroom.

*Plots.* `fig_lm_tts_frontier`: accuracy (y) vs generated tokens per problem (x, log) for every method and for the thinking-mode curve (single sample as a point, voted as a line); one panel per dataset (`chain` unseen, `chain` harder, GSM8K (public), MATH-500 (public), AIME (public, thinking only)); the oracle is dashed gray with the label "oracle (diagnostic, not deployable)". A second figure `fig_lm_tts_gte` repeats the `chain` unseen panel with GTE on the x axis so verifier tokens are charged. `fig_lm_tts_correlation`: measured majority accuracy vs $N$ with the two independence predictions. `fig_lm_tts_coverage`: $1-\textrm{pass@}N$ (log–log) with the fitted power law and its extrapolation. Bands: paired-bootstrap 95% over problems (`cluster=problem_id`) at each $n$; every band is for the difference to the single-sample baseline, stated in the caption.

*Expected outcome.* Directions with sources. The self-consistency paper reports large gains over greedy chain-of-thought on arithmetic and commonsense benchmarks; Large Language Monkeys reports coverage growing roughly log-linearly over about four orders of magnitude of samples and selectors (majority, reward model) plateauing on tasks without a verifier; Snell et al. report that the best test-time method depends on difficulty. Magnitudes on these pools: to be measured.

*Noise check.* Problems are the unit of resampling (clustered bootstrap), random subsets are paired across methods (shared RNG), and the number of subset trials (200) is chosen so that subset noise is below the problem-level interval (verify by repeating with 1,000 trials on one panel). The AIME panel has 60 problems; its intervals are shown and no ranking claim is made from it.

*T1 reduction.* Analysis on committed pools at CPU; live mini-run on MATH-500 with Qwen3-0.6B, $N=16$, 100 problems (wide intervals; labeled).

*Artifacts committed for T2.* `lm-tts-pools-v1` (all pools), `gen_pools.py`, fitted $(a,b)$ per pool, the frontier tables as JSON.

#### Figures and tables

- `fig_lm_tts_frontier`, `fig_lm_tts_gte`, `fig_lm_tts_correlation`, `fig_lm_tts_coverage` (headline).
- `fig_lm_tts_majority_theory`: D1 exact curves for the three cases (binary, three-way, failure), with the Chernoff bound as dashed lines.
- `tab_lm_tts_headroom`: per dataset and generator, pass@1, majority@$N$, best scorer@$N$, oracle@$N$, GTE per problem.
- `tab_lm_tts_wallclock`: tokens/s and wall-clock for $N\in\{1,4,16\}$ on the mini-run, next to the FLOP ratio.

#### Pitfalls and misconceptions

- Reporting pass@$N$ as accuracy; it is an oracle.
- Not charging for discarded samples and for the verifier's tokens.
- Voting over strings rather than equivalence classes: `1/2`, `0.5` and `\frac{1}{2}` are one answer; unnormalized voting splits the majority.
- Comparing voted accuracy with greedy decoding that was run at a different temperature; use the single-sample mean at the sampling temperature.
- Reading the independence formula for majority accuracy as a prediction for a population: correlation caps it.
- More samples from a low-temperature or prefix-sharing sampler are not independent samples.
- Majority voting can reduce accuracy when the modal answer is wrong (D1 check iii).
- Treating `n_trials` noise as problem-level noise: only the cluster bootstrap over problems measures the latter.
- Using public benchmarks as clean measurements; the chapter's policy applies.

#### Summary (draft bullets)

- Parallel compute buys coverage (pass@$N$), but a deployed system gets only what its selector extracts.
- Under independence, majority voting converges to the right answer iff it is the modal answer, at exponential rate $(\sqrt p-\sqrt q)^2$ per sample.
- Samples are correlated through problem difficulty and shared misconceptions; majority accuracy saturates at the fraction of problems where the right answer is modal.
- Coverage grows as a power law in $N$ whose exponent is the lower-tail exponent of the difficulty distribution.
- An imperfect verifier caps best-of-$N$ at $\alpha p/(\alpha p+\beta(1-p))$ in the binary model.
- Compare methods per generated token (and per GTE when verifiers run), with problem-clustered intervals and the oracle labeled as a diagnostic.

#### Exercises

1. *(check)* A problem has $p=0.35$ and three wrong answers with $q=(0.4,0.15,0.1)$. Does majority voting converge to the correct answer? What is the accuracy at $N=1$ and as $N\to\infty$? *Sketch:* $p<\max q=0.4$, so the limit is $0$; $N=1$ gives $0.35$; for moderate $N$ it first decreases.
2. *(check)* Explain why best-of-$N$ with a perfect verifier on a problem with $p=0.01$ needs about $N\approx\ln2/0.01\approx69$ samples to reach $50\%$. *Sketch:* $1-0.99^N=0.5\Rightarrow N=\ln0.5/\ln0.99\approx69$.
3. *(derive)* For Beta$(a,b)$ difficulty and odd $N$, show that binary majority accuracy tends to $\Pr[p_i>1/2]=1-I_{1/2}(a,b)$ (regularized incomplete Beta) and give a Beta example in which accuracy *decreases* with $N$. *Sketch:* for each $p>1/2$ the binomial majority probability is nondecreasing along odd $N$ and for $p<1/2$ nonincreasing, so the mixture has no monotonicity guarantee; the limit is the indicator by D1(b). For $\textrm{Beta}(2,3)$ the mean is $0.4$ at $N=1$ while $\Pr[p>1/2]=1-(6x^2-8x^3+3x^4)|_{x=1/2}=0.3125$, so accuracy falls.
4. *(derive)* Show that the Chernoff bound of D1(c) is never larger than $\sum_j\exp(-N(\sqrt p-\sqrt{q_j})^2)$ and compare it with the exact binary tail for $p=0.55$, $N=101$. *Sketch:* $1-x\le e^{-x}$; the exact error at $p=0.55$, $N=101$ is $\approx0.156$, while the bound $[2\sqrt{0.55\cdot0.45}]^{101}$ is $\approx0.60$ (values from the author's check), loose by a polynomial factor as D1 says.
5. *(measure)* Raise the sampling temperature from $0.5$ to $1.0$ on the mini-run ($N=16$): predict the direction of pass@1, majority@16 and pass@16, then run. Report intervals and the fitted $a$.
6. *(measure)* On the `chain` unseen split compare majority@$N$ of Qwen3-0.6B, 1.7B and 4B at *equal FLOPs* (use `nonembedding_params`); predict where the curves cross and plot accuracy vs FLOPs (this previews 9.6).
7. *(open)* Build a voting scheme for code generation: cluster programs by outputs on a set of probe inputs, vote over clusters, and compare with unit-test-verified best-of-$N$ on a LiveCodeBench subset dated after the model's cutoff (if none exists, say so and use a labeled-contaminated set). Define the cost of clustering.

#### Further reading

- Wang et al., [Self-Consistency](https://arxiv.org/abs/2203.11171): sampling reasoning paths and marginalizing over answers.
- Snell et al., [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314): when search/selection beats a larger model.
- Brown et al., [Large Language Monkeys](https://arxiv.org/abs/2407.21787): coverage scaling, the gap between coverage and selection.
- [Inference Scaling Laws](https://arxiv.org/abs/2408.00724) (new, verified): compute-optimal inference with sampling and reranking for small vs large models; the empirical frame for D1–D3 and 9.6.
- [Theoretical guarantees on the best-of-n alignment policy](https://arxiv.org/abs/2401.01879) (new, verified): the KL cost of best-of-$N$ and why the common formula is an upper bound.
- Code: Raschka's [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) (self-consistency from scratch), Hugging Face [search-and-learn](https://github.com/huggingface/search-and-learn); blog: Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/).

#### Risks and open decisions

- *Pool generation cost.* Three generators × five datasets × 64 samples is large; the pool definition above is the maximum. Fallback: $N_{\max}=32$ for 4B and for MATH-500; the plots state the actual $N_{\max}$.
- *Thinking-mode traces exceeding the cap.* Truncated samples have no answer; they are counted as wrong *and* reported separately; raising the cap changes the GTE axis, so the cap is part of the pool name.
- *No scorers outside GSM8K/`chain`.* MATH-500 and AIME panels use majority, oracle and the zero-shot generative verifier only; this is stated in the captions.
- *Answer equivalence on MATH-500.* Depends on 8.5's verifier; fallback: exact-string classes plus a small normalizer, with the disagreement rate between the two on 500 problems reported.
- *MATH-500 license tag.* Missing on the Hub entry; resolved as "MIT upstream" (PRM800K and MATH are both MIT, checked 2026-10-29) and listed so in the section's license table, with attribution to both.
- *AIME datasets.* Problems are third-party material; if license review rejects redistribution, the AIME panel loads from the dataset hosts at run time and the pool stores only per-sample correctness.

### 9.6 Sequential and adaptive test-time compute

- **Label:** `sec_lm_sequential_tts` · **Tier:** T1 (reduced T1 configuration: the T1 run is the reduced configuration: all sweeps and the allocation fit run on committed traces and pools at CPU speed; a live budget-forcing demo on 20 `chain` problems with Qwen3-1.7B in thinking mode, at most 2,048 thinking tokens, to be measured, target ≤ ~15 min and ≤ 15 GiB) · **Budget:** analysis minutes on CPU; live demo to be measured · **Codex units:** 11.5, 11.6
- **Builds on (book):**
  - `sec_decoding` (`generate`, stop criteria): the generation loop that budget forcing edits.
  - `sec_scaling-laws`: $2N_{\textrm{ne}}$ FLOPs per generated token (the unit for the model-size comparison).
  - `sec_kv-cache`, `sec_perf_model`: decode is memory-bound, so the FLOP ordering of model sizes and the wall-clock ordering differ; one paragraph and one measured table with `d2l.Benchmark`.
  - `sec_mdl-statistics`: validation/test discipline and bootstrap intervals; `sec_baselines` (zero-mean lemma only for the sequential stopping derivation's martingale step, one sentence).
- **Builds on (this part):** 9.5 (`d2l.load_tts_pool`, `d2l.answer_classes`, `d2l.majority_vote`, `d2l.accuracy_vs_cost`, pools `lm-tts-pools-v1`), 9.4 (`d2l.select_by_score`, scorer columns), 9.3 (thinking vs non-thinking behavior statistics, `d2l.behavior_stats`), 9.1 (`ComputeLedger`, `nonembedding_params`, `make_problems`), 1.5 (`d2l.paired_bootstrap`, `d2l.EvalSuite`), 8.5 (`d2l.sample_responses` for the live demo).
- **Exports** (`#@save`, new):
  - `d2l.budget_force(engine, prompts, budget, extend=0, wait='Wait,', stop_note=None, answer_tokens=256, ledger=None) -> list[dict]`: runs a thinking-mode generation under a thinking-token `budget`; if the model closes its thinking early and `extend>0`, strips the end-of-thinking marker, appends `wait` and continues (at most `extend` times, never beyond `budget`); if the budget is reached first, closes the block (optionally with `stop_note`) and generates the answer. Returns text, thinking tokens, answer tokens, number of extensions, whether the budget bound.
  - `d2l.vote_lead_stop(classes_stream, lead, min_n=2, max_n=64) -> (winner, n_used)`: draws answers one at a time and stops when the leading answer class has at least `lead` votes more than the runner-up.
  - `d2l.greedy_allocate(gain_fn, p, budget) -> np.ndarray`: allocates integer sample counts per problem by repeatedly granting a sample to the largest marginal gain (heap), `gain_fn(p, n)` marginal gain of the $(n+1)$st sample.
  - `d2l.AllocationPolicy`: `.signal` name, `.edges` (bin edges), `.configs` (one config per bin; a config is `{'kind': 'single'|'majority'|'thinking', 'n': int, 'budget': int}`), `.assign(signal) -> list[config]`, `.to_json(path)`, `.from_json(path)`; immutable after `freeze()`.
  - `d2l.fit_allocation(val, configs, signal, n_bins, lam_grid, cost_cap=None) -> list[AllocationPolicy]`: for each $\lambda$ chooses per bin the config maximizing accuracy $-\lambda\cdot$cost on the validation set; returns the policies along the frontier; asserts that no test problem id appears in `val`.
  - `d2l.evaluate_policy(policy, test, charge_signal=True, n_boot=10000) -> dict`: accuracy, mean cost (signal-pilot tokens charged), bootstrap interval; evaluates **once** per frozen policy (a counter raises on a second call for the same policy file unless `force_reuse=True` is passed and logged).
  - Data: `DATA_HUB['lm-tts-budget-v1']` (budget-forcing traces: per problem, thinking trace up to 6,144 tokens with token offsets, forced-answer completions at $B\in\{256,512,1024,2048,4096\}$ and with $m\in\{0,1,2,4\}$ extensions; Parquet), `lm-alloc-policy-v1` (the frozen policy JSON).
- **Models and data:**
  - Thinking-mode model: `Qwen/Qwen3-1.7B` (apache-2.0; thinking by default, temperature 0.6, top-$p$ 0.95, top-$k$ 20, min-$p$ 0). The notebook also reads `Qwen/Qwen3-4B-Thinking-2507` traces only if present in the committed artifact (262,144-token native context; recommended output lengths 32,768 and 81,920 for hard problems per its card), otherwise skips them.
  - Size ladder (non-thinking, committed pools from 9.5 plus one extension): `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, and `Qwen/Qwen3-8B` ($N_{\max}=16$ on the `chain` splits only; a pool extension generated by the same T2 job as 9.5). Non-embedding parameters from the model cards: 0.44B, 1.4B, 3.6B, 6.95B (checked against `nonembedding_params`).
  - Problems: `chain` `test_unseen_templates` and `test_harder` split by problem id into 500 validation and 500 test (disjoint, seeded), `openai/gsm8k` test and `HuggingFaceH4/MATH-500` flagged public/possibly seen with a 250/250 split. The test half is touched once per frozen policy.
- **Packages:** `numpy`, `pandas`; `vllm` or 8.6's engine for the live demo.

#### Narrative arc

1. **The question.** Opening cell (committed traces): for Qwen3-1.7B in thinking mode on `chain` unseen templates, plot the distribution of thinking length (tokens) for correct and incorrect answers and print the fraction of problems whose thinking exceeds 2,048 tokens. The symptom to measure: lengths vary by an order of magnitude across problems and wrong answers tend to be longer; one fixed length budget cannot fit both. "Which problems deserve more compute, and how do we know before spending it?"
2. **Sequential compute as a knob.** Budget forcing: (i) *truncate*: stop thinking at $B$ and force an answer; (ii) *extend*: when the model closes its thinking, replace the end marker by "Wait," and continue. Implement `budget_force` and show its trace on one problem, token by token around the intervention. The Qwen3 report describes a thinking-budget mechanism of the first kind (truncate and instruct the model to answer; section reference unverified); s1 introduced both and reports test-time scaling by extension.
3. **Budget sweep.** Accuracy and mean tokens vs $B$ and vs number of extensions $m$, on the committed traces; the fraction of answers that are *missing* or malformed at small $B$, the fraction that flip from correct to incorrect after an extension, and the opposite flip. The point: a sweep needs the *answer-extraction* confound separated from the reasoning effect, so malformed answers are a separate column.
4. **Revision.** A prompted self-revision round ("review your answer; if you find an error, give a corrected answer") on non-thinking samples: report flips in both directions; compare with revision that receives the verifier's verdict (an oracle feedback arm, labeled as such). Cite the "cannot self-correct reasoning yet" paper for the intrinsic case; the learned-revision model of Snell et al. is not reproduced (no open revision-trained model; exercise).
5. **Stopping early (derivation D2).** The vote-lead rule for adaptive parallel sampling, with the gambler's-ruin formulas, and a thinking-trace version: probe the answer at fixed token intervals and stop when $k$ consecutive probes agree (one cell, labeled as an illustrative rule, not from a paper).
6. **Allocating compute across problems (derivation D1).** The optimal split of a budget across problems; the cost of estimating difficulty; a toy first, then the real fit.
7. **Fit and freeze (headline).** Difficulty signal selection, config space, policy fit on validation, JSON freeze, single evaluation on test.
8. **Small model many times vs larger model once.** FLOPs-matched curves for 0.6B, 1.7B, 4B, 8B with majority voting and for thinking mode; wall-clock beside it.

#### Derivations

**D1. Optimal allocation of a sample budget across problems.**

*Statement.* There are $M$ problems with success probabilities $p_i$ (known for now); allocating $n_i$ i.i.d. samples to problem $i$ with a perfect verifier gives expected accuracy $a_i(n_i)=1-(1-p_i)^{n_i}$. (a) Maximizing $\sum_ia_i(n_i)$ subject to $\sum_in_i=B$, $n_i\in\mathbb{N}$ is solved exactly by granting samples one by one to the problem with the largest marginal gain $g_i(n)=p_i(1-p_i)^n$. (b) For small budgets this gives nothing to problems with $p_i\approx0$ and little to problems with $p_i\approx1$; the first sample goes to the easiest problem. (c) With a Lagrange multiplier $\lambda$ the continuous solution satisfies $g_i(n_i)=\lambda$ for all funded problems, so $n_i=\max\{0,\ln(\lambda/p_i)/\ln(1-p_i)\}$. (d) If $p_i$ is replaced by an estimate $\hat p_i$ from $n_0$ pilot samples (charged to the budget), expected gain uses the posterior $\textrm{Beta}(a+c_i,b+n_0-c_i)$; the plug-in rule is worse than the posterior-mean rule when $n_0$ is small.

*Assumptions.* Independent samples, perfect verifier (so that accuracy is coverage; with an imperfect selector replace $a_i$ by the selector's accuracy curve, still concave in the regimes of interest, checked empirically), all samples cost the same (otherwise use gain per token).

*Steps.*
1. $a_i$ is concave in $n$ ($g_i$ is decreasing in $n$), so the problem is separable concave resource allocation; for such problems the greedy rule on marginal gains is optimal (exchange argument: if some feasible solution gives $n_j$ to a problem whose next marginal gain is smaller than another's, moving one unit improves the objective).
2. (b) follows from $g_i(0)=p_i$ and $g_i(n)\to0$.
3. (c) Equalize marginal gains: the greedy rule stops funding problem $i$ when $g_i(n_i)=p_i(1-p_i)^{n_i}\le\lambda$; solve for $n_i$ and clip at 0. (Differentiating the real-valued $a_i(n)$ instead gives the gain $-\ln(1-p_i)(1-p_i)^{n}$, which differs by the factor $-\ln(1-p_i)/p_i\approx1$ for small $p_i$; the integer rule is the one the code implements.)
4. (d) Marginal gain of the next sample under the posterior: $\mathbb{E}[p(1-p)^n]=\frac{B(a'+1,b'+n)}{B(a',b')}$ with $(a',b')$ the posterior parameters. Plug-in $\hat p(1-\hat p)^n$ is biased by Jensen.
5. Cost of estimation: the pilot samples are part of the budget; with $n_0=0$ one has no signal, and the best fixed allocation is uniform.

*Numerical check (cell `alloc-greedy`, CPU).* (i) Four problems $p=(0.05,0.2,0.5,0.8)$, $B=10$: exhaustive search over all compositions of 10 equals the greedy allocation and its objective to $10^{-12}$. (ii) $300$ problems with $p_i\sim\textrm{Beta}(0.5,1)$ and $B=600$: greedy coverage vs uniform ($2$ each): the author's check gives $0.532$ vs $0.474$ (to be re-measured; seeded). (iii) With $\hat p_i$ estimated from $n_0\in\{2,4,8\}$ pilot samples and the budget reduced by the pilot cost, report coverage of plug-in greedy, posterior-mean greedy and uniform: the cell prints the table; the assertion is only that oracle greedy $\ge$ every other rule.

**D2. Adaptive stopping by vote lead.**

*Statement.* Consider two competing answer classes $A$ (correct) and $B$ with per-sample probabilities $p_A>p_B$ (other classes only delay the walk). Draw samples one at a time and stop when the vote difference $D$ reaches $+L$ or $-L$. (a) $\Pr[\textrm{stop at }-L]=\dfrac{\rho^L}{1+\rho^L}$ with $\rho=p_B/p_A$. (b) The expected number of draws is $\mathbb{E}T=\dfrac{L\,(1-\rho^L)/(1+\rho^L)}{p_A-p_B}$. (c) Easy problems (small $\rho$) stop after about $L/(p_A-p_B)$ draws; hard ones (ratio near 1) run to the cap. (d) A fixed $N$ spends $N$ on every problem.

*Assumptions.* Independent samples, class probabilities fixed, a cap `max_n` that truncates the walk (the formulas are for the uncapped walk).

*Steps.*
1. Conditional on a vote for $A$ or $B$, $D$ is a simple random walk with up-probability $u=p_A/(p_A+p_B)$ and down-probability $d=1-u$, so $d/u=\rho$.
2. Gambler's ruin on $[-L,L]$ started at 0 gives $\Pr[\textrm{hit }+L\textrm{ first}]=\frac{1-\rho^L}{1-\rho^{2L}}=\frac1{1+\rho^L}$; hence (a).
3. $D_t-(u-d)t$ is a martingale (`sec_baselines`, zero-mean increments); optional stopping gives $\mathbb{E}D_T=(u-d)\,\mathbb{E}T_{\textrm{inf}}$, where $T_{\textrm{inf}}$ counts informative votes; with $\mathbb{E}D_T=L(1-2\Pr[\textrm{err}])=L\frac{1-\rho^L}{1+\rho^L}$ and $u-d=\frac{p_A-p_B}{p_A+p_B}$ we get $\mathbb{E}T_{\textrm{inf}}$; dividing by $p_A+p_B$ converts informative votes into total draws, which is (b).
4. Example: $p_A=0.8$, $p_B=0.2$, $L=3$: error $\rho^L/(1+\rho^L)=0.0154$, $\mathbb{E}T=4.85$ draws. A fixed majority of $N=5$ at the same $p$'s has a different error; the cell computes both.

*Numerical check (cell `vote-lead`, CPU).* Simulate $10^5$ walks for $(p_A,p_B)\in\{(0.8,0.2),(0.6,0.3),(0.5,0.4)\}$, $L\in\{2,3,5\}$ (without cap): error and mean draws agree with (a)–(b) within 3 SE (the author's check at $(0.8,0.2),L=3$ gives $0.0157$ and $4.81$ from $4\times10^4$ walks vs $0.0154$ and $4.85$). Then a capped version (`max_n=64`) on a committed pool: compare error vs mean samples with fixed-$N$ majority on the same problems; report the accuracy at equal mean number of samples.

#### From-scratch implementation

Budget ≤ ~150 lines.

- `budget_force` (≈55 lines): a loop over the engine in chunks of 64 tokens with a stop at the end-of-thinking token; bookkeeping of thinking and answer tokens; extension logic; ledger entries for every generated token (including the stripped marker). Tests with a stub engine that emits a scripted token stream: (i) budget reached before `</think>` → exactly `budget` thinking tokens, then the forced marker, then the answer; (ii) `extend=2` with an early `</think>` → two "Wait," insertions, total thinking tokens $\le$ budget; (iii) ledger tokens equal the sum of the stub's emitted tokens.
- `vote_lead_stop`, `greedy_allocate` (≈30 lines). Tests: D1(i) and D2 checks; `greedy_allocate` with a heap equals brute force for 4 problems.
- `AllocationPolicy`, `fit_allocation`, `evaluate_policy` (≈50 lines): quantile bins on the signal using validation data only; argmax per bin; JSON round trip; leakage guard (set intersection of ids); evaluation counter. Tests: round trip equality; guard raises when a validation id appears in test; on a synthetic pool where the signal is uninformative the fitted policy performs at most the best fixed config on test (within the bootstrap interval) over 50 seeds in at least 90% of runs (a null test of the procedure).
- `flops_per_problem` (≈10 lines): tokens $\times\,2N_{\textrm{ne}}$ with the ledger.

#### In production

- vLLM and SGLang expose stop strings and logit processors; budget forcing in production is a *logits processor* that bans the end-of-thinking token until the minimum thinking length and forces it at the maximum, plus a prompt-side `Wait`; Qwen3's chat template exposes `enable_thinking` and soft switches (`/think`, `/no_think`) as the coarse knob. The notebook shows the logits-processor version on 5 problems and checks that its outputs match `budget_force` token for token under greedy decoding with the same budget (greedy used only for this equality test, which is why the model card's warning about greedy decoding does not apply to the experiment).
- Hosted reasoning models expose an effort or token-budget parameter; the allocation policy of this section is how one would choose it per request. Treated in one paragraph.
- s1's released code (`simplescaling/s1`, Apache-2.0) is the reference for budget forcing with `vllm` and for the evaluation harness; the notebook cites the pinned commit and states which flags map to `budget` and `extend`.

#### Headline experiment

*Hypothesis.* (H1) Accuracy as a function of thinking budget saturates, and extension beyond the model's natural stopping point helps on some problems and hurts on others, so the net gain is small relative to its cost; (H2) a policy fitted on validation that spends more on problems of intermediate estimated difficulty beats the best fixed configuration at the same *average* cost on test; (H3) at equal FLOPs, many samples from a small model beat one answer from a larger model on easy and intermediate problems but not on the hardest. None is assumed; the notebook reports which hold.

*Setup.* Committed traces and pools; the 500/500 validation/test split of each dataset by problem id. Config space for the policy: `single` (non-thinking, one sample), `majority` with $n\in\{4,8,16\}$, `thinking` with $B\in\{1024,2048,4096\}$ and no extension. Difficulty signals: (s1) agreement fraction among $n_0=4$ non-thinking samples, (s2) mean ORM score of those 4 samples (where 9.4 scorers exist), (s3) problem step count (`difficulty` field, available only on `chain`; the notebook marks it as an oracle-like feature unavailable in the wild). The signal is chosen on validation by AUROC for predicting failure of a single sample; the pilot tokens are charged to every problem.

*Procedure.* (1) Budget sweep on thinking traces: accuracy, mean tokens, missing-answer rate, flip rates, for $B$ and $m$. (2) Fit the policy with `fit_allocation` for a grid of $\lambda$ on validation only; freeze to `lm-alloc-policy-v1`; select the operating point whose *validation* average cost equals each fixed baseline's; evaluate each frozen policy once on test. (3) Baselines at matched average cost: the best single fixed config; a uniform-$n$ majority; the vote-lead rule; an *oracle* allocation that knows each problem's correctness (labeled as an upper bound, not deployable). (4) Size ladder: majority@$n$ for 0.6B, 1.7B, 4B, and 8B at $n\in\{1,2,4,8,16\}$ plotted against FLOPs per problem ($2N_{\textrm{ne}}\times$ generated tokens) with the Qwen3-1.7B thinking points; tokens/s measured with `d2l.Benchmark` for each size at batch 16 on the build box.

*Plots.* `fig_lm_seq_budget`: accuracy (y) vs mean total tokens (x, log) for truncation at $B$ (a line), extension $m=0..4$ at a fixed $B$ (a second line per $B$), and natural stopping (a point). `fig_lm_seq_alloc`: test accuracy vs average cost for each fitted policy (a frontier), the fixed baselines, the vote-lead rule, and the oracle allocation (dashed, labeled "oracle (needs correctness), not deployable"); bands from the paired bootstrap on test problems. `fig_lm_seq_size`: accuracy vs FLOPs per problem (x, log) for the four sizes plus thinking; a second panel with wall-clock instead of FLOPs. `fig_lm_seq_flips`: stacked bars of correct→incorrect, incorrect→correct, unchanged for each extension count and for prompted revision.

*Expected outcome.* Directions with sources. s1 reports that extending thinking with "Wait" improves accuracy beyond the budget the model was trained on, in a fine-tuned 32B model; whether a 1.7B model trained with a different recipe benefits is to be measured. Snell et al. report that the best allocation depends on question difficulty and that, for easy and intermediate questions under a modest inference workload, test-time compute on a smaller model can outperform a much larger model in FLOPs-matched comparisons (check the paper for the exact conditions); Wu et al. report compute-optimal inference patterns that favor smaller models with sampling in some regimes. Magnitudes: to be measured.

*Noise check.* Test evaluation once per frozen policy (the counter in `evaluate_policy`); paired bootstrap over test problems (the same problems for all baselines); intervals reported for differences to the best fixed config at matched cost; the null-test of the fitting procedure (shuffled validation labels) is run and shown as `fig_lm_seq_null`. Validation noise is controlled by 5 repeated random 500/500 splits on `chain` (the committed split is split 0); the notebook reports how many of the 5 splits agree in sign.

*T1 reduction.* This section is T1: analysis on committed traces; live demo on 20 problems (labeled, no conclusion).

*Artifacts committed for T2.* `lm-tts-budget-v1`, the frozen policy JSON, the pool extension with Qwen3-8B, tokens/s tables.

#### Figures and tables

- `fig_lm_seq_budget`, `fig_lm_seq_alloc`, `fig_lm_seq_size`, `fig_lm_seq_flips`, `fig_lm_seq_null` (headline).
- `fig_lm_seq_trace`: a single traced problem with the position of each intervention.
- `fig_lm_seq_alloc_toy`: D1 toy: uniform vs greedy vs oracle with the allocation histogram over $p_i$.
- `tab_lm_seq_policy`: the frozen policy (bins, edges, config per bin, validation cost).
- `tab_lm_seq_speed`: tokens/s and FLOPs per token for each size (from `d2l.Benchmark` and `nonembedding_params`).

#### Pitfalls and misconceptions

- Truncating thinking and then reading an *empty* answer as wrong mixes a reasoning effect with an extraction effect; count missing answers separately and include the forced-answer prompt in the cost.
- Budget forcing changes the distribution the model was trained on; a model not trained for "Wait" may repeat itself; check for loops (repeated $n$-grams) in extended traces.
- Evaluating the allocation policy on the data it was fitted on; the code guards it, but test-set reuse across policy variants is the quiet version of the same error.
- Charging nothing for the difficulty signal; a pilot of $n_0$ samples is real cost.
- FLOP-matched is not time-matched: small models are less efficient per FLOP at decode (memory-bound) so wall-clock favors them less than FLOPs suggest.
- Mixing thinking and non-thinking tokens in one budget without saying so; thinking tokens are generated tokens.
- Confusing oracle allocation (knows correctness) with a deployable policy.
- Majority over thinking-mode samples multiplies already-long generations; cap and report truncation.

#### Summary (draft bullets)

- Sequential compute lengthens or revises one trajectory; budget forcing truncates or extends thinking and must separate answer-extraction effects from reasoning effects.
- Prompted self-revision flips answers in both directions; measure both before trusting it.
- With concave returns, the best split of a sample budget funds problems by marginal gain; difficulty must be estimated and its cost charged.
- A vote-lead stopping rule has closed-form error and expected cost; easy problems stop early.
- Fit allocation rules on validation data, freeze them, evaluate once.
- Compare many small-model samples with one large-model answer at equal FLOPs and report wall-clock beside it.

#### Exercises

1. *(check)* With $p_A=0.7$, $p_B=0.2$ (and $0.1$ spread over other answers) and lead $L=2$, compute the error rate and the expected number of draws. *Sketch:* $\rho=2/7$, error $=\rho^2/(1+\rho^2)=0.0755$; $\mathbb{E}T=L\frac{1-\rho^L}{1+\rho^L}/(p_A-p_B)=2\cdot\frac{0.918}{1.082}/0.5\approx3.40$ draws in total.
2. *(check)* Explain why granting the first sample to the *easiest* problem is optimal at small budgets but the wrong policy when a budget is large. *Sketch:* marginal gain $p_i(1-p_i)^n$ is largest for the largest $p_i$ at $n=0$; as easy problems saturate, their gains fall below those of medium problems.
3. *(derive)* Show that for the continuous relaxation in D1(c) the funded set is $\{i:p_i>\lambda\}$ and give $n_i$ for one funded problem. *Sketch:* $n_i>0$ iff $g_i(0)=p_i>\lambda$.
4. *(derive)* Show that the plug-in estimate $\hat p(1-\hat p)^n$ (with $\hat p$ the posterior mean) underestimates the posterior-mean marginal gain when $\varphi(p)=p(1-p)^n$ is convex over the posterior's mass, and overestimates it when $\varphi$ is concave there. *Sketch:* Jensen's inequality on $\varphi(p)=p(1-p)^n$, whose second derivative is $n(1-p)^{n-2}[(n+1)p-2]$: convex for $p>2/(n+1)$ (plug-in underestimates) and concave below it (plug-in overestimates, the counterexample region).
5. *(measure)* Vary the extension phrase ("Wait," vs "Hmm," vs "Let me double-check."), predict the direction on flip rates, and run on 20 problems per phrase with a paired comparison (wide intervals; say so).
6. *(measure)* Add the 4B thinking model (if committed) to the size ladder and find the FLOPs at which its single thinking answer overtakes majority@$n$ of Qwen3-1.7B non-thinking; report with an interval.
7. *(open)* Design a learned difficulty predictor (a small classifier on the question embedding from 2.3, or on the first sample's hidden state) and test whether it beats the pilot-agreement signal at equal cost. State what features you allow and keep the test split untouched.

#### Further reading

- Muennighoff et al., [s1](https://arxiv.org/abs/2501.19393) and its [code](https://github.com/simplescaling/s1) (Apache-2.0, verified): budget forcing, test-time scaling curves.
- Snell et al., [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314): difficulty-adaptive allocation; revisions and search; small model with more compute vs larger model.
- Huang et al., [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) (new, verified): intrinsic self-correction and why oracle feedback must be labeled.
- [Inference Scaling Laws](https://arxiv.org/abs/2408.00724) (new, verified): compute-optimal inference across model sizes and strategies.
- Qwen team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (new): the thinking-budget mechanism in a released family.
- Blog and code: Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/); Hugging Face [search-and-learn](https://github.com/huggingface/search-and-learn).

#### Risks and open decisions

- *Qwen3-1.7B thinking traces are long on hard problems.* The committed traces cap at 6,144 thinking tokens; any sweep above the cap is not available; the plots end at the cap.
- *Extension may not help a model not trained for it.* Expected possibility; the section then teaches the flip-rate analysis and the null result, and exercise 5 explores phrases.
- *Difficulty signal availability.* (s3) uses a field unavailable in the wild; kept only as an oracle-like reference and never selected by the fit.
- *Test-set single use.* If the editor wants multiple policy variants evaluated, use more validation splits instead of reusing the test half.
- *Qwen3-8B pool extension.* It enlarges the T2 pool job of 9.5; fallback: drop the 8B series and use 4B as the "larger model" (the claim then covers two ratios only).
- *Thinking-budget section reference in the Qwen3 report* is unverified; replace by a quoted passage once located.

### 9.7 Efficient reasoning

- **Label:** `sec_lm_efficient_reasoning` · **Tier:** T2 (reduced T1 configuration: the 8.5 Qwen3-0.6B-Base GRPO checkpoint with LoRA continuation for 30 steps as a smoke test, plus committed T2 logs and frontier tables for every plot) · **Budget:** T2: a few hours on 1× A100 40 GB for all arms (the arms are short LoRA continuations of the 8.5 checkpoint, which is a LoRA adapter on frozen bf16 `Qwen/Qwen3-1.7B-Base` (3.44 GB), so the adapter is trained further beside a vLLM engine at `gpu_memory_utilization` ≈ 0.4; rollouts dominate; peak memory to be measured, target ≤ 38 GiB; $M_2$'s SFT uses LoRA too, and $M_3$ needs no second model because its teacher is the same weights under `concise_prompt`); T1: to be measured, target ≤ ~15 min and ≤ 15 GiB · **Codex units:** 10.6, 11.7
- **Builds on (book):**
  - `sec_regularized` (`eq_kl_objective`, `eq_kl_optimum`): the length-penalized objective is a reward-shaped KL-regularized problem; D1 uses the closed form.
  - `sec_rl_sequences` (`subsec_rl_sequences_hacking`): reward shaping creates exploitable shortcuts (short wrong answers); `subsec_pg_loss_divisor` and `sec_baselines`: the length-normalization choices that interact with length rewards (8.5 derived them).
  - `sec_scaling-laws`: the FLOP accounting for the amortization formula.
  - `sec_mdl-statistics`: paired comparisons.
- **Builds on (this part):** 8.5 (`d2l.grpo_train`, `d2l.verify_math`, and the 8.5 headline checkpoints), 1.4 (`d2l.extract_answer`), 9.1 (`ComputeLedger`, `make_problems`), 9.2 (`d2l.OnPolicyDistiller`, `d2l.privileged_prompt` pattern for the concision-conditioned teacher), 9.3 (`d2l.eval_families`, `d2l.behavior_stats`, `d2l.compute_matched_grid`, thinking-mode logs), 9.6 (`d2l.budget_force`, committed traces `lm-tts-budget-v1`, `d2l.AllocationPolicy` as the deployable alternative), 1.4–1.5 (`d2l.pass_at_k`, `d2l.paired_bootstrap`).
- **Use of 8.5's trainer.** The length reward is group-relative (it needs the minimum and maximum length in the group), and `d2l.grpo_train` accepts a batch-level `reward_fn(texts, items) -> list[float]`. 9.7 passes a closure that calls `d2l.verify_math` per item, derives `group_ids` from the prompt-major sample order (`idx // cfg.K`) and calls `length_reward`. One-line fallback: if the sample order is not documented as prompt-major, group by the `items` entries' prompt ids.
- **Exports** (`#@save`, new):
  - `d2l.length_reward(correct, lengths, group_ids, scheme='kimi', mu=1.0, l_max=None) -> np.ndarray`: per-completion length reward. `scheme='kimi'`: within each group, $\lambda_i=0.5-\frac{\ell_i-\ell_{\min}}{\ell_{\max}-\ell_{\min}}$, reward $\lambda_i$ if correct and $\min(0,\lambda_i)$ if wrong (as described in the Kimi k1.5 paper; the notebook cites the equation and states the check against the paper); `scheme='linear'`: $-\mu\,\ell_i/\ell_{\max}$ for all; `scheme='correct_only'`: $-\mu\,\ell_i/\ell_{\max}$ if correct, 0 if wrong.
  - `d2l.first_correct_position(text, gold, tokenizer) -> int | None`: token index of the first point in the response at which a stated answer equals `gold` (answers extracted from sentence-level segments with `d2l.extract_answer`); `None` if never.
  - `d2l.overthinking_stats(responses, golds, tokenizer) -> DataFrame`: per response `n_tokens`, `first_correct`, `tail_fraction = 1 - first_correct/n_tokens`, `answered_early` (first correct before the final answer), following the outcome-efficiency and process-efficiency ideas of the overthinking paper (definitions restated in the notebook).
  - `d2l.pareto_front(points, x='tokens', y='acc') -> DataFrame`: non-dominated points (lower $x$, higher $y$).
  - `d2l.match_budget(curve, target_tokens) -> dict`: the truncation budget $B$ (from a monotone sweep) whose mean tokens equals `target_tokens`, with the per-problem correctness at that $B$ for paired comparison.
  - `d2l.stratified_frontier(logs, strata, weights='equal') -> DataFrame`: accuracy and tokens per stratum and macro-averaged with equal stratum weights.
  - `d2l.amortized_breakeven(c_train_flops, tokens_saved_per_query, n_ne) -> float`: queries after which training pays back, $Q^\star=C_{\textrm{train}}/(2N_{\textrm{ne}}\,\Delta\ell)$ (D2).
  - `d2l.concise_prompt(problem) -> str`: the teacher-side context for the concision-conditioned self-distillation arm.
  - Data: `DATA_HUB['lm-len-penalty-q17b']`, `lm-shortest-sft-q17b`, `lm-concise-opsd-q17b` (T2 checkpoints), `lm-effic-logs-v1` (JSONL logs and eval tables).
- **Models and data:**
  - Baseline model: the 8.5 headline checkpoint (`DATA_HUB['lm-grpo-qwen3-1p7b-base']` from 8.5 at T2; T1: `lm-grpo-qwen3-0p6b` from 8.5), the `chain` family and, if 8.5 trained on GSM8K-style data, the same train stream.
  - Reference thinking models for the opening overthinking measurement (committed traces from 9.6/9.3, no new generation at T1): `Qwen/Qwen3-1.7B` in thinking mode and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (MIT).
  - Teacher-free data for shortest-of-$N$ SFT: samples of the baseline itself on `chain` train problems.
  - Evaluation: `chain` strata by `difficulty` $k$: easy $\{2,3,4\}$, medium $\{5,6\}$, harder $\{7,8\}$ (the 9.1 splits `test_unseen_templates` and `test_harder`), plus `openai/gsm8k` test (public, possibly seen; labeled).
- **Packages:** as in 9.2 and 9.3.

#### Narrative arc

1. **The question.** Opening cell on committed thinking traces (Qwen3-1.7B thinking and DeepSeek-R1-Distill-Qwen-1.5B) over `chain` problems grouped by difficulty: fraction of tokens that come *after* the first appearance of the correct answer (`tail_fraction`) and mean tokens per problem, by stratum. The symptom to measure: on easy problems most of a long trace is spent after the answer is already stated, and the ratio of tokens to difficulty is far from proportional. (Numbers are to be measured.)
2. **What "efficient" can mean.** Four ways to cut tokens at fixed model: truncate at inference (9.6), penalize length in RL, distill shorter traces (shortest-of-$N$ SFT), distill a concision-conditioned teacher into the student (on-policy self-distillation). A table `tab_lm_effic_methods` with what each changes (decode-time rule, policy, data, objective) and what each costs.
3. **The frontier as the object.** Accuracy vs mean tokens, per difficulty stratum and macro-averaged with fixed equal weights; dominance defined; the truncation frontier of the baseline (from `d2l.budget_force`) as the reference every trained variant must beat at matched tokens. Cell: build the baseline frontier and print dominated points.
4. **Length penalties (derivation D1).** The exchange rate, early-exit threshold, hard-problem abandonment, and why the Kimi reward gates the penalty by correctness.
5. **Training the variants.** Four arms (below); recipes in a table with the exact reward or loss; the controls: continue the 8.5 recipe unchanged for the same steps (training-time control) and truncate the baseline at matched tokens (inference-time control).
6. **Measuring honestly.** Stratified frontiers with intervals; non-attempt and truncation rates; the macro-average with fixed weights so dropping hard items cannot raise the score; paired comparison against the truncation baseline at matched tokens (`match_budget`).
7. **Is it worth training? (derivation D2).** Amortization of training cost against per-query savings; a plot over deployment volume.
8. **Headline experiment** (below).

#### Derivations

**D1. Length penalties: exchange rate, early exit, and abandoning hard problems.**

*Statement.* Let a policy choose a response length $\ell$ for a problem of difficulty $\tau$ with accuracy curve $a_\tau(\ell)$, concave increasing. Consider the reward $R_\mu=R-\mu\,\ell$ ($\mu$ in accuracy per token). (a) The maximizer of $\mathbb{E}[R_\mu]$ is the solution of $a_\tau'(\ell^\star)=\mu$ if $a_\tau'(0)>\mu$ and $\ell^\star=0$ otherwise. (b) It is the Lagrangian of $\max\ \mathbb{E}R$ s.t. $\mathbb{E}\ell\le B$; $\mu$ is the multiplier, i.e. the slope of the accuracy–tokens frontier at the chosen operating point, *the exchange rate between a point of accuracy and tokens*. (c) For $a_\tau(\ell)=1-e^{-\ell/\tau}$: $\ell^\star=\tau\ln\frac1{\mu\tau}$ when $\mu\tau<1$ and $\ell^\star=0$ when $\mu\tau\ge1$: easy problems are shortened, medium problems keep more tokens, and problems harder than $1/\mu$ are *abandoned* (zero reasoning, near-zero accuracy). (d) With the KL-regularized objective of `eq_kl_objective` the optimal policy is $\pi^\star(y)\propto\pi_{\textrm{ref}}(y)\exp\big((R(y)-\mu\ell(y))/\beta\big)$ (`eq_kl_optimum`): the penalty tilts the reference by a factor $e^{-\mu\ell/\beta}$ that is geometric in length. (e) Penalizing wrong answers as well as correct ones ($\textrm{scheme}=\texttt{linear}$) rewards short wrong answers over long wrong answers; the Kimi reward and the `correct_only` scheme remove that incentive on failures.

*Assumptions.* $a_\tau$ smooth, concave, $a_\tau(0)=0$; length continuous (the integer case changes nothing material); the policy can choose $\ell$ freely.

*Steps.*
1. (a) $\frac{d}{d\ell}[a_\tau(\ell)-\mu\ell]=a_\tau'(\ell)-\mu$ is decreasing; it vanishes at $\ell^\star$ or is negative at 0.
2. (b) Standard Lagrangian duality for a concave program (`sec_mdl-constrained-optimization-duality`): the multiplier of the budget constraint is the marginal value of tokens. Sweeping $\mu$ traces the frontier; its local slope is $-\mu$.
3. (c) $a_\tau'(\ell)=e^{-\ell/\tau}/\tau=\mu$ gives $\ell^\star=\tau\ln\frac1{\mu\tau}$; positive iff $\mu\tau<1$. Resulting accuracy $1-\mu\tau$ at $\ell^\star$. As $\tau\uparrow1/\mu$, $\ell^\star\downarrow0$ and the accuracy falls to $0$. Consequence: a global penalty gives each problem its own length, but the same $\mu$ makes the policy give up on the hardest problems; an average accuracy over *only the problems that are still attempted* would hide this, hence the fixed-weight stratified macro-average.
4. (d) Insert reward $R-\mu\ell$ into `eq_kl_optimum`.
5. (e) For a wrong answer $R=0$, so reward $=-\mu\ell$ under `linear`: among wrong responses the shortest scores best, so the gradient pushes toward early guessing on problems where $p$ is small: shortening a response of length $\ell$ to $\ell-\delta$ loses accuracy $a'(\ell)\delta$ and gains $\mu\delta$; early exit is optimal exactly when $a'(\ell)<\mu$, the same threshold as (a), which is fine for the exchange-rate view but *the policy learns it from noisy sampled rewards*, with the length of the *wrong* samples carrying the signal. Under `correct_only` or Kimi, length matters only through correct samples.
6. Reading: the penalty weight is a product decision (tokens per accuracy point), not a hyperparameter to tune for the score; report the frontier, not a point.

*Numerical check (cell `effic-penalty`, CPU, seconds).* For $\mu=0.004$ and $\tau\in\{10,30,100,300\}$, brute-force the integer $\ell\in[0,2000]$ maximizing $1-e^{-\ell/\tau}-\mu\ell$ and compare with $\tau\ln\frac1{\mu\tau}$: $32$ vs $32.2$, $64$ vs $63.6$, $92$ vs $91.6$, and $0$ vs $0$ for $\tau=300$ (author's check; the accuracies at the optimum are $0.959,0.882,0.601,0$). Assert $|\ell^\star_{\textrm{brute}}-\ell^\star_{\textrm{formula}}|\le1$ and that the $\tau=300$ problem is abandoned. A second check on a softmax policy over $\ell\in\{0,\dots,512\}$ (logits trained by exact gradient ascent on $a_\tau(\ell)-\mu\ell+\beta H(\pi)$ with $\beta=0.05$) converges to the tilted distribution $\propto\exp((a_\tau(\ell)-\mu\ell)/\beta)$ within total variation $10^{-4}$ (D1(d) with a uniform reference).

**D2. When does training a shorter model pay back? Amortization.**

*Statement.* Let the base model use mean $\ell_0$ tokens per query and the trained variant $\ell_1<\ell_0$, both with $N_{\textrm{ne}}$ non-embedding parameters; training the variant costs $C_{\textrm{train}}$ FLOPs (the ledger: rollouts plus updates plus any teacher). Ignoring attention FLOPs, the variant is cheaper in total after $Q^\star=C_{\textrm{train}}/\big(2N_{\textrm{ne}}(\ell_0-\ell_1)\big)$ queries. (b) If the variant also loses accuracy $\Delta a$, comparing at equal accuracy requires either extra samples ($\approx\Delta a/a'$ tokens by the exchange rate of D1(b)) or a different deployment value; the break-even then uses $\ell_1+\Delta a/\mu$ in place of $\ell_1$ where $\mu$ is the exchange rate. (c) For a *decode-bound* service, wall-clock savings are proportional to tokens saved only if the batch composition is unchanged; shorter responses also free KV cache and raise throughput (`sec_kv-cache`), a second-order gain noted but not modeled.

*Steps.* (1) Per-query FLOPs $2N_{\textrm{ne}}\ell$; savings per query $2N_{\textrm{ne}}(\ell_0-\ell_1)$; break-even by division. (2) Exchange-rate correction from D1(b). (3) Example with the notebook's own ledger numbers (filled at run time): $C_{\textrm{train}}$ from `lm-effic-logs-v1`, $\ell_0-\ell_1$ from the evaluation. *Check (cell `effic-breakeven`):* given a toy ledger ($C_{\textrm{train}}=10^{17}$ FLOPs, $N_{\textrm{ne}}=1.4\times10^9$, $\ell_0-\ell_1=300$) the function returns $Q^\star=1.19\times10^{5}$ queries (arithmetic: $10^{17}/(2\cdot1.4\times10^9\cdot300)=10^{17}/(8.4\times10^{11})$); verify against direct summation of per-query FLOPs; the illustration is labeled as arithmetic, not a measurement.

#### From-scratch implementation

Budget ≤ ~150 lines.

- `length_reward` (≈35 lines): vectorized by `group_ids`; guards for groups with identical lengths ($\ell_{\max}=\ell_{\min}$: reward 0.5 for all by convention, documented) and for groups with no correct sample (penalty applies as `min(0, λ)`). Tests: the paper's example (a three-sample group with lengths $(100,200,300)$ and mixed correctness gives rewards $(0.5,0,-0.5)$ for correct samples and $(0,0,-0.5)$ for wrong ones; computed from the formula above); permutation invariance within a group; `linear` equals $-\mu\ell/\ell_{\max}$.
- `first_correct_position` and `overthinking_stats` (≈40 lines): split on sentence and newline boundaries, extract the answer candidate with `d2l.extract_answer`, map the segment end to a token index via the tokenizer's offset mapping. Tests on 12 hand-labeled traces (first correct answer at known token offsets), including a trace whose answer appears only in the final line (`tail_fraction = 0`) and one that never states the right answer (`None`).
- `pareto_front`, `match_budget`, `stratified_frontier`, `amortized_breakeven` (≈45 lines). Tests: `pareto_front` on a hand-made table; `match_budget` returns a $B$ whose mean tokens differ from the target by $\le2\%$ on the committed baseline sweep; `stratified_frontier` with equal weights: removing the hardest half of a stratum's problems from the *answered* set and counting them as non-attempts (failures) leaves that stratum's accuracy unchanged or lower, never higher; a naive average over attempted items does go up, which the test also shows.
- The training arms reuse 8.5's trainer, 9.1's SFT/`kd_loss`, and 9.2's `OnPolicyDistiller`; new code is the reward closure (≈10 lines) and the shortest-of-$N$ selection (≈10 lines).

#### In production

- Hosted reasoning APIs expose an effort level or token cap; open stacks expose thinking budgets (9.6) and sampling-time length controls; the Kimi long-to-short recipes (shortest rejection sampling, DPO on short/long pairs, model merging, length-penalized RL) are the research reference; TRL's GRPO reward functions accept custom length rewards through `reward_funcs`, the notebook shows the same reward expressed as a TRL reward function and checks that it returns the same numbers as `d2l.length_reward` on a 64-sample batch (the TRL reward-function signature is to be verified in the pinned release).
- L1-style *length-controllable* models (a target length in the prompt) turn one trained model into a frontier at inference time; mentioned in one paragraph and offered as an optional arm.

#### Headline experiment

*Hypothesis.* (H1) Length-penalized RL and shortest-of-$N$ SFT each produce frontiers that dominate *truncating the baseline* at equal tokens on the easy and medium strata; (H2) a global penalty degrades the harder stratum disproportionately (D1(c)); (H3) after accounting for the training cost, the break-even volume $Q^\star$ is within reach of a realistic service only for the cheapest arm. None is assumed; the notebook reports measured outcomes.

*Setup.* Baseline $M_0$ = the 8.5 checkpoint. Arms (all start from $M_0$; same train stream of `chain` train problems; same maximum response length; seeds fixed):

| Arm | Update | Notes |
|---|---|---|
| $M_0^+$ (control) | 8.5's GRPO recipe, $S$ more steps, no length term | isolates the effect of extra training steps |
| $M_1$ Kimi-style | GRPO with `scheme='kimi'` added to the correctness reward, $S$ steps | group-relative length reward |
| $M_2$ shortest-of-$N$ | sample $N=8$ per train problem from $M_0$, keep the shortest correct, 1 epoch SFT | no teacher, no RL |
| $M_3$ concision self-distillation | `OnPolicyDistiller` with the teacher = the same weights under `concise_prompt`, $S$ steps | our concision-conditioned variant of 9.2's OPSD, motivated by CRISP (9.2 code) |
| $M_4$ (optional) length-conditioned | GRPO with a target-length prompt and reward $-|\ell-\ell_{\textrm{target}}|$ | L1-style; one model gives a frontier |
| truncation | $M_0$ with `budget_force`, $B\in\{64,\dots,1024\}$ | inference-time control from 9.6 |

$S$ is set from the 8.5 ledger so that one arm costs a fraction of 8.5's training FLOPs (the fraction is a config; value to be set after timing). Each trained arm is evaluated at three checkpoints during training so the *trajectory* of the frontier is visible.

*Procedure.* Evaluate every arm and checkpoint on 500 problems per stratum (easy, medium, harder) and on GSM8K test (public): accuracy (sampled at temperature 0.6, $n=4$ samples per problem to reduce evaluation noise), mean generated tokens, truncation and non-attempt rates, `tail_fraction` from `overthinking_stats`. Truncation arm: sweep $B$. Compute `d2l.match_budget` for each trained arm and paired accuracy differences against the truncated baseline at matched mean tokens. Ledger: training FLOPs per arm; `amortized_breakeven` relative to $M_0$ using the *measured* equal-accuracy token savings.

*Plots.* `fig_lm_effic_frontier`: accuracy (y) vs mean generated tokens (x, log), one panel per stratum plus a macro-averaged panel (equal weights); series: truncation sweep of $M_0$ (line), $M_0$, $M_0^+$ (points), $M_1$–$M_4$ (points with arrows along training checkpoints); bands: paired bootstrap on accuracy; dominated points drawn hollow (`pareto_front`). `fig_lm_effic_overthink`: `tail_fraction` vs difficulty for $M_0$ and each trained arm. `fig_lm_effic_breakeven`: total FLOPs (training + $Q$ queries) vs $Q$ (log–log) for $M_0$ and the arms at (approximately) equal accuracy; crossing points marked. `tab_lm_effic_summary`: per arm: macro accuracy, tokens, tokens saved vs $M_0$, matched-budget paired difference with interval, training FLOPs, $Q^\star$.

*Expected outcome.* Directions with sources. The overthinking paper reports that o1-like models spend many tokens on redundant solution rounds for easy problems and that simple self-training reduces tokens with maintained accuracy on easy benchmarks; Kimi k1.5 reports that long-to-short methods (shortest rejection sampling, length-penalized RL, merging) give token-efficient models, and that the length reward must be gated by correctness to avoid collapse; L1 reports precise length control with RL; the CRISP paper (compressed reasoning through iterative self-policy distillation) is the motivation for $M_3$; its exact formulation is to be read from the paper and the notebook lists where $M_3$ differs. Whether these effects appear for a 1.7B checkpoint trained on a templated family is to be measured; the baseline's responses may be short already, in which case the frontier gains are small and the notebook says so.

*Noise check.* Three seeds per trained arm at the T1 0.6B configuration and one seed at T2 (the part's single-run convention), with the T1 seed spread used as the yardstick: a T2 difference is called real only if it exceeds twice the T1 spread and the matched-budget paired interval excludes 0. Evaluation noise is reduced by $n=4$ samples per problem; intervals are over problems, clustered by template on the unseen split.

*T1 reduction.* `scale='t1'`: 0.6B checkpoint, LoRA, 30 steps for $M_1$ and $M_2$, evaluation on 100 problems per stratum (intervals wide, stated); every plot is rebuilt from the committed T2 logs.

*Artifacts committed for T2.* The checkpoints named above, `lm-effic-logs-v1`, frontier tables (JSON), the shortest-of-$N$ SFT set, the matched-budget tables.

#### Figures and tables

- `tab_lm_effic_methods`: methods, what each changes, cost, failure mode.
- `fig_lm_effic_overthink` (opening cell and per-arm), `fig_lm_effic_frontier`, `fig_lm_effic_breakeven`, `tab_lm_effic_summary` (headline).
- `fig_lm_effic_penalty_toy`: D1 toy: $\ell^\star(\tau)$ for several $\mu$ with the abandonment cliff at $\tau=1/\mu$ (cell `effic-penalty`).
- `fig_lm_effic_traj`: for each trained arm, mean length and accuracy vs training step (from logs).

#### Pitfalls and misconceptions

- Reporting accuracy only on attempted problems; a policy that abandons hard problems raises the "attempted" accuracy. Use fixed-weight strata and count non-attempts as failures.
- Penalizing length on wrong answers (shortest wrong wins) rewards guessing.
- Not controlling for extra training steps: part of any change comes from continuing to train.
- Comparing at different token budgets; always compare with the truncation frontier at matched tokens.
- Length measured in characters or words instead of tokens; the unit is tokens.
- Conflating *shorter reasoning* with *less reasoning when needed*: on harder strata a good variant keeps its tokens.
- Treating a single penalty weight as "the" efficient model; the deliverable is a frontier.
- Ignoring training cost when the savings are small or the deployment volume is low (D2).

#### Summary (draft bullets)

- Efficiency is a frontier of accuracy against tokens, per difficulty stratum, compared with truncating the same model at matched tokens.
- A length penalty has an exchange-rate interpretation; the same penalty abandons problems harder than $1/\mu$ tokens-per-accuracy, so strata must be fixed in advance.
- Penalties should apply to correct answers (or be group-relative and gated) so that short wrong answers are not rewarded.
- Shorter traces can come from RL, from shortest-of-$N$ SFT, or from on-policy self-distillation of a concise context; each has a different cost.
- Training a shorter model pays back after $C_{\textrm{train}}/(2N_{\textrm{ne}}\Delta\ell)$ queries, corrected for any accuracy loss.
- Measure overthinking directly: the fraction of tokens after the first correct answer.

#### Exercises

1. *(check)* Under D1(c) with $\mu=0.004$, which of $\tau\in\{50,200,260\}$ are abandoned? What is $\ell^\star$ for the others? *Sketch:* abandoned iff $\mu\tau\ge1\iff\tau\ge250$, so only $\tau=260$; $\ell^\star(50)=50\ln5=80.5$, $\ell^\star(200)=200\ln1.25=44.6$ (note the non-monotonicity: $\ell^\star$ rises then falls with $\tau$).
2. *(check)* Why is the Kimi length reward $\min(0,\lambda_i)$ for wrong answers rather than $\lambda_i$? *Sketch:* with $\lambda_i$ a short wrong answer would get a positive reward; $\min(0,\cdot)$ never rewards a wrong answer for being short and only penalizes long wrong ones.
3. *(derive)* Show that $\ell^\star(\tau)=\tau\ln\frac1{\mu\tau}$ is maximized at $\tau=\frac1{e\mu}$ and give the maximum. *Sketch:* derivative $\ln\frac1{\mu\tau}-1=0\Rightarrow\tau=\frac1{e\mu}$; $\ell^\star_{\max}=\frac1{e\mu}$.
4. *(derive)* Using D1(d) with a uniform reference over lengths, show that the expected length of the tilted policy is decreasing in $\mu$ and give the limit $\mu\to\infty$. *Sketch:* $\frac{d}{d\mu}\mathbb{E}_{\pi_\mu}[\ell]=-\mathrm{Var}_{\pi_\mu}(\ell)/\beta\le0$ (exponential-family identity); limit is the shortest length with positive reward.
5. *(measure)* Run $M_1$ with $\mu$ (the Kimi weighting) multiplied by $\{0.5,1,2\}$ and plot the three resulting frontier points per stratum; predict which stratum loses accuracy first and check against D1(c). Three seeds at T1.
6. *(measure)* Replace the shortest correct sample in $M_2$ by a *random* correct sample (same $N$) and by the median-length correct sample; predict the change in tokens and in `tail_fraction`, and measure with paired intervals.
7. *(open)* Design an efficiency evaluation for a *code* reasoning model (unit-test verified): define difficulty strata, a token unit, a truncation baseline, and a contamination-safe test set. No reference answer.

#### Further reading

- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187) (the "Overthinking" entry of v5): outcome and process efficiency, redundancy in o1-like traces.
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599): length penalty and long-to-short methods.
- [CRISP: Compressed Reasoning via Iterative Self-Policy Distillation](https://arxiv.org/abs/2603.05433) (v5; title re-checked here): self-distillation for shorter reasoning; the Codex plan titles it "On-Policy Self-Distillation for Reasoning Compression", which is not the arXiv title.
- [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://arxiv.org/abs/2503.04697) (new, verified): length-conditioned RL and the frontier of a single model.
- Muennighoff et al., [s1](https://arxiv.org/abs/2501.19393) (9.6): inference-time control as the baseline.
- Code and blogs: [Open R1](https://github.com/huggingface/open-r1) (GRPO with custom rewards); Thinking Machines, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (9.2); Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/).

#### Risks and open decisions

- *8.5's checkpoint may already produce short responses.* Then overthinking is small on the chosen family and the section's measured gains small; fallback: add a thinking-style baseline (the 9.3 SFT→RL checkpoint, which has long traces) as a second $M_0$ and report both. This changes the training cost and is flagged as an open decision for the editor.
- *Trainer API.* Batch-level reward function (assumption above); fallback closure.
- *Three seeds at T2 are expensive.* The T1 seed spread is the stated yardstick; if it exceeds the effect size the section reports "not resolved".
- *First-correct-position extraction.* Depends on the answer extractor and sentence segmentation; the hand-labeled test fixes the behavior on the `chain` format only. For free-form math traces, the notebook reports the extractor's agreement with a manual check on 20 traces.
- *CRISP reproduction detail.* The notebook implements a concision-conditioned teacher in the spirit of 9.2's OPSD; it is not claimed to reproduce CRISP, and the differences (to be read from the paper) are listed in the text.
- *Amortization example uses arithmetic, not data.* Labeled as such; the run-time cell uses the measured ledger.
