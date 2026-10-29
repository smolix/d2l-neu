# Chapter 8: Preference Optimization and Reinforcement Learning — Blueprint

**Label:** `chap_lm_rl` · **Arc:** IV. Post-training · **Sections:** 8.1 `sec_lm_reward_models` (T2), 8.2 `sec_lm_rlhf_ppo` (T2), 8.3 `sec_lm_dpo` (T2), 8.4 `sec_lm_po_family` (T1), 8.5 `sec_lm_rlvr` (T2), 8.6 `sec_lm_rl_systems` (T3), 8.7 ◇ `sec_lm_overoptimization` (T1) · **Tiers:** T1–T3 · **Assignment:** A7 (SFT → DPO → GRPO), shared with Chapter 7 as the producer of its input checkpoint.

## Chapter overview

### Draft introduction (for the author to polish)

Chapter 7 produced a model that follows instructions because it imitated demonstrations. Imitation cannot tell the model that one acceptable answer is better than another, nor that a fluent answer is wrong. This chapter replaces demonstrations with comparisons and with checks. We fit a reward model to human or AI preferences (8.1), optimize against it with PPO (8.2), remove the reward model with DPO (8.3), and survey the methods that change one assumption of DPO at a time (8.4). We then drop learned rewards entirely when the answer can be verified (8.5), and ask what a training system must look like when generating samples costs more than learning from them (8.6). The last section asks the question that every earlier one invites: when does the number that went up stop meaning that the model got better (8.7)? The theory is not new. Policy gradients, baselines, PPO, the KL-regularized optimum and the token MDP were taught in Advanced; here we pay the price of running them on a 0.6B-parameter language model, where the measured symptoms (length drift, collapsing entropy, stale samples, memory for four models) decide which variant of each method is usable.

### Learning objectives (measurable; refine v5's)

After the chapter a student can:
1. Train a reward model with the Bradley–Terry loss on a language-model backbone, and report held-out pairwise accuracy, expected calibration error, and the correlation of reward with length; construct perturbation tests that expose what it rewards. (8.1)
2. Implement per-token KL penalties, GAE with response masks and whitening, and a PPO-RLHF step with four resident models; verify advantages against hand-computed trajectories, and state which of k1, k2, k3 is unbiased for the KL value and which gradient each one estimates. (8.2)
3. Derive DPO from the KL-regularized optimum and the Bradley–Terry model, implement it, and verify it against a library loss; explain likelihood displacement with a worked example. (8.3)
4. Place IPO, KTO, SimPO, ORPO and online DPO by the single assumption each changes, and run an offline-versus-online comparison with a paired interval. (8.4)
5. Run GRPO with an exact verifier; show how the aggregation divisor (sample, token, constant) and the standard-deviation normalization change the optimized objective; ablate one choice and read accuracy, length and entropy together. (8.5)
6. Attribute the time of an RL step to generation, scoring, update and weight synchronization, predict the effect of placement and asynchrony with a timing model, and measure policy lag and effective sample size. (8.6)
7. (◇) Fit the Gao et al. overoptimization curve to a best-of-n sweep, and evaluate a policy trained on a biased reward with an independent checker. (8.7)

### Prerequisites

Book (Advanced, Mathematics) — what each is used for and never re-taught:

| Label | Used for | Not re-taught here |
|---|---|---|
| `sec_policygradient` | score function, REINFORCE, `eq_traj_prob`, `eq_softmax_score` | the log-derivative trick |
| `sec_baselines` | zero-mean lemma, leave-one-out (`subsec_leave_one_out`), the $(K-1)/K$ shrinkage of the group mean, divisor choices (`subsec_pg_loss_divisor`), `d2l.normalize` | any proof of baseline unbiasedness |
| `sec_actorcritic` | TD error, GAE (`eq_gae_deltas`), critic as sampled policy evaluation | the GAE telescoping identity |
| `sec_ppo` | ratio $\rho_t$, clip (`eq_ppo_clip`), asymmetric bands (`subsec_ppo_asymmetric`), length-normalized ratio, diagnostics | the surrogate and trust-region theory |
| `sec_regularized` | `eq_bradley_terry`, `eq_kl_objective`, `eq_kl_optimum`, identifiability, Goodhart curve (`subsec_kl_budget`), KL direction (`subsec_kl_direction`) | the closed-form optimum; the gridworld Goodhart experiment |
| `sec_rl_sequences` | token MDP, `eq_seq_factorization`, group baseline, `eq_dpo_reward`, hacking (`subsec_rl_sequences_hacking`), `tab_rl_notation_contract` | the factorization proposition |
| `sec_mdl-divergences-distances` | KL and f-divergences | |
| `sec_mdl-constrained-optimization-duality` | Lagrangian view of KL constraint vs penalty (8.3) | |
| `sec_mdl-bayesian-computation` | importance sampling, effective sample size (8.6) | |
| `sec_mdl-statistics` | bootstrap | |
| `sec_perf_model`, `sec_hardware`, `sec_multi_gpu` | roofline and collectives for 8.6 | |

This part: Chapter 1 (1.1 `d2l.load_hf`, 1.2 `d2l.generate`, 1.3 chat templates, 1.4 `d2l.pass_at_k`, `d2l.extract_answer`, 1.5 `d2l.EvalSuite`, `d2l.paired_bootstrap`, `d2l.judge_pairwise`); 5.1 `d2l.chip_specs()`, 6.1 `d2l.launch` and 6.2 (FSDP) for 8.6 only. Chapter 7: 7.2 (the SFT checkpoint and its chat serialization, completion-only masks), 7.4 (`d2l.LoRALinear`), 7.3 (data), 7.7 (TRL conventions).

### Notation (quoted verbatim from `tab_rl_notation_contract`)

The chapter uses these symbols without redefinition. The overview of the written chapter must reproduce the table as is.

:The notation contract. Symbols the post-training chapters use without redefinition.
:label:`tab_rl_notation_contract`

| symbol | meaning | built in |
|:--|:--|:--|
| $x$, $y$ | prompt, response | this section |
| $\tau$ | trajectory | :numref:`sec_mdp` |
| $\mu_0$ | start-state, i.e. prompt, distribution | :numref:`sec_mdp` |
| $\pi_\theta$, $\pi_{\textrm{ref}}$ | policy; frozen reference | :numref:`sec_policygradient`, :numref:`sec_regularized` |
| $\hat{G}_t$ | reward-to-go | :numref:`sec_baselines` |
| $A$ | advantage | :numref:`sec_valueiter`, :numref:`sec_baselines` |
| $K$ | group size | :numref:`sec_baselines` |
| $\rho_t$ | importance ratio $\pi_\theta / \pi_{\theta_{\textrm{old}}}$ | :numref:`sec_ppo` |
| $\epsilon$ | clip half-width, exploration rate; never a numerical constant | :numref:`sec_ppo`, :numref:`sec_qlearning` |
| $\beta$ | KL coefficient | :numref:`sec_regularized` |
| $\delta_t$ | TD error | :numref:`sec_qlearning` |
| $w$, $w^-$ | critic parameters; target copy | :numref:`sec_actorcritic`, :numref:`sec_dqn` |
| $D_{\textrm{KL}}(P \Vert Q)$, $\mathbf{1}(\cdot)$ | KL divergence; indicator | :numref:`sec_regularized` |

Additions the chapter must introduce explicitly and keep consistent (they are not in the contract, so each is defined once, in 8.1, in a short "chapter-local symbols" box, and never reused for anything else):
- $r_\phi(x,y)$: the learned scalar reward; $y_w, y_l$: preferred and rejected response of a pair; $\sigma$: the logistic function (as in `eq_bradley_terry`).
- $h_\theta(y)=\log\pi_\theta(y\mid x)-\log\pi_{\textrm{ref}}(y\mid x)$: the sequence log-ratio (as in the book's exercise on DPO). The *implicit reward* is $\beta h_\theta$.
- $|y|$: response length in tokens (EOS included). $L_{\max}$: the fixed constant divisor of 8.5.
- $\pi_b$: the behavior policy that produced the sample (differs from $\pi_{\theta_{\textrm{old}}}$ only in 8.6, where it is the inference engine).
- Do **not** use $w$ for anything but critic parameters (contract); the preferred response is $y_w$ only as a subscript, and the DPO weights are written $\lambda$.
- The clip half-width is always $\epsilon$ (asymmetric: $\epsilon_{\textrm{low}},\epsilon_{\textrm{high}}$); never write a numerical constant such as `1e-8` as $\epsilon$ (write `eps` in code, "a small constant" in prose).

### Section dependency graph

```
8.1 reward models ──► 8.2 RLHF/PPO ──────────┐
   │  (pairs, RM, token_logprobs toolkit)    │
   ├────────────► 8.3 DPO ──► 8.4 family     │
   │                  │                      ▼
   └────────────► 8.5 RLVR/GRPO ──► 8.6 systems
                      │                │
                      └──────► 8.7 ◇ overoptimization (uses 8.1 RM, 8.5 GRPO, 8.3 DPO)
```
Course paths in v5 take 8.1, 8.3, 8.5 without 8.2; therefore every shared utility (pair loader, `token_logprobs`, adapter toggling, response masks) is exported by 8.1, not by 8.2. 8.5 does not import from 8.2 or 8.3. 8.4 imports `d2l.dpo_loss` from 8.3. 8.6 imports the GRPO loop of 8.5. 8.7 imports 8.1's reward model, 8.3's DPO, 8.5's GRPO.

### Running examples and shared artifacts

Used:
- **R1** is not used directly; the chapter works on Hugging Face models (Qwen3-0.6B family) because RL needs `generate` speed and vLLM compatibility. Nothing in the chapter requires `d2l.GPT`.
- **π_SFT** (from 7.2): Qwen3-0.6B-Base after instruction tuning, chat template and completion-only masks as taught in 7.2. `DATA_HUB` key `lm-sft-qwen3-0p6b` (7.2 reference run) (a safetensors checkpoint plus tokenizer, `chat_template` included). The part's template is `d2l_chatml_v1` (7.2: ChatML, no `<think>` blocks, generation prompt exactly `<|im_start|>assistant\n`); the main path renders every prompt, pair and rollout with it. Fallback if absent: `Qwen/Qwen3-0.6B` (post-trained), which keeps *Qwen's own* shipped template with `enable_thinking=False` (that template renders an empty think block at the start of the assistant turn); a config flag `template='qwen3_nothink'` selects it, the completion-only mask and the `d2l.token_logprobs` shift are unchanged, and the notebook prints the first assistant tokens under both templates so the difference is visible. Never mix the two within one run (a DPO pair, a reference and a policy must be rendered by the same template).
- **7.4** `d2l.LoRALinear`, `d2l.apply_lora`, `d2l.lora_disabled`; **1.4** `d2l.pass_at_k`, `d2l.extract_answer`; **1.5** `d2l.EvalSuite`, `d2l.paired_bootstrap`, `d2l.judge_pairwise`. Return types used throughout: `d2l.generate` → `list[d2l.Generation]` (use `.text`); `paired_bootstrap` → `BootstrapResult` (`.mean`, `.ci`); `EvalSuite([...], n=..., seed=...).run(model, tok)` → `EvalResult` (use `.scores()`); `judge_pairwise` → `JudgeResult` (`.pref_a`, `.verdict`).

Produced (all committed under `DATA_HUB`, safetensors, LoRA adapters only unless stated):

| Artifact | Key (proposed) | Made in | Used in |
|---|---|---|---|
| Reward model on UltraFeedback pairs (backbone + head, ~0.6B) | `lm-rm-qwen3-0p6b-uf` | 8.1 | 8.2, 8.4, 8.7 |
| Reward model on HH-RLHF helpful-base (head+LoRA) | `lm-rm-qwen3-0p6b-hh` | 8.1 | 8.1 (noise study) |
| Tokenized pair set with masks (parquet) | `lm-pairs-uf-tok` | 8.1 | 8.3, 8.4 |
| PPO-RLHF LoRA adapter | `lm-ppo-qwen3-0p6b` | 8.2 | 8.2 |
| DPO adapter on π_SFT | `lm-dpo-qwen3-0p6b` | 8.3 | 8.4, A7 |
| Judged-preference tables (jsonl) | `lm-judgments-*` | 8.3, 8.4 | analysis cells |
| Arithmetic task generator spec and eval split (json) | `lm-arith-tasks` | 8.5 | 8.5–8.7, A7 |
| GRPO adapter, π_SFT + LoRA, arm A, seed 0 (T1/T2) | `lm-grpo-qwen3-0p6b` | 8.5 | 8.6, A7, 9.3, 9.7 |
| GRPO adapter, R1-Zero-like Qwen3-1.7B-Base replicate, arm A, seed 0 (T2) | `lm-grpo-qwen3-1p7b-base` | 8.5 | 9.3, 9.7 |
| RL systems step logs and timelines (jsonl) | `lm-rlsys-logs` | 8.6 (T3) | 8.6 analysis cells |
| Best-of-n gold/proxy tables, RL curves | `lm-overopt-logs` | 8.7 | 8.7 |

### Compute plan

Caps are budgets of the plan, not predictions; measured values replace them in the written chapter ("to be measured").

| § | Tier | Reduced T1 configuration | Cap (T2/T3) | Cap (T1) | Peak memory (T2) | Peak memory (T1) |
|---|---|---|---|---|---|---|
| 8.1 | T2 | LoRA r=16 reward model on π_SFT, ≤ 4k pairs, max 512 tokens | ≤ 1 h | ≤ 15 min | to be measured | ≤ 15 GiB (target) |
| 8.2 | T2 | LoRA actor, reference = adapters off, critic = LoRA+value head on the shared frozen backbone, frozen RM | ≤ 2 h | ≤ 15 min | four-role table (full, LoRA) to be measured; full roles are 2 × 9.5 GB (actor, critic at 16 B/param) + 2 × 1.2 GB (reference, reward) = 21.5 GB before activations, within one A100 40 GB | ≤ 15 GiB (target) |
| 8.3 | T2 | LoRA, ≤ 4k pairs, reference logprobs precomputed once | ≤ 1 h | ≤ 15 min | to be measured | ≤ 15 GiB |
| 8.4 | T1 | native | ≤ 15 min | ≤ 15 min | | |
| 8.5 | T2 | Qwen3-0.6B-Base/SFT, LoRA, ≤ 40 steps × 32 prompts × K=8, ≤ 256 new tokens | ≤ 3 h (one A100 40 GB) | ≤ 15 min | to be measured; 1.7B-Base replicate: LoRA, target ≤ 38 GiB | ≤ 15 GiB |
| 8.6 | T3 | Qwen3-0.6B on one Ada slot: colocated (sleep mode) vs two-process split sharing the GPU | ≤ 2 h on 8× A100 40 GB (one run per layout) | ≤ 15 min | per-GPU table to be measured; target ≤ 38 GiB per GPU | ≤ 15 GiB |
| 8.7 | T1 | BoN with n ≤ 64 on 200 prompts, 60-step RL | ≤ 15 min (T1 only) | ≤ 15 min | | ≤ 15 GiB |

Committed artifacts for T2+: the checkpoints and logs in the table above, plus one `config.json` per run (seed, revisions, commit of the notebook), so analysis cells never need a GPU.

Section budget summary: of a 15 GiB build-box slot, a frozen BF16 0.6B backbone is 596,049,920 parameters × 2 B = about 1.2 GB (Hugging Face API safetensors count for `Qwen/Qwen3-0.6B-Base`); everything else in T1 is activations, adapters, optimizer state for adapters, and the vLLM KV cache.

## Assignment

### A7. SFT → DPO → GRPO on Qwen3-0.6B (Chapters 7–8; T1–T2)

**Label:** `sec_lm_a7` (Chapter 8 owns this heading; part 1, the SFT stage, is `sec_lm_a7_sft` in Chapter 7 and is referenced with `:numref:`sec_lm_a7_sft``).

v5 attaches A7 to Chapters 7–8 and lists it under Chapter 8. This blueprint places the full specification here; Chapter 7's blueprint must only promise the SFT checkpoint (see "Consumption of Chapter 7" below).

**Learning goal.** Run the pipeline that most open post-training recipes share — SFT, then preference optimization, then RL on verifiable rewards — on one small model, implement the two losses yourself, and show with intervals that each stage improved what it was meant to improve without destroying the other.

**Consumption of Chapter 7's output.** The starter kit contains `a7_config.yaml` with:
- `sft_ckpt`: the `DATA_HUB` key `lm-sft-qwen3-0p6b` (the staff-provided 7.2 reference run), or a student's own 7.2 output directory (a safetensors checkpoint with its tokenizer; the chat template must be the one 7.2 prints).
- Preflight cell `check_sft()`: (i) loads the checkpoint with `d2l.load_hf`; (ii) renders a conversation with the tokenizer's chat template and asserts that the training prefix and the inference prompt are token-identical (the 7.2 invariant); (iii) computes held-out SFT loss on 200 conversations of the 7.2 data and asserts it is within the staff's recorded value ± a stated tolerance (set from the reference run, "to be measured"); (iv) generates 20 completions and asserts each ends with the end-of-turn token or is truncated at `max_new_tokens`. A student whose 7.2 checkpoint fails (iii) uses the reference checkpoint; nothing downstream depends on its quality, only on its format.
- The reference model for DPO and the KL term is this checkpoint, frozen.

**Tasks.**
1. *Baselines.* Evaluate π_SFT on the two evaluation sets (below) and write `baseline.json`.
2. *DPO.* Implement `dpo_loss` and `implicit_reward` (from the 8.3 contract). Train LoRA (`d2l.apply_lora(model, target_modules='all', r=16, alpha=32)`) on the first 4,000 pairs of `lm-pairs-uf-tok` (prompt-grouped, no prompt overlap with the evaluation prompts), 1 epoch, effective batch 32 pairs, β chosen by the student from {0.05, 0.1, 0.3}, reference logprobs precomputed once (under `d2l.lora_disabled`). Save `dpo_adapter`.
3. *GRPO.* Implement `group_advantages` and `ppo_loss` with its three aggregation modes (from the 8.2/8.5 contracts). Starting from π_DPO (merged adapter), train LoRA (fresh r=16) on `ArithmeticTasks` for a fixed budget of S=100 steps × 64 prompts × K=8 responses, `max_new_tokens=256` (T1: S=40, 32 prompts). Report which aggregation and which advantage normalization you used and why.
4. *Evaluate* both stages and the final model, with paired bootstrap intervals.
5. *Written analysis.*

**Evaluation sets (fixed by the starter kit).**
- *Preference win rate:* 300 prompts from `HuggingFaceH4/ultrafeedback_binarized` split `test_prefs` (first 300 by `prompt_id` after sorting), greedy decoding, `max_new_tokens=384`. Judge: `Qwen/Qwen3.5-9B` (bf16, text-only, non-thinking; the part's judge, loaded after the policies are freed, so evaluation claims a whole 24 GB GPU) through `d2l.judge_pairwise`, both orders; a pair counts as a win only if the judge prefers the candidate in both orders, a tie (0.5) if the two orders disagree. Independent check: `Skywork/Skywork-Reward-V2-Qwen3-1.7B` scores both responses; report agreement with the judge. Judge and trained models share no weights with the training reward (the 8.1 RM).
- *Verifiable accuracy:* (a) 500 held-out `ArithmeticTasks` instances (seed 2026, difficulty profile of the starter), (b) a 500-problem subset of `openai/gsm8k` test (fixed indices) with the 8.5 verifier. Report greedy accuracy and, with 8 samples, `d2l.pass_at_k` for k=1, 4.

**Deliverables.** `dpo.py` and `grpo.py` (or the notebook cells), `a7_report.md` with the tables below, three plots (8.3-style chosen/rejected logprobs; 8.5-style accuracy/length/entropy; a final bar chart of all metrics with intervals), the two adapters, and `baseline.json`, `stage_dpo.json`, `stage_grpo.json`.

**Autograder (hidden tests unless stated).**
1. *Unit tests, float64, CPU, ≤ 2 min:*
   - `dpo_loss` on 8 fixed random tensors equals a reference implementation to `atol=1e-10`; gradient with respect to each of the four log-probability inputs equals the analytic $\mp\beta\,\sigma(-\beta m)$ to `atol=1e-10`; at policy = reference the loss is $\log 2$ to `1e-12` and both implicit rewards are 0.
   - `group_advantages`: three fixtures (mixed group, all-equal group, K=2) to `atol=1e-6`; the all-equal group returns exactly zeros; `std=True` uses the population standard deviation (divide by K) — the test fixture makes the choice observable; `loo=True` returns $\frac{K}{K-1}(r_i-\mu)$.
   - `ppo_loss`: on a hand-built toy (3 responses of lengths 2, 3, 5; tabular logits) the three aggregation modes reproduce enumerated gradients to `atol=1e-8`; `clip` matches `eq_ppo_clip` on 6 ratio/advantage sign cases; masked (padding) tokens have exactly zero gradient.
   - `token_logprobs` / masks: a token-by-token fixture of 4 edge cases (empty response, response that is only EOS, truncated response, left-padded batch) must match.
2. *Integration (GPU, T1):* recomputed trainer log-probabilities for 32 sampled responses agree with the log-probabilities of the sampler within a tolerance set by the staff from the reference solution (3 × the reference run's mean absolute difference, "to be measured"); DPO loss decreases over the first 50 steps (slope of a linear fit < 0 with p<0.05 over steps).
3. *Outcome gates under the fixed budget:*
   - DPO: win rate of π_DPO against π_SFT has paired-bootstrap 95% lower bound > 0.5.
   - GRPO: held-out arithmetic accuracy of π_GRPO minus π_DPO has paired-bootstrap 95% lower bound > 0.
   - Retention: win rate of π_GRPO against π_SFT has lower bound ≥ 0.5 − δ, with δ stated in the starter config from the reference run. A solution that gains accuracy by shortening or truncating answers fails the response-length sanity check (mean length within ×[0.5, 2] of π_SFT's).
   - The thresholds are *floors set from the reference solution*; the staff run the reference solution five times with different seeds before the term and publish the floors at half the reference gain, so that noise cannot fail a correct implementation.

**Written analysis (answers graded by rubric, ≤ 2 pages).**
1. Using your plot of chosen and rejected log-probabilities, did the chosen response become more or less likely during DPO? Relate it to the win rate, and name the mechanism of 8.3 you think explains it.
2. Which aggregation divisor and normalization did you use in GRPO? Report the response-length trend and say what the objective-bias derivation of 8.5 predicts for it.
3. What fraction of GRPO groups had zero reward variance at the start and at the end of training? What does that imply for the useful sample count of your budget?
4. Give one example from your DPO outputs where the judge and the independent reward model disagree and say which you trust and why.
5. Where would you spend the next unit of compute: more pairs, more GRPO steps, or a better verifier?

**T1 path.** Everything above uses LoRA and the reference-offload tricks of 8.2 and 8.3; the numbers marked T1 apply (4,000 pairs; 40 GRPO steps × 32 prompts × K=8). The readers' 24 GB GPU is sufficient; the reference solution fits a 15 GiB build slot with sequence lengths capped at 512 tokens. **Estimated student hours:** about 15–20, dominated by debugging masks and by waiting for generation.

## Teaching sources

Keep all of v5's, with what to take from each:
- Nathan Lambert, [RLHF Book](https://rlhfbook.com/): the sequence of topics and the vocabulary of implicit rewards, rejection sampling, and "direct alignment algorithms"; read the chapters on reward models, policy gradients and regularization alongside 8.1–8.5.
- HF, [Illustrating RLHF](https://huggingface.co/blog/rlhf): the four-model picture that 8.2 turns into a memory table.
- John Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html): the k1/k2/k3 estimators; the derivation of 8.2 extends it to gradients.
- Lilian Weng, [Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/): the catalogue of failure modes for 8.1 and 8.7.
- [Stanford CS336](https://cs336.stanford.edu/) (alignment assignment): the shape of a graded SFT/DPO/GRPO assignment; A7 differs by using one 0.6B model and intervals.
- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/): policy-gradient background that the book chapters already cover; cite for lectures on off-policy correction (8.6).
- New: HF [TRL](https://github.com/huggingface/trl) documentation pages for `RewardTrainer`, `PPOTrainer`, `DPOTrainer`, `GRPOTrainer` as the library oracle in every "In production" block (pin the release; the PyPI latest at blueprint time is 1.16.0).

## Sections

### 8.1 Reward models for language

- **Label:** `sec_lm_reward_models` · **Tier:** T2 (reduced T1 config: LoRA r=16 on π_SFT, ≤ 4,000 pairs, max 512 tokens total length, micro-batch 4 with gradient accumulation) · **Budget:** T2 ≤ 1 h wall-clock for 3 reward models (full fine-tune, 1 epoch of the filtered pairs), peak memory to be measured (full fine-tune with AdamW: BF16 weights + FP32 master + two FP32 moments is about 16 B per parameter by the memory accounting of `sec_memory_precision`, i.e. ~9.5 GB for 596M parameters before activations); T1 ≤ 15 min, ≤ 15 GiB · **Codex units:** 9.1
- **Builds on (book):** `sec_regularized` (`eq_bradley_terry`, identifiability and the per-prompt baseline, Dense versus Terminal Rewards, the fitted-reward Goodhart experiment): reused for the loss and for the statement that a reward is identified only up to $f(x)$; **not re-taught:** the BT model, the gridworld Goodhart curve. `sec_rl_sequences` ("Learned Rewards from Preferences"): the terminal-reward reading of $r_\phi(x,y)$. `sec_mdl-statistics`: bootstrap intervals for accuracy. `sec_softmax`: calibration (via 1.4).
- **Builds on (this part):** 7.2 (π_SFT, chat serialization and the response-mask idiom), 7.4 (`d2l.LoRALinear`), 1.1 (`d2l.load_hf`), 1.3 (chat templates), 1.5 (`d2l.paired_bootstrap`).
- **Exports (`#@save`, all new, all consumed by later sections):**
  - `d2l.PreferenceData(name, split, tokenizer, max_len=512, n=None, seed=0)` → iterable of dicts `{prompt_ids, chosen_ids, rejected_ids, chosen_mask, rejected_mask, prompt_id}` where masks mark response tokens (EOS/end-of-turn included, prompt excluded). Sources: `'uf'` (UltraFeedback-binarized), `'hh'` (HH-RLHF helpful-base). Drops pairs longer than `max_len` and reports the fraction dropped.
  - `d2l.collate_pairs(batch, pad_id, side='right')` → padded tensors for both responses.
  - `d2l.RewardModel(backbone, hidden_size)` : `forward(input_ids, attention_mask) -> (B,)` reward at the last non-pad token; `RewardModel.from_pretrained(path)`.
  - `d2l.bt_loss(r_w, r_l, margin=0.0, center_coef=0.0)` → scalar mean of $-\log\sigma(r_w-r_l-m)+\lambda(r_w+r_l)^2$ and the pair accuracy.
  - `d2l.reliability(p, y, n_bins=15)` → `(bin_conf, bin_acc, bin_count, ece)`.
  - `d2l.token_logprobs(model, input_ids, attention_mask, response_mask=None, temperature=1.0)` → `(B, T-1)` log-probabilities of tokens `input_ids[:, 1:]`, zeroed outside `response_mask[:, 1:]`; logits divided by `temperature` before the softmax; computed in chunks over the vocabulary dimension to avoid a `(B,T,V)` FP32 tensor for V=151,936.
  - `d2l.sequence_logprob(logps)` → sum over tokens (the $\log\pi_\theta(y\mid x)$ of `eq_seq_factorization`).
  - Data artifacts: `lm-pairs-uf-tok`, `lm-rm-qwen3-0p6b-uf`, `lm-rm-qwen3-0p6b-hh` (above).
- **Imported from 7.4 (not defined here):** `d2l.apply_lora` (alias of `d2l.add_lora`), signature `(model, target_modules='all', r=8, alpha=16.0, dropout=0.0, rslora=False, exclude=('lm_head',), adapter_dtype=torch.float32) -> list[str]`, which wraps the matching `nn.Linear` modules in `d2l.LoRALinear` in place and freezes the rest; this chapter calls it as `d2l.apply_lora(model, target_modules='all', r=16, alpha=32)`. `d2l.lora_disabled(model)`, a context manager that turns every adapter off (via `LoRALinear.enabled = False`), so the same network gives $\pi_{\textrm{ref}}$ with no second copy of the model. `d2l.LoRALinear` also offers `merge()/unmerge()`.
- **Models and data:**
  - Backbone/policy init: π_SFT (7.2) — `Qwen/Qwen3-0.6B-Base` after SFT, 596,049,920 parameters, Apache-2.0.
  - Preference data: `HuggingFaceH4/ultrafeedback_binarized`, splits `train_prefs` (61,135), `test_prefs` (2,000), MIT; prompts and responses from the UltraFeedback pool scored by GPT-4 (AI labels). `Anthropic/hh-rlhf`, subset `helpful-base` (human labels, MIT; train/test files `helpful-base/{train,test}.jsonl.gz`).
  - Evaluation: `allenai/reward-bench` (ODC-By), per-category accuracy; reference RM `Skywork/Skywork-Reward-V2-Qwen3-0.6B` (Apache-2.0, 596M) and `-1.7B` (the independent "gold" in 8.7).
- **Packages:** `transformers>=4.5x` (Qwen3ForSequenceClassification used only as the test oracle; pin the version printed by 1.1), `peft`, `trl` (RewardTrainer oracle), `datasets`, `matplotlib`.

#### Narrative arc

1. **The question.** *A reward model that agrees with the labels 70% of the time: what is it agreeing with?* Opening cell: load 2,000 `test_prefs` pairs, print (i) the fraction of pairs whose chosen response is longer, (ii) the accuracy of the one-feature predictor "prefer the longer response", and (iii) the same two numbers on HH-RLHF helpful-base. The symptom is a length-only predictor that is far above 50% on one data set and near chance on the other (direction expected from Singhal et al.; actual values to be measured).
2. **Preference data: where the labels come from** (≈ 25 lines of prose, two cells).
   - Pairs, prompts, and who labeled them: HH-RLHF (crowd workers choosing between two Anthropic-model replies in a conversation) versus UltraFeedback (GPT-4 scores per response; chosen = highest-scored, rejected = a random lower one, per the dataset card — the author must re-read the card and quote its recipe exactly).
   - Annotation instructions as part of the data: "helpful" vs "harmless" criteria; ties; noise. Print five pairs per source with the chosen/rejected highlighted.
   - Splits: held-out *prompts* (no prompt in both splits; check `prompt_id`), plus an out-of-distribution split (HH helpful-base test, RewardBench).
   - Cell: `PreferenceData` statistics table (`tab_lm_rm_data`): pairs, mean response length (tokens), fraction dropped by `max_len`, fraction chosen-longer.
3. **A reward head on a language-model backbone.**
   - Architecture: backbone → hidden state at the last non-pad token (the end-of-turn token) → `Linear(hidden, 1, bias=False)`; initialization with std 1/√hidden·small so initial rewards ≈ 0; the reward is read at the end token because causal attention makes only that position see the whole response. Padding side matters (cell shows left vs right padding give identical rewards only when the index of the last token is computed from the mask).
   - From scratch (`RewardModel`) vs Hugging Face `AutoModelForSequenceClassification` (`num_labels=1`): the weights of the two heads are copied and the outputs compared (test below).
   - Initialize from π_SFT (not from the base model): the reward model shares the policy's tokenizer and template; this is the standard choice (InstructGPT, Llama 2) and is what lets 8.2 share a backbone.
4. **The loss and the training loop.**
   - `eq_lm_bt_loss`: $\mathcal L(\phi)=-\mathbb E\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$ (`eq_bradley_terry` applied to responses; derivation block below gives its gradient, its invariance, its behaviour under label noise).
   - Training choices to show and justify: one epoch (a cell plots training vs held-out loss for 3 epochs: held-out accuracy typically stops improving after the first epoch — direction to be measured), learning rate roughly an order of magnitude below SFT (grid {5e-6, 1e-5, 2e-5} full; {1e-4, 2e-4} LoRA), no dropout, centering regularizer $\lambda(r_w+r_l)^2$ with $\lambda\in\{0,10^{-3}\}$ and the Llama-2 margin $m$ (Touvron et al.) as a pair-dependent input (set from UltraFeedback score gap `score_chosen-score_rejected`, scaled so that the largest margin is 1).
   - Full fine-tune on T2; LoRA on T1; both through the same `bt_loss`.
5. **Measuring a reward model: accuracy.**
   - Pairwise accuracy with a bootstrap interval over pairs (binomial standard error as a sanity check); report in-distribution (UF test), out-of-distribution (HH test) and RewardBench per category (chat, chat-hard, safety, reasoning).
   - Against references: `Skywork-Reward-V2-Qwen3-0.6B` (same size, trained on a much larger curated preference set; quote its size from the report only after reading it): the table `tab_lm_rm_accuracy` frames what a 4k–60k-pair RM can and cannot do.
6. **Calibration.** Reliability diagram of $\sigma(r_w-r_l)$ against the label; ECE with 15 bins; one-parameter temperature scaling fit on a held-out slice; why the *absolute* reward scale is arbitrary but the *difference* is meaningful (identifiability, `sec_regularized`) and why calibration to noisy labels is not calibration to the truth (derivation D3 below).
7. **Length and style bias.**
   - Spearman correlation of reward with response length, over all responses and within-prompt (rank prompts' responses by length first to remove prompt effects).
   - Length-stratified accuracy: bin pairs by $\log(|y_w|/|y_l|)$ and plot accuracy per bin; a length-only baseline line; "RM-prefers-longer rate" vs "labels-prefer-longer rate".
   - A cheap length-debiasing experiment: add the regularizer "penalize the covariance of reward with length" or train on length-matched pairs (|log ratio|<0.1); report accuracy and Spearman before/after. Pointer: Singhal et al.
8. **Probing failures with perturbation tests** (`tab_lm_rm_perturb`). Apply a label-preserving or label-reversing edit to the *rejected* or *chosen* response and measure the reward change with paired bootstrap CIs and the *flip rate* (fraction of pairs whose predicted order flips):
   - append a neutral filler paragraph (length without content);
   - reformat into bullets / add markdown headers (style);
   - prepend "Certainly! Great question." (sycophantic opener);
   - truncate the last sentence (completeness);
   - replace a number or named entity with a wrong one (factuality; applied to chosen in the 300 pairs with a numeric answer);
   - for a user message that states a false belief, answer agreeing vs correcting (sycophancy; 40 handwritten items, listed in `data/rm_probe.jsonl` — author writes them, each with a one-line rationale).
9. **Accuracy is not usefulness** (the optimization target). Save RM checkpoints at 25%, 50%, 100% and 3 epochs of training; for each, run best-of-8 on 200 held-out prompts with samples from π_SFT and score the selected responses with the independent gold RM (`Skywork-Reward-V2-Qwen3-1.7B`); plot gold reward of the selected response against the proxy's held-out accuracy (`fig_lm_rm_acc_vs_bon`). Message: accuracy ranks reward models, it does not say how much optimization they survive — the lead-in to 8.7.

#### Derivations

**D1. The Bradley–Terry reward-model loss: gradient, invariance, and weights.** *Statement.* For pairs $(x,y_w,y_l)$ with $\Delta_\phi=r_\phi(x,y_w)-r_\phi(x,y_l)$ and loss $\ell=-\log\sigma(\Delta_\phi)$: (a) $\ell$ is the exact NLL of `eq_bradley_terry` when the label is "$y_w$ preferred"; (b) $\nabla_\phi\ell=-\big(1-\sigma(\Delta_\phi)\big)\big(\nabla_\phi r_\phi(x,y_w)-\nabla_\phi r_\phi(x,y_l)\big)$; (c) $\ell$ is invariant under $r_\phi(x,\cdot)\mapsto r_\phi(x,\cdot)+f(x)$; (d) $\ell$ is convex in $\Delta$ and, for a linear reward $r_\phi=\phi^\top g(x,y)$, convex in $\phi$.
*Assumptions.* Labels are independent given $(x,y_w,y_l)$; one comparison per pair.
*Steps.*
1. Under `eq_bradley_terry` the probability of the observed label is $\sigma(\Delta_\phi)$; take $-\log$.
2. Differentiate: $\frac{d}{d\Delta}[-\log\sigma(\Delta)]=-(1-\sigma(\Delta))=-\sigma(-\Delta)$ using $\sigma'=\sigma(1-\sigma)$; chain rule through $\Delta_\phi$ gives (b).
3. Read (b): a pair contributes in proportion to $\sigma(-\Delta)$, the model's current probability of the *wrong* order — correctly ordered pairs with large margins stop contributing; this is the same weight that reappears in DPO (8.3) with $\Delta=\beta(h_w-h_l)$.
4. (c): $\Delta_\phi$ contains $f(x)$ twice with opposite signs.
5. Because $\Delta$ is unchanged by (c), a centering penalty $\lambda\,\mathbb E[(r_w+r_l)^2]$ selects the member of the equivalence class with zero mean pair reward and does not change $\Delta$ (its gradient pushes $r_w+r_l\to0$ only).
6. (d): $\log(1+e^{-\Delta})$ has second derivative $\sigma(\Delta)\sigma(-\Delta)\ge0$; composition with a linear map keeps convexity.
*Numerical check (cell, float64, seed 0).* Random linear reward on 5 pairs: autograd gradient equals formula (b) with `atol=1e-10`; adding a random $f(x)$ changes the loss by < 1e-12; the Hessian of $\ell$ in $\Delta$ equals $\sigma(\Delta)\sigma(-\Delta)$ at 20 random $\Delta$ (finite differences, `atol=1e-6`).

**D2. Pair weights as hard-example mining, and the margin variant.** *Statement.* With margin $m$, $\ell_m=-\log\sigma(\Delta-m)$ has weight $\sigma(m-\Delta)$ and a minimizer for a deterministic pair at $\Delta=\infty$ (no finite optimum); with label noise (D3) the finite optimum appears. *Steps.* Same as D1 with $\Delta\to\Delta-m$; the Llama 2 use of $m$ as a function of the annotator's rating gap emphasizes pairs the annotators found clear. *Check.* Cell plots weight vs margin for $m\in\{0,0.5,1\}$ and verifies the formula by autograd.

**D3. Label noise shrinks the learned reward, and fixes an accuracy ceiling.** *Statement.* Suppose each pair has a true logit gap $\Delta^\ast$ (the BT model holds exactly) and the *observed* label is flipped with probability $\varepsilon<1/2$, independently of the pair. Let the model class be rich (one free logit per pair type). Then (a) the population-optimal model has $\sigma(\hat\Delta)=(1-2\varepsilon)\sigma(\Delta^\ast)+\varepsilon$, i.e. $|\hat\Delta|<|\Delta^\ast|$ with the same sign; (b) the maximum achievable agreement with the *noisy* labels is $\varepsilon+(1-2\varepsilon)\,\mathbb E[\sigma(|\Delta^\ast|)]$ — so accuracy on noisy test labels is bounded below 1 even for the true reward; (c) predicted probabilities $\sigma(\hat\Delta)$ are calibrated to the noisy labels but under-confident with respect to the true ones.
*Steps.*
1. Probability of observing "$y_w$ preferred" for a pair of type $\Delta^\ast$ is $p=(1-\varepsilon)\sigma(\Delta^\ast)+\varepsilon\sigma(-\Delta^\ast)=\varepsilon+(1-2\varepsilon)\sigma(\Delta^\ast)$ (using $\sigma(-u)=1-\sigma(u)$).
2. The expected log-likelihood per type is $p\log\sigma(\Delta)+(1-p)\log\sigma(-\Delta)$, a Bernoulli cross-entropy in $\sigma(\Delta)$; Gibbs' inequality gives the maximizer $\sigma(\hat\Delta)=p$. This proves (a); monotonicity in $\Delta^\ast$ gives the same sign.
3. (b): a Bayes-optimal predictor picks the more likely noisy label, which agrees with probability $\max(p,1-p)$; average over types with $\sigma(\Delta^\ast)\ge1/2$ (take $\Delta^\ast\ge0$ w.l.o.g.): $\max(p,1-p)=p=\varepsilon+(1-2\varepsilon)\sigma(|\Delta^\ast|)$.
4. (c): by construction $\sigma(\hat\Delta)=p$ equals the empirical frequency of the noisy label, which is exactly calibration.
*Numerical check (cell, numpy, seed 0).* Pair types $\Delta^\ast\in\{-2,-1,0,1,2\}$, $\varepsilon=0.2$, $N=2\cdot10^5$ draws per type; fit one logit per type by Newton's method; assert $|\sigma(\hat\Delta)-p|<3\sqrt{p(1-p)/N}$ for each type, and assert that the empirical agreement of the fitted model equals formula (b) within 3 standard errors. (Exercise D-derive 2 extends it to the "noisy" HH labels.)

#### From-scratch implementation

Components and budget (≤ ~140 lines total):
- `PreferenceData`, `collate_pairs` (≈ 40 lines): applies the chat template of 7.2 to `(prompt messages, response)`, builds response masks from template offsets (not by string search), asserts that the concatenation of prompt ids and response ids equals the ids of the templated full conversation.
- `RewardModel` (≈ 25 lines), `bt_loss` (≈ 12), `reliability` (≈ 12).
- `token_logprobs` (≈ 25 lines, chunked over vocabulary using `torch.logsumexp` on slices of the output hidden states times the unembedding rows; falls back to full logits when V·B·T is small), `sequence_logprob` (≈ 20 lines; `apply_lora` and `lora_disabled` are imported from 7.4).
- Training loop (≈ 25 lines; AdamW, cosine schedule with warmup from `sec_scheduler`, gradient accumulation).

Tests (each a cell that prints PASS):
1. `bt_loss(torch.zeros(8), torch.zeros(8))` equals $\log2$ to 1e-6 (fp32); accuracy 0.5 on ties is defined (document the convention: strict `>`).
2. D1 gradient identity, as above (fp64).
3. Head equivalence: for 4 sequences, copy the `RewardModel` head into `AutoModelForSequenceClassification(num_labels=1)` (same backbone weights) and compare; tolerance: 2× the BF16 difference observed between running the same sequences alone vs in a padded batch (to be measured; document it, do not hard-code).
4. `token_logprobs` vs `-F.cross_entropy(logits[:, :-1].float().transpose(1,2), ids[:, 1:], reduction='none')` on 4 sequences incl. a left-padded one: `atol=1e-3` in BF16 inputs, `1e-5` when the model is cast to FP32 (seed 0; model `Qwen/Qwen3-0.6B-Base`).
5. Mask invariants on the 4 edge cases of the A7 autograder (empty response, EOS-only, truncated, left-padded).
6. `lora_disabled` (7.4) returns bitwise the base model's log-probs (`torch.equal` after merging check: with all B matrices zero the adapter model equals the base to 0.0).

#### In production

- TRL `RewardTrainer` (dataset columns `chosen`/`rejected` in conversational format; `center_rewards_coefficient`; `max_length`): run the same 4,000 pairs for one epoch and compare the first 20 losses and the final held-out accuracy with the from-scratch trainer (the loss curves match within the noise of seeds; do not claim identity).
- `AutoModelForSequenceClassification` as the serving form of the RM; RewardBench's `rewardbench` evaluation script for the OOD table (verify its CLI at build time).
- The from-scratch `bt_loss`, `RewardModel` and `token_logprobs` are the oracle for 8.2/8.3 pipelines.

#### Headline experiment

- **Hypothesis.** A reward model trained for one epoch on UltraFeedback pairs beats the length-only baseline on UF test pairs, transfers poorly to HH helpful-base (direction: below the in-distribution score), and shows a positive within-prompt rank correlation of reward with length that is stronger than the human labels' own length preference.
- **Setup.** π_SFT backbone; T2: full fine-tune, lr 1e-5 (grid above on a 2k-pair dev slice), batch 64 pairs, 1 epoch of the length-filtered 61k pairs (T1: LoRA r=16, lr 2e-4, 4,000 pairs); bf16 autocast; seeds {0,1,2}.
- **Procedure.** Train; evaluate on UF test (2,000), HH test (helpful-base, first 2,000), RewardBench; compute calibration, length and perturbation metrics; train the HH RM (head + LoRA, 1 epoch) for the noise study; run the BoN-gold study of step 9.
- **Metrics.** accuracy with 95% bootstrap CI; ECE; Spearman (global and within-prompt); flip rate per perturbation; gold reward of BoN selection.
- **Plots.** `fig_lm_rm_calibration` (reliability diagram, two data sets); `fig_lm_rm_length` (accuracy by length-ratio bin with length-only baseline); `fig_lm_rm_perturb` (bars: mean Δreward with CI per perturbation); `fig_lm_rm_acc_vs_bon`.
- **Expected outcome.** Directions as in the hypothesis; magnitudes: "to be measured". Singhal et al. (A Long Way to Go) report that length explains a large share of RLHF reward improvement; Skywork-Reward-V2 and RewardBench 2 give reference scores for much stronger RMs — compare only the ordering of categories.
- **Noise check.** Three seeds; report per-seed accuracy and the paired bootstrap of (RM_a − RM_b) on shared test pairs (`d2l.paired_bootstrap`); only differences whose interval excludes 0 are discussed. Perturbation effects use paired bootstrap over pairs.
- **T1 reduction.** LoRA, 4,000 train pairs, 500 test pairs per set, 1 seed for the grid and 3 seeds for the headline.
- **Artifacts (T2).** `lm-rm-qwen3-0p6b-uf`, `lm-rm-qwen3-0p6b-hh`, per-pair scores (`rm_scores.parquet`: pair id, r_w, r_l, lengths), probe results.

#### Figures and tables

- `tab_lm_rm_data` — data sources, sizes, mean length, chosen-longer rate (cell: statistics of `PreferenceData`).
- `fig_lm_rm_calibration`, `fig_lm_rm_length`, `fig_lm_rm_perturb`, `fig_lm_rm_acc_vs_bon` — specified above.
- `tab_lm_rm_accuracy` — rows: length-only baseline, RM (UF), RM (HH), Skywork-V2-0.6B; columns: UF test, HH test, RewardBench categories.
- `tab_lm_rm_perturb` — perturbation, Δreward mean ± CI, flip rate.

#### Pitfalls and misconceptions

- Reading the reward at the last *position* rather than the last *non-pad* position under left padding (silently wrong rewards).
- Accuracy ties counted as correct (inflates by up to the tie rate); the convention must be fixed and printed.
- Treating reward magnitudes across prompts as comparable: only within-prompt differences are identified.
- Believing 70% accuracy means 70% usefulness: label noise caps accuracy (D3) and says nothing about off-distribution behaviour.
- Splitting by pair rather than by prompt when prompts repeat (true for HH multi-turn contexts): leaks near-duplicates.
- Training several epochs "to converge": reward models overfit quickly.

#### Summary (draft bullets)

- A reward model is a language model with a scalar head, trained with the pairwise Bradley–Terry loss on comparisons from the same prompt.
- The loss weights each pair by the probability of the wrong order, is invariant to per-prompt shifts, and under label noise learns shrunken but calibrated-to-the-labels scores.
- Held-out accuracy is bounded by label noise and says little about behaviour under optimization.
- Length and style are the first things a small reward model learns; perturbation tests make this visible.
- The reward model is an instrument to be tested before it is an objective.

#### Exercises

1. **(check)** Under left padding, a batch contains sequences of lengths 5, 3 and 7. Which index must the head read for each, and what changes under right padding? *Sketch:* last non-pad index per row computed from the attention mask (`mask.sum(1)-1` only for right padding; with left padding it is `T-1`); using `T-1` under right padding reads a pad token.
2. **(check)** A reward model assigns rewards (2.1, 1.7) and (−0.5, −0.9) to the two pairs (chosen, rejected). Compute the BT loss of each pair and the gradient weights $\sigma(-\Delta)$. Which pair does the model "care about"? *Sketch:* $\Delta=0.4$ for both; both have the same loss $0.513$ and the same weight $0.401$ — the loss depends only on differences (invariance D1c).
3. **(derive)** Extend D3 to *prompt-dependent* noise $\varepsilon(x)$ and show that the learned $\hat\Delta(x)$ differs across prompts by the factor in (a); what does this do to a ranking of *prompts* by reward margin? *Sketch:* $\sigma(\hat\Delta(x))=\varepsilon(x)+(1-2\varepsilon(x))\sigma(\Delta^\ast)$; margins are compressed more on noisier prompts, so comparing margins across prompts confounds signal and noise.
4. **(derive)** Show that for a Gaussian reward gap $\Delta^\ast\sim\mathcal N(0,s^2)$ the ceiling (b) is $\varepsilon+(1-2\varepsilon)\mathbb E\,\sigma(|\Delta^\ast|)$ and give its large-$s$ limit. *Sketch:* $\sigma(|\Delta|)\to1$ as $s\to\infty$, so the ceiling $\to1-\varepsilon$; for $s\to0$ it tends to $1/2$.
5. **(measure)** Predict, then measure, how in-distribution accuracy and the length Spearman correlation change if you train on length-matched pairs only. State your prediction first; run `d2l.paired_bootstrap` against the unmatched RM.
6. **(measure)** Train with and without the centering penalty and compare the standard deviation of rewards on 1,000 π_SFT samples and the BoN-selected gold reward. Predict whether reward scale matters for BoN (it should not: BoN is rank-based) and for PPO (it does).
7. **(open)** Design a perturbation test for a bias you suspect (e.g., a language, a formatting style, a refusal phrasing), build 50 items, and show whether your reward model, and Skywork-V2, respond to it. No reference answer.

#### Further reading

- InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)): the RM-then-PPO pipeline and the one-epoch finding for reward models.
- RewardBench ([arXiv:2403.13787](https://arxiv.org/abs/2403.13787)), RewardBench 2 ([arXiv:2506.01937](https://arxiv.org/abs/2506.01937)): how to evaluate an RM by category.
- Anthropic [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf): human-labeled pairs and the limits to discuss before training.
- New: Learning to summarize from human feedback ([arXiv:2009.01325](https://arxiv.org/abs/2009.01325)) — the first RM+PPO-on-LMs recipe; A Long Way to Go ([arXiv:2310.03716](https://arxiv.org/abs/2310.03716)) — length correlations; Llama 2 ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)) — margin loss and two reward models; Secrets of RLHF Part II ([arXiv:2401.06080](https://arxiv.org/abs/2401.06080)) — label noise and RM ensembles; Skywork-Reward-V2 ([arXiv:2507.01352](https://arxiv.org/abs/2507.01352)) — data curation for strong RMs.
- Code: [TRL](https://github.com/huggingface/trl) `RewardTrainer`; [reward-bench](https://github.com/allenai/reward-bench).

#### Risks and open decisions

- *UltraFeedback pairs are AI-labeled and length-correlated*, so the length result may be stronger than for human labels; fallback: report both data sets (already planned).
- *π_SFT unavailable*: fall back to `Qwen/Qwen3-0.6B` (instruct, non-thinking) as backbone.
- *Skywork-V2 models are gated or renamed*: fallback gold RM `Skywork/Skywork-Reward-V2-Qwen3-4B` (checked present, Apache-2.0) or a judge via `d2l.judge_pairwise`.
- *RewardBench file format changes*: pin the dataset revision.
- *Memory of the chunked log-prob*: if `torch.compile` is unavailable the chunked path is 1.5–2× slower — to be measured, acceptable.

### 8.2 RLHF with PPO for language models

- **Label:** `sec_lm_rlhf_ppo` · **Tier:** T2 (reduced T1 config: LoRA actor via `d2l.apply_lora(model, target_modules='all', r=16, alpha=32)`; reference = `d2l.lora_disabled`; critic = linear value head on the actor's own final hidden state (shared frozen backbone, "hydra" layout); frozen RM from 8.1; 60 PPO steps × 16 prompts × 1 sample, 192 new tokens) · **Budget:** T2 ≤ 2 h (200 steps × 64 prompts, 256 new tokens), peak memory from the role table below, to be measured; T1 ≤ 15 min, ≤ 15 GiB · **Codex units:** 9.2, 9.4
- **Builds on (book):** `sec_ppo` (`eq_ppo_clip`, ratio $\rho_t$, clip, diagnostics `subsec_ppo_diagnostics`, advantage choice and GAE $\lambda=0.95$) and `sec_actorcritic` (`eq_gae_deltas`, critic as sampled policy evaluation, moving target `subsec_actorcritic_two_timescales`): reused as stated; **not re-taught:** surrogate objective, GAE derivation, the clip ablation. `sec_rl_sequences` (token MDP, terminal reward, `eq_seq_factorization`, the contract's $\hat G_t$, $A$, $\rho_t$, $\epsilon$, $\beta$, $\delta_t$, $w$): the notation. `sec_regularized` (`eq_kl_objective`, `subsec_kl_direction`): the objective that the per-token penalty implements. `sec_mdl-divergences-distances`: KL as expectation.
- **Builds on (this part):** 8.1 (`d2l.PreferenceData`, `d2l.RewardModel`, `d2l.token_logprobs`, π_SFT, `lm-rm-qwen3-0p6b-uf`), 7.4 (`d2l.apply_lora`, `d2l.lora_disabled`, `d2l.LoRALinear`), 1.5 (`d2l.paired_bootstrap`, returning a `BootstrapResult` with `.mean` and `.ci`).
- **Exports (`#@save`):**
  - `d2l.kl_estimators(logp, logp_ref)` → dict `{'k1','k2','k3'}` of per-token tensors with $\log r=\texttt{logp\_ref}-\texttt{logp}$.
  - `d2l.rlhf_rewards(logp, logp_ref, reward, resp_mask, beta, eos_ok)` → per-token reward $r_t=-\beta(\log\pi_{\textrm{old}}-\log\pi_{\textrm{ref}})$ with the RM score added at the last response token; for responses with `eos_ok=False` (truncated) the score is replaced by a constant penalty.
  - `d2l.gae(rewards, values, resp_mask, gamma=1.0, lam=0.95)` → `(advantages, returns)`, backward scan restricted to masked tokens.
  - `d2l.masked_whiten(x, mask, shift_mean=True, eps=1e-8)` using the population standard deviation over valid tokens.
  - `d2l.ppo_loss(logp, logp_old, adv, resp_mask, eps_low=0.2, eps_high=0.2, aggregation='token', l_max=None)` → `(loss, diagnostics)`: the clipped surrogate of the book's `eq_ppo_clip` with per-token advantages `adv` (shape `(B,T)`, or `(B,)` broadcast over tokens), plus diagnostics (`clipfrac_low`, `clipfrac_high`, `approx_kl`). `aggregation` selects how per-token terms are averaged: `'sample'` = mean over responses of the per-response token mean (GRPO's $1/|y_i|$), `'token'` = sum over all valid tokens divided by their total count in the batch (DAPO), `'const'` = sum divided by `B*l_max` (Dr. GRPO). The three modes are derived in 8.5 D1; 8.2 uses `'token'`.
  - `d2l.value_loss(values, values_old, returns, resp_mask, clip=0.2)`.
  - `d2l.PhaseTimer` : context-manager timers around phases (`with t('generate'): …`) that call `torch.cuda.synchronize()` and record wall-clock and `torch.cuda.max_memory_allocated()` deltas per phase; `.table()` returns a DataFrame. Reused by 8.5/8.6.
  - `d2l.rlhf_memory(n_params, roles, weight_bytes=2, optimizer='adamw_fp32_master')` → bytes per role (weights, gradients, optimizer state) using the accounting of `sec_memory_precision`; roles: `actor`, `critic`, `reference`, `reward`.
  - `d2l.ppo_rlhf_train(policy, rm, prompts, cfg)` → log (list of dicts per step).
- **Models and data:** π_SFT (596,049,920 parameters, `Qwen/Qwen3-0.6B-Base` after 7.2); RM `lm-rm-qwen3-0p6b-uf` (8.1); independent gold `Skywork/Skywork-Reward-V2-Qwen3-1.7B` (1,720,577,024 parameters, Apache-2.0); prompts: the prompts of `HuggingFaceH4/ultrafeedback_binarized` `train_prefs` not used for RM training (disjoint prompt ids), held-out 200 from `test_prefs`.
- **Packages:** `transformers`, `peft`; `trl` only as an oracle/comparison (check whether `PPOTrainer` is shipped in the pinned release; if it has moved to an experimental namespace, import it from there and say so); `vllm` optional (T1 uses `transformers.generate`; 8.5 introduces vLLM rollouts).

#### Narrative arc

1. **The question.** *How many models does one RLHF step touch, and where does the time go?* Opening cell: `d2l.rlhf_memory` for the 0.6B model (full roles) and, for scale, an 8B model (parameter count from `Qwen/Qwen3-8B` config via `d2l.config_memory`/1.1), printing bytes per role; then a 3-step dry run with `PhaseTimer` that prints seconds in generate / score / ref-logprob / value / update. The symptom: four models' weights and two optimizer states resident, and generation taking the larger share of the step (the fraction is "to be measured").
2. **A one-paragraph recap and the token-level view** (`:numref:` to `sec_ppo`, `sec_rl_sequences`): actor $\pi_\theta$, critic $V_w$, frozen $\pi_{\textrm{ref}}$ and $r_\phi$; response = trajectory $\tau$; sample $y\sim\pi_{\theta_{\textrm{old}}}(\cdot\mid x)$. A diagram `fig_lm_ppo_roles`: one PPO step as data flow with, per box, whether it needs gradients, its weights/optimizer bytes, and the phase name used by `PhaseTimer`.
3. **The four roles and their memory.** Table `tab_lm_ppo_memory` from `rlhf_memory`: actor and critic carry weights + gradients + AdamW state; reference and reward models carry weights only; the actor's KV cache for generation is additional. Three ways to cut it, each tested in the notebook: (a) LoRA actor with adapters-off reference (no second copy); (b) shared backbone with a value head (critic bytes ≈ one linear layer, but actor and critic gradients now interact — a cell logs the cosine of the policy and value gradients on the shared parameters); (c) CPU/disk offload or sequential loading of the reward model. Peak memory per role is measured with `PhaseTimer`, never assumed.
4. **Rewards on tokens: the per-token KL penalty.**
   - $r_t=-\beta\,(\log\pi_{\theta_{\textrm{old}}}(y_t\mid x,y_{<t})-\log\pi_{\textrm{ref}}(y_t\mid x,y_{<t}))+\mathbf 1(t=T)\,r_\phi(x,y)$ (`eq_lm_ppo_token_reward`).
   - D2 below shows the sum is the sequence-level regularized return of `eq_kl_objective`.
   - Where the KL goes (reward vs loss), and what each estimator's gradient is: D1 below. This is the place where GRPO's choice (k3 as a loss term, 8.5) is justified or criticized.
   - Adaptive KL controller (Ziegler et al.) versus fixed $\beta$: one cell implements the controller, ablated in an exercise.
5. **Reward on the final token, truncation and whitening.**
   - Terminal reward: $\hat G_t$ is the same for all $t$ when $\beta=0$ (`sec_rl_sequences`); with the KL term the returns differ per token, so GAE does real work.
   - EOS handling: responses that hit `max_new_tokens` without EOS receive a fixed penalty instead of the RM score, so the policy cannot exploit an RM that likes unfinished text; show the rate of truncated responses rising when this is off.
   - Whitening: `masked_whiten` of advantages over valid tokens of the *whole batch* (population std) — and the known interaction with reward scale: whitening makes PPO invariant to a global rescaling of the RM, but also hides a collapsing reward signal. Rewards are *not* whitened by default here (a switch is ablated).
6. **The implementation details that matter** (`tab_lm_ppo_details`; each row has: detail, what breaks, the test or ablation). Each is in the notebook as a toggle in `cfg`, with the three or four most consequential ablated in the headline experiment:
   - dropout off in actor, critic, reference (else ratios ≠ 1 at the first epoch; test: ratio ≡ 1 to 1e-4 in FP32 before any update);
   - logits divided by the *sampling temperature* before computing $\log\pi$ (test: log-probs from `generate(output_scores=True)` equal `token_logprobs(temperature=T)`);
   - left padding with correct `position_ids` during generation and the same positions in the training forward;
   - value head init from the RM backbone vs from scratch; value clipping;
   - mask-aware GAE (padding after EOS contributes zero; test: the 3-token fixture below padded to length 8 gives the same advantages);
   - whiten advantages, not returns; population vs unbiased std (the book's `d2l.normalize` uses `x.std()`, unbiased in PyTorch — state which each place uses);
   - one or two PPO epochs, 4 minibatches, no data reuse beyond that;
   - FP32 for the softmax/log-softmax of logits even when the model runs in BF16;
   - EOS/truncation penalty; minimum response length (off).
   Source of the catalogue: Huang et al., *The N+ Implementation Details of RLHF with PPO* (cite; do not copy its count of details — read the paper and quote its own list).
7. **Hand-computable advantages.** Cell `check_advantages` runs `rlhf_rewards` and `gae` on the fixture below and prints PASS against the numbers (derived by hand in the text so a reader can follow).
8. **Training run and diagnostics.** Headline experiment below. Diagnostics panel: proxy reward, gold reward, sampled KL to the reference (sequence-level, from k1), response length, entropy, `clipfrac`, `approx_kl`, value loss, explained variance of the critic.
9. **Compare with TRL/OpenRLHF** in production (below).

**Fixture (hand-computable; verified in the blueprint preparation by a CPU script).** One response of $T=3$ tokens, $\gamma=1$, $\lambda=0.95$, $\beta=0.1$, RM score $R=1.0$, values $V=(0.5,0.4,0.3)$ with $V_{T}=0$, log-ratios $\log(\pi_{\textrm{old}}/\pi_{\textrm{ref}})=(0.2,-0.1,0.3)$.
- Rewards $r_t=-\beta\log(\pi_{\textrm{old}}/\pi_{\textrm{ref}})$, plus $R$ on the last token: $(-0.02,\ 0.01,\ -0.03+1.0)=(-0.02,\ 0.01,\ 0.97)$.
- TD errors $\delta_t=r_t+V_{t+1}-V_t$: $(-0.12,\,-0.09,\,0.67)$.
- GAE: $A_2=0.67$; $A_1=-0.09+0.95\cdot0.67=0.5465$; $A_0=-0.12+0.95\cdot0.5465=0.399175$. Returns $A+V=(0.899175,\,0.9465,\,0.97)$.
- Limits: $\lambda=1$ gives $A=\hat G_t-V_t=(0.46,\,0.58,\,0.67)$ with $\hat G=(0.96,0.98,0.97)$; $\lambda=0$ gives $A=\delta$.
- Whitened (population std over the 3 tokens; the batch mean of $A$ is $0.5385583$, std $0.1107064$): $(-1.25904,\,0.07174,\,1.18730)$. With the unbiased std ($0.1355870$) the values differ — the test fixes the convention.
- Pad to length 8: identical advantages on the 3 valid positions and exact zeros on padding.

#### Derivations

**D1. KL estimators k1, k2, k3: value bias, variance, and the gradients they estimate (v5 derivation).**
*Statement.* Let $q=\pi_\theta(\cdot\mid x)$ (the sampling distribution), $p=\pi_{\textrm{ref}}(\cdot\mid x)$ with $\operatorname{supp}q\subseteq\operatorname{supp}p$, and $\varrho(y)=p(y)/q(y)$. Define $k_1=-\log\varrho$, $k_2=\tfrac12(\log\varrho)^2$, $k_3=(\varrho-1)-\log\varrho$.
(a) *Value:* $\mathbb E_q[k_1]=\mathbb E_q[k_3]=D_{\textrm{KL}}(q\Vert p)$; $\mathbb E_q[k_2]$ is the $f$-divergence with $f(u)=\tfrac12(\log u)^2$, which agrees with $D_{\textrm{KL}}(q\Vert p)$ to second order in $\varrho-1$ and is biased otherwise.
(b) *Variance:* $k_2,k_3\ge0$ pointwise; $k_1$ takes both signs and its standard deviation is of order $\sqrt{\text{KL}}$ when KL is small, versus order KL for $k_2$ and $k_3$.
(c) *Gradients as losses* (samples $y\sim q_{\theta}$ treated as data, differentiate only the explicit $\theta$-dependence): $\mathbb E_q[\nabla_\theta k_1]=0$; $\mathbb E_q[\nabla_\theta k_2]=\nabla_\theta D_{\textrm{KL}}(q\Vert p)$; $\mathbb E_q[\nabla_\theta k_3]=\nabla_\theta D_{\textrm{KL}}(p\Vert q)$ (the *forward* KL).
(d) *k1 as a reward:* $\mathbb E_q[k_1\,\nabla_\theta\log q]=\nabla_\theta D_{\textrm{KL}}(q\Vert p)$ — the correct reverse-KL gradient is obtained only if $k_1$ multiplies the score (it sits in the reward of PPO-RLHF), not if it is added to the loss.
*Steps.*
1. (a) $\mathbb E_q[-\log\varrho]=\sum q\log(q/p)$. $\mathbb E_q[\varrho-1]=\sum_y p(y)-1=0$ (support condition), so $\mathbb E_q[k_3]=\mathbb E_q[k_1]$. For $k_2$ expand $\log\varrho=\delta-\delta^2/2+O(\delta^3)$ with $\delta=\varrho-1$: $\tfrac12(\log\varrho)^2=\tfrac12\delta^2+O(\delta^3)$, and $\mathbb E[-\log\varrho]=\mathbb E[\delta^2/2]+O(\delta^3)$ since $\mathbb E\delta=0$.
2. (b) $k_3=\varrho-1-\log\varrho\ge0$ because $\log u\le u-1$; $k_2$ is a square. For small shifts $\delta$ is $O(\sqrt{KL})$ so $k_1\approx-\delta$ has std $\sim\sqrt{\text{KL}}$ while $k_2,k_3\approx\delta^2/2$ have mean and std $\sim\text{KL}$.
3. (c) With $y$ held fixed, $\nabla\varrho=-\varrho\nabla\log q$, so $\nabla k_1=\nabla\log q$, $\nabla k_2=(\log q-\log p)\nabla\log q$, $\nabla k_3=(1-\varrho)\nabla\log q$. Take $\mathbb E_q$: $\sum q\nabla\log q=\sum\nabla q=0$ gives $k_1$; $\sum q\log(q/p)\nabla\log q=\nabla D_{\textrm{KL}}(q\Vert p)$ (the extra term $\sum q\nabla\log q$ vanishes) gives $k_2$; $\sum q(1-p/q)\nabla\log q=-\sum p\nabla\log q=\nabla\sum p\log(p/q)$ gives $k_3$.
4. (d) is step 3's $k_2$ computation with the weight $\log(q/p)=k_1$.
5. Token-level use: apply (a)–(d) per state through `eq_seq_factorization`; for a sequence the estimators are summed or averaged over tokens, so (c) holds statewise, and "k3 in the loss" optimizes a forward-KL-like penalty at the visited states.
*Numerical check (cell, numpy/torch fp64, seed 0).* A 5-outcome categorical with random logits and a random $p$. Enumerate: assert (c) and (d) against autograd of the exact $D_{\textrm{KL}}(q\Vert p)$ and $D_{\textrm{KL}}(p\Vert q)$ with `atol=1e-9` (prepared run: $k_1$ loss gradient $\sim10^{-17}$; $k_2$ and $k_3$ gradients match the reverse and forward KL gradients to $\sim10^{-10}$, while $k_3$ differs from the reverse-KL gradient by $0.44$ in max-norm). Monte Carlo for (a)–(b): $q=\mathcal N(0,1)$, $p=\mathcal N(\mu,1)$, $2\cdot10^6$ samples, seed 0: for $\mu=0.1$ (KL=0.005) the prepared run gives means $(0.0049,0.0050,0.0050)$ and std $(0.100,0.0071,0.0071)$ for $(k_1,k_2,k_3)$; for $\mu=1$ (KL=0.5): means $(0.501,0.626,0.500)$, std $(1.00,0.87,0.84)$ — i.e. $k_2$ is biased by about $+25\%$ at KL=0.5, $k_3$ is unbiased with the lowest variance. The cell must recompute these numbers and print them; the text quotes the printed values.

**D2. The per-token KL penalty is the sequence-level regularized return.**
*Statement.* With $r_t=-\beta\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\textrm{ref}}(y_t\mid s_t)}$ for $t<T$ and an extra $r_\phi$ at $T$, the undiscounted return is $R_\phi(x,y)-\beta\,h_\theta(y)$, whose expectation over $y\sim\pi_\theta$ is the objective of `eq_kl_objective` at sequence level.
*Steps.* (1) Sum the token terms; by `eq_seq_factorization` $\sum_t\log\pi_\theta(y_t\mid s_t)=\log\pi_\theta(y\mid x)$ and likewise for $\pi_{\textrm{ref}}$. (2) Taking $\mathbb E_{y\sim\pi_\theta}$ of $-\beta h_\theta(y)$ gives $-\beta D_{\textrm{KL}}(\pi_\theta(\cdot\mid x)\Vert\pi_{\textrm{ref}}(\cdot\mid x))$ (sequence KL). (3) The reward-to-go $\hat G_t$ keeps only the penalty terms at or after $t$ (causality, `sec_baselines`), which is why the penalty is placed per token rather than only at the end: variance is lower with the same expectation.
*Numerical check.* A two-token, three-symbol policy (tabular) where all 9 sequences are enumerated; compute $\mathbb E[R-\beta h]$ exactly and via 20,000 Monte Carlo samples; assert the enumerated value equals $\mathbb E_q[r_\phi]-\beta\,D_{\textrm{KL}}$ computed from the sequence distributions to 1e-12, and the Monte Carlo agrees within 3 standard errors.

#### From-scratch implementation

Budget ≤ ~150 lines: `kl_estimators` (8), `rlhf_rewards` (15), `gae` (15), `masked_whiten` (8), `ppo_loss` + `value_loss` (35), `PhaseTimer` (25), `rlhf_memory` (15), `ppo_rlhf_train` (≈ 60: sample with `generate`, compute old/ref log-probs under `torch.no_grad()` and `d2l.lora_disabled`, values, RM score, rewards, GAE, then minibatch epochs; logging).
Tests: (1) fixture above to 1e-6; padding invariance; (2) D1 and D2 checks; (3) ratio ≡ 1 to 1e-4 at the first minibatch with dropout off (FP32 forward); (4) `ppo_loss` equals the book's clipped objective (`d2l.ppo_epochs` in `sec_ppo`) on a tabular problem — the same function value, not the same loop; (5) `rlhf_memory` reproduces `torch.cuda.memory_allocated` after loading each role within 10% (to be measured; state the discrepancy).

#### In production

- OpenRLHF ([arXiv:2405.11143](https://arxiv.org/abs/2405.11143), [repo](https://github.com/OpenRLHF/OpenRLHF)): Ray actors for actor/critic/reference/reward, vLLM generation; run a 0.6B config and compare the diagnostics panel (not the final reward) with the from-scratch run. TRL `PPOTrainer` if present in the pinned release; otherwise TRL's reward-model + the from-scratch loop. The from-scratch `gae`, `rlhf_rewards` and `ppo_loss` are the oracles: feed both implementations the same recorded batch and compare advantages (`atol=1e-5` FP32) and loss (`rtol=1e-4`).
- Unpacking DPO and PPO ([arXiv:2406.09279](https://arxiv.org/abs/2406.09279)) as the controlled comparison to read after 8.3.

#### Headline experiment

- **Hypothesis.** Under PPO against the 8.1 reward model, the proxy reward rises monotonically while the *gold* reward (independent RM) rises and then flattens or falls as sampled KL grows; mean response length drifts upward; a larger $\beta$ delays but may not remove the flattening. (A preview of 8.7; directions only.)
- **Setup.** π_SFT actor; RM `lm-rm-qwen3-0p6b-uf`; T2: LoRA r=32 on all linear layers *and* a full-parameter variant for the memory table only; 200 steps × 64 prompts × 1 sample, 256 new tokens, temperature 1.0, 2 PPO epochs × 4 minibatches, lr 1e-5 (LoRA 1e-4), GAE γ=1, λ=0.95, $\epsilon=0.2$; $\beta\in\{0,0.05\}$; whitening on. Seeds {0,1,2} for the T1 version, {0,1} for T2.
- **Procedure.** Train both $\beta$; every 10 steps evaluate on 200 held-out prompts: mean proxy reward, mean gold reward, mean length, sequence KL; log `PhaseTimer` tables for the first 5 steps and the last 5.
- **Metrics.** proxy/gold reward, sequence KL (k1 average, and k3 as a cross-check of D1), length, entropy, clipfrac, per-role seconds and peak memory.
- **Plots.** `fig_lm_ppo_curves` (2×2 grid vs step: proxy and gold reward; KL; length; entropy; series $\beta\in\{0,0.05\}$ mean ± range over seeds); `fig_lm_ppo_frontier` (gold reward vs $\sqrt{\text{KL}}$ — the Gao-style axes, used again in 8.7); `fig_lm_ppo_time` (stacked bars of seconds per role per step).
- **Expected outcome.** Directions above; the book's `subsec_kl_budget` shows the same non-monotonic true-return curve in a tabular setting and Gao et al. report it at LM scale. Magnitudes to be measured.
- **Noise check.** Paired bootstrap over the 200 held-out prompts for gold-reward differences between the two β at the last step, and the per-seed spread of the curves; a claim of "flattening" requires the late-step difference from the peak to exceed the seed range.
- **T1 reduction.** 60 steps × 16 prompts, one β pair, 3 seeds, gold evaluated on 100 prompts.
- **Artifacts (T2).** `lm-ppo-qwen3-0p6b` adapters, step logs (jsonl), `PhaseTimer` tables.

#### Figures and tables

`fig_lm_ppo_roles`, `tab_lm_ppo_memory` (bytes per role for 0.6B and 8B, full vs LoRA, computed), `tab_lm_ppo_details`, `fig_lm_ppo_curves`, `fig_lm_ppo_frontier`, `fig_lm_ppo_time`, `tab_lm_ppo_kl` (k1/k2/k3 values and stds from D1's Monte Carlo).

#### Pitfalls and misconceptions

- Putting $k_3$ in the loss and believing it optimizes the reverse KL (it estimates the forward-KL gradient; D1c).
- Computing ref log-probs *with* adapters enabled after the first update (the reference silently follows the policy; the KL term is then ≈ 0).
- Whitening over padded positions, or with the unbiased standard deviation of a 1-token batch (division by zero).
- Sampling at temperature ≠ 1 but scoring log-probs at temperature 1 (ratios ≠ 1, biased gradient).
- Reading a rising proxy reward as progress; always plot an independent reward.
- Counting memory by model size only: optimizer state and KV cache dominate.

#### Summary (draft bullets)

- PPO-RLHF touches four models; two carry gradients and optimizer state, and generation precedes every update.
- The per-token KL penalty is the sequence-level regularized return, delivered with lower variance.
- $k_1$ in the reward, $k_2$ as a loss, and $k_3$ as a loss estimate different gradients (reverse, reverse, forward KL); $k_3$ is the unbiased low-variance *value* estimator.
- Details such as masks, temperature, dropout and truncation penalties decide whether the advantage code is correct; hand-computable fixtures catch them.
- The gold reward, not the proxy, tells whether PPO helped.

#### Exercises

1. **(check)** Which of $k_1,k_2,k_3$ can be negative per sample, and which is unbiased for $D_{\textrm{KL}}(q\Vert p)$? *Sketch:* only $k_1$ can be negative; $k_1$ and $k_3$ are unbiased, $k_2$ is not.
2. **(check)** In the fixture, which tokens' advantages change if $R$ becomes $2.0$ and $\lambda=0.95$, and by how much? *Sketch:* $r_T$ grows by 1, so $A_2$ grows by 1, $A_1$ by $0.95$, $A_0$ by $0.9025$.
3. **(derive)** Compute $\mathbb E_q[k_3\,\nabla\log q]$ (k3 used as a *reward* multiplying the score) and say which divergence's gradient it is. *Sketch:* $k_3=\varrho-1+\log(q/p)$; $\mathbb E_q[\varrho\nabla\log q]=\sum p\nabla\log q=-\nabla D_{\textrm{KL}}(p\Vert q)$, $\mathbb E_q[\nabla\log q]=0$, and $\mathbb E_q[\log(q/p)\nabla\log q]=\nabla D_{\textrm{KL}}(q\Vert p)$; the result is $\nabla D_{\textrm{KL}}(q\Vert p)-\nabla D_{\textrm{KL}}(p\Vert q)$, neither pure direction.
4. **(derive)** For tabular softmax $q$ and the loss $\beta\,\mathbb E_q[k_2]$ (reverse-KL gradient), derive the logit gradient and show it vanishes exactly at $q=p$. *Sketch:* from D1(c) the logit gradient is $q\odot(\log(q/p)-D_{\textrm{KL}})$, zero iff $\log(q/p)$ is constant, i.e. $q=p$.
5. **(measure)** Ablate whitening (on/off) and the truncation penalty (on/off) in the T1 run with 3 seeds; predict which changes the truncated-response rate; report paired intervals.
6. **(measure)** Replace the fixed $\beta$ by the adaptive controller targeting a sampled KL; compare the gold-vs-$\sqrt{\text{KL}}$ frontier with fixed $\beta$.
7. **(open)** Replace the critic with a group baseline (anticipating 8.5) in the same PPO code and compare memory, time and gold reward; defend which role you would drop first at 8B.

#### Further reading

- Primary: Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html); [The N+ Implementation Details of RLHF with PPO](https://arxiv.org/abs/2403.17031); [OpenRLHF](https://arxiv.org/abs/2405.11143); InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)).
- New: A Comedy of Estimators ([arXiv:2512.21852](https://arxiv.org/abs/2512.21852)) and On the Design of KL-Regularized Policy Gradient Algorithms ([arXiv:2505.17508](https://arxiv.org/abs/2505.17508)) — the gradient-level analysis that D1 reproduces in the one-step case; Unpacking DPO and PPO ([arXiv:2406.09279](https://arxiv.org/abs/2406.09279)).
- Tutorials: HF [Illustrating RLHF](https://huggingface.co/blog/rlhf); Lambert, RLHF Book (policy-gradient chapters).
- Code: [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF); [TRL](https://github.com/huggingface/trl).

#### Risks and open decisions

- *Generation speed at T1 with `transformers.generate`* limits steps; fallback: shorter responses (128 tokens), or vLLM as in 8.5.
- *TRL's PPO API unstable across releases*: the production comparison is by recorded batches, so any version suffices.
- *Shared-backbone critic degrades the policy*: the cosine-of-gradients cell diagnoses it; fallback is a separate LoRA for the critic.
- *Gold RM disagrees with the proxy on length only*: add the length-controlled view of 8.7 to the plot.

### 8.3 Direct preference optimization

- **Label:** `sec_lm_dpo` · **Tier:** T2 (reduced T1 config: LoRA r=16 via `d2l.apply_lora(model, target_modules='all', r=16, alpha=32)`, 4,000 pairs, reference log-probabilities precomputed once with adapters disabled, max length 512) · **Budget:** T2 ≤ 1 h (full fine-tune, 20,000 pairs, 1 epoch, 3 values of β), peak memory to be measured (policy + optimizer state ≈ the 16 B/parameter accounting of 8.1, reference as cached tensors rather than a second model); T1 ≤ 15 min, ≤ 15 GiB · **Codex units:** 9.3
- **Builds on (book):** `sec_regularized` (`eq_kl_objective`, `eq_kl_optimum`, `eq_bradley_terry`, identifiability): the two ingredients that DPO combines; **not re-taught:** the closed-form optimum or the BT model. `sec_rl_sequences` (`eq_dpo_reward`, `eq_seq_factorization`, the exercise "DPO on the finite instance"): this section *delivers* the derivation that `sec_rl_sequences` defers, and must refer to that exercise for the finite-data failure on a tabular policy. `sec_mdl-constrained-optimization-duality`: β as the Lagrange multiplier of a KL budget.
- **Builds on (this part):** 8.1 (`d2l.PreferenceData`, `d2l.collate_pairs`, `d2l.token_logprobs`, `d2l.sequence_logprob`, `d2l.bt_loss` for the comparison with the reward-model loss), 1.2 (`d2l.generate`, returning `list[Generation]`; use `.text`), 1.5 (`d2l.judge_pairwise`, returning a `JudgeResult`; `.pref_a`, `.verdict`; `d2l.paired_bootstrap`, returning a `BootstrapResult`; `.mean`, `.ci`), 7.2 (π_SFT), 7.4 (`d2l.apply_lora`, `d2l.lora_disabled`).
- **Exports (`#@save`):**
  - `d2l.dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1, label_smoothing=0.0)` → `(loss, stats)`; inputs are sequence log-probabilities (sums over response tokens); `stats` has `reward_w`, `reward_l` (the implicit rewards $\beta h_\theta$), `margin`, `acc`. Contract: returns $-\log\sigma(\beta[(\pi_w-\textrm{ref}_w)-(\pi_l-\textrm{ref}_l)])$ (with the cDPO smoothing $\varepsilon$: $-(1-\varepsilon)\log\sigma(\cdot)-\varepsilon\log\sigma(-\cdot)$).
  - `d2l.implicit_reward(pol_logp, ref_logp, beta)` → $\beta(\log\pi_\theta-\log\pi_{\textrm{ref}})$.
  - `d2l.cache_ref_logps(model, pairs, path)` → writes/reads a tensor file of $(\log\pi_{\textrm{ref}}(y_w),\log\pi_{\textrm{ref}}(y_l))$ keyed by `pair_id`; asserts the model is in eval mode and inside `d2l.lora_disabled`.
  - `d2l.dpo_train(policy, pairs, ref_cache, cfg, loss_fn=None)` (`loss_fn` defaults to `d2l.dpo_loss`; 8.4 passes its own variants) → log (list of dicts) including per-step mean chosen and rejected log-probabilities.
  - `d2l.win_rate(outputs_a, outputs_b, prompts, judge, tokenizer)` → `(rate, ci, tie_rate, per_item)` built on `d2l.judge_pairwise` and `d2l.paired_bootstrap`; a win requires the same verdict in both orders, a disagreement counts 0.5.
  - `d2l.ches(model, pairs)` → centered hidden-embedding similarity per pair (see narrative step 7).
  - Data artifact: `lm-dpo-qwen3-0p6b` (adapter or merged weights), `lm-judgments-dpo` (per-prompt verdicts).
- **Models and data:** π_SFT (596,049,920 parameters); pairs from `lm-pairs-uf-tok` (8.1); judge `Qwen/Qwen3.5-9B` (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache-2.0) in bf16, text-only and non-thinking (the part's judge, `00_conventions.md`); independent RM `Skywork/Skywork-Reward-V2-Qwen3-1.7B`; evaluation prompts = first 300 `test_prefs` prompts by `prompt_id`.
- **Packages:** `peft`, `transformers`; `trl` for the oracle (`DPOTrainer`, `DPOConfig` with `loss_type="sigmoid"`; pin the release; PyPI latest at blueprint time 1.16.0).

#### Narrative arc

1. **The question.** *DPO has no reward model and no rollouts, and its loss goes down. What did the model learn, and what did it lose?* Opening cell: 100 steps of the LoRA DPO on 800 pairs, then print for 100 held-out pairs the mean change in $\log\pi(y_w)$, $\log\pi(y_l)$ and in the implicit-reward accuracy. The measured symptom: accuracy up, *both* log-probabilities down (to be measured; direction expected from Razin et al.).
2. **From the regularized objective to a classification loss** (core derivation, D1): five boxed steps with `eq_lm_dpo_reward` ($r=\beta h_\theta+\beta\log Z$), `eq_lm_dpo_loss`, `eq_lm_dpo_grad`. Remind (one paragraph, `:numref:` to `sec_regularized` and `sec_rl_sequences`) that these are the equations the book already stated; here we derive the missing step, the substitution into the preference likelihood, and the consequences for language models. Duality aside (`sec_mdl-constrained-optimization-duality`): $\beta$ is the multiplier of a KL budget, so a sweep over $\beta$ traces the reward–KL frontier of `subsec_kl_budget`.
3. **Implicit rewards and what to log.** Implicit reward $\beta h_\theta(y)$ (identified up to $f(x)$, as in 8.1); reward accuracy (fraction of pairs with positive margin), margin, and the *two* log-probabilities separately — the notebook never logs only the margin, because the margin hides the symptom of step 1.
4. **Dependence on the reference model.** The loss is a function of $h_\theta$, so the reference enters twice: it is the prior in the KL and the origin of the implicit rewards. Cells: (a) the reference is π_SFT (default); (b) reference = the base model (before SFT); (c) reference-free ($\pi_{\textrm{ref}}$ uniform, all $\log\pi_{\textrm{ref}}=0$), which turns the loss into SimPO-like margin training without length normalization (forward pointer to 8.4); report win rate and KL for each. Also the on-policy question: UltraFeedback responses were not sampled from π_SFT, so the data are off-policy for DPO; show the initial gap $\log\pi_{\textrm{SFT}}(y_w)-\log\pi_{\textrm{SFT}}(y_l)$ per token. (Forward pointer: Tajwar et al. and Tang et al. on on-policy data, 8.4.)
5. **The training loop in production shape.** Precompute reference log-probs once (`cache_ref_logps`: a pass over the data with adapters off, stored as 2 floats per pair); the policy forward over chosen and rejected concatenated in one batch; learning rate 10–100× below SFT; β ∈ {0.05, 0.1, 0.3}; one epoch; length filtering. A `PhaseTimer` table (8.2) shows the training step cost with and without the cache.
6. **What the optimum looks like with finite data.** A tabular cell (reusing the book's four-prompt instance): deterministic preferences drive $h_w-h_l\to\infty$; $\sigma(-\beta m)$ weights vanish; probability can leave *both* responses. This is the `sec_rl_sequences` exercise's observation; connect to step 7.
7. **Likelihood displacement.** Observed in the LM run: `fig_lm_dpo_logps` (mean chosen and rejected log-probabilities vs step). Mechanism: D2 (a linear-softmax toy with a test vector). Then the empirical hook: implement CHES, the centered hidden-embedding similarity of the preferred and dispreferred responses defined by Razin et al., and plot $\Delta\log\pi(y_w)$ against it for 1,000 pairs (`fig_lm_dpo_ches`); remedies: filter or down-weight high-similarity pairs, add an NLL term on the chosen response, lower the learning rate. *The author must read the paper's definition and implement it exactly; the blueprint's description is a pointer, not a definition.*
8. **Evaluating the result.** Win rate vs π_SFT (judge, both orders), independent-RM win rate, length, length-controlled win rate, sampled KL to the reference (k1 from 8.2), implicit-reward accuracy on held-out pairs. Table `tab_lm_dpo_results` over β; the best β by judged win rate is *not* chosen from the same 300 prompts that produce the headline interval (a 100-prompt dev slice picks β).
9. **Preview of the family** (8.4) in a half-page: which assumption each variant changes.

#### Derivations

**D1. DPO from the KL-regularized optimum and the Bradley–Terry model (required).**
*Statement.* Let $r^\ast(x,y)$ be a reward, $\beta>0$, $\pi_{\textrm{ref}}$ have full support, and preferences follow `eq_bradley_terry`: $P(y_w\succ y_l\mid x)=\sigma(r^\ast(x,y_w)-r^\ast(x,y_l))$. Let $\pi^\ast$ maximize $\mathbb E_{y\sim\pi}r^\ast-\beta D_{\textrm{KL}}(\pi\Vert\pi_{\textrm{ref}})$. Then (a) $P(y_w\succ y_l\mid x)=\sigma\big(\beta h^\ast(y_w)-\beta h^\ast(y_l)\big)$ with $h^\ast=\log\pi^\ast/\pi_{\textrm{ref}}$; (b) the maximum-likelihood fit over an unrestricted policy class is the minimizer of $\mathcal L_{\textrm{DPO}}(\theta)=-\mathbb E\log\sigma\big(\beta h_\theta(y_w)-\beta h_\theta(y_l)\big)$; (c) $\nabla_\theta\mathcal L=-\beta\,\mathbb E\big[\sigma(\beta h_\theta(y_l)-\beta h_\theta(y_w))\,(\nabla_\theta\log\pi_\theta(y_w)-\nabla_\theta\log\pi_\theta(y_l))\big]$.
*Assumptions.* Sequence-level responses; unrestricted policy class (the finite-capacity caveat is step 7); the data are comparisons of two responses to the same prompt; one label per pair.
*Steps.*
1. By `eq_kl_optimum`, $\pi^\ast(y\mid x)=\pi_{\textrm{ref}}(y\mid x)\,e^{r^\ast(x,y)/\beta}/Z(x)$.
2. Solve for the reward: $r^\ast(x,y)=\beta\log\frac{\pi^\ast(y\mid x)}{\pi_{\textrm{ref}}(y\mid x)}+\beta\log Z(x)$ (`eq_dpo_reward` with $c(x)=\beta\log Z(x)$).
3. Substitute in $\sigma(r^\ast(y_w)-r^\ast(y_l))$: the $\beta\log Z(x)$ terms cancel because both responses belong to the same prompt (this is the identifiability of 8.1/`sec_regularized`: only differences matter). This is (a).
4. Reparameterize: any reward is, up to $f(x)$, of the form $\beta\log(\pi/\pi_{\textrm{ref}})$ for exactly one full-support policy $\pi$ (invert step 2 for $\pi_r\propto\pi_{\textrm{ref}}e^{r/\beta}$). Fitting the reward by BT-maximum likelihood over *all* rewards is therefore the same as fitting a policy $\pi_\theta$ over *all* full-support policies, with $r_\theta:=\beta\log\pi_\theta/\pi_{\textrm{ref}}$. The optimizer of the resulting loss is (b); by the invariance of 8.1 D1, its policy is the $\pi^\ast$ of the fitted reward.
5. Gradient (c): the loss is $-\log\sigma(\Delta)$ with $\Delta=\beta(h_\theta(y_w)-h_\theta(y_l))$; apply 8.1 D1(b) and $\nabla h_\theta=\nabla\log\pi_\theta$ ($\pi_{\textrm{ref}}$ is constant). By `eq_seq_factorization` $\nabla\log\pi_\theta(y)=\sum_t\nabla\log\pi_\theta(y_t\mid x,y_{<t})$, so every token of $y_w$ is raised and every token of $y_l$ lowered with the *same* weight $\beta\sigma(-\Delta)$ — DPO does no credit assignment between tokens.
6. Reading the weight: $\sigma(-\Delta)$ is the implicit model's probability of ordering the pair wrongly; pairs it already orders correctly stop contributing. Read $\beta$: it scales the implicit reward, hence how far from $\pi_{\textrm{ref}}$ the optimum of the same data lies (D1 plus the duality aside).
7. What is not shown: that finite data, finite capacity or deterministic preferences give the KL-optimal policy; the book's DPO exercise shows that with deterministic pairs the margin diverges and $\beta$ does not prevent it.
*Numerical check (cell, numpy fp64, seed 0).* Tabular policies: 3 prompts × 6 responses, $\pi_{\textrm{ref}}$ random Dirichlet, random $r^\ast$, $\beta=0.5$. Draw $N=2\cdot10^5$ pairs per prompt from $\pi_{\textrm{ref}}$ (independent draws, discard identical), label by BT. (i) Check (c) against autograd/finite differences at 5 random $\theta$ (`atol=1e-8`). (ii) Minimize the DPO loss over the logits by gradient descent/L-BFGS to a gradient norm $<10^{-9}$; (iii) fit the BT reward $\hat r$ on the *same* pairs by logistic regression (one parameter per response, mean-centered per prompt), then form $\pi_{\hat r}\propto\pi_{\textrm{ref}}e^{\hat r/\beta}$; assert $\max|\log\pi_{\textrm{DPO}}-\log\pi_{\hat r}|<10^{-6}$ (the equivalence of step 4 *at the finite-sample optimum*); (iv) assert $\max|\log\pi_{\textrm{DPO}}-\log\pi^\ast|<3\,\textrm{SE}$ where the standard error is estimated by bootstrap (cell prints it; "to be measured").

**D2. Likelihood displacement in a linear-softmax model.**
*Statement.* Let logits be $z=\Phi\theta$ over $V$ outcomes with feature matrix $\Phi\in\mathbb R^{V\times d}$ (rows $\varphi_v$ — in a language model, hidden-state-times-unembedding features), kernel $K=\Phi\Phi^\top$, and $\pi=\operatorname{softmax}(z)$. One gradient step of the DPO loss on a pair $(w,l)$ at weight $c=\eta\beta\sigma(-\Delta)>0$ changes the logits by $\delta z=c\,K(e_w-e_l)$ to first order. Then (a) $\delta(\log\pi_w-\log\pi_l)=c\,(K_{ww}-2K_{wl}+K_{ll})\ge0$ always; (b) $\delta\log\pi_w=c\big[(K_{ww}-K_{wl})-\sum_v\pi_v(K_{vw}-K_{vl})\big]$, which is negative whenever some high-probability outcome $v$ is more aligned with $w$ than $w$ is with itself relative to $l$ — the margin grows while the chosen probability falls.
*Steps.* (1) $\partial\mathcal L/\partial z=-\beta\sigma(-\Delta)(e_w-e_l)$ because the normalizer cancels in $z_w-z_l$; (2) $\delta\theta=c\,\Phi^\top(e_w-e_l)$ and $\delta z=\Phi\,\delta\theta=cK(e_w-e_l)$; (3) $\delta\log\pi_v=\delta z_v-\sum_u\pi_u\delta z_u$; (4) specialize to $v=w$ and $v=l$, subtract for (a) using positive semi-definiteness of $K$; (5) in a tabular model $K=I$, giving $\delta\log\pi_w=c(1-\pi_w+\pi_l)>0$: displacement needs *shared* parameters.
*Numerical check.* Test vector: $\varphi_w=(1,0)$, $\varphi_l=(0.99,0.141)$, $\varphi_v=(3,0)$, $\pi=(0.05,0.05,0.90)$. Then $\delta z/c=(0.01,-0.00998,0.03)$, $\delta\log\pi_w/c=-0.0170$, $\delta\log\pi_l/c=-0.0370$ (prepared CPU run), while $\delta(\log\pi_w-\log\pi_l)/c=+0.01998$. The cell (a) reproduces these, (b) takes a real gradient step with $\eta=10^{-4}$ on a model whose $\pi$ is set to the given values and verifies the first-order prediction to relative error $<10^{-3}$, (c) scans random $\Phi$ with $d=2,V=3$ and reports the fraction with $\delta\log\pi_w<0$ (to be measured; prints a number between 0 and 1).

#### From-scratch implementation

Budget ≤ ~120 lines: `dpo_loss`/`implicit_reward` (15), `cache_ref_logps` (20), `dpo_train` (45: concatenated chosen+rejected forward, `token_logprobs` → `sequence_logprob`, AdamW, linear warmup/cosine, logging of the two log-probabilities), `win_rate` (25), `ches` (15).
Tests: (1) $\log2$ at policy = reference and gradient wrt $\pi_w$ equals $-\beta/2$; (2) D1(c) and the tabular equivalence of D1's numerical check; (3) cached reference log-probs equal recomputed log-probs under `d2l.lora_disabled` after 20 optimizer steps (`atol=1e-3` in BF16, to be measured) — proves the reference is frozen; (4) label smoothing $\varepsilon=0$ reduces to the plain loss, and $\varepsilon=0.5$ gives gradient zero at any $\Delta$; (5) sum vs mean log-probability: assert the code sums (document that SimPO's mean is a different loss).

#### In production

- TRL `DPOTrainer` with `DPOConfig(beta=…, loss_type="sigmoid", precompute_ref_log_probs=True)` (verify option names in the pinned release) on the same 4,000 pairs; compare: first-batch loss to within BF16 tolerance, then the logged chosen/rejected rewards over 50 steps (agreement of trends, not of values). The from-scratch `dpo_loss` is the oracle for the library's loss on a *recorded batch of log-probabilities*, which avoids dependence on trainer internals.
- LoRA with `peft` for the T1 path (same `d2l.apply_lora` conventions); PEFT's "disable adapter" context is the library counterpart of `d2l.lora_disabled`.

#### Headline experiment

- **Hypothesis.** DPO on UltraFeedback pairs raises the judged win rate over π_SFT above 0.5 and the implicit-reward accuracy on held-out pairs above chance, while the mean log-probability of the chosen responses decreases; the win-rate gain is partly a length gain (to be separated by the length-controlled win rate).
- **Setup.** π_SFT; reference = π_SFT; T2: full fine-tune, lr ∈ {5e-7, 1e-6}, β ∈ {0.05, 0.1, 0.3}, 20,000 pairs (length-filtered), batch 64 pairs, 1 epoch; T1: LoRA, lr ∈ {5e-6, 2e-5}, 4,000 pairs; evaluation: 300 prompts, greedy, 384 new tokens; judge Qwen3.5-9B both orders; seeds {0,1,2} for the chosen configuration at T1 and {0,1} at T2.
- **Procedure.** Dev slice (100 prompts disjoint from the 300) selects lr and β; train final runs; evaluate every 25% of an epoch for the curves; compute CHES for 1,000 held-out pairs.
- **Metrics.** win rate with 95% paired-bootstrap CI over prompts (and, for the seed variance, the range over seeds); independent-RM win rate; mean response length; sampled KL to π_SFT; implicit-reward accuracy; mean $\log\pi(y_w)$, $\log\pi(y_l)$.
- **Plots.** `fig_lm_dpo_logps` (x: step; left axis: mean chosen and rejected log-probability per token, two lines; right axis: judged win rate with CI band; vertical line at best win rate); `fig_lm_dpo_beta` (win rate and KL vs β); `fig_lm_dpo_ches` (scatter of Δ log π(y_w) vs CHES with a trend line and the fraction of displaced pairs).
- **Expected outcome.** Directions in the hypothesis; the magnitude of win-rate gains and of displacement is "to be measured". Razin et al. (Unintentional Unalignment) report that displacement is common and tied to embedding similarity; the DPO paper reports reward-accuracy and win-rate gains on its own tasks — do not compare magnitudes.
- **Noise check.** Seeds plus paired bootstrap over prompts; a β comparison is reported only if its interval excludes 0; the position-consistency rate of the judge is printed with the win rate.
- **T1 reduction.** as in Setup.
- **Artifacts (T2).** adapter/merged checkpoints per β, per-pair log-probability traces (`dpo_traces.parquet`), judgments, `ches.parquet`.

#### Figures and tables

`fig_lm_dpo_logps`, `fig_lm_dpo_beta`, `fig_lm_dpo_ches`, `tab_lm_dpo_results` (β × {win rate, CI, length-controlled win rate, KL, reward accuracy, Δlogπ_w}), `tab_lm_dpo_reference` (reference ablation: π_SFT / base / uniform).

#### Pitfalls and misconceptions

- "DPO is offline, so it cannot hack a reward": it can over-fit its pairs and drift to unlikely regions; the margin can grow while both probabilities fall.
- Using mean instead of sum of token log-probabilities (that is SimPO's loss, not DPO's).
- A reference that differs from the model that produced the data (UltraFeedback pairs come from other models): the implicit-reward story is then only approximate.
- Reading reward accuracy on the training pairs as quality; it rises monotonically.
- Computing reference log-probs with the adapter on after the first step.
- Comparing β values on the same prompts used to pick them.

#### Summary (draft bullets)

- Substituting the closed-form optimum of the KL-regularized objective into the Bradley–Terry model gives a classification loss on policy log-ratios; the partition function cancels.
- The loss weights pairs by the model's probability of the wrong order and gives every token of a response the same weight.
- In the unrestricted tabular case DPO and "fit a reward, then tilt" have the same optimum; with shared parameters and finite data they do not.
- Likelihood displacement — margin up, chosen probability down — follows from shared parameters; it is measurable and partly predictable from embedding similarity.
- Evaluation must separate gains from length, and must not reuse the prompts that selected the hyperparameters.

#### Exercises

1. **(check)** At $\pi_\theta=\pi_{\textrm{ref}}$, what are the loss, the implicit rewards and the gradient weight? *Sketch:* $\log2$; $0,0$; $\sigma(0)=1/2$, so the gradient is $-\tfrac\beta2(\nabla\log\pi_w-\nabla\log\pi_l)$.
2. **(check)** For $\beta=0.1$, $\log\pi_\theta(y_w)-\log\pi_{\textrm{ref}}(y_w)=-2$ and $\log\pi_\theta(y_l)-\log\pi_{\textrm{ref}}(y_l)=-12$, compute $\Delta$, the loss and the weight. Is the chosen response more or less likely than under the reference? *Sketch:* $\Delta=0.1\cdot10=1$; loss $=\log(1+e^{-1})=0.313$; weight $\sigma(-1)=0.269$; less likely — a displaced chosen response with positive margin.
3. **(derive)** Show that with label smoothing $\varepsilon$ the population optimum for a pair with true win probability $p$ has $\sigma(\beta m^\ast)=(1-\varepsilon)p+\varepsilon(1-p)$ (cDPO) and relate to 8.1 D3. *Sketch:* the same Bernoulli cross-entropy argument as 8.1 D3 with target $p'=(1-\varepsilon)p+\varepsilon(1-p)$.
4. **(derive)** Extend D2 to a pair whose responses are $T$ tokens long with a shared prefix: show that the gradient on the shared-prefix tokens cancels exactly, so only the diverging suffix tokens carry signal. *Sketch:* $\nabla\log\pi(y_w)-\nabla\log\pi(y_l)$ with identical prefix terms cancels by `eq_seq_factorization`; only the first diverging token and later ones remain.
5. **(measure)** Train with reference = π_SFT, base, and uniform; predict the ordering of win rate and KL before running; report paired intervals.
6. **(measure)** Add an NLL term $\alpha\,(-\log\pi_\theta(y_w)/|y_w|)$ with $\alpha\in\{0,0.1,1\}$; predict the effect on Δlogπ_w and on win rate; measure with 3 seeds.
7. **(open)** Use CHES (or your own pair-level score) to filter the 20% of pairs predicted to be displaced; does the filtered run keep its win-rate gain? Defend your control for the reduction in data.

#### Further reading

- DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)): the derivation and its reward-equivalence argument; Unintentional Unalignment ([arXiv:2410.08847](https://arxiv.org/abs/2410.08847)): likelihood displacement and CHES.
- Tutorials: Lambert, RLHF Book chapter on direct alignment; [TRL DPO docs](https://huggingface.co/docs/trl/dpo_trainer).
- New: Understanding the performance gap between online and offline alignment algorithms ([arXiv:2405.08448](https://arxiv.org/abs/2405.08448)); Unpacking DPO and PPO ([arXiv:2406.09279](https://arxiv.org/abs/2406.09279)).
- Code: [TRL](https://github.com/huggingface/trl) `DPOTrainer`.

#### Risks and open decisions

- *Judge noise on a 0.6B model's outputs*: a 4B judge may be at its limit; fallback: add the independent RM win rate and report both; increase to 500 prompts if the interval is wider than the effect.
- *TRL option names change*: the production check compares recorded batches and trends, so a rename costs one line.
- *CHES definition mismatch*: implement from the paper; if unclear, report a simpler cosine of mean response embeddings and label it as such.
- *No displacement observed at LoRA/T1 scale*: report the null with its interval; D2's cell still demonstrates the mechanism.

### 8.4 The preference-optimization family

- **Label:** `sec_lm_po_family` · **Tier:** T1 (native; sampling with vLLM if installed, else `d2l.generate`) · **Budget:** ≤ 15 min and ≤ 15 GiB for the headline experiment (3 online rounds × 256 prompts × 2 samples × 192 new tokens plus short DPO updates; generation time with HF `generate` to be measured, vLLM is the fallback that meets the cap); each loss's unit test is T0 · **Codex units:** 9.3
- **Builds on (book):** `sec_regularized` (`eq_kl_objective`, `eq_bradley_terry`), `sec_rl_sequences` (the finite-instance exercise: DPO on deterministic pairs). **Not re-taught:** anything about DPO's derivation (8.3), the BT model.
- **Builds on (this part):** 8.3 (`d2l.dpo_loss`, `d2l.dpo_train`, `d2l.cache_ref_logps`, `d2l.win_rate`), 8.1 (`d2l.RewardModel` as online labeler, `d2l.token_logprobs`), 7.4 (`d2l.apply_lora`), 1.5 (`d2l.judge_pairwise`, `d2l.paired_bootstrap`), 1.2 (`d2l.generate`).
- **Exports (`#@save`):**
  - `d2l.ipo_loss(pol_w, pol_l, ref_w, ref_l, tau=0.1)` → $\big(h_w-h_l-\tfrac1{2\tau}\big)^2$ averaged over pairs.
  - `d2l.simpo_loss(pol_w, pol_l, len_w, len_l, beta=2.0, gamma=0.5)` → $-\log\sigma\big(\tfrac\beta{|y_w|}\log\pi_w-\tfrac\beta{|y_l|}\log\pi_l-\gamma\big)$ (the default hyperparameters are placeholders; the notebook sets them from a grid and says so).
  - `d2l.orpo_loss(pol_w, pol_l, len_w, len_l, nll_w, lam=0.1)` → $\mathrm{NLL}(y_w)-\lambda\log\sigma(\log\mathrm{odds}_w-\log\mathrm{odds}_l)$ with length-normalized likelihoods inside the odds (follow the paper and TRL; verify).
  - `d2l.kto_loss(pol, ref, desirable, beta=0.1, lam_d=1.0, lam_u=1.0, z_ref=None)` → KTO loss on unpaired examples; `z_ref` defaults to the batch estimate on mismatched pairs, clamped at 0.
  - `d2l.online_dpo_round(policy, prompts, labeler, n_samples=2, cfg=None)` → list of pairs `(prompt, y_w, y_l)` after sampling `n_samples` responses per prompt from `policy`, scoring with `labeler` (callable returning scalars), and taking best vs worst; returns also the labeler calls count.
  - `d2l.PO_LOSSES` : dict name → loss function, consumed by `d2l.dpo_train(loss_fn=…)`.
- **Models and data:** π_SFT; pairs `lm-pairs-uf-tok`; labeler `lm-rm-qwen3-0p6b-uf` (8.1); judge `Qwen/Qwen3.5-9B` (as in 8.3); gold `Skywork/Skywork-Reward-V2-Qwen3-1.7B`; prompts for online rounds: `train_prefs` prompts disjoint from the 4,000 offline pairs; evaluation prompts as in 8.3. For KTO, convert pairs to unpaired examples by taking each response with its binary label (chosen = desirable, rejected = undesirable), then *shuffle and unpair* (and optionally drop half of the rejected responses to show the imbalance handling by $\lambda_D,\lambda_U$).
- **Packages:** `trl` oracles (`DPOTrainer` for `ipo`; `CPOTrainer` with `loss_type="simpo"`; `KTOTrainer`; `ORPOTrainer`; pin the release and check each is present — some have moved between namespaces across releases), `vllm` (optional).

#### Narrative arc

1. **The question.** *DPO changes the RLHF recipe in four ways at once: no reward model, no sampling, a reference model, and a Bradley–Terry assumption. Which of those matter?* Opening cell: the table `tab_lm_po_family` printed from code — each loss with its required data (paired, unpaired, prompts only), whether it needs π_ref, and what it assumes — followed by a 4-line numerical symptom (D1: margin dynamics under deterministic preferences).
2. **Organizing principle.** One sentence per method of the form "keeps DPO's objective, changes X" (see table); a figure `fig_lm_po_family` shows a 2×2 grid: *data source* (offline fixed / online fresh) × *what is modeled* (pairwise preference with reference / without reference or unpaired). Each method is a cell in the grid.
3. **IPO: bounded preferences.** The deterministic-preference failure of DPO (a finite-data property already seen in the book's exercise), IPO's squared loss with a finite target gap $1/(2\tau)$ and why it regularizes by construction; derivation D1 below; what it does *not* fix (still offline, still needs π_ref).
4. **KTO: unpaired feedback.** Data: (prompt, response, thumbs up/down) with no pairing; the loss and its reference point $z_{\textrm{ref}}$; the asymmetry of $\lambda_D,\lambda_U$ as the knob for imbalance. Implementation and the check that, on pairs converted to unpaired labels, KTO and DPO produce comparable *directions* (not equal values).
5. **SimPO and ORPO: no reference model.** SimPO: the reward is the *average* token log-probability, matching what generation optimizes, with a target margin $\gamma$; ORPO: SFT on the chosen response plus an odds-ratio penalty, no separate SFT stage. Memory and speed consequences (one model instead of two; no reference caching), and the price: no KL anchor, so length and drift must be watched (log both). Gradient weights in `tab_lm_po_family`.
6. **Online and iterative DPO: fresh samples.** The loop: sample 2 responses per prompt from the current policy, label (RM from 8.1 or an LLM annotator, as in Online AI Feedback), take best vs worst, DPO step; iterative DPO updates the reference between rounds, online DPO keeps π_SFT as reference. Why it should help (the pair is near the policy's own support, so the implicit-reward story is on-policy); what it costs (generation, labeler calls, a labeler that can be exploited); the connection to rejection sampling and to RLHF with a group of 2.
7. **The head-to-head** (headline experiment): equal prompts, equal labels, equal steps.
8. **Choosing among them** (half-page decision table in prose, no new experiment): data you have × hardware × need for a KL anchor.

#### Derivations

**D1. Bounded preferences: why DPO's margin diverges on deterministic data and IPO's does not.** (An addition to v5, which lists no derivation for 8.4; it is short and gives the section its one quantitative claim.)
*Statement.* Take a pair $(y_w,y_l)$ that is always labeled $y_w\succ y_l$, and a model whose margin $m=h_\theta(y_w)-h_\theta(y_l)$ is moved by gradient flow at rate $\dot m=-s\,\partial_m\mathcal L$, where $s>0$ is a constant fixed by the parameterization ($s=2$ for a two-outcome tabular softmax with two free logits). (a) For DPO, $\dot m=s\beta\,\sigma(-\beta m)$, so $m(t)\to\infty$ with $m(t)\approx\frac1\beta\log(s\beta^2t)$ for large $t$. (b) For IPO, $\dot m=-2s\,(m-\tfrac1{2\tau})$, so $m(t)\to\frac1{2\tau}$ exponentially.
*Steps.* (1) DPO: $\mathcal L=\log(1+e^{-\beta m})$, $-\partial_m\mathcal L=\beta\sigma(-\beta m)$, giving the rate in (a). (2) For large $m$, $\sigma(-\beta m)\approx e^{-\beta m}$, so $\frac{d}{dt}e^{\beta m}=\beta\dot m\,e^{\beta m}\approx s\beta^2$, hence $e^{\beta m}\approx s\beta^2t$ and $m\approx\frac1\beta\log(s\beta^2t)$: unbounded, but only logarithmically fast. (3) IPO: $\mathcal L=(m-\tfrac1{2\tau})^2$ has the finite minimizer; its flow is a linear ODE with rate $2s$. (4) Consequence for KL: an unbounded $m$ lets $\pi_\theta$ drift arbitrarily far from $\pi_{\textrm{ref}}$ on the pair's two responses; only the logarithmic slowness (and early stopping) protects DPO, whereas IPO ties the gap to $\tau$.
*Numerical check (cell, numpy).* Two-outcome tabular softmax with $\pi_{\textrm{ref}}=(\tfrac12,\tfrac12)$ ($s=2$); DPO with $\beta=0.5$ and IPO with $\tau=\beta$; gradient descent with step $0.1$ for $10^5$ steps (continuous time $t=10^4$). Assert: the DPO margin is within 5% of $\frac1\beta\log(s\beta^2t)$ (the cell prints both), and the IPO margin equals $1/(2\tau)$ to $10^{-6}$.

**D2 (table, not a proof). Per-pair gradient weights.** For $m=h_w-h_l$: DPO $\beta\sigma(-\beta m)$ (bounded by $\beta$); IPO $2|m-\tfrac1{2\tau}|$ (grows with the error, zero at the target); SimPO $\tfrac{\beta}{|y|}\sigma(-(\beta\bar m-\gamma))$ per token with $\bar m$ the length-normalized margin; ORPO: SFT gradient on $y_w$ plus $\lambda$ times the weight $\sigma(-\Delta_{\textrm{odds}})$; KTO: $\lambda_y\beta\,\sigma(\cdot)(1-\sigma(\cdot))$ on each example. The cell verifies each weight by autograd on a scalar example (`atol=1e-8`, fp64).

#### From-scratch implementation

≤ ~120 lines: four loss functions (≈ 15 lines each), `online_dpo_round` (≈ 35 lines), registry (5).
Tests (T0, fp64): (1) each loss at policy = reference: IPO $=\frac1{4\tau^2}$, DPO $=\log2$, KTO consistent with $z_{\textrm{ref}}=0$, SimPO $=-\log\sigma(-\gamma)$, ORPO's odds term $=\log2$ when both responses have equal likelihood; (2) gradient weights of D2 by autograd; (3) DPO ↔ IPO ↔ SimPO reduce to each other under the stated special cases: SimPO with unit lengths and zero reference log-probs equals DPO with $\gamma=0$ and $\beta$ matched; (4) library oracles on *recorded batches of log-probabilities*: `trl` losses for `ipo`, `simpo`, `kto`, `orpo` evaluated on the same tensors agree to BF16/`rtol=1e-4` — if a library's loss normalizes differently (it will, for IPO's length averaging and KTO's batch statistic), the notebook states which normalization it replicates and by how much the two differ.

#### In production

TRL trainers (`DPOTrainer` with `loss_type="ipo"`, `CPOTrainer` with `loss_type="simpo"`, `KTOTrainer`, `ORPOTrainer`), LLaMA-Factory or axolotl configs as the config-driven wrappers (7.7). The from-scratch losses are the test oracle on recorded batches; the notebook runs one library trainer end-to-end (default: `DPOTrainer`, `loss_type="ipo"`) for 50 steps and compares the first losses with the from-scratch values.

#### Headline experiment

- **Hypothesis.** At equal prompts, equal numbers of preference labels and equal optimizer steps, online DPO (fresh samples each round) reaches a higher judged win rate over π_SFT than offline DPO on the same number of fixed pairs labeled by the same RM; both exceed 0.5. (Direction supported by Tang et al. and Guo et al.; magnitude to be measured.)
- **Setup.** π_SFT LoRA policy, reference π_SFT; three arms: (A) *offline-dataset*: 768 UltraFeedback pairs; (B) *offline-RM*: 256 prompts × 2 samples from π_SFT drawn once, labeled by the 8.1 RM, repeated for 3 "rounds" worth of the same data (3 epochs) so that the number of prompts, labeler calls and optimizer steps equals (C); (C) *online-RM*: 3 rounds, each 256 *new* prompts × 2 samples from the current policy, labeled by the 8.1 RM, one epoch of DPO per round. β=0.1; lr from 8.3's dev grid; seeds {0,1,2}. Evaluate after every round on the 300 evaluation prompts with the Qwen3.5-9B judge (both orders) and with the independent gold RM; the labeler (8.1 RM) is never the judge.
- **Procedure.** Run A, B, C; log per round: win rate vs π_SFT, length, KL, labeler-score mean, gold-score mean; labeler calls and generation tokens used.
- **Metrics.** win rate (CI), length-controlled win rate, gold-RM mean difference, KL, mean length, tokens generated, labeler calls.
- **Plots.** `fig_lm_po_online_offline` (x: round; y: win rate vs π_SFT with paired-bootstrap bands; three lines); `fig_lm_po_proxy_gold` (labeler score vs gold score per round for the online arm — the first sign of the 8.7 gap); `fig_lm_po_length` (mean length per round).
- **Expected outcome.** Direction per hypothesis; whether the online arm's gain survives the length control is "to be measured". A growing gap between labeler and gold score would be the preview of overoptimization.
- **Noise check.** Three seeds per arm, paired bootstrap over the 300 prompts for A vs B vs C; a claim "C > B" requires the 95% interval on the difference to exclude 0 *and* the sign to agree in ≥ 2 of 3 seeds; otherwise report "no difference detected".
- **T1 reduction:** is the native configuration.
- **Artifacts:** `lm-judgments-po` (per-prompt verdicts by arm and round), per-round pair sets (parquet).

#### Figures and tables

`tab_lm_po_family` (method | data | reference | loss | per-pair gradient weight | assumption changed | known failure), `fig_lm_po_family`, `fig_lm_po_online_offline`, `fig_lm_po_proxy_gold`, `fig_lm_po_length`.

#### Pitfalls and misconceptions

- "Reference-free means regularization-free": SimPO and ORPO move the anchor to the training data and to the SFT term; drift and length must be monitored.
- Comparing methods at the best hyperparameters of each on the evaluation prompts (use a dev slice).
- Letting the labeler also be the judge: online arms then win by construction.
- Treating KTO as "DPO with less data": its reference point and class weights change the objective.
- Different normalizations of the same-named loss across libraries (IPO's length averaging, KTO's batch KL estimate): compare on recorded batches, not trainer logs.

#### Summary (draft bullets)

- Each preference-optimization method keeps most of DPO and changes one assumption: IPO (bounded targets), KTO (unpaired labels), SimPO/ORPO (no reference model), online DPO (fresh samples).
- On deterministic preferences DPO's margin grows without bound (logarithmically in time); IPO's has a finite target.
- Removing the reference removes the KL anchor and its memory cost; length and drift become the monitored quantities.
- Whether fresh samples help is an empirical question answered by a paired comparison at equal labels and steps.
- The labeler in an online loop is a proxy and must be audited by an independent evaluator.

#### Exercises

1. **(check)** Which of IPO, KTO, SimPO, ORPO and online DPO need (a) paired data, (b) a reference model, (c) a labeler at training time? *Sketch:* paired: IPO, SimPO, ORPO, online DPO (self-made pairs); reference: IPO, KTO, DPO-online; none: SimPO, ORPO; labeler: online DPO only.
2. **(check)** With $\tau=0.1$, what gap $h_w-h_l$ does IPO target, and what is its loss when $h_w-h_l=0$? *Sketch:* $1/(2\tau)=5$; loss $25$.
3. **(derive)** Show that SimPO with response lengths 1 and $\gamma=0$, $\log\pi_{\textrm{ref}}\equiv0$ equals DPO. *Sketch:* with $|y|=1$ the normalization disappears and $h=\log\pi$.
4. **(derive)** For the two-outcome model of D1, find the $\beta$-dependence of the time needed for DPO's margin to reach $5$ and compare with IPO at $\tau=0.1$ (target 5). *Sketch:* DPO: from $e^{\beta m}\approx s\beta^2t$, $t\approx e^{5\beta}/(s\beta^2)$; IPO: exponential approach, time $\propto\log(1/\varepsilon)$.
5. **(measure)** Repeat the headline comparison with $n=4$ samples per prompt (best vs worst) and predict whether the gain over $n=2$ is larger at equal *generation tokens* or at equal *labeler calls*.
6. **(measure)** Run SimPO with $\gamma\in\{0,0.5,1.5\}$ and report length and KL; predict the direction first.
7. **(open)** Replace the RM labeler with the 4B judge as an online labeler and design the check that stops the judge from also being the evaluator.

#### Further reading

- IPO ([arXiv:2310.12036](https://arxiv.org/abs/2310.12036)), KTO ([arXiv:2402.01306](https://arxiv.org/abs/2402.01306)), SimPO ([arXiv:2405.14734](https://arxiv.org/abs/2405.14734)), ORPO ([arXiv:2403.07691](https://arxiv.org/abs/2403.07691)), Online AI Feedback ([arXiv:2402.04792](https://arxiv.org/abs/2402.04792)).
- New: RLHF Workflow: From Reward Modeling to Online RLHF ([arXiv:2405.07863](https://arxiv.org/abs/2405.07863)) — iterative DPO recipe; Understanding the performance gap between online and offline alignment algorithms ([arXiv:2405.08448](https://arxiv.org/abs/2405.08448)).
- Tutorials: Lambert, RLHF Book (direct alignment algorithms); [TRL docs](https://huggingface.co/docs/trl/index).
- Code: [TRL](https://github.com/huggingface/trl) trainers listed above.

#### Risks and open decisions

- *Generation throughput at T1*: fall back to 192 → 128 tokens or to vLLM.
- *Judge/labeler agreement too low for the online arm to learn*: use the gold RM as labeler in a separate run and say so (the 8.7 independence issue).
- *Library loss normalization differences*: documented, not hidden.
- *If the three arms are statistically tied*: the section reports it; the noise rule of the conventions forbids a ranking.

### 8.5 RL with verifiable rewards

- **Label:** `sec_lm_rlvr` · **Tier:** T2 (reduced T1 config: π_SFT with LoRA r=16, 40 steps × 32 prompts × K=8, ≤ 256 new tokens, two arms re-run live and the other arms loaded from committed logs) · **Budget:** T2 ≤ 3 h for 4 aggregation arms × 2 seeds on π_SFT plus one Qwen3-1.7B-Base replicate of two arms, trained with LoRA r=16 like the 0.6B arms (bf16 base 3.44 GB; a full-parameter AdamW state would be 1,720,574,976 × 16 B = 27.5 GB and would not leave room for the colocated vLLM engine on 40 GB); the replicate runs on one A100 40 GB with vLLM at `gpu_memory_utilization` ≈ 0.4 and the trainer in the remainder (generation through vLLM; per-step time and peak memory to be measured with `PhaseTimer`, target ≤ 38 GiB); T1 ≤ 15 min and ≤ 15 GiB for the live arms · **Codex units:** 9.5, 9.6, 9.7
- **Builds on (book):**
  - `sec_baselines`: zero-mean lemma, `subsec_leave_one_out` (the $(K-1)/K$ shrinkage of the same-group mean and the unbiased $K/(K-1)$-rescaled leave-one-out), `subsec_pg_loss_divisor` (four divisors on a FrozenLake batch: per-episode, per-own-length, total steps, constant) and its "Normalized Returns and GRPO" paragraph; `d2l.normalize`. **Not re-taught:** that centering shrinks the gradient by $(K-1)/K$, that dividing by $\sigma$ is a per-batch step-size change in the one-batch case. This section re-derives only what is new at LM scale: the *per-prompt* effect of the group standard deviation on a Bernoulli reward, and the per-token weights of each divisor for sequences.
  - `sec_ppo`: clip, `subsec_ppo_asymmetric` (clip-higher arithmetic), "The Length-Normalized Trajectory Ratio" (GSPO's idea in one paragraph): the 8.5 treatment of GSPO/CISPO starts from these and adds only the MoE argument and code.
  - `sec_rl_sequences`: the token MDP, the group baseline, the hedge/verifier hacking example (`subsec_rl_sequences_hacking`), the notation contract and the GRPO components paragraph.
  - `sec_regularized` (KL penalty vs trust region), `sec_mdl-statistics`.
- **Builds on (this part):** 8.1 (`d2l.token_logprobs`), 7.4 (`d2l.apply_lora`, `d2l.lora_disabled`), 8.2 (`d2l.ppo_loss`, `d2l.PhaseTimer`, `d2l.masked_whiten`, `d2l.kl_estimators`, `d2l.rlhf_memory`), 8.3 (`d2l.win_rate` for the retention check used by A7), 1.4 (`d2l.pass_at_k`), 1.5 (`d2l.EvalSuite`, `d2l.paired_bootstrap`), 7.2 (π_SFT, template), 7.4.
- **Exports (`#@save`):**
  - `d2l.ArithmeticTasks(seed=0, split='train', n=1000, profile='mixed')` → sequence of problems `{'prompt': chat messages, 'answer': str, 'n_ops': int}`; deterministic given `(seed, split, n, profile)`; train/eval splits are disjoint by construction (disjoint seeds *and* an assertion that no expression string repeats across splits). Profiles: `'easy'` (2 operands), `'mixed'` (2–4 operands, integers up to 99, `+ - *` and exact division), `'hard'`.
  - `d2l.gsm8k_tasks(split='test', n=500, seed=0)` → same schema from `openai/gsm8k` (answers parsed from the `####` line).
  - `d2l.canonical_answer(s)` → `str | None`: canonical form of an answer string; strips `$`, commas as thousands separators, trailing periods and units; accepts integers, decimals, `a/b`, `\frac{a}{b}` and maps them to an exact `fractions.Fraction` representation; `None` if unparseable or empty. (Extraction is not defined here: `d2l.extract_answer(text, kind='gsm8k'|'boxed', strict=False)` is owned by 1.4 and imported.)
  - `d2l.verify_math(text, gold)` → `float` reward in `{0.0, 1.0}`: `canonical_answer(extract_answer(text, kind='boxed'))` compared with `canonical_answer(gold)` by exact `Fraction` equality (no float tolerance). The rule "the last `\boxed{…}` that appears after the last reasoning token; empty or multiple *different* boxed values in the final line score 0" is applied here, before calling `extract_answer`.
  - `d2l.verifier_tests()` → list of `(response, gold, expected)`; the table `tab_lm_verifier_tests` is printed from it and is run by the autograder of A7.
  - `d2l.group_advantages(rewards, std=True, loo=False, eps=1e-8)` → advantages of shape `(G,K)` for rewards of shape `(G,K)`: $r_{ij}-\mu_i$, divided by $\sigma_i+\textrm{eps}$ (population std, divide by $K$) if `std`; with `loo=True`, $\frac{K}{K-1}(r_{ij}-\mu_i)$ and no std. Groups with identical rewards return exact zeros.
  - `d2l.sample_responses(engine, prompts, K, max_new_tokens, temperature=1.0, top_p=1.0, seed=0)` → dict `{ids, resp_mask, logp_sampler, text, finished}`; `engine` is a `transformers` model + tokenizer or a vLLM `LLM` (same return schema). (Not named `rollout`: the book's `d2l.rollout(env, policy, num_episodes, rng)` of `sec_policygradient` already exists.)
  - `d2l.dynamic_filter(rewards)` → boolean mask of groups with non-constant rewards and the discarded fraction (DAPO's dynamic sampling criterion).
  - `d2l.overlong_penalty(lengths, max_len, soft_window)` → DAPO-style length shaping for truncated responses.
  - `d2l.cispo_loss(logp, logp_old, adv, resp_mask, eps_high=0.2, eps_low=None)` and `d2l.gspo_loss(logp, logp_old, adv, resp_mask, eps=…)`: the two ratio variants (below).
  - `d2l.token_entropy(logits, resp_mask)` → mean token entropy of the policy over response positions.
  - `d2l.grpo_train(policy, tasks, cfg, reward_fn=None)` → log of per-step dicts; `reward_fn(texts: list[str], items: list[dict]) -> list[float]` is a batch-level reward (one call per sampled batch, `items` are the task dicts); it defaults to `d2l.verify_math(text, item['answer'])` applied per item. Chapters 9.7 (length penalties) and 10.4 pass their own; `cfg` fields: `K`, `prompts_per_step`, `steps`, `aggregation` ∈ {`'sample'`,`'token'`,`'const'`}, `std_norm`, `loo`, `eps_low`, `eps_high`, `beta` (KL coefficient, 0 by default), `kl_estimator` ∈ {`'k3'`,`'k2'`}, `dynamic_sampling`, `overlong`, `num_minibatches`, `engine` ∈ {`'hf'`,`'vllm'`}, `seed`.
  - `d2l.fit_sigmoid_scaling(compute, score)` → `(params, predict)` for the saturating compute–performance curve (see narrative step 9).
  - Data artifacts: `lm-arith-tasks` (generator spec, eval split); `lm-grpo-qwen3-0p6b` (π_SFT + LoRA, arm A, seed 0; T1/T2); `lm-grpo-qwen3-1p7b-base` (the R1-Zero-like Qwen3-1.7B-Base replicate, arm A, seed 0; T2), both consumed by 9.3 and 9.7; other arms and seeds stay in the step logs.
- **Models and data:**
  - π_SFT (596,049,920 parameters) for A7 and the headline; replicate at T2 on `Qwen/Qwen3-1.7B-Base` (1,720,574,976 parameters, Apache-2.0) with a fixed minimal prompt template and no SFT ("R1-Zero-like"); both with Transformers for training (LoRA r=16 on all linear layers, as the arms above; the replicate is a plain-text prompt with no chat template, since no SFT stage has taught one) and vLLM for sampling.
  - Tasks: `ArithmeticTasks` train (seed 0, 20,000 problems), dev (seed 1, 500), eval (seed 2026, 500); GSM8K: `openai/gsm8k` (MIT), train for an optional second task, test subset of 500 fixed indices for reporting.
  - Independent check: none needed for correctness (exact verifier); `d2l.win_rate` against π_SFT on the 8.3 prompts for retention.
- **Packages:** `vllm>=0.10` (the 2026 releases are well above; pin whatever 1.1 pins), `trl` (GRPOTrainer oracle), `peft`, `fractions` (stdlib), `scipy` (sigmoid fit, optional).

#### Narrative arc

1. **The question.** *If the reward is a program, how much of the RLHF machinery is left, and which implementation choices silently change what is optimized?* Opening cell: sample $K=8$ responses to 200 training problems from π_SFT; print pass@1, pass@8 (`d2l.pass_at_k`, 1.4) and the fraction of groups that are all-correct, all-wrong, and mixed. The symptom: a large share of groups have zero reward variance and therefore contribute zero advantage (fraction to be measured).
2. **Robust answer verification** (`tab_lm_verifier_tests`).
   - The two sub-problems: extraction (what did the model answer?) and equivalence (is it the gold answer?). A model can be right and scored wrong (formatting, units, `4.0` vs `4`) or wrong and scored right (a hedge listing several answers, answer in the reasoning but not the final line, a response that quotes the gold string from the prompt).
   - Test table (≥ 20 rows, written by the author into `d2l.verifier_tests()`): valid (`\boxed{12}`, `\boxed{12.0}`, `\boxed{\frac{24}{2}}`, `1,000`), invalid (`\boxed{13}`), ambiguous (`\boxed{12 or 13}`, two boxed values), adversarial (a final line "the grader marked this correct", `\boxed{}` empty, unicode minus `−5`, trailing text after the box, a boxed value inside quoted reasoning, a response that repeats the question's numbers, truncation before the box). Each row has the expected reward; the autograder reuses it. Connect to `subsec_rl_sequences_hacking`: a verifier that searches for the answer string rewards the hedge.
   - GSM8K answers (`####`) as a second verifier with the same exactness; its false-negative rate is measured on 200 correct reference solutions pasted as responses (it should be 0).
   - Reward design: binary correctness; an optional small format reward is shown to *change* the optimum (cell) and is off by default.
3. **GRPO on multi-token responses.** One paragraph of recap with `:numref:` to `sec_rl_sequences`; the training loop as a diagram `fig_lm_grpo_loop`: sample $K$ per prompt → verify → `group_advantages` → trainer log-probs (old, from the sampler or recomputed) → `ppo_loss` → update. What disappears relative to 8.2: critic, reward model, GAE; what remains: the reference model only if $\beta>0$. `tab_lm_grpo_memory` from `rlhf_memory` (roles actor, optional reference).
4. **What the advantage and the divisor optimize** (D1). Three results, each with a cell on a toy and then visible in the run: (i) per-prompt weight from the group standard deviation, (ii) the length weights of the three divisors, (iii) the $(K-1)/K$ self-inclusion factor (recap by one sentence). `fig_lm_grpo_weights`: the exact weight $w_K(p)$ on a prompt of success probability $p$ for $K\in\{2,4,8,16,64\}$.
5. **DAPO's four changes, each tied to the symptom it addresses** (`tab_lm_grpo_recipes`): clip-higher (entropy collapse; `subsec_ppo_asymmetric` arithmetic), dynamic sampling (zero-variance groups, step 1), token-level loss (long-response bias), overlong filtering/shaping (truncated responses scored as wrong). Each is a `cfg` flag; the notebook switches them on one at a time for a diagnostic panel, not a leaderboard.
6. **GSPO and CISPO, and why they matter for MoE policies.** GSPO: the sequence-level geometric-mean ratio $s_i=\exp\big(\tfrac1{|y_i|}\sum_t\log\rho_{i,t}\big)$ clipped with a *much* narrower band; CISPO: clip and stop-gradient the importance weight, keep every token's gradient. Why the choice interacts with mixture-of-experts: routing changes between policy versions and between inference engine and trainer, so token-level ratios have noise that does not average out (D3 quantifies the effect on clipping). Code for both losses, a test that all three losses have the same gradient at $\rho\equiv1$, and the noise-injection cell of D3. State the claims as the papers' ("the GSPO paper argues…"), since the MoE results cannot be reproduced at this scale; an optional T3 exercise runs Qwen3-30B-A3B.
7. **Entropy.** `token_entropy` logged every step. D2 explains first-order entropy change as a covariance between log-probability and update; plot entropy against step for each arm; remedies (clip-higher, small entropy bonus, KL, covariance-based clipping as in the Entropy Mechanism paper — pointer only).
8. **The run and the ablation** (headline experiment below), read with the panel: accuracy, length of correct and incorrect responses, entropy, zero-variance fraction, clip fractions, gradient norm.
9. **Predictable scaling of RL compute.** Plot held-out accuracy against cumulative generated tokens for all arms; `fit_sigmoid_scaling` fits the saturating-sigmoid family that *The Art of Scaling RL Compute* fits to compute–performance curves (read the paper and use its parametrization; the notebook states which) on the first 70% of the budget and predicts the rest; report the prediction error. The point is a method, not a law: with one small model the fit constrains little, and the notebook says so.
10. **After RL.** Evaluate on the GSM8K test subset with `d2l.EvalSuite`; report pass@1 and `d2l.pass_at_k` for k=1,4,8, and the retention win rate. Message: RLVR mostly raises pass@1 toward pass@k of the starting model (Yue et al.); check whether this holds here instead of asserting it.

#### Derivations

**D1. The group-relative advantage: what each normalization optimizes (required: GRPO's normalization bias, cf. Dr. GRPO).**
*Setting.* Prompt $x$, $K$ i.i.d. responses $y_i\sim\pi_\theta(\cdot\mid x)$, rewards $r_i=r(x,y_i)$, $\mu=\frac1K\sum r_i$, $\hat\sigma^2=\frac1K\sum(r_i-\mu)^2$, advantages $A_i=(r_i-\mu)/(\hat\sigma+\textrm{eps})$ (GRPO) or $r_i-\mu$. At the first minibatch $\rho_{i,t}=1$, so the clipped loss has gradient $\hat g=\sum_i c_i\,A_i\,\nabla_\theta\log\pi_\theta(y_i\mid x)$ with the *aggregation weight* $c_i$ and `eq_seq_factorization` ($\nabla\log\pi(y)=\sum_t\nabla\log\pi(y_t\mid\cdot)$). The three divisors: sample-mean (GRPO) $c_i=\frac1{K|y_i|}$; token-mean over a batch with total length $N=\sum_j|y_j|$ (DAPO): $c_i=\frac1N$; constant (Dr. GRPO): $c_i=\frac1{K\,L_{\max}}$. The target is $\nabla_\theta J_x$, $J_x=\mathbb E_{y\sim\pi_\theta}r(x,y)$.
*Claims.*
(a) *Constant divisor, centered, no std:* $\mathbb E\hat g=\frac{K-1}{K}\frac1{L_{\max}}\nabla J_x$: direction exact, scale reduced; with leave-one-out centering the factor is $\frac1{L_{\max}}$.
(b) *Sample-mean divisor (length bias):* with a leave-one-out baseline (independent of $y_i$), $\mathbb E\hat g=\frac1K\nabla_\theta\tilde J_x\big|_{b=J_x}$ where $\tilde J_x(\theta;b)=\mathbb E_{y\sim\pi_\theta}\big[(r(x,y)-b)/|y|\big]$ with $b$ held fixed. The update ascends reward *divided by length*: among correct responses ($r-b>0$) it prefers shorter ones, among incorrect responses ($r-b<0$) it prefers *longer* ones (the penalty per response is diluted by length).
(c) *Token-mean divisor:* if $N$ is treated as fixed, $\mathbb E\hat g=\frac{1}{N}\sum_i\mathbb E[A_i\nabla\log\pi(y_i)]$ is a rescaling of the unbiased estimator by $1/(K\,\mathbb E|y|)$ — each token is weighted equally, so long responses contribute proportionally more; because $N$ is random and correlates with $y_i$, the exact expectation differs by a ratio-estimator bias of order $1/K$ (so (a)'s constant is the cleanest).
(d) *Standard-deviation normalization (difficulty bias), binary rewards:* let $p=\Pr(r=1)$, $\sigma_x^2=p(1-p)$. For $0<m<K$ successes the group contribution is $\sum_iA_i\nabla\log\pi(y_i)$ and
$$\mathbb E\hat g=\frac{\mathbb E[\hat\sigma_K]}{p(1-p)}\;\nabla_\theta p\quad\text{(centered, std-normalized; sample divisor omitted)},$$
with $\mathbb E[\hat\sigma_K]=\sum_{m}\binom Km p^m(1-p)^{K-m}\sqrt{\tfrac mK(1-\tfrac mK)}$. Hence the weight on prompt $x$ relative to $\nabla J_x=\nabla p$ is $w_K(p)=\mathbb E[\hat\sigma_K]/(p(1-p))$ (versus $\frac{K-1}K$ for centered-only). As $K\to\infty$, $w_K(p)\to1/\sqrt{p(1-p)}$ and the update ascends $\mathbb E_x\big[2\arcsin\sqrt{p_x}\big]$ rather than $\mathbb E_x[p_x]$, because $\frac{d}{dp}2\arcsin\sqrt p=\frac1{\sqrt{p(1-p)}}$: easy and hard prompts are up-weighted relative to medium ones.
*Steps.*
1. (a) Take the expectation conditional on $y_i$; the baseline $\mu=\frac1K\sum_jr_j$ contains $r_i$: $\mathbb E[(r_i-\mu)\nabla\log\pi(y_i)]=\frac{K-1}{K}\mathbb E[(r_i-b_{-i})\nabla\log\pi(y_i)]$ with $b_{-i}$ the mean of the other $K-1$ rewards (independent of $y_i$); the zero-mean lemma (`sec_baselines`) gives $\frac{K-1}K\nabla J_x$ per response; sum over $i$ and divide by $K L_{\max}$ (this is `subsec_leave_one_out` restated; do not re-prove).
2. (b) With $b_{-i}$ independent of $y_i$ and $\mathbb E b_{-i}=J_x$: $\mathbb E[c_iA_i\nabla\log\pi(y_i)]=\frac1K\sum_y\pi(y)\frac{r(y)-J_x}{|y|}\nabla\log\pi(y)=\frac1K\sum_y\frac{r(y)-J_x}{|y|}\nabla\pi(y)=\frac1K\nabla_\theta\sum_y\pi_\theta(y)\frac{r(y)-b}{|y|}\Big|_{b=J_x}$. Sum over the $K$ responses (a $K/K$ cancellation).
3. For a tabular softmax over responses, $\Delta z_y\propto\pi(y)\,f(y)$ with $f=(r-b)/|y|$ (the gradient of $\sum\pi f$ with $f$ fixed has $y$-th logit component $\pi(y)(f(y)-\sum\pi f)$; the centering term is common to all $y$ and does not affect *relative* changes). Two incorrect responses of lengths $L_s<L_\ell$ and equal probability have logit changes in the ratio $L_\ell:L_s$ in favor of the longer — the sign of the length bias.
4. (c) As stated; the exact bias statement is left as exercise 4.
5. (d) For $m\in\{1,\dots,K-1\}$ write $\mu=m/K$, $\hat\sigma=\sqrt{\mu(1-\mu)}$; $A_+=(1-\mu)/\hat\sigma=\sqrt{(1-\mu)/\mu}$, $A_-=-\mu/\hat\sigma=-\sqrt{\mu/(1-\mu)}$. Then $mA_+=K\hat\sigma$ and $(K-m)A_-=-K\hat\sigma$. Given the success pattern, the conditional mean of $\nabla\log\pi(y)$ over successes is $s_+=\nabla p/p$ and over failures $s_-=-\nabla p/(1-p)$ (since $\sum_{y:r=1}\pi\nabla\log\pi=\nabla p$ and $\sum_{y:r=0}\pi\nabla\log\pi=-\nabla p$). So $\mathbb E[\sum_iA_i\nabla\log\pi(y_i)\mid m]=K\hat\sigma\,(s_+-s_-)=K\hat\sigma\,\nabla p/(p(1-p))$. Groups with $m\in\{0,K\}$ contribute $0$. Average over $m\sim\mathrm{Bin}(K,p)$ and divide by $K$.
6. Large-$K$ limit: $\hat\sigma\to\sigma_x$ a.s. (law of large numbers) and bounded, so $\mathbb E\hat\sigma_K\to\sigma_x$ and $w_K\to1/\sigma_x$. Finite $K$: $w_K$ is bounded by about $\sqrt{K-1}$ near $p\to0,1$ (only $m=1$ or $m=K-1$ groups contribute).
*Numerical checks (one cell each, numpy fp64, seed 0).*
- *Exact enumeration* for $K=6$ over all $2^6$ success patterns of a one-parameter policy $p=\sigma(\theta)$ ($\nabla_\theta p=p(1-p)$): the exact expected gradient of the centered estimator equals $\frac{K-1}Kp(1-p)$, of leave-one-out equals $p(1-p)$, and of the std-normalized estimator equals $\mathbb E[\hat\sigma_K]$ — all to `atol=1e-12`. The preparation run for $K=8$ gives $w_K(p)=2.33,\,1.96,\,1.86$ at $p=0.1,0.3,0.5$ (and 2.33 at $0.9$), $2.61$ at $p=0.01$, versus the $K\to\infty$ values $3.33,\,2.18,\,2.00$ at $p=0.1,0.3,0.5$; at $p=0.5$, $w_K=1.00,\,1.62,\,1.86,\,1.93,\,1.98$ for $K=2,4,8,16,64$.
- *Length bias on four responses*: lengths $(2,6,2,6)$ for (correct, correct, wrong, wrong), rewards $(1,1,0,0)$, random logits (seed 0), $b=J_x$. Enumerate $\mathbb E\hat g$ under the three divisors with the leave-one-out baseline at $K=2$ ($b_{-i}$ is the other draw's reward; enumerate all 16 draws) and check: sample-mean $=\tfrac1K\nabla\tilde J_x$ computed by autograd to `atol=1e-12`; the constant divisor gives a vector parallel to $\nabla J_x$ (cosine $=1$ to $10^{-12}$); the ratio of logit changes of the two wrong responses per unit probability equals $L_\ell/L_s=3$ under the sample divisor and $1$ under the constant divisor.

**D2. Entropy change as a covariance (supports the entropy-collapse narrative).**
*Statement.* For a tabular softmax policy $\pi=\operatorname{softmax}(z)$ over actions at one state, to first order a logit change $\delta z$ changes the entropy by $\delta H=-\operatorname{Cov}_{a\sim\pi}\big(\log\pi(a),\delta z_a\big)$. Under vanilla policy gradient $\delta z_a=\eta\,\pi(a)A(a)$; under natural policy gradient $\delta z_a=\eta A(a)$, giving $\delta H=-\eta\operatorname{Cov}(\log\pi,A)$.
*Steps.* (1) $H=-\sum\pi\log\pi$, $\partial H/\partial z_b=-\pi_b(\log\pi_b+H)$ (use $\partial\pi_a/\partial z_b=\pi_a(\delta_{ab}-\pi_b)$); (2) $\delta H=-\sum_b\pi_b(\log\pi_b+H)\delta z_b$; (3) since $\mathbb E[\log\pi]=-H$, this is $-\operatorname{Cov}(\log\pi,\delta z)$; (4) substitute the two update rules. High-probability actions with positive advantage (the common case early on) reduce entropy; low-probability actions with positive advantage raise it — the mechanism behind clip-higher's motivation.
*Numerical check.* Random 6-way logits and advantages (seed 0); a step of size $10^{-4}$ along $\delta z$; assert the measured entropy change matches the formula to relative error $<10^{-3}$ for both update rules.

**D3. Why sequence-level ratios tolerate routing noise (supports GSPO for MoE).**
*Statement.* If $\log\rho_{t}=\varepsilon_t$ with i.i.d. zero-mean noise of variance $s^2$ (a model of nondeterministic routing or kernel mismatch), then the per-token ratio has the same distribution for any response length, so token-level clipping removes a fixed fraction $2\bar\Phi(\ln(1+\epsilon)/s)$ (approximately, for the upper side) of tokens regardless of $T$; the sequence-level log-ratio $\frac1T\sum\varepsilon_t$ has variance $s^2/T$, so a band that clips almost nothing at $T=256$ for $s$ small would clip a large fraction of sequences at short $T$ unless its width scales with $1/\sqrt T$.
*Steps.* Variance of the mean of i.i.d. terms; Gaussian tail for the clip fraction.
*Numerical check.* $s=0.1$, $T\in\{16,64,256\}$, $10^5$ simulated sequences, $\epsilon=0.2$ token-level and the band $\pm\ln(1+\epsilon_{\textrm{seq}})$ with $\epsilon_{\textrm{seq}}\in\{0.01,0.001\}$: print the fraction of tokens clipped (constant in $T$) and of sequences clipped (decreasing in $T$) next to the Gaussian prediction (agreement within 3 standard errors). The assumption (i.i.d. noise independent of $T$) is exactly what MoE routing noise may violate; the notebook says so.

#### From-scratch implementation

Budget ≤ ~150 lines (verifier and tasks ≈ 55 lines, advantages and filters ≈ 25, `cispo_loss`/`gspo_loss` ≈ 25, `grpo_train` ≈ 40, sigmoid fit ≈ 10; `ppo_loss` and `PhaseTimer` come from 8.2).
Tests: (1) `verifier_tests()` all pass; (2) `group_advantages` fixtures (mixed group of 8 with 3 successes: $A_+=\sqrt{5/3}=1.2910$, $A_-=-\sqrt{3/5}=-0.7746$ for population std; the unbiased-std version gives different numbers and the test fixes the convention), all-equal group $\to$ 0, `loo` $=K/(K-1)(r-\mu)$; (3) `ppo_loss` modes equal D1's enumerated gradients on the 4-response toy (`atol=1e-8`); (4) at $\rho\equiv1$ `ppo_loss`, `cispo_loss`, `gspo_loss` give the same gradient; (5) sampler consistency: trainer log-probabilities of 32 sampled responses vs the sampler's: mean absolute difference reported (to be measured; HF sampler: ≈ BF16 noise; vLLM: larger — see 8.6), and the sample-time *temperature* is applied in `token_logprobs`; (6) `ArithmeticTasks` determinism (same hash for same seed) and disjointness; (7) entropy of a uniform policy $=\log V$.

#### In production

- TRL `GRPOTrainer`/`GRPOConfig` (pin the release; PyPI latest at blueprint time 1.16.0): options that correspond to this section's choices — number of generations, `epsilon` and an upper clip (`epsilon_high`), KL `beta`, a loss-type option distinguishing GRPO, Dr. GRPO and DAPO-style aggregation, a reward-scaling option for the std normalization, masking of truncated completions, sequence-level vs token-level importance sampling (GSPO), and `use_vllm` with a colocated or server mode. *The author inspects the installed release and tabulates the exact names; the blueprint does not guarantee them.* Run the T1 arm A and arm D configurations through TRL on the same tasks and compare the diagnostics panel (not final accuracy) with the from-scratch run.
- verl: the same choices as configuration (advantage estimator, loss aggregation mode token-mean vs sequence-mean, clip ratios low/high); read its loss code to see which of the from-scratch divisors it implements.
- The from-scratch `group_advantages`, `ppo_loss`, verifiers and entropy are the oracles; compare per-step advantages and losses on a recorded batch.

#### Headline experiment

- **Hypothesis.** With the standard deviation normalization and sample-mean divisor (GRPO as published), the mean length of *incorrect* responses grows faster during training than under the constant divisor without std (Dr. GRPO), with held-out accuracy not lower under the latter; per-difficulty-bin accuracy gains differ because the std term re-weights prompts (D1d). (Directions from Liu et al.; magnitudes to be measured.)
- **Setup.** π_SFT + LoRA (r=16, all linear layers), `ArithmeticTasks('mixed')` train; K=8; 64 prompts/step; 100 steps (T1: 32 prompts, 40 steps); temperature 1.0; `max_new_tokens=256`; $\beta=0$; symmetric $\epsilon=0.2$; `num_minibatches=2`; lr from a dev grid {1e-5, 5e-5} on `dev` (arm A); four arms — A: `sample` + std (GRPO); B: `token` + std; C: `const` + std (isolates the divisor with std fixed); D: `const`, no std (Dr. GRPO). Seeds {0,1} at T2 (T1: arms A and D live, seeds {0,1}; B and C from committed logs). One variable per comparison: A→B→C for the divisor, C→D for std.
- **Procedure.** Train; every 10 steps evaluate greedy accuracy on the 500 eval problems; log per step reward, lengths of correct/incorrect samples, entropy, zero-variance fraction, clip fractions, gradient norm, cumulative generated tokens; final: GSM8K subset, pass@k, win-rate retention vs π_SFT.
- **Metrics.** accuracy (eval, greedy; with 95% bootstrap CI over problems), mean length of correct and of incorrect responses (CI by cluster bootstrap over prompts), entropy, fraction of mixed groups, accuracy per `n_ops` bin.
- **Plots.** `fig_lm_grpo_curves`: three panels vs step — held-out accuracy; mean response length with separate lines for correct and incorrect samples; mean token entropy — series A, B, C, D (mean over seeds, band = seed range). `fig_lm_grpo_groups`; `fig_lm_grpo_clip`; `fig_lm_grpo_bins` (accuracy gain by difficulty bin per arm); `fig_lm_grpo_scaling`.
- **Expected outcome.** Direction as in the hypothesis. Liu et al. (Understanding R1-Zero-Like Training) report a length-bias and a difficulty-bias effect and the Dr. GRPO fix; DAPO reports gains from its four changes on a much larger model — none of these magnitudes transfer to 0.6B arithmetic; "to be measured".
- **Noise check.** 2–3 seeds; paired bootstrap over eval problems for each pair of arms at the last step; the length claim requires the late-step difference in incorrect-response length between A and D to exceed the seed range of each arm; otherwise reported as "no detectable difference". A *shuffled-reward control* (random verdicts, one seed) shows what the curves do without signal.
- **T1 reduction.** as in Setup.
- **Artifacts (T2).** adapters per arm and seed, step logs (jsonl), eval tables; committed explicitly: `lm-grpo-qwen3-0p6b` (arm A, seed 0) and `lm-grpo-qwen3-1p7b-base` (arm A, seed 0).

#### Figures and tables

`tab_lm_verifier_tests`, `fig_lm_grpo_loop`, `tab_lm_grpo_memory`, `fig_lm_grpo_weights`, `tab_lm_grpo_aggregation` (per-token weight of each divisor for a batch of lengths (50, 200) — printed from code), `tab_lm_grpo_recipes` (method | what changes | symptom | known cost), `fig_lm_grpo_curves`, `fig_lm_grpo_groups`, `fig_lm_grpo_clip`, `fig_lm_grpo_bins`, `fig_lm_grpo_scaling`, `tab_lm_grpo_ablation` (arm × accuracy with CI, length correct/incorrect, entropy at last step).

#### Pitfalls and misconceptions

- `torch.std` is unbiased (divides by $K-1$) while `numpy.std` is population: the book's `d2l.normalize` uses `x.std()`, so its PyTorch and NumPy behaviours differ by $\sqrt{K/(K-1)}$; the chapter always uses the population std in `group_advantages` and says so.
- Believing the group mean is unbiased (it shrinks the gradient by $(K-1)/K$) and that std normalization is "just a step size" (it re-weights prompts, D1d).
- Dividing by response length and then reading falling loss per token as progress; the optimized quantity is reward per token.
- Discarding zero-variance groups changes the prompt distribution the statistics average over; report it.
- A verifier with a bug is an optimization target: run `verifier_tests()` before every training run.
- Reading higher pass@1 as new capability: compare pass@k.
- Sampler/trainer log-prob mismatch (vLLM vs HF) makes $\rho\ne1$ at the first minibatch; see 8.6.

#### Summary (draft bullets)

- With a verifier the machinery reduces to sampling groups, scoring, normalizing within the group and applying a clipped policy gradient; there is no critic and no reward model.
- Group-mean centering shrinks the gradient by $(K-1)/K$; dividing by the group standard deviation re-weights prompts by $\approx1/\sqrt{p(1-p)}$ (a different objective); dividing by response length optimizes reward per token and favors long wrong answers.
- A constant divisor and no std keep the direction of the policy gradient.
- DAPO's changes and GSPO/CISPO each address a measured symptom — entropy collapse, empty groups, length bias, truncation, noisy token ratios.
- Entropy changes by a covariance between log-probability and update; watching it is part of the training loop.
- Accuracy vs compute is smooth enough to fit, but the fit is only as good as the held-out check.

#### Exercises

1. **(check)** A group of 4 has rewards (1, 0, 0, 0). Compute the GRPO advantages (population std) and the leave-one-out advantages. *Sketch:* $\mu=0.25$, $\hat\sigma=\sqrt{0.1875}=0.4330$; $A=(1.7321,-0.5774,-0.5774,-0.5774)$; LOO: $\frac43(r-\mu)=(1,-0.3333,\dots)$.
2. **(check)** Two correct responses of lengths 100 and 400 and an incorrect response of 100, 400 in a group with mean reward 0.5: write the per-token weight each receives under the three divisors. *Sketch:* sample divisor $A/L$; token divisor $A/N$; constant $A/L_{\max}$; the per-token weight of the 400-token incorrect response is $\tfrac14$ of the 100-token one under the sample divisor, equal under the others.
3. **(derive)** Prove (d) for $K=2$: $w_2(p)=1$ for all $p$ when std-normalized. *Sketch:* for $m=1$, $\hat\sigma=\tfrac12$, $\mathbb E\hat\sigma=2p(1-p)\cdot\tfrac12=p(1-p)$, so $w=1$.
4. **(derive)** Compute the exact bias of the token-mean divisor for $K=2$ and two possible lengths $\{L_1,L_2\}$ independent of reward. *Sketch:* $\mathbb E[A_i/N]$ with $N=|y_1|+|y_2|$; since lengths and rewards are independent, $\mathbb E[A_i\nabla\log\pi(y_i)/N]$ factorizes through $\mathbb E[1/N\mid\cdot]$ and is unbiased in direction; the bias appears only when length correlates with reward — show it equals $\operatorname{Cov}$ terms in the numerator.
5. **(measure)** Predict then measure the fraction of zero-variance groups at step 0 and at the end for K=4, 8, 16 at equal generated tokens; relate to the sample-efficiency curve.
6. **(measure)** Switch on clip-higher ($\epsilon_{\textrm{high}}=0.28$) in arm D and measure entropy and accuracy over 3 seeds; use D2 to predict the sign of the entropy effect.
7. **(open)** Design a verifier for a task where exact match is impossible (e.g., code with unit tests or a constrained-format instruction) and write the adversarial test table; show one response that fools it and fix it.

#### Further reading

- DeepSeekMath ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)): GRPO. Understanding R1-Zero-Like Training ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)): Dr. GRPO. DAPO ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)). GSPO ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)). MiniMax-M1 ([arXiv:2506.13585](https://arxiv.org/abs/2506.13585)): CISPO. The Art of Scaling RL Compute ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786)).
- New: The Entropy Mechanism of RL for Reasoning LMs ([arXiv:2505.22617](https://arxiv.org/abs/2505.22617)); Back to Basics: Revisiting REINFORCE Style Optimization ([arXiv:2402.14740](https://arxiv.org/abs/2402.14740)) — RLOO; DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)); Does RL Really Incentivize Reasoning Capacity Beyond the Base Model? ([arXiv:2504.13837](https://arxiv.org/abs/2504.13837)); Tülu 3 ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) — the name RLVR and an open recipe; Thinking Machines, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — LoRA for RL.
- Tutorials: Lambert, RLHF Book; [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer).
- Code: [TRL](https://github.com/huggingface/trl); [verl](https://github.com/verl-project/verl).

#### Risks and open decisions

- *Zero-variance groups dominate at the start* (task too hard) or at the end (too easy): `ArithmeticTasks` profiles are calibrated on π_SFT so that the initial pass rate lies in a band (target 0.2–0.6; measured); fallback: filter prompts by pass rate with a 16-sample probe.
- *HF `generate` too slow at T1*: switch `engine='vllm'`; the log-probability mismatch it adds is then part of the lesson (8.6).
- *A seed-range larger than the arm differences*: report null results; add seeds only at T2.
- *π_SFT already saturates arithmetic*: use the `hard` profile or GSM8K train for the live arms.
- *Library option names*: tabulate from the installed release at authoring time.

### 8.6 RL training systems

- **Label:** `sec_lm_rl_systems` · **Tier:** T3 (reduced T1 config: Qwen3-0.6B with LoRA on one Ada GPU — colocated with vLLM sleep mode versus a two-process "split" on the same GPU, plus the T0 simulator fed by logs) · **Budget:** T3 ≤ 2 h wall-clock on 8× A100 40 GB for four layouts in five runs (each a fixed number of optimizer steps after a calibration run; per-GPU peak memory to be measured, target ≤ 38 GiB of the card's 40 GB). Access to the T3 node is limited: every run happens once and its logs are committed; 4B full-parameter training needs the trainer to span at least 3 GPUs (64.4 GB of state, 21.5 GB per GPU at 3, 32.2 GB at 2, see the Setup); T1 ≤ 15 min, ≤ 15 GiB; T0 simulator seconds · **Codex units:** 9.8, 14.7
- **Builds on (book):** `sec_ppo` (importance ratio, the per-step surrogate vs the product of ratios, clipping), `sec_actorcritic` ("The Critic's Moving Target and Data Freshness": staleness in the one-GPU setting), `sec_mdl-bayesian-computation` (importance sampling, effective sample size), `sec_perf_model` and `sec_hardware` (roofline; decode is memory-bound; the accelerator specification table), `sec_kv-cache` (prefill vs decode, cache bytes per token), `sec_multi_gpu` (α–β model and broadcast cost). **Not re-taught:** the roofline, the cost of a collective, the importance-sampling identity; only their application to the RL step.
- **Builds on (this part):**
  - 5.1 `d2l.chip_specs()` (peak BF16 FLOP/s, HBM bandwidth and capacity, interconnect bandwidth for the A100 row — columns `chip`, `p16_tflops`, `mem_bw_tbs`, `mem_gb`, `scaleup_gbs` (5.1; note `mem_bw_tbs` is in TB/s, `scaleup_gbs` in GB/s); the datasheet values are the fallback: 312 TFLOPS BF16 dense, 1,555 GB/s HBM for the A100 40 GB PCIe card, NVLink bridge 600 GB/s for a pair, PCIe Gen4 64 GB/s, from the NVIDIA A100 datasheet; the T3 node's own interconnect is read from `nvidia-smi topo -m` and decides which row applies);
  - 6.1 `d2l.launch(script, nodes=1, gpus_per_node=8, **env) -> LaunchResult`, 6.2 `d2l.fsdp_wrap`/`d2l.fsdp_train_step` (FSDP2 trainer sharding; the trainer in this section is a 6.2 FSDP module);
  - 8.2 `d2l.PhaseTimer`, `d2l.rlhf_memory`, `d2l.ppo_loss`; 8.5 `d2l.grpo_train`, `d2l.sample_responses`, `d2l.group_advantages`, `d2l.ArithmeticTasks`, `d2l.verify_math`, `d2l.token_logprobs` (8.1);
  - 1.5 `d2l.paired_bootstrap`.
  - Forward pointers only (not prerequisites): 12.2 and 13.2–13.3 for how the rollout engine batches and pages the KV cache.
- **Exports (`#@save`):**
  - `d2l.Timeline`: span recorder usable from several processes (`tl.span(worker, phase, version=None)` context manager writing `(worker, phase, t0, t1, version)` to a jsonl file with `time.time_ns()` after `torch.cuda.synchronize()`); `.frame()` → DataFrame; `.gantt(ax)`; `.busy_fraction(workers)`.
  - `d2l.rl_step_model(...)` → dict of predicted seconds per phase and per layout (see D1): arguments `n_params`, `kv_bytes_per_token` (from `d2l.config_memory`, 1.1), `prompt_len`, `resp_len_dist` (a sample of lengths), `n_seq`, `chip` (a row of `chip_specs`), `G_gen`, `G_train`, `layout` ∈ {`'colocated'`,`'split_sync'`,`'split_async'`}, `max_staleness`, `mfu_train`, `decode_eff`, `sync_bw`, `swap_s`.
  - `d2l.simulate_pipeline(t_gen, t_train, t_sync, layout, max_staleness, n_steps, t_swap=0.0)` → `Timeline` of a discrete-event schedule (T0).
  - `d2l.is_weights(logp_new, logp_behavior, resp_mask, level='token', clip=None)` → importance weights (token or sequence level, optionally truncated at `clip`); `d2l.ess(w, mask=None)` → $(\sum w)^2/\sum w^2$ and its fraction of $n$.
  - `d2l.VersionedBatch` (dataclass: `ids, resp_mask, logp_behavior, rewards, version`) and `d2l.policy_lag(batch, current_version)` → mean/max lag in optimizer steps.
  - `d2l.sync_weights(trainer, engine, mode)` → `(seconds, bytes)`; `mode` ∈ {`'nccl'`, `'ipc'`, `'lora_only'`}; a thin wrapper around vLLM's documented weight-reload and sleep/wake-up hooks (version-sensitive; isolated in this one function).
  - `d2l.rl_system_script(cfg)` → text of the `%%writefile` launcher run through `d2l.launch`.
  - Data artifacts: `lm-rlsys-logs` (timeline jsonl, step logs, memory snapshots, eval tables).
- **Models and data:** T3 headline: `Qwen/Qwen3-4B-Base` (4,022,468,096 parameters, Apache-2.0 on its own Hugging Face Hub entry, checked 2026-10-29) full-parameter training under FSDP2 with a minimal R1-Zero-like template and the 8.5 arithmetic task; the run is limited-access and happens once; fallback `Qwen/Qwen3-1.7B-Base` (1,720,574,976 parameters, Apache-2.0) if the 4B state does not fit the trainer GPUs. T1: π_SFT (0.6B). Tasks: `ArithmeticTasks('mixed')` train/dev/eval of 8.5.
- **Packages:** `vllm` (current release at build time; the PyPI latest at blueprint time is 0.32.0 — pin and record), `torch` ≥ the version pinned by 1.1 and 6.1 (FSDP2 `fully_shard`), `ray` only for the comparison with OpenRLHF (optional), `nvidia-ml-py` for utilization.

#### Narrative arc

1. **The question.** *In an RL step, which GPUs are idle, and when?* Opening cell: read the committed T3 `Timeline` of the naive colocated layout (or, at T1, run 3 steps) and print GPU-seconds by phase and the idle fraction; draw the Gantt (`fig_lm_rls_gantt_naive`). The symptom: generation occupies most of the step wall-clock, while the training GPUs wait (fractions to be measured; the prediction from D1 is printed next to them).
2. **Anatomy of an RL step and why generation dominates.**
   - Phases: sample (K responses per prompt) → score (verifier, CPU) → recompute log-probabilities with the trainer (old and, if $\beta>0$, reference) → advantages → update → synchronize weights to the rollout engine.
   - The roofline argument, applied (not re-derived): training/prefill FLOPs are $\approx 6N$ per token (fwd+bwd; $8N$ with activation recomputation) at a high MFU; decoding at batch $b$ is memory-bound until $b$ reaches the ridge point — the arithmetic intensity of decode is $\approx b$ FLOP per byte of weights — so tokens/s per GPU is set by HBM bandwidth over (weights + KV bytes). A cell computes, for 0.6B and 4B on the A100 row of `chip_specs`, the predicted seconds to *generate* the tokens of one step versus the seconds to *train* on them (the author states the assumptions: batch per replica, mean length, MFU 0.35–0.5 for training — a *range to be measured*, not a claim).
   - The length tail: GRPO groups are synchronous barriers; one 1,000-token straggler idles the rest of its batch. `fig_lm_rls_lengths`: histogram of generated lengths from a committed run and the fraction of decode steps run at batch < 25% of the maximum.
3. **Placement** (`fig_lm_rls_layouts`, Gantt cartoons of the same four-phase step under each layout):
   - *Colocated, time-multiplexed:* all GPUs run generation, then all run training; the rollout engine sleeps (vLLM sleep mode: level 1 keeps weights in CPU memory, level 2 discards them and is recommended when the trainer will push new weights) while the trainer holds memory; the cost is the wake/sleep or reload time and the need to fit trainer state *and* engine memory (or offload one).
   - *Disaggregated, synchronous:* separate GPU sets; trainer idles during generation and vice versa; the cost is idle capacity and a weight transfer over the interconnect each step.
   - *Disaggregated, asynchronous:* generation and training overlap with a bounded staleness $k$ (steps); the cost is off-policy data (D2).
   - Hybrid engines (HybridFlow): one resident copy sharded for training and re-sharded for inference; what it saves (a copy, a transfer) and what it needs (a resharding path).
4. **Weight synchronization.** Bytes: $2N$ for BF16 full weights (≈ 8.0 GB for the 4B model, 1.2 GB for 0.6B), or adapter bytes only for LoRA (`mode='lora_only'`, orders of magnitude smaller, to be measured). Paths: NCCL broadcast from trainer rank(s) to engine workers, CUDA IPC handles when colocated, shared filesystem as the slow baseline. Resharding: FSDP shards must be gathered layer by layer (peak memory = one layer's full parameters, not the whole model). Cell: measure seconds and bytes for each mode, compare with the α–β prediction of `sec_multi_gpu` (latency + bytes/bandwidth).
5. **vLLM in the loop.** Parity of sampling parameters with the trainer (temperature, top-p/top-k, stop tokens); requesting per-token log-probabilities of the sampled tokens (the behavior policy's $\log\pi_b$ — essential for D2); prefix caching so the $K$ samples of a prompt share prefill; `gpu_memory_utilization` and KV-cache sizing when colocated; the sleep/wake and reload hooks isolated in `sync_weights`; *train–inference mismatch*: even at lag 0 the engine's log-probabilities differ from the trainer's (different kernels and precision), so the first minibatch has $\rho\neq1$; cell: histogram of $\log(\pi_{\textrm{trainer}}/\pi_{\textrm{engine}})$ per token on 256 sampled responses with mean, std and the fraction beyond ±0.1 (to be measured). Cite the Yao et al. note on this mismatch and truncated importance sampling as the fix (unverified page title; see Further reading).
6. **Asynchronous rollouts and their correction** (D2): version tags on every sample (and on every *token* if generation is interrupted for a weight update, as AReaL does — then one response mixes versions); bounded staleness; importance weights relative to the behavior policy; the decoupled objective (clip around the proximal policy, correct staleness with a separate, possibly truncated, weight); effective sample size as the diagnostic; the interaction with the clip of 8.5.
7. **The timing model and the simulator** (D1): `rl_step_model` predicts per-phase seconds; `simulate_pipeline` produces the Gantt of any layout from phase durations; the optimal GPU split falls out of D1; cell compares prediction with the measured layouts of the headline experiment (relative error printed; the A5-style tolerance of ±15% is a *target*, "to be measured").
8. **The T3 run** (headline experiment).
9. **Systems in the wild.** `tab_lm_rls_systems`: verl (HybridFlow; hybrid engine with a controller; backends), OpenRLHF (Ray; separate or colocated actor/critic/reference/reward groups; vLLM engines), AReaL (fully asynchronous; staleness bound; interruptible generation), TRL (`use_vllm` with colocate or server mode). Columns: placement options, weight-sync path, async support, what to read in the code. No claims about performance across systems are made; the notebook does not benchmark them against each other.

#### Derivations

**D1. A timing model for the three layouts, and the optimal GPU split.** *(Required content for v5's "attribute time" and "colocated vs split".)*
*Setting.* A step generates $n$ responses with total response tokens $M$ and prompt tokens $P$, trains on $M+P$ tokens with $N$ parameters. Let $t_g$ be the single-GPU generation time of the step's tokens (at the engine's operating batch size), $t_\tau$ the single-GPU training time of the step (forward+backward+old log-prob pass), $s$ the weight-synchronization time (independent of the split for broadcast over an interconnect; $\approx\alpha+2N/B_{\textrm{link}}$ with the α–β model), $c$ the colocated swap overhead (sleep/wake or reload). Assume perfect data-parallel scaling in each phase: generation on $G_g$ GPUs takes $t_g/G_g$, training on $G_t$ takes $t_\tau/G_t$ (a stated approximation; the straggler tail and the FSDP communication make it optimistic).
*Claims.*
(a) *Colocated* on $G$ GPUs: $T_{\textrm{col}}=\frac{t_g+t_\tau}{G}+c+s_{\textrm{local}}$.
(b) *Split, synchronous* ($G_g+G_t=G$): $T_{\textrm{sync}}=\frac{t_g}{G_g}+\frac{t_\tau}{G_t}+s$. Minimizing over $G_g$ gives $G_g^\ast=G\frac{\sqrt{t_g}}{\sqrt{t_g}+\sqrt{t_\tau}}$ and $T^\ast_{\textrm{sync}}=\frac{(\sqrt{t_g}+\sqrt{t_\tau})^2}{G}+s$.
(c) *Split, asynchronous* with staleness bound $k\ge1$ in steady state: $T_{\textrm{async}}=\max\!\big(\frac{t_g}{G_g},\,\frac{t_\tau}{G_t}+s\big)$; minimizing over $G_g$ gives $G_g^\ast\approx G\frac{t_g}{t_g+t_\tau}$ and $T^\ast_{\textrm{async}}\approx\frac{t_g+t_\tau}{G}+\frac{t_g}{t_g+t_\tau}s$ (up to GPU integrality).
(d) Hence at the balanced split the async layout costs the same as an *overhead-free colocated* layout, plus a fraction of the sync time; it beats the synchronous split by the factor $T^\ast_{\textrm{sync}}/T^\ast_{\textrm{async}}\to\frac{(\sqrt{t_g}+\sqrt{t_\tau})^2}{t_g+t_\tau}\in[1,2]$, with maximum 2 at $t_g=t_\tau$. Colocation beats async only when $c<$ the integrality and sync losses of async; async costs staleness (D2).
*Steps.* (1) Sum the phase times for (a),(b). (2) For (b), set $\frac{d}{dG_g}\big[\frac{t_g}{G_g}+\frac{t_\tau}{G-G_g}\big]=0$: $\frac{t_g}{G_g^2}=\frac{t_\tau}{G_t^2}$, so $G_g/G_t=\sqrt{t_g/t_\tau}$; substitute. (3) For (c) the two stages form a two-stage pipeline whose steady-state period is the larger stage time; the max over $G_g$ of two monotone functions is minimized where they are equal: $\frac{t_g}{G_g}=\frac{t_\tau}{G-G_g}+s$ — approximate $s\ll$ to get $G_g^\ast$, then $T^\ast=t_g/G_g^\ast$. (4) Ratio in (d): divide (b) by (c) with $s=0$: $\frac{(\sqrt{t_g}+\sqrt{t_\tau})^2}{t_g+t_\tau}=1+\frac{2\sqrt{t_gt_\tau}}{t_g+t_\tau}$, maximized (value 2) at $t_g=t_\tau$ by AM–GM. (5) With staleness bound $k=1$ the pipeline never stalls if $G_g$ is at the balance point; with $k=0$ it degenerates to (b).
*Numerical check (cell, T0).* `simulate_pipeline` with $t_g=480$ and $t_\tau=160$ GPU-seconds, $G=8$, $s=3$ s, $c=5$ s, $n=1000$ steps: (i) the simulated mean step time equals (a), (b), (c) to relative $10^{-3}$ after discarding 5 warm-up steps (for (b) and (c) at the integer $G_g$ used); (ii) a scan over integer $G_g\in\{1,\dots,7\}$ finds the argmin of (b) at the integer nearest $G^\ast=5.07$ (i.e. 5) and of (c) at 6 (the integer nearest $G\,t_g/(t_g+t_\tau)=6$); (iii) the printed speedup of async over sync at the integer optima is compared with the closed form $1+2\sqrt{t_gt_\tau}/(t_g+t_\tau)=1.87$ and the difference is attributed to integrality and to $s$. The values $t_g,t_\tau$ are placeholders in the *test*; the headline experiment replaces them with measured numbers.

**D2. Off-policy correction for stale rollouts: unbiasedness, variance and effective sample size.**
*Statement.* Let samples be drawn from a behavior policy $\pi_b$ (the engine at version $v-k$) and the update be for $\pi_\theta$. (a) The sequence-level estimator $\hat g=w(y)\,A(y)\,\nabla\log\pi_\theta(y\mid x)$ with $w=\pi_\theta(y\mid x)/\pi_b(y\mid x)$ is unbiased for $\nabla_\theta J(\theta)$; without $w$ the estimator is biased; (b) the per-token surrogate with $\rho_t=\pi_\theta(y_t\mid\cdot)/\pi_b(y_t\mid\cdot)$ drops the later ratios (the surrogate of `sec_ppo`) and is biased but has smaller variance; (c) if $\log w$ is Gaussian with variance $\varsigma^2$ and mean $-\varsigma^2/2$ (so $\mathbb E_b w=1$), the effective sample size fraction is $\mathrm{ESS}/n\to e^{-\varsigma^2}$; for i.i.d. per-token log-ratios of variance $s^2$ over $T$ tokens, $\varsigma^2=Ts^2$, so ESS decays exponentially with response length; (d) truncating the weight at $c$ (replace $w$ by $\min(w,c)$) changes the expectation by $-\mathbb E_b[(w-c)_+Af]$, bounded by $\sup|Af|\cdot\mathbb E_b[(w-c)_+]$, and caps the weight variance by $c^2$ — the bias-variance trade-off behind truncated importance sampling and CISPO's clipped weights; (e) decoupled objective: $\frac{\pi_\theta}{\pi_b}=\frac{\pi_\theta}{\pi_{\textrm{prox}}}\cdot\frac{\pi_{\textrm{prox}}}{\pi_b}$ — PPO's clip applies to the first factor (around the proximal policy computed by the trainer at the start of the update), and the second factor is the staleness weight.
*Steps.* (1) (a): $\mathbb E_b[w\,A\nabla\log\pi_\theta]=\sum_y\pi_b\frac{\pi_\theta}{\pi_b}A\nabla\log\pi_\theta=\sum_yA\nabla\pi_\theta=\nabla J$ with $A$ fixed (`eq_change_of_measure` of `sec_ppo`); without $w$ the expectation is $\sum_y\pi_bA\nabla\log\pi_\theta\ne\nabla J$ unless $\pi_b=\pi_\theta$. (2) (b) is the book's per-step surrogate; do not re-derive. (3) (c): $\mathbb E[w^2]=e^{\varsigma^2}$ for the stated log-normal ($\mathbb E e^{2X}$ with $X\sim\mathcal N(-\varsigma^2/2,\varsigma^2)$ equals $e^{-\varsigma^2+2\varsigma^2}$), and $\mathrm{ESS}/n=(\mathbb Ew)^2/\mathbb E w^2=e^{-\varsigma^2}$. (4) (d): $\mathbb E[\min(w,c)f]=\mathbb E[wf]-\mathbb E[(w-c)_+f]$. (5) (e) is an identity; its content is the *allocation* of the two jobs (trust region vs staleness) to two factors.
*Numerical checks (cell, numpy fp64, seed 0).* (i) A 5-outcome categorical with two logits vectors ($\pi_\theta,\pi_b$), a random advantage $A$: exact expectations by enumeration — uncorrected gradient differs from $\nabla J$ by a nonzero norm (printed), IS-corrected equals $\nabla J$ to $10^{-12}$, truncated at $c\in\{1.5,3\}$ differs by exactly $\mathbb E_b[(w-c)_+A\nabla\log\pi_\theta]$ to $10^{-12}$. (ii) Monte Carlo $N=2\cdot10^6$ log-normal weights: ESS fraction vs $e^{-\varsigma^2}$ for $\varsigma\in\{0.3,0.6,1.0\}$ agree within 1% (preparation run: 0.914/0.914, 0.698/0.698, 0.367/0.368). (iii) The product identity (e) on random tensors to $10^{-12}$.

#### From-scratch implementation

Budget ≤ ~150 lines: `Timeline` (35), `simulate_pipeline` (35), `rl_step_model` (30), `is_weights` + `ess` + `policy_lag` (20), `sync_weights` (20 — the only vLLM-version-sensitive code), `rl_system_script` (10 lines of template; the training script itself reuses `grpo_train`).
Tests: (1) D1 and D2 checks above; (2) `Timeline`: spans recorded by two processes merge and are ordered; `busy_fraction` on a synthetic schedule equals the hand-computed value; (3) `is_weights` at identical log-probs gives all ones; sequence-level weights equal the product of token weights (`atol=1e-6`); (4) `sync_weights` round trip: after `sync_weights`, engine log-probabilities of 8 fixed sequences match the trainer's within the BF16 tolerance measured at lag 0 in the mismatch cell (to be measured; a failing sync shows up as a large mean $|\log\rho|$); (5) `rl_step_model` reproduces the closed forms of D1 for the toy inputs of the numerical check.

#### In production

- **verl** ([repo](https://github.com/verl-project/verl); HybridFlow, [arXiv:2409.19256](https://arxiv.org/abs/2409.19256)): run the same T1 task as a verl config with the hybrid-engine option on and off; compare the phase table and, using the recorded batch of log-probabilities, check the advantage and loss values against the from-scratch `group_advantages`/`ppo_loss`.
- **OpenRLHF** ([arXiv:2405.11143](https://arxiv.org/abs/2405.11143)): the colocated vs separated actor/vLLM placements; read its launch flags in the pinned release (verify names).
- **AReaL** ([arXiv:2505.24298](https://arxiv.org/abs/2505.24298), [repo](https://github.com/areal-project/AReaL)): the asynchronous design; map its staleness bound and decoupled objective to D2. (The repository was verified to resolve to `areal-project/AReaL`; the v5 text cites it under that name.)
- **TRL** GRPO with vLLM in colocate and server modes: the shortest route to the same two placements in one library; time them with the notebook's `Timeline`.
- The from-scratch `Timeline`, `simulate_pipeline`, `ess` and `is_weights` are the *oracles*: every library's logged step times and ratios are re-derived from recorded batches rather than trusted from its own dashboards.

#### Headline experiment

- **Hypothesis.** (i) Generation takes the largest share of GPU-seconds of a GRPO step on this task and model; (ii) at a balanced allocation the asynchronous split has a shorter step time than the synchronous split (D1d predicts up to 2× at balance; the realized factor is smaller because of integrality, stragglers and sync) and approaches the colocated layout; (iii) bounded staleness $k\le2$ changes held-out accuracy per optimizer step by less than the seed-to-seed spread, while the ESS fraction falls below 1; (iv) D1's predicted step times agree with the measured ones within ±15%.
- **Setup.** `Qwen3-4B-Base` full-parameter under FSDP2 (n_train GPUs), vLLM replicas with TP=1 (n_gen GPUs), `ArithmeticTasks('mixed')`, K=8, 128 prompts/step, 512 new tokens max, BF16; GRPO with the 8.5 default (constant divisor, no std) to avoid the confound of 8.5's ablation; layouts: L1 colocated 8 GPUs with sleep mode; L2 split-sync 4+4; L3 split-async 4+4 with $k=1$; L4 split-async at the D1-optimal split from the calibration phase with $k=2$ (one run; the trainer keeps at least 3 GPUs, so $G_g^\ast\le5$ even if D1 predicts 6, and the departure from D1's optimum is reported with D1's prediction for the split actually run; if the optimum is 4+4, L4 differs from L3 only by $k$); every layout runs a calibration phase of 5 steps, then a fixed number of optimizer steps chosen so that the whole batch fits 2 h (to be set after calibration), two seeds for L1 only and one run for each other layout: five runs in all. *Memory on 8× 40 GB* (arithmetic by `d2l.rlhf_memory` at 16 B per parameter, $4{,}022{,}468{,}096\times16$ B $=64.4$ GB of weights, gradients and AdamW state, sharded by FSDP2): 8.04 GB per GPU on 8 trainer GPUs (L1, beside the engine's 8.04 GB of bf16 weights and its KV cache, which is why the engine sleeps during the update), 16.1 GB on 4 (L2, L3), 21.5 GB on 3 (L4 at the floor); on 2 trainer GPUs the state alone is 32.2 GB, which with a gathered layer, activations and logits leaves no margin under 38 GiB, so no layout uses 2. Each generation GPU holds one TP=1 vLLM replica (8.04 GB of weights plus KV cache). Per-GPU peaks are to be measured and recorded (`torch.cuda.max_memory_allocated` per rank). *Cut for the 40 GB, limited-access node (run once, so no run without a stated claim):* the former L4 runs at $k=1$ and $k=4$ (claim (iii) covers $k\le2$, which L3 and L4 span), the second seed of L3 (the seed spread that (iii) is judged against comes from L1's two seeds), and any 2-GPU-trainer layout (does not fit); the former plan was eight runs, this one five.
- **Procedure.** `Timeline` spans for every worker; per-step records of tokens generated, tokens trained, version used/generated, $\log\pi_b$ and trainer $\log\pi$ for a 5% subsample of tokens, ESS; every 10 steps a held-out accuracy on 500 eval problems (done by the engine, costed separately); memory via `torch.cuda.max_memory_allocated` per rank.
- **Metrics.** step time; tokens/s (generated and trained); busy fraction of each GPU group; seconds per phase; sync bytes and seconds; policy lag (mean, max, in optimizer steps) and mean $|\log\rho|$; ESS fraction; peak memory per GPU; held-out accuracy vs step and vs wall-clock.
- **Plots.** `fig_lm_rls_gantt` (Gantt of three consecutive steps for each layout, workers on the y-axis, phases coloured); `fig_lm_rls_breakdown` (stacked bars of seconds per phase per layout, with D1's prediction as markers); `fig_lm_rls_lag` (policy-lag histogram and ESS fraction vs $k$); `fig_lm_rls_wallclock` (held-out accuracy vs wall-clock and vs optimizer step for L1, L3, L4 ($k=2$); the band is L1's two-seed range only, drawn on all three as the yardstick).
- **Expected outcome.** Directions above; no magnitude is asserted. AReaL reports a training-speed gain for asynchronous execution with staleness control over synchronous baselines on its own workloads; that figure does not transfer to this setup and is not quoted.
- **Noise check.** Step times: median and inter-quartile range over the steady-state steps (after calibration), 3 repeats of the 5-step calibration for the layout comparison; layout differences claimed only when ranges do not overlap. Accuracy: paired bootstrap over eval problems for equal-step comparisons, and seed range of L1 (two seeds; the other layouts have one run); a claim of "no accuracy loss" is made only with an interval that includes 0 *and* is narrower than the effect that would matter (stated beforehand).
- **T1 reduction.** `Qwen3-0.6B` π_SFT + LoRA on one Ada GPU: L1 (colocated, vLLM `enable_sleep_mode`) and L3 emulated by two processes sharing the GPU (vLLM at a fixed `gpu_memory_utilization` fraction, trainer in the remainder); 15 steps of 32 prompts × K=8, two seeds for the timing; staleness $k\in\{0,1\}$; claims are limited to attribution (what fraction each phase takes) and to the correctness of the correction code (ESS, lag), *not* to throughput of the layouts (time-sliced sharing of one GPU is not a split). The D1 simulator runs with the T1-measured phase durations (T0) to illustrate what the split would give.
- **Artifacts (T2+).** `lm-rlsys-logs`: timeline jsonl per layout, step logs, memory snapshots, eval tables, `config.json` (vLLM, torch, transformers versions, seed, git hash); all plotting and fitting cells run on these files.

#### Figures and tables

`fig_lm_rls_gantt_naive`, `fig_lm_rls_lengths`, `fig_lm_rls_layouts`, `tab_lm_rls_memory` (bytes per GPU role for 0.6B, 4B, 30B-class from `rlhf_memory` — computed, with the 30B row labelled illustrative), `fig_lm_rls_gantt`, `fig_lm_rls_breakdown`, `fig_lm_rls_lag`, `fig_lm_rls_wallclock`, `tab_lm_rls_systems`, `tab_lm_rls_model_vs_measured` (D1 prediction vs measurement per phase and layout with relative error).

#### Pitfalls and misconceptions

- Timing GPU phases without `torch.cuda.synchronize()` (asynchronous launches make a phase look instantaneous and the next one slow).
- Comparing layouts at different numbers of GPUs or different batch sizes, or comparing "steps per hour" when asynchrony changes what a step means; compare tokens trained per GPU-hour *and* accuracy per wall-clock.
- Assuming the engine and the trainer agree on log-probabilities at lag 0: they do not, in general; measure it.
- Treating staleness as free: ESS and the lag histogram are part of the result.
- Forgetting that the verifier (CPU) and the weight sync are phases too; at small models and short responses they are not negligible.
- Calling "two processes on one GPU" a disaggregated layout.
- Scaling intuition from the 4B run to 70B+ models without redoing memory and sync arithmetic (`tab_lm_rls_memory`).

#### Summary (draft bullets)

- In an RL step generation is memory-bound decode and dominates GPU-seconds; training is compute-bound and shorter, so a naive loop leaves one of the two resources idle.
- Placement is a choice among colocation (swap cost), synchronous split (idle capacity) and asynchronous split (staleness).
- Under ideal scaling the optimal split is $G_g^\ast\propto\sqrt{t_g}$ (sync) or $\propto t_g$ (async), and async at balance costs about the same as an overhead-free colocated layout.
- Stale data are corrected with importance weights relative to the behavior policy; their effective sample size decays exponentially in response length, so weights are truncated or applied per token.
- Even at zero lag the engine's and the trainer's log-probabilities differ; the first minibatch is slightly off-policy.
- Every systems claim should be backed by a timeline and by a prediction from a model.

#### Exercises

1. **(check)** Which of the three layouts has idle GPUs during generation, and which has data from policy versions older than the trainer's? *Sketch:* split-sync idles the trainer GPUs during generation; async uses stale data; colocated idles nothing but pays swap time.
2. **(check)** A step generates $4\cdot10^5$ tokens with a 4B model; weights are 8.0 GB (BF16); an A100 40 GB PCIe has 1,555 GB/s of HBM bandwidth. What is the weights-only lower bound on seconds per decode step, and how many decode steps per second per GPU is that? *Sketch:* $8.0\ \textrm{GB}/1555\ \textrm{GB/s}\approx5.1$ ms per step, about 194 steps/s (ignoring KV and compute); with batch $b$ each step emits $b$ tokens.
3. **(derive)** Extend D1 to $G_g^\ast$ for the synchronous split when the sync time grows with $G_t$ (e.g., $s=\alpha+\beta\log_2G_t$). *Sketch:* add $s'(G_t)$ to the stationarity condition; the optimum shifts toward fewer trainer GPUs.
4. **(derive)** If each token's log-ratio is independent with variance $s^2$, show that truncating the *sequence-level* weight at $c=2$ discards a fraction of the weight mass that tends to 1 as $Ts^2\to\infty$, while truncating *token-level* weights affects a fraction of tokens that does not depend on $T$. *Sketch:* $\log w\sim\mathcal N(-Ts^2/2,Ts^2)$ has median $e^{-Ts^2/2}\to0$ but mean 1, so the mean is carried by rare large weights: $\mathbb E[w\mathbf 1(w>c)]=\Pr_{\tilde b}(w>c)$ under the tilted law $\mathcal N(+Ts^2/2,Ts^2)$, which tends to 1; the per-token tail probability $\Pr(\rho_t>c)$ is the same for every $T$.
5. **(measure)** Predict the optimal $G_g$ from a 3-step calibration of $t_g$ and $t_\tau$, then measure the step time at $G_g\in\{3,4,5\}$ (the 4B trainer cannot go below 3 GPUs on 40 GB cards) and report where the minimum is and the relative error of the model.
6. **(measure)** Add `lora_only` weight sync to the 0.6B T1 run and report seconds per sync, bytes, and the effect on step time; predict first.
7. **(open)** Build a straggler-mitigation strategy (e.g., over-sampling groups and dropping the slowest, or interruptible generation) and show, with timelines and an ESS table, what it changes in throughput and in the training distribution.

#### Further reading

- HybridFlow/verl ([arXiv:2409.19256](https://arxiv.org/abs/2409.19256)); OpenRLHF ([arXiv:2405.11143](https://arxiv.org/abs/2405.11143)); AReaL ([arXiv:2505.24298](https://arxiv.org/abs/2505.24298)); [verl](https://github.com/verl-project/verl).
- New: Asynchronous RLHF ([arXiv:2410.18252](https://arxiv.org/abs/2410.18252)) — one-step-off-policy generation with PPO/DPO; vLLM [sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) documentation — levels 1 and 2 and the weight-reload recipe; NVIDIA [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) — the peak numbers used in D1; Yao et al., *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training* ([Notion note](https://fengyao.notion.site/off-policy-rl); the URL returns 200 but the page is client-rendered and its title could not be verified, so mark "(unverified title)").
- Tutorials/courses: [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) lectures on off-policy methods; Lambert, RLHF Book (infrastructure notes).
- Code: [vLLM](https://github.com/vllm-project/vllm); [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF); [TRL](https://github.com/huggingface/trl).

#### Risks and open decisions

- *vLLM API churn* (sleep mode, weight reload, log-prob return): all contact points live in `sync_weights` and `sample_responses`; fallback: HF `generate` as the engine with a smaller step (the attribution still works; the claim "generation dominates" must then be re-measured).
- *FSDP2 plus vLLM on one node with NCCL contention* may deadlock or slow down: fall back to LoRA-only weights (small sync) and a trainer on 2–4 GPUs; report the failure mode.
- *A100 nodes without NVLink* (the T3 node's `nvidia-smi topo -m` output is recorded and decides) change the sync cost by an order of magnitude: the notebook measures the broadcast bandwidth first and feeds it to D1; the PCIe case is an exercise.
- *Two-hour budget too tight for four layouts in five runs*: shorten to 20 optimizer steps per layout and rely on step-time statistics; accuracy-vs-wall-clock is then shown only for L1 and L3.
- *The 4B trainer state does not fit 40 GB GPUs at 3 trainer GPUs* (peak above 38 GiB once activations and the gathered layer are counted, to be measured): fall back to `Qwen/Qwen3-1.7B-Base` (27.5 GB of state, 9.2 GB per GPU at 3), the stated fallback; the run is limited-access and happens once, so the calibration phase checks the memory peak on 3 GPUs before the layouts start.
- *T3 hardware unavailable*: the committed logs and the simulator carry the notebook; the T1 reduction runs live.

### 8.7 ◇ Overoptimization, safety and evaluating post-trained models

- **Label:** `sec_lm_overoptimization` · **Tier:** T1 (best-of-n sweep with n ≤ 64 on 200 prompts, GRPO against a deliberately biased verifier for ≤ 40 steps; other arms loaded from committed logs) · **Budget:** ≤ 15 min and ≤ 15 GiB, except the judge cells, which claim a whole 24 GB GPU (sampling 200 × 64 responses of ≤ 192 tokens with vLLM; HF `generate` is the slower fallback, to be measured); the principle-judge and over-refusal cells add a few minutes of 4B-judge inference · **Codex units:** 9.9
- **Builds on (book):** `sec_regularized` (Goodhart's law, the true-return-vs-KL-budget curve `subsec_kl_budget`, `eq_kl_optimum`: reused for the interpretation; **not re-taught:** the gridworld experiment or the closed form), `sec_rl_sequences` (`subsec_rl_sequences_hacking`: the hedge and the threshold $\beta=1/4$ in a four-prompt instance, which this section reproduces with a language model and a real KL budget), `sec_mdl-statistics` (bootstrap).
- **Builds on (this part):** 8.1 (`d2l.RewardModel`, the reward models and their seeds, `d2l.PreferenceData`, perturbation probes), 8.2 (`d2l.kl_estimators`), 8.3 (`d2l.dpo_train`, for the DPO overoptimization remark), 8.5 (`d2l.grpo_train`, `d2l.ArithmeticTasks`, `d2l.verify_math`, `d2l.sample_responses`), 1.4 (`d2l.extract_answer`), 1.5 (`d2l.judge_pairwise`, `d2l.paired_bootstrap`, `d2l.EvalSuite`), 1.4 (`d2l.pass_at_k`).
- **Exports (`#@save`):**
  - `d2l.bon_kl(n)` → $\log n-(n-1)/n$ (the KL of best-of-$n$ from the reference, in nats).
  - `d2l.bon_expected_gold(proxy, gold, n)` → expected gold score of the proxy-best of $n$ samples drawn without replacement from the $N$ available per prompt (unbiased "all subsets" estimator: with samples sorted ascending by proxy, the item of rank $i$ is the best of a random $n$-subset with probability $\binom{i-1}{n-1}/\binom Nn$); arrays of shape `(prompts, N)`; returns per-prompt values.
  - `d2l.gao_fit(d, gold, form='bon')` → `(alpha, beta, ci)` for $R(d)=d(\alpha-\beta d)$ (`form='bon'`) or $R(d)=d(\alpha-\beta\log d)$ (`form='rl'`), least squares with a prompt-level bootstrap interval.
  - `d2l.length_residualize(reward, length, fit_on=None)` → reward minus a fitted function of $\log$ length (linear in $\log$ length; fitted on `fit_on` pairs).
  - `d2l.rm_ensemble(models, agg='mean')` with `agg` ∈ {`'mean'`, `'min'`, `'mean-k*std'`} → callable scoring `(prompt, response)`.
  - `d2l.sloppy_math(text, gold)` → 1.0 if the gold string occurs anywhere inside the final `\boxed{…}` content (the language-model version of the book's `sloppy` grader); `d2l.hedge_rate(texts)` → fraction of final boxes with ≥ 2 distinct numbers.
  - `d2l.sycophancy_gap(model, tokenizer, items)` → accuracy on neutral prompts minus accuracy when the user asserts a wrong answer (items from the 8.1 probe set extended to 40 arithmetic/fact items).
  - `d2l.over_refusal_rate(model, tokenizer, prompts, judge)` → fraction of benign prompts answered with a refusal (regex pre-filter plus judge confirmation).
  - `d2l.principle_judge(judge, tokenizer, constitution, prompt, a, b)` → preference labels from a judge that is shown a short list of written principles.
  - Data artifacts: `lm-overopt-logs` (BoN tables, RL logs per arm), `data/overrefusal.jsonl` (100 benign-but-edgy prompts; written by the author, each with a note on why it is benign), `data/constitution.md` (5 principles written by the author); the should-refuse set: `data/should_refuse_categories.md` (category descriptions) and `data/should_refuse_rubric.md` (the judge rubric) are published, the prompts themselves stay private (not committed, not in `DATA_HUB`, shipped only to the autograder and the build host), and the notebook publishes only per-category and overall aggregate refusal rates (`lm-overopt-logs` carries those rates and no prompt or completion text of this set).
- **Models and data:** policy π_SFT (0.6B); proxy RMs: the 8.1 RM (and two more seeds trained in 8.1's headline); gold: `Skywork/Skywork-Reward-V2-Qwen3-1.7B` (Apache-2.0, 1,720,577,024 parameters), with `Skywork/Skywork-Reward-V2-Qwen3-4B` as a second gold in an exercise; judge `Qwen/Qwen3.5-9B` (the part's judge, as in 8.3; loaded after the policy is freed, so the judge cells claim a whole 24 GB GPU); prompts: 200 held-out `HuggingFaceH4/ultrafeedback_binarized` prompts (BoN) and `ArithmeticTasks` train/eval (RL arm); regression suite: the 1.5 development suite subset (`d2l.EvalSuite`).
- **Packages:** `vllm` (sampling), `transformers`, `peft`; `trl` not needed.

#### Narrative arc

1. **The question.** *A reward went up. Did the model get better?* Opening cell: best-of-$n$ against the 8.1 reward model for $n\in\{1,\dots,64\}$ on 200 prompts, printing the proxy score and the gold score of the selected responses side by side, and the correlation of both with length. The symptom: the proxy score increases with $n$ while the gold score first rises and then flattens or falls (to be measured; the Gao et al. shape).
2. **Overoptimization as a measurement problem.** The gold-reward-vs-$\sqrt{\textrm{KL}}$ plot (`fig_lm_oo_gao`): best-of-$n$ has an analytic KL (`d2l.bon_kl`, D1a), so no KL estimation is needed; fit $d(\alpha-\beta d)$ (`d2l.gao_fit`) with a prompt-level bootstrap interval; what a positive $\beta$ means (curvature of gold in the proxy, D1c). Comparison with the RL form $d(\alpha-\beta\log d)$ on the PPO and GRPO runs of 8.2/8.5 (committed logs): different functional forms and different coefficients, same qualitative story. Pointer to the scaling of the coefficients with reward-model size and data in Gao et al. and, for direct-alignment methods, Rafailov et al. (cite; the notebook does not reproduce the scaling study).
3. **Exploits specific to language models.** Three studies, each a cell with a counterfactual test:
   - *Length.* Reward residualization against $\log|y|$ before best-of-$n$ (`d2l.length_residualize`), and the Spearman correlation of gold with length before/after; the Singhal et al. finding as the reason to look first. State the limit of residualizing: it removes a *linear* length effect only.
   - *Sycophancy.* `d2l.sycophancy_gap` on π_SFT, on the DPO policy (8.3) and on the best-of-64 policy under the 8.1 RM; the RM-level version is the 8.1 perturbation test (agree vs correct a false belief). Pointer: Sharma et al. (Sycophancy), who tie it to human and reward-model preferences.
   - *Format and verifier loopholes.* The hedge: `d2l.sloppy_math` accepts any boxed content that contains the gold answer. GRPO (8.5 code, $\beta$ swept) against `sloppy_math` while evaluating with exact `verify_math`; the notebook logs sloppy reward, exact accuracy, hedge rate, mean length and sampled KL (k1 from 8.2). D2 predicts a *hedge threshold* in $\beta$ from the reference hedge probability; the notebook measures $h$ (probe of 2,000 samples), computes the prediction, and compares it with the observed suppression threshold — this reproduces at LM scale the four-prompt instance of `subsec_rl_sequences_hacking`. If the hedge is never discovered in the step budget (to be measured), the fallback is to seed 5% hedge examples into a 20-step SFT warm-start and say so.
4. **Mitigations, measured.**
   - *Reward-model ensembles* (`d2l.rm_ensemble`; mean, min and mean − $k\cdot$std over the three 8.1 RMs): gold-vs-$d$ curves for best-of-$n$; the prediction from Coste et al. that ensembles mitigate but do not eliminate, tested at this scale.
   - *KL penalty vs early stopping.* On the GRPO/sloppy arm, plot gold accuracy against $\sqrt{\textrm{KL}}$ for $\beta\in\{0,\beta_1,\beta_2\}$ and for early stopping of the $\beta=0$ run; ask whether the penalty moves the frontier (Gao et al. examine this; report what *this* experiment shows with intervals).
   - *Length control* as in step 3; *held-out proxy* (train on RM A, select with RM B) as a cheap overoptimization detector.
   - Heavy-tailed errors: one paragraph on Catastrophic Goodhart (cite): KL regularization cannot bound the damage when the reward error is heavy-tailed; no experiment.
5. **Training against written principles.** Constitutional AI (critique–revision and AI-labeled preferences) and deliberative alignment (a model that reads a specification in context and is trained on its own spec-grounded reasoning): a one-page explanation and one cell — `d2l.principle_judge` labels 200 UltraFeedback pairs with the 5-line constitution; report agreement with the dataset labels, with the gold RM, and with the unprincipled judge (position-consistent verdicts only, `d2l.judge_pairwise` convention). What the cell does *not* show: that training on these labels is safe; it shows what the labels encode.
6. **Evaluating a post-trained model.** The suite (`tab_lm_oo_suite`): (a) task metric with exact verifier; (b) independent reward/judge win rate with both orders; (c) length and format statistics; (d) over-refusal rate on `overrefusal.jsonl` and the refusal rate on a small private set of should-refuse requests *published as category descriptions, a judge rubric and aggregate rates only (prompts stay private; the editor reviews the set before freeze, approved 2026-10-29; see Risks)*; (e) a regression subset of the 1.5 suite before/after each stage (π_SFT, π_DPO, π_GRPO from A7), each with `d2l.paired_bootstrap`; (f) the counterfactual tests above. Message: report a vector, not a number.
7. **What would convince you?** A closing checklist (half a page): independent evaluator, held-out prompts, counterfactual edits, intervals, and a stated stopping rule.

#### Derivations

**D1. Reward-overoptimization scaling (required).**
*Statement.* (a) For continuous proxy reward and $n$ i.i.d. samples, the best-of-$n$ distribution $\pi_n$ satisfies $D_{\textrm{KL}}(\pi_n\Vert\pi_{\textrm{ref}})=\log n-\frac{n-1}{n}$, independent of the reward distribution. (b) In a jointly Gaussian proxy–gold model with correlation $\rho$ (proxy $\hat r\sim\mathcal N(0,1)$ under the reference, gold $r=\rho\hat r+\sqrt{1-\rho^2}\varepsilon$), the expected gold reward of best-of-$n$ is $\rho\,\mathbb E[\max_n\hat r]$: increasing in $n$ — Gaussian *linear* misspecification produces no overoptimization. (c) If gold is a concave quadratic function of the proxy, $r=a\hat r-c\hat r^2$ ($c>0$), and the proxy under the policy is Gaussian with mean $m$ and variance $s^2$ (the KL-regularized optimum with Gaussian proxy), then the gold gain relative to the reference is $d(\alpha-\beta d)$ in $d=\sqrt{D_{\textrm{KL}}}$ with $\alpha=\sqrt2\,a\,s$, $\beta=2cs^2$ (up to the reference-level offset). For best-of-$n$ the same form holds approximately, with $\mathbb E[\max_n\hat r]\approx\sqrt2\,d\cdot\kappa_n$, $\kappa_n\in[0.91,0.95]$ for $n$ from 2 to $10^4$ (prepared CPU run). (d) The Gao et al. RL form $d(\alpha-\beta\log d)$ is an empirical fit and is not derived here; the notebook fits both forms to the committed PPO/GRPO logs and reports which has the smaller bootstrap residual.
*Assumptions.* Continuous rewards without ties for (a); independence of the gold noise from the proxy for (b); the second-order Taylor form of gold as a function of the proxy and a Gaussian proxy law for (c) — stated as a *model*, not as a theorem about language models.
*Steps.*
1. (a) The probability that sample $y$ is selected as the best of $n$ is $n\,F(\hat r(y))^{n-1}\pi_{\textrm{ref}}(y)$ with $F$ the proxy cdf under the reference, so $\pi_n(y)/\pi_{\textrm{ref}}(y)=nF^{n-1}$. Then $D_{\textrm{KL}}=\mathbb E_{\pi_n}[\log n+(n-1)\log F]$. Under $\pi_n$, $U=F(\hat r)$ has density $nu^{n-1}$ (the maximum of $n$ uniforms), and $\mathbb E[\log U]=-1/n$. Hence $D_{\textrm{KL}}=\log n-\frac{n-1}n$.
2. (b) $\mathbb E[r\mid\hat r]=\rho\hat r$ (the regression of gold on proxy), and the best-of-$n$ selection depends only on $\hat r$, so $\mathbb E[r]=\rho\,\mathbb E[\max_n\hat r]$, which increases with $n$. With $\mathbb E[\max_n\hat r]\approx\sqrt{2\log n}$ for large $n$ and $d\approx\sqrt{\log n}$, gold $\approx\sqrt2\rho\,d$.
3. (c) For the KL-regularized optimum of a Gaussian proxy, tilting $\mathcal N(0,s^2)$ by $e^{\hat r/\beta_{\textrm{KL}}}$ gives $\mathcal N(s^2/\beta_{\textrm{KL}},s^2)$, with KL $=\frac{s^2}{2\beta_{\textrm{KL}}^2}$; so $d=\sqrt{\textrm{KL}}=\frac{s}{\sqrt2\,\beta_{\textrm{KL}}}$ and the proxy mean is $m=s^2/\beta_{\textrm{KL}}=\sqrt2\,s\,d$. Gold mean change: $a(m-0)-c(\mathbb E\hat r^2-s^2)=a\sqrt2sd-c\,m^2=a\sqrt2\,s\,d-2cs^2d^2$.
4. For best-of-$n$ with $s=1$: $\mathbb E\max_n\hat r=\kappa_n\sqrt2 d$ with $\kappa_n$ slowly varying; the quadratic form follows with $\mathbb E\hat r^2$ computed exactly by quadrature; the fit is checked numerically.
5. Interpretation: $\beta>0$ is the curvature of gold in the proxy; the peak of the curve is at $d^\ast=\alpha/(2\beta)$ with value $\alpha^2/(4\beta)$; a policy trained past $d^\ast$ loses gold reward while the proxy keeps rising.
*Numerical checks (cell, numpy fp64).*
- (a) For $n\in\{2,4,16,64,256,1000\}$ compute the best-of-$n$ density of a standard normal proxy on a grid of $2\cdot10^5$ points over $[-9,9]$ and integrate $\int\pi_n\log(\pi_n/\varphi)$; assert agreement with $\log n-(n-1)/n$ to $10^{-4}$ (prepared run, e.g. $n=64$: 3.1745 both ways; $n=1000$: 5.9088).
- (b) Print $\mathbb E[\max_n\hat r]$ and $\sqrt{2d^2}$ and their ratio $\kappa_n$ (prepared run: 0.908 at $n=2$, 0.913 at 4, 0.930 at 64, 0.943 at 1000, 0.950 at $10^4$).
- (c) Take $a=1$, $c=0.08$, compute exact gold gain $a\mathbb E\max_n-c(\mathbb E\max_n^2-1)$ for $n\in\{1,2,\dots,16384\}$ (18 values), fit $d(\alpha-\beta d)$ by least squares; assert the max absolute residual is below 1% of the peak (prepared run: $\alpha=1.341$, $\beta=0.133$ for $n\le64$, max residual $4.7\cdot10^{-4}$; $\alpha=1.346$, $\beta=0.136$ over the whole range, max residual $5.1\cdot10^{-3}$ against a peak of 2.78).
- (d) The "all subsets" estimator `bon_expected_gold` equals brute-force enumeration over all $\binom83$ subsets for $N=8,n=3$ to $10^{-12}$.

**D2. How much KL suppresses a hedge: the threshold in $\beta$.**
*Statement.* A response pattern ("hedge") has reference probability $h$ and reward $1$; all other responses have reward $0$. At the KL-regularized optimum (`eq_kl_optimum`, sequence level) $\pi^\ast_h=\frac{h\,e^{1/\beta}}{h\,e^{1/\beta}+1-h}$. (a) $\pi^\ast_h>h$ for every finite $\beta$: the penalty can reduce the hedge only toward its reference level, never below it. (b) $\pi^\ast_h=\tfrac12$ at $\beta_{1/2}=1/\ln\frac{1-h}h$. (c) $D_{\textrm{KL}}(\pi^\ast\Vert\pi_{\textrm{ref}})=\pi^\ast_h\ln\frac{\pi^\ast_h}h+(1-\pi^\ast_h)\ln\frac{1-\pi^\ast_h}{1-h}$, increasing as $\beta$ falls.
*Steps.* (1) Apply `eq_kl_optimum` to the two-cell partition {hedge, rest}: the tilt multiplies the hedge by $e^{1/\beta}$ and the rest by $1$; normalize. (2) $\pi^\ast_h>h\iff e^{1/\beta}>1$. (3) Solve $he^{1/\beta}=1-h$. (4) Substitute into the KL definition. In the book's four-prompt instance (`subsec_rl_sequences_hacking`) the threshold $\beta=1/4$ is the value at which the hedge is as probable as each of eight alternatives, i.e. a different partition and a different criterion — the section states which.
*Numerical check.* Enumerate a 9-response categorical with $h=0.002$ for the hedge and the rest uniform, compute $\pi^\ast$ for $\beta\in\{0.05,0.1,0.25,0.5\}$ and verify (a)–(c) to $10^{-12}$; print $\beta_{1/2}$ for $h\in\{0.002,0.02\}$ (= 0.161, 0.256). The LM cell computes $h$ from 2,000 samples and the predicted $\beta_{1/2}$, then compares with the sweep; the sequence-level KL optimum is over *all* responses, so the match is expected to be qualitative (stated in the text).

#### From-scratch implementation

Budget ≤ ~120 lines: `bon_kl` (3), `bon_expected_gold` (15), `gao_fit` (25), `length_residualize` (8), `rm_ensemble` (12), `sloppy_math` + `hedge_rate` (15), `sycophancy_gap` (15), `over_refusal_rate` (12), `principle_judge` (15).
Tests: (1) D1(a)–(d) and D2 checks above; (2) `sloppy_math` fixtures: boxed `12` vs gold `12` → 1; boxed `12, 13, 64` with gold `64` → 1 (the loophole), `hedge_rate` → 1; exact `verify_math` → 0; (3) `gao_fit` recovers $(\alpha,\beta)$ from synthetic data generated with known parameters and noise (error within the bootstrap interval, seed 0); (4) `length_residualize` output is uncorrelated with log length (|Spearman| < 0.02 on the fit set); (5) `rm_ensemble(agg='min')` equals the elementwise minimum of the member scores.

#### In production

There is no single library for overoptimization studies. TRL `RewardTrainer` (8.1) and `GRPOTrainer` (8.5) supply the pipeline pieces; the *gold-vs-proxy protocol* is the practice: held-out proxies, independent gold, fixed evaluation prompts. The from-scratch estimators and the verifier tests are the oracles; RewardBench (`allenai/reward-bench`, 8.1) is the community check that a gold RM is itself reliable in the categories used. For the over-refusal and safety side, the notebook names the public benchmark families in the Further reading and does not rely on any of them as ground truth.

#### Headline experiment

- **Hypothesis.** (i) Best-of-$n$ against the 8.1 reward model yields a gold-vs-$d$ curve that is concave with a positive fitted $\beta$ (95% interval excluding 0), and the curve for the length-residualized proxy has a smaller $\beta$ or none; (ii) GRPO against `sloppy_math` drives the hedge rate and the sloppy reward up while exact accuracy falls below the starting policy's, and a KL coefficient above the predicted $\beta_{1/2}$ keeps the hedge near its reference rate at the price of slower gains; (iii) ensembles raise the gold curve at large $d$ but do not remove the decline.
- **Setup.** (BoN) π_SFT, 200 held-out prompts, $N=64$ samples each (temperature 1.0, 192 new tokens), proxy = 8.1 RM (and each of three RMs for the ensemble), gold = Skywork-V2-1.7B; $n\in\{1,2,4,8,16,32,64\}$ with the all-subsets estimator. (RL) π_SFT + LoRA, `ArithmeticTasks('mixed')`, K=8, 32 prompts/step, 40 steps, `sloppy_math` as the training reward, exact `verify_math` and `hedge_rate` as evaluation, $\beta\in\{0,\beta_1,\beta_2\}$ with $\beta_1<\beta_{1/2}<\beta_2$ (k3 loss term, 8.5 `kl_estimator`); seeds {0,1} for $\beta=0$ live, other arms from committed logs.
- **Procedure.** Sample, score with proxy(s) and gold, compute curves and fits; run the RL arms; evaluate every 5 steps on 200 eval problems; run the sycophancy, over-refusal and regression cells for π_SFT, π_DPO (8.3), π_GRPO(exact reward, 8.5) and π_GRPO(sloppy reward).
- **Metrics.** gold and proxy score vs $d$ with bootstrap bands; fitted $(\alpha,\beta)$ with intervals; exact accuracy, sloppy reward, hedge rate, length, sampled KL per RL step; sycophancy gap; over-refusal rate; regression-suite deltas with intervals.
- **Plots.** `fig_lm_oo_gao` (x: $\sqrt{\textrm{KL}}$ (best-of-$n$ analytic); y: mean score; lines: proxy, gold, gold under length-residualized proxy, gold under ensemble; bands: bootstrap over prompts; the fitted $d(\alpha-\beta d)$ dashed with its peak marked); `fig_lm_oo_hedge` (3 panels vs step: sloppy reward and exact accuracy; hedge rate; sampled KL; series by $\beta$, with the predicted $\beta_{1/2}$ in the caption); `fig_lm_oo_frontier` (exact accuracy vs $\sqrt{\textrm{KL}}$ for the $\beta$ runs and for early-stopped $\beta=0$); `fig_lm_oo_suite` (post-training stage × metric heat-map with intervals).
- **Expected outcome.** Directions in the hypothesis. Gao et al. report that best-of-$n$ gold reward follows $d(\alpha-\beta d)$ and RL follows $d(\alpha-\beta\log d)$ with coefficients that scale smoothly with proxy-model size and data; this section reports *its own* fitted values and intervals ("to be measured"), not theirs.
- **Noise check.** Bootstrap over prompts for every curve; for the RL arms, 2 seeds for $\beta=0$ and the range as the noise floor — a suppression claim requires the hedge rate of the penalized run to fall below the unpenalized seed range; the fit-form comparison (BoN vs RL form) uses the bootstrap residual, never a single fit.
- **T1 reduction.** native (T1 tier); T2 variant: the full 1,000-prompt BoN curve and a second proxy size, committed.
- **Artifacts.** `lm-overopt-logs`: BoN score tables, per-arm RL logs, suite results.

#### Figures and tables

`fig_lm_oo_gao`, `fig_lm_oo_hedge`, `fig_lm_oo_frontier`, `fig_lm_oo_suite`, `tab_lm_oo_suite` (metric families and what each guards against), `tab_lm_oo_exploits` (exploit | symptom | counterfactual test | mitigation), `tab_lm_oo_threshold` (D2: measured $h$, predicted $\beta_{1/2}$, observed suppression β).

#### Pitfalls and misconceptions

- Using the proxy's own held-out accuracy as the overoptimization detector.
- A gold model trained on the same data family as the proxy (shared biases): the two agree on exactly the artifacts that matter. Check gold on RewardBench categories and by human reading of the top-$n$ responses.
- Plotting against reward rather than against $\sqrt{\textrm{KL}}$: curves from different methods are not comparable.
- Estimating best-of-$n$ KL with the sampled-KL estimators (it has the exact formula).
- Concluding "the penalty works" from a single β, or "does not work" from a single seed.
- Reading a falling refusal rate as success without the over-refusal and should-refuse sets together.
- Treating a constitution-labeled preference set as safe because the principles are good.

#### Summary (draft bullets)

- Overoptimization is a gap between proxy and gold along a KL axis; best-of-$n$ has an exact KL, so its curve needs no estimation.
- With Gaussian linear misspecification there is no overoptimization; a concave gold-in-proxy relation produces $d(\alpha-\beta d)$ under both best-of-$n$ and the KL-regularized optimum.
- Language-model exploits are concrete and testable: length, agreement with the user, and format loopholes such as the hedge.
- A KL penalty can return a hedge toward its reference rate but not below it; its threshold is a computable function of the reference probability.
- Ensembles, length control and held-out proxies mitigate; none removes the effect; independent evaluators detect it.
- A post-trained model is evaluated by a vector of metrics with intervals.

#### Exercises

1. **(check)** Why does the best-of-$n$ KL not depend on the reward model? *Sketch:* selection depends only on the *ranks* of the proxy, and the rank distribution of $n$ i.i.d. draws is the same for any continuous reward.
2. **(check)** The fitted curve is $d(1.2-0.3d)$. Where is its peak, and what gold gain is attained there? *Sketch:* $d^\ast=\alpha/(2\beta)=2$; value $\alpha^2/(4\beta)=1.2$.
3. **(derive)** For the hedge model of D2, compute the KL cost of reaching $\pi^\ast_h=1/2$. *Sketch:* $D=\tfrac12\ln\frac1{2h}+\tfrac12\ln\frac1{2(1-h)}=\tfrac12\ln\frac1{4h(1-h)}$; for $h=0.002$ it is $\tfrac12\ln125.3=2.41$ nats.
4. **(derive)** Extend D1(b) to correlation $\rho$ plus an additive gold bias that grows with the proxy, $r=\rho\hat r-\gamma\hat r^2$, and find the $n$ at which best-of-$n$ stops improving. *Sketch:* gold $=\rho\mathbb E\max_n-\gamma\mathbb E\max_n^2$; with $\mathbb E\max_n\approx\sqrt{2\log n}$ the stationarity gives $\sqrt{2\log n^\ast}\approx\rho/(2\gamma)$.
5. **(measure)** Repeat the best-of-$n$ sweep with the 4B gold model; predict whether the fitted peak $d^\ast$ moves left or right and measure the interval.
6. **(measure)** Replace `sloppy_math` by a verifier that accepts only the *first* number in the box; predict the new loophole, find it by reading 20 samples at the end of training, and measure it.
7. **(open)** Write a five-item constitution for a domain you care about, label 200 pairs with `principle_judge`, and design the test that shows whether training on these labels changes behaviour in a way the constitution did not intend.

#### Further reading

- Reward Model Overoptimization ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760)): the gold/proxy protocol and the two functional forms; Sycophancy ([arXiv:2310.13548](https://arxiv.org/abs/2310.13548)); Constitutional AI ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073)); Deliberative Alignment ([arXiv:2412.16339](https://arxiv.org/abs/2412.16339)).
- New: Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms ([arXiv:2406.02900](https://arxiv.org/abs/2406.02900)); Helping or Herding? Reward Model Ensembles ([arXiv:2312.09244](https://arxiv.org/abs/2312.09244)); Catastrophic Goodhart ([arXiv:2407.14503](https://arxiv.org/abs/2407.14503)); A Long Way to Go ([arXiv:2310.03716](https://arxiv.org/abs/2310.03716)).
- Tutorials: Lilian Weng, [Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/); Lambert, RLHF Book (over-optimization chapter).
- Code: [reward-bench](https://github.com/allenai/reward-bench); [TRL](https://github.com/huggingface/trl).

#### Risks and open decisions

- *Safety content.* The should-refuse set is written by the author and kept private. What is published is the category descriptions, the judge rubric and aggregate refusal rates (per category and overall); never the prompts and never harmful completions. The editor reviews the set before it is frozen (approved 2026-10-29), so the review is part of the authoring plan and no longer a condition for keeping the set. Because readers cannot rerun the private prompts, the notebook loads the aggregate table from committed logs (the `over_refusal_rate` code path is exercised live on the public over-refusal set). Consequence stated in the text: the published rates are not independently reproducible, which is why the section reports them next to over-refusal and sycophancy as a vector and never as a score.
- *Gold and proxy share failure modes* (both are reward models): add the 9B judge pairwise as a second gold and report both.
- *The hedge is not discovered at 0.6B in 40 steps*: the warm-start fallback of narrative step 3; or use `profile='hard'` so that exact matching is rarely achieved and the hedge pays more.
- *Optional section:* if 8.7 is cut, move the BoN sweep (narrative step 2) and D1 into 8.1's headline experiment as a short "preview", and the hedge study into the 8.5 exercises.
