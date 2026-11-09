# Further Reading
:label:`sec_lm_rl_fr`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "FR Further Reading: Post-Training · T0").
Blueprint: none (the plan adds this section).
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** The post-training results that fail the teachable-core test but that a practitioner will meet, each with the one clause saying what it adds.
**Opening:** The reader has run PPO, DPO, GRPO, a lag sweep and a safety evaluation; this notebook names what each of those left out and where the field moved in 2026.
**Teaches:**
- Asynchronous corrections beyond 8.5's paragraph: single-rollout asynchronous optimization (one rollout per prompt, trained GLM-5.2), off-policy RL without importance ratios, the ICML 2026 finding that clipping and masking hold at low lag and none is robust at high lag.
- Training against written principles: constitutional AI (principles label the pairs, then RL from AI feedback) and deliberative alignment (the model reasons over the rules before answering); what each changes in 8.7's two rates.
- Reward-model ensembles (what they mitigate, what they do not) and RL scaling studies (sigmoid-shaped compute curves, which recipe choices change the asymptote and which only the speed).
- Watermarking: a green-list bias at sampling time, detectable with a z-score and robust to light edits; why it is a sampling-time, not a training-time, method.
**Derivation:** none.
**Notebook:** none (reading); 1,000 to 2,000 words, no training run.
**Figures:** none.
**Restate / forward:** none; nothing depends on this notebook.
**Exercises:** none.
**Writer notes:** Each paragraph names one work and its one contribution; no method is taught, no numbers from secondary coverage are quoted (the sota file's GLM-5.2 and speedup claims are secondary). Cite the ICML poster as the primary for the clipping-versus-masking comparison. Keep deliberative alignment's method description to the primary paper's own words. The watermark paragraph states the detection statistic in one sentence and points to 1.4's scoring conventions.
**Sources:**
- Papers: [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508) (SAO); [LLMs Can Learn to Reason Via Off-Policy RL](https://arxiv.org/abs/2602.19362) (OAPL, no importance ratios); [Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs](https://arxiv.org/abs/2602.17616) (VCPO); [Constitutional AI](https://arxiv.org/abs/2212.08073); [Deliberative Alignment](https://arxiv.org/abs/2412.16339); [Helping or Herding?](https://arxiv.org/abs/2312.09244) (ensembles); [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786) (RL scaling); [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226).
- Code and documentation: [AReaL repository](https://github.com/inclusionAI/AReaL) (an asynchronous system to read alongside the corrections); [verl documentation](https://verl.readthedocs.io/en/latest/).
- Courses, tutorials and blogs: [ICML 2026 poster, Stable Asynchrony](https://icml.cc/virtual/2026/poster/62012); [RLHF Book](https://rlhfbook.com/) (its chapters on constitutional AI and over-optimization).
-->
