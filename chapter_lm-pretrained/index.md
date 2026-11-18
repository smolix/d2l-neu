# Working with Pretrained Language Models
:label:`chap_lm_pretrained`

By now you can build a language model from nothing. In :numref:`sec_gpt` we
trained a small GPT on a few hundred megabytes of Wikipedia, and we loaded
the released weights of GPT-2 into the same class with a name map that we
wrote by hand (:numref:`subsec_gpt-loading`). Neither is a model that
anybody would consult. The models that people do consult were trained on
more text and more machines than any of us commands. Few readers of this
book will ever train one. All of them can download one.

That changes what the work is. Since about 2019, when a shared library and
public repositories of weights made the pretrained model the usual place to
start, the effort has moved from training to checking and measuring. What
did we download? Does it compute on our machine what its authors computed on
theirs? When one model scores two points above another, is that a difference
between the models? These questions sound like housekeeping, and they are
where results go wrong. A score, it turns out, belongs to a model together
with a prompt, a rule for reading the answer and a sample of questions.
Every later chapter of this part compares two scores, of a fine-tuned model
and its base or of a compressed model and the original, and this chapter
builds the instruments for that.

The chapter has three movements. The first opens a checkpoint:
:numref:`sec_lm_hf_models` reads its files until the configuration alone
accounts for the bytes of its weights and its cache and bounds its speed,
and :numref:`sec_lm_d2l_checkpoint` runs the same tensors in the book's own
code, which forces us to say how closely two correct programs can agree. The
second takes a score apart in the order in which it is made: the prompt
(:numref:`sec_lm_prompting`), the rule that turns outputs into a number
(:numref:`sec_lm_scoring`), and the error bar that says whether a difference
would survive another set of questions, with a model as the judge where no
answer key exists (:numref:`sec_lm_eval_suites`). The third looks at a model
from both sides. With the weights in hand we can overwrite an activation and
watch what changes, which is how a claim about the inside of a model is
tested (:numref:`sec_lm_interpretability`). The largest models we cannot
hold at all, and :numref:`sec_lm_api` asks what can still be measured when
only text comes back.

Every model in this chapter keeps the weights it was released with.
Fine-tuning has to wait for :numref:`chap_lm_finetuning`, architectures
beyond what a configuration file shows for :numref:`chap_lm_architectures`,
and serving engines for :numref:`chap_lm_serving`. Models that turn text
into vectors, and retrieval, follow directly in :numref:`chap_lm_encoders`.
For licenses, version pins and the choice of hardware we point to the Tools
appendix (:numref:`sec_software_ecosystem`, :numref:`sec_hardware_buyers`).

We assume the transformer of :numref:`chap_transformers` with its key-value
cache (:numref:`sec_kv-cache`), and a first acquaintance with confidence
intervals. We ran the notebooks on one GPU with 24 GB of memory, and none of
them requires one. The long computations were done once and published with
the book. A notebook downloads their results, repeats the cheap
measurements, and without a GPU falls back on a smaller model or on fewer
examples, and says so. One notebook is a recording on every machine: the
requests of :numref:`sec_lm_api` cost money and cannot be repeated exactly,
so its cells replay the responses of one dated session.

```toc
:maxdepth: 2

hf-models
d2l-checkpoint
prompting
scoring
eval-suites
interpretability
api
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter, and each names
the section it goes with. All are freely accessible online. If you read only
three, read the note by Miller on error bars, the guide by Heimersheim and
Nanda on patching, and the essay by He on nondeterminism. The chapter has no
notebook of further reading: fine-tuning, architectures, serving and
retrieval, the subjects it leaves out, have chapters of their own in this
part.

**Courses**

- [Hugging Face LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1) is a tour of the Transformers interface. Read it first if the loading calls and the `Auto` classes of :numref:`sec_lm_hf_models` are new to you.
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) has lectures on evaluation that make a habit of reporting what was held fixed, the habit behind the grid of conventions in :numref:`sec_lm_scoring` and the report card of :numref:`subsec_lm_reproduce_protocol`.
- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) sets an assignment on benchmarking and evaluating language models, the course-side counterpart of :numref:`sec_lm_scoring` and :numref:`sec_lm_eval_suites`, and its guest lecture on interpretability pairs with :numref:`sec_lm_interpretability`.
- [CMU 11-711: Advanced Natural Language Processing](https://cmu-l3.github.io/anlp-spring2026) and [Berkeley CS288](https://cal-cs288.github.io/sp26) each give a lecture to experimental design and human annotation, and both ask students to reproduce a published result, the task that :numref:`subsec_lm_reproduce_protocol` works through on one number.
- [ARENA](https://github.com/ARENA-education/ARENA_materials) is a curriculum of programming exercises whose chapter on the interpretability of transformers treats patching, sparse autoencoders and steering at greater length. :numref:`sec_lm_interpretability` follows it in spirit, on one released model and with plain hooks.

**Prompting and evaluation**

- Lilian Weng's [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) surveys few-shot prompting, the choice and order of the examples, and the label biases that the calibration of :numref:`subsec_lm_calibration` is meant to remove. :numref:`sec_lm_prompting` measures these effects on two checkpoints.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the program behind many published scores of open models, and its authors' [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) explains its design. :numref:`sec_lm_scoring` reproduces its scores item by item, and :numref:`subsec_lm_protocol_example` borrows its protocol for GSM8K.
- [OLMES](https://aclanthology.org/2025.findings-naacl.282/) is a standard that fixes the prompt format, the choice of demonstrations and the normalization of a multiple-choice score, the decisions that :numref:`sec_lm_scoring` varies one at a time.
- Evan Miller's [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) sets out standard errors, paired differences, clustered questions and power for the evaluation of language models. It is the statistics of :numref:`sec_lm_eval_suites` in compact form, with formulas where the section uses the bootstrap.
- The tutorials on the evaluation of language models at [NeurIPS 2024](https://neurips.cc/virtual/2024/tutorial/99524) and [ICML 2024](https://icml.cc/virtual/2024/tutorial/35227) survey the benchmarks and methods of the field, the background of :numref:`subsec_lm_benchmark_lifecycle`.
- Hamel Husain's [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) builds a small suite of tests from the failures of a real product, the development suite of :numref:`subsec_lm_dev_reporting` as a practitioner keeps it.

**Ports and interpretability**

- Sebastian Raschka's [Qwen3 from scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) writes the model of :numref:`sec_lm_d2l_checkpoint` in plain PyTorch and loads the released weights, a second independent port to set beside ours.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) offers recording and overwriting of activations for many models. The `d2l.intervene` of :numref:`sec_lm_interpretability` writes the same operations out for one.
- Stefan Heimersheim and Neel Nanda's [How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255) discusses the choices that :numref:`subsec_lm_patching` has to make: the corrupted prompt, the metric, and the direction of the patch.
- [Neuronpedia](https://www.neuronpedia.org/) is a gallery of the features of published sparse autoencoders with the contexts on which they fire. Browse it after :numref:`subsec_lm_sae_features`, and ask of every label there what that subsection asks of ours.

**Hosted models and reproducibility**

- Horace He's [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) traces the run-to-run drift of a served model to kernels whose arithmetic depends on the batch. It is the explanation behind :numref:`subsec_lm_nondeterminism` and behind the third experiment of :numref:`sec_lm_api`.
- [openai-python](https://github.com/openai/openai-python) is the client package that the `d2l.ChatAPI` of :numref:`subsec_lm_api_client` wraps.
- Marc Brooker's [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) simulates the retry schedules of :numref:`subsec_lm_api_client` for many competing clients and is the source of full jitter.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §18]{.kicker}

Checking and measuring released models<br>
**files · port · prompts · scoring · error bars · inside · behind an API**
:::
:::

::: {.slide title="A Model Is a Download Away; the Work Is to Check and Measure It"}
[The problem]{.kicker}

We can build a small GPT and load GPT-2 by hand. The models that people
consult we can only download, and then three questions are ours to answer.

| question | why it is not obvious |
|---|---|
| **What did we download?** | the name of one model says 2B, and its file of 4.55 GB holds 2,274,069,824 numbers |
| **Does it run the same in our code?** | two correct programs do not agree to the last bit, and a wrong one may agree to the third digit |
| **Is a difference between two scores real?** | a score belongs to a model, a prompt, a rule for reading the answer and a sample of questions |

::: {.d2l-note}
A released checkpoint becomes an instrument once we can answer all three.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- read a configuration into parameters, cache bytes and decode time;
- run a released checkpoint in the book's code, and show agreement with
  its reference to rounding;
- name the prompt and the scoring convention that a score depends on;
- put a paired interval and a minimum detectable effect on a difference;
- test a claim about the inside of a model by overwriting an activation;
- measure the cost, the throughput and the limits of a hosted model.

**Running example:** open checkpoints of 0.6 to a few billion parameters
on one GPU with 24 GB; 500 science questions and 100 arithmetic problems.

**Idea:** know what the files hold, run the tensors yourself, and put an
error bar on every score.
:::

::: {.slide title="Open a Checkpoint, Measure It, Look Inside and from Outside"}
[Chapter map]{.kicker}

| movement | section | question |
|---|---|---|
| open a checkpoint | Pretrained Models with Hugging Face | what is in a checkpoint, and what does it cost to run? |
| | A Released Checkpoint in the Book's Own Code | do its tensors run the same in our code? |
| measure it | Prompting and In-Context Learning | how much does the prompt decide? |
| | Scoring Language Models | which convention gives which score? |
| | Evaluation Suites, Judges and Error Bars | is a difference real? |
| inside and outside | Interpretability | how is a claim about the inside tested? |
| | Models behind an API | what does a hosted model withhold? |

- **Uses:** `d2l.GPT` with GPT-2's weights, the key-value cache.
- **Skips:** fine-tuning, new architectures, serving engines.
:::
