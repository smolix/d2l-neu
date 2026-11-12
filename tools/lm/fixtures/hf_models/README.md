# Hugging Face metadata fixtures

Test fixtures for the Language Models part (`make test-lm`). Each
`<org>/<repo>/` directory holds files of the Hugging Face model repository of
that name at the commit pinned in `d2l.HF_REVISIONS`
(`chapter_lm-pretrained/hf-models.md`), resolved on 2026-11-12:

| Repository | Commit | Files |
|---|---|---|
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | `15852e8c16360a2fea060d615a32b45270f8a8fc` | `config.json`, `header.json` |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | `config.json` |
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `config.json` |
| [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | `config.json` |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | `1cfa9a7208912126459214e8b04321603b3df60c` | `config.json` |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` | `config.json` |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | `4adf15748de9e1a0097b6274cee9c15836c41c37` | `config.json`, `header.json` |

`config.json` is the repository's file, byte for byte. `header.json` is
derived from the repository's safetensors headers (read by HTTP range
requests; no tensor data): the shard names and, for every tensor, its dtype
and shape (`data_offsets` and `__metadata__` dropped).

Attribution: the configurations and tensor metadata are part of the model
releases of the Qwen team (Alibaba Cloud) and of Google DeepMind (Gemma 4),
each published under the Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>). They are reproduced here
unmodified (configs) or reduced to the fields named above (headers), for
testing only. No model weights are included.

Regenerate after moving a pin: `.venv-lm/bin/python tools/lm/fetch_hf_fixtures.py`.
