# Further Reading
:label:`sec_transformers-further-reading`

The sections of this chapter build one block, one language model, one cache,
one encoder and one encoder--decoder, one vision model, one mixture of
experts, and one small scaling study. This section points to topics they had
no room for: other ways to keep the block stable, the training recipe in
more detail, the systems that serve a cache, the uses of each wiring, the
recipes that make vision transformers work on modest data, routing beyond
the layer of :numref:`sec_moe`, and the questions that a scaling law leaves
open. Several of these topics belong to the Language Models part of the
book, and their entries here only say what they change and where they are
treated. Implementations, tutorials, and courses for the material that the
chapter does teach are listed in its introduction,
:numref:`chap_transformers`. Nothing else in the book depends on this
section.

## The Block

### Bounds on the Logits

QK-norm, in :numref:`subsec_qk-norm`, bounds the attention logits by
normalizing queries and keys. Released models arrived at it from different
starting points: Qwen3 :cite:`Yang.Li.Yang.ea.2025` removed the biases of
its query, key, and value projections and added QK-norm, OLMo 2
:cite:`OLMo.2025` added it together with a loss on the output normalizer,
and Gemma 3 :cite:`Gemma.Team.2025` adopted it in place of the soft-capping
of Gemma 2 :cite:`Gemma.Team.2024`. Soft-capping squashes a logit $z$ as
$c\tanh(z/c)$, with $c = 50$ for the attention logits and $c = 30$ for the
output logits. It bounds the logits without normalizing anything. Fused
attention kernels, however, never form the logits, so soft-capping does not
fit them. The
output-side counterpart is the z-loss of PaLM :cite:`chowdhery2022palm`, a
penalty of about $10^{-4} (\log Z)^2$ on the normalizer $Z$ of the output
softmax that keeps it near one. :citet:`Wortsman.Liu.Xiao.ea.2024` show
that small models already exhibit the divergence of output logits that
z-loss prevents, once the weight decay is low.

### Other Placements and No Normalization

:numref:`fig_norm-taxonomy` drew four placements of the normalization. OLMo
2 :cite:`OLMo.2025` uses a fifth: it normalizes the output of each branch
before adding it to the stream, rather than its input. DeepNet
:cite:`Wang.Ma.Dong.ea.2024` keeps the post-norm arrangement of
:numref:`sec_transformer-block` but multiplies the stream by a constant
larger than one before adding the branch, and scales down the branch's
initialization; with these two changes, post-norm stacks of a thousand
layers train. Dynamic
tanh :cite:`Zhu.Chen.He.ea.2025` removes the normalization altogether: it
replaces every LayerNorm by an elementwise $\tanh(\alpha x)$ with a learned
scalar $\alpha$, so that no statistic is computed over the token. At the
reported scales it matches normalized models in vision and language.

### Initialization of Deep Stacks

`GPT` in :numref:`sec_gpt` uses GPT-2's initialization, with the projections
that write to the stream scaled by $1/\sqrt{2L}$. SmallInit
:cite:`Nguyen.Salazar.2019` draws the attention weights with a smaller
standard deviation and, together with pre-norm, trains translation models
without warmup. OLMo 2 :cite:`OLMo.2025` went the other way: it dropped its
depth-scaled scheme for a plain normal distribution with standard deviation
0.02, and reports that the scales of activations and gradients are better
preserved through training. muP :cite:`Yang.Hu.Babuschkin.ea.2022`, which
:numref:`sec_scaling` teaches, makes both the initialization and the
learning rates depend on width, so that hyperparameters tuned on a small
model carry over to a large one.

### Modifications That Transfer

Many changes to the block were proposed between 2017 and 2020, each with a
gain in its own paper. :citet:`Narang.Chung.Tay.ea.2021` reimplemented
dozens of them in one codebase and found that few improve on the original
transformer when the setting changes. RMSNorm and the gated feed-forward
networks of :numref:`sec_transformer-block` are among the few that do.

## The Training Recipe

### Adam's Second Moment at Small Batch

The runs of :numref:`sec_gpt` use PyTorch's default $\beta_2 = 0.999$,
while reports of large pretraining runs use $0.95$.
:citet:`Porian.Wortsman.Jitsev.ea.2024` find that for models of 5M to 220M
parameters trained with batches of 128 sequences or fewer, $0.95$ is a poor
choice and values between $0.99$ and $0.999$ work better. A small batch
gives noisy gradients, and the longer average of a larger $\beta_2$ smooths
the estimate of their second moment.

### Constant Rates and Cooldowns

Cosine decay, which `LMOptimizer` uses, needs the length of the run in
advance. A constant learning rate followed by a short cooldown matches a
tuned cosine schedule, and a cooldown over the last fifth of the steps with
a $1 - \sqrt{t}$ shape works best :cite:`Hagele.Bakouch.Kosson.ea.2024`.
MiniCPM :cite:`Hu.Tu.Han.ea.2024` named this schedule warmup--stable--decay.
For scaling studies this changes the cost: one long run with cooldowns
branched off at several points yields the losses of many data budgets,
where :numref:`sec_scaling-laws` trains a separate run for each.
:numref:`sec_scheduler` teaches the schedule itself.

### Pretraining Data

GPT-2 was trained on WebText, about 40 GB of web pages linked from Reddit,
and later models train on filtered web crawls of trillions of tokens. How
such corpora are filtered, deduplicated, and mixed, and how models are
adapted after pretraining, is the subject of the Language Models part.

## Generation and the Cache

### Serving Engines

A server answering many requests at once must decide which sequences share
a forward pass and where their caches live. Continuous batching
:cite:`Yu.Jeong.Kim.ea.2022` schedules at the granularity of one decoding
step, admitting and retiring sequences at every step instead of waiting for
the longest member of a batch. Paged memory :cite:`Kwon.Li.Zhuang.ea.2023`
stores each cache in fixed blocks located through a table, like the pages of
virtual memory, instead of reserving the maximum length for every sequence
as `init_cache` of :numref:`sec_kv-cache` does. Prefix caching
:cite:`Zheng.Yin.Xie.ea.2024` keeps the caches of finished requests in a
tree keyed by their tokens and reuses them when a new request shares a
prefix, such as a system prompt. The Language Models part builds these
mechanisms; :numref:`sec_model_serving` shows them in use.

### Speculative Decoding

Decoding reads all weights to produce one token
(:numref:`subsec_kv-cache-prefill-decode`). Speculative decoding
:cite:`Leviathan.Kalman.Matias.2023,Chen.Borgeaud.Irving.ea.2023` lets a
small draft model propose several tokens, which the large model scores in
one pass. Let $p$ and $q$ be the distributions of the large and the draft
model. Accepting each proposed token with probability $\min(1, p(x)/q(x))$,
and resampling a rejected one from the normalized $\max(0, p - q)$, leaves
the output distribution exactly that of the large model. Medusa
:cite:`Cai.Li.Geng.ea.2024` replaces the draft model by extra decoding
heads, EAGLE :cite:`Li.Wei.Zhang.ea.2024` drafts from the large model's
features, and multi-token prediction :cite:`Gloeckle.Idrissi.Roziere.ea.2024`
trains the model to predict several tokens at once. The Language Models
part treats these methods.

### Compressing a Trained Model's Cache

:numref:`subsec_kv-cache-low-rank` compressed GPT-2's cache with a fixed
basis and :numref:`subsec_kv-cache-sinks` with a window. Methods applied
after training go further along both lines. Palu :cite:`Chang.Lin.Lin.ea.2025`
factorizes the key and value projections of a trained model into low-rank
pairs and caches the latent, and Loki :cite:`Singhania.Singh.He.ea.2024`
keeps the full cache but uses a low-dimensional key subspace to choose the
tokens that receive exact attention. H2O :cite:`Zhang.Sheng.Zhou.ea.2023`
keeps the recent tokens plus those that have accumulated the most attention,
and SnapKV :cite:`Li.Huang.Yang.ea.2024` chooses positions per head from the
attention of the last prompt tokens. A benchmark of single questions hides
the cost of such evictions. SCBench :cite:`Li.Jiang.Wu.ea.2025` finds that
methods whose memory grows more slowly than the context fail when later
turns of a conversation need what an earlier turn evicted. KVQuant
:cite:`Hooper.Kim.Mohammadzadeh.ea.2024` extends the quantization of the
section's exercise to three bits per entry, quantizing keys per channel
before RoPE and keeping a few outliers exact. The Language Models part
treats compression at inference time.

### Multi-Head Latent Attention in Full

:numref:`sec_kv-cache` sketches the latent cache of DeepSeek-V2
:cite:`DeepSeek-AI.2024`. In full, keys and values of all heads are
reconstructed from one compressed vector per token, and the up-projection of
the keys can be absorbed into the query projection, so that attention runs
on the latent directly. Absorption fails for a rotated key, because the
rotation depends on the position and sits between the two matrices, so the
model adds a small rotary key shared by all heads beside the latent.
DeepSeek-V2 reports a cache 93% smaller than that of its dense 67B
predecessor. The Language Models part treats the design in its own section.

### Gated Attention in Full

:numref:`subsec_kv-cache-sinks` describes the sigmoid gate of
:citet:`Qiu.Wang.Zheng.ea.2025` as the design that makes sinks unnecessary.
The authors compared 30 variants and found the best to be an elementwise
gate, computed from the layer input, on each head's output before the output
projection. In models of 1.7B dense and 15B mixture-of-experts parameters
trained on 3.5T tokens, it lowered perplexity, tolerated larger learning
rates, and reduced the attention on the first token from about half to
about a twentieth. Qwen3-Next :cite:`Qwen.Team.2025` uses it.

## Wirings and Images

### Uses of Each Wiring

The three wirings of :numref:`sec_transformer` divide the work of current
systems. Encoders such as ModernBERT :cite:`Warner.Chaffin.Clavie.ea.2024`
serve embedding, retrieval, reranking, and classification, where each
document is encoded once with context on both sides. Encoder--decoders
serve tasks with a distinct input: the T5 family
:cite:`raffel2020exploring` frames every task as text to text, and Whisper
:cite:`radford2023whisper` decodes text from an encoded recording. Decoders
alone serve generation and in-context learning :cite:`brown2020language`.
:citet:`Wang.Roberts.Hesslow.ea.2022` compared the wirings at more than 5B
parameters and found the decoder best for zero-shot use after pretraining
alone, and an encoder with full visibility of the input best after
multitask fine-tuning. T5Gemma :cite:`Zhang.Moiseev.Ainslie.ea.2025`
converts pretrained decoders into encoder--decoders and finds them stronger
at a similar cost of inference. The Language Models part treats pretrained
encoders in depth.

### Learned Queries as an Interface

The Perceiver of :numref:`sec_transformer` reads a long input through a few
learned queries. Vision--language models use the same device to hand images
to a language model. Flamingo :cite:`alayrac2022flamingo` compresses a
variable number of image and video features into 64 visual tokens, and
BLIP-2's Q-Former :cite:`Li.Li.Savarese.ea.2023` is a small transformer of
32 learned queries between a frozen image encoder and a frozen language
model. DETR :cite:`Carion.Massa.Synnaeve.ea.2020` used learned object
queries earlier, for detection: each of a hundred queries cross-attends into
the image features and produces one object or none. :numref:`chap_cv`
treats detection.

### Vision Transformers on Modest Data

A vision transformer lacks the locality that a convolution builds in, and
:numref:`sec_vision-transformer` showed what that costs on a small dataset.
Several recipes close the gap without more data. DeiT
:cite:`touvron2021training` trained on ImageNet-1k alone with strong
augmentation and regularization and a distillation token that learns from a
convolutional teacher. :citet:`Beyer.Zhai.Kolesnikov.2022` replaced the
class token by average pooling and the learned positions by fixed 2-D
sinusoids and reached 80% top-1 in 300 epochs, and
:citet:`Steiner.Kolesnikov.Zhai.ea.2022` found that with the right
augmentation and regularization, more compute gives about the same accuracy
as ten times more data. Compact transformers :cite:`Hassani.Walton.Shah.ea.2021`
replace the patch stem by a small convolutional tokenizer and the class
token by attention pooling; with small patches and 300 epochs, models of
under 4M parameters beat ResNet-18 on CIFAR-10.

### Windows, Two-Dimensional Rotations, and Registers

Swin :cite:`liu2021swin` restricts attention to local windows that shift
between blocks, and merges patches between stages. This restores locality
and a pyramid of resolutions, and makes the cost linear in the number of
patches, as detection and segmentation at high resolution require. RoPE in
two dimensions :cite:`Heo.Park.Han.ea.2024` rotates by the row and the
column of a patch and extrapolates to higher resolutions better than a
learned table. The register tokens of :numref:`sec_vision-transformer` need
not be trained: :citet:`Jiang.Dravid.Efros.ea.2025` trace the outlier
tokens to a few neurons and move their activation into an extra token at
test time, with nearly the effect of trained registers.

### Self-Supervised and Image--Text Encoders

The ViT of the chapter learns from labels. Masked autoencoders
:cite:`he2022masked` learn by reconstructing masked patches from the
quarter of patches that remain, and DINOv2
:cite:`Oquab.Darcet.Moutakanni.ea.2024` by self-distillation on a curated
image collection. CLIP :cite:`radford2021learning` trains an image encoder
against a text encoder with a contrastive loss over the batch, and SigLIP
:cite:`Zhai.Mustafa.Kolesnikov.ea.2023` replaces the softmax over the batch
by a sigmoid loss per pair, which decouples the loss from the batch size.
Encoders of this kind are the vision towers of most vision--language
models. :numref:`chap_cv` and the Language Models part treat them.

## Experts

### Router Stability

Large sparse models are prone to loss spikes that start in the router,
whose logits can grow without bound. ST-MoE
:cite:`Zoph.Bello.Kumar.ea.2022` adds a router z-loss, a penalty on the
squared log-sum-exp of the router logits, which keeps them small without
changing which expert wins. With it, ST-MoE trains a sparse model of 269B
parameters at the computation per token of a 32B dense model.

### Expert-Choice Routing

In the layer of :numref:`sec_moe` each token chooses its experts, so loads
must be balanced afterwards. Expert-choice routing
:cite:`Zhou.Lei.Liu.ea.2022` reverses the roles: each expert picks the
tokens it scores highest, up to its capacity, so loads are balanced by
construction. A token may then be picked by no expert or by many, and its
experts depend on the other tokens of the batch, which complicates decoding
one token at a time.

### Sigmoid Gates and Limited Groups

DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` scores each expert with its own
sigmoid instead of a softmax over experts, normalizes the scores of the
selected experts, and restricts the experts of each token to a few groups of
devices, which bounds the communication per token. The layer of
:numref:`sec_moe` keeps the softmax router of the Switch transformer and
adopts only DeepSeek-V3's bias controller.

### Sparsity and Expert Width

:numref:`sec_moe` trains one configuration. :citet:`Clark.Casas.Guy.ea.2022`
fit scaling laws for routed models and find that routing helps at every
dense size up to 1.3B parameters, with gains that shrink as the dense model
grows. :citet:`Krajewski.Ludziejewski.Adamczewski.ea.2024` treat the width
of the experts as a hyperparameter and scale the tokens with the model; the
gap between mixtures and dense models then grows with compute, and experts
as wide as the dense feed-forward network are almost never optimal. Later
work finds an optimal sparsity that grows with compute
:cite:`Abnar.Shah.Busbridge.ea.2025` and mixtures that beat dense models of
the same memory, not only of the same computation
:cite:`Ludziejewski.Pioro.Krajewski.ea.2025`. Released models have become
sparser accordingly: Qwen3-Next :cite:`Qwen.Team.2025` activates about 3B of
its 80B parameters, with 512 experts per layer, and Kimi K2
:cite:`Kimi.Team.2025` combines latent attention with 384 experts per layer.
The Language Models part treats sparsity in scaling laws.

### Experts across Devices

When the experts do not fit on one accelerator, different experts live on
different devices, and the tokens travel to their experts and back in an
all-to-all exchange before and after each expert layer. This is expert
parallelism, introduced with GShard :cite:`Lepikhin.Lee.Xu.ea.2021`. Fixed
capacities, as in the layer of :numref:`sec_moe`, make the exchange static.
The Language Models part treats expert parallelism and the serving of
mixtures.

## Scaling

### Reconciling Kaplan and Chinchilla

:citet:`kaplan2020scaling` found that the compute-optimal model size grows
as $C^{0.73}$, and Chinchilla :cite:`hoffmann2022training` as $C^{0.50}$.
:numref:`sec_scaling-laws` quotes the finding of
:citet:`Pearce.Song.2024` that counting only non-embedding parameters at
small scale explains most of the difference.
:citet:`Porian.Wortsman.Jitsev.ea.2024` account for the rest: the FLOPs of
the output layer, the length of the warmup, and optimizer settings tuned to
each scale. With these corrected, the two fits agree.

### Repeated Data

:numref:`sec_scaling-laws` quotes the finding of
:citet:`Muennighoff.Rush.Barak.ea.2023` that up to about four passes over
the data are nearly as good as fresh data. Their fit gives the whole curve:
$U$ unique tokens repeated $R$ times are worth
$U + U R^* (1 - e^{-R/R^*})$ fresh tokens, with $R^* \approx 15$, so the
value of further passes decays to zero. Budgets beyond the compute optimum
are treated in the Language Models part.

### Precision as a Scaling Variable

:citet:`Kumar.Ankner.Spector.ea.2025` treat the precision of training as a
reduction of the effective parameter count. Quantizing a model after
training hurts more the longer it was trained, so that beyond some point
more pretraining data makes the quantized model worse. They also find that
the compute-optimal precision for training lies around seven to eight bits.

### Loss and Abilities

Loss falls smoothly with scale, but some abilities measured downstream
appear to switch on abruptly as it falls :cite:`wei2022emergent`. Whether
such jumps are a property of the models or of metrics that give no credit
for partial answers is debated.

### Departures from the Shared Block

The configurations of :numref:`tab_modern-recipe` share one block. Three
directions depart from it. Hybrids keep full attention in a fraction of the
layers and replace the rest with recurrent mixers of linear cost
:cite:`Kimi.Team.2025b`, the subject of :numref:`chap_modern_rnn`. Long
contexts combine the positional rescaling of
:numref:`sec_positional-information`, the cache policies of
:numref:`sec_kv-cache`, and systems work; Llama 4 :cite:`Meta.2025`
interleaves attention layers without positional encoding. Learned sparse
attention chooses the keys by content. The indexer of DeepSeek-V3.2
:cite:`Liu.Mei.Lin.ea.2025`, for example, selects 2,048 key--value tokens
per query; :numref:`sec_attention-further-reading` describes it.

[Discussions](https://d2l.discourse.group/t/36853)
