# Further Reading
:label:`sec_transformers-further-reading`

After this chapter you can build a current language model at a size that
trains in minutes on one GPU. Between that model and the ones in use lie
several thousand papers, and most of them can wait. This section collects
the topics the chapter had no room for and says which of them matter to
whom.

If you will train models, read "The Block" and "The Training Recipe"; they
hold settings that current models use. If you will serve them, read
"Generation and the Cache". If you work with images, read the last entry
of "Wirings and Images", on encoders trained without labels or trained to
match images with their captions. How to route experts and how far the
scaling laws reach are research in progress, and the last two groups are
best read as such. The Language Models part takes several of these threads
further, in its chapters on architectures (:numref:`chap_lm_architectures`),
pretraining (:numref:`chap_lm_pretraining`), inference
(:numref:`chap_lm_inference`) and serving (:numref:`chap_lm_serving`).
Nothing else in the book depends on this section.

## The Block

The first two entries fix two ways in which a softmax misbehaves, and
both fixes have reached current models. The placement of the normalization
is still moving. Initialization matters on the day a deep stack refuses to
train, and the last entry is a warning about reading all the others.

### Bounds on the Logits

Attention logits that grow during training saturate the softmax, whose
gradient then vanishes, and long runs spike or stall
(:numref:`subsec_qk-norm`). QK-norm bounds the logits by normalizing
queries and keys, and Qwen3 :cite:`Yang.Li.Yang.ea.2025` and OLMo 2
:cite:`OLMo.2025` added it. Gemma 3 :cite:`Gemma.Team.2025` adopted it in
place of the *soft-capping* of Gemma 2 :cite:`Gemma.Team.2024`, which
squashes a logit $z$ into $c\tanh(z/c)$, with $c = 50$ for the attention
logits and $c = 30$ for the output logits. Soft-capping must act on every
logit inside the fused attention kernel; QK-norm acts on the queries and
keys before the kernel and needs no change to it.

The output logits have a problem of their own. Adding a constant to all of
them leaves the softmax unchanged, so the cross-entropy does nothing to
stop them from drifting together. The *z-loss* of PaLM
:cite:`chowdhery2022palm` pins their level with a penalty of about
$10^{-4} (\log Z)^2$ on the normalizer $Z$ of the output softmax, which
keeps $Z$ near one; PaLM reports that it makes training more stable.

### Gated Attention

Softmax must put each query's weight somewhere, so a head with nothing to
retrieve puts it on the first token (:numref:`subsec_kv-cache-sinks`).
:citet:`Qiu.Wang.Zheng.ea.2025` compared 30 variants of gating. The best
was an elementwise sigmoid gate on each head's output, computed from the
layer's input, which lets a head close its gate instead. In models of 1.7B
dense and 15B mixture-of-experts parameters trained on 3.5T tokens, the
gate lowered perplexity, tolerated larger learning rates, and cut the
attention on the first token from about half to about a twentieth.
Qwen3-Next :cite:`Qwen.Team.2025` uses it.

### Placements of the Normalization

:numref:`sec_transformer-block` compared post-LN and pre-LN, and its block
uses pre-LN. Pre-LN fixes the scale of what each branch reads, but not of
what it writes: a branch whose weights grow during training adds ever
larger vectors to the stream. :numref:`fig_norm-taxonomy` adds the two
placements that recent open models chose. OLMo 2 normalizes what each
branch *writes* before adding it to the stream, and its report credits
this change, together with QK-norm, with smaller and less spiky gradient
norms. Gemma 3 normalizes each branch both before and after the sublayer.
Only post-LN normalizes the stream itself; the other three leave the path
from the first block to the last an identity.

![Four placements of normalization around one sublayer, with the residual stream running bottom to top; the FFN sublayer is treated the same way. Post-LN normalizes the stream itself after each addition, pre-LN, the default of the chapter's block, normalizes what the branch reads, OLMo 2 normalizes what the branch writes, and Gemma 3 normalizes the branch on both sides.](../img/mdl-transformers-norm-taxonomy.svg)
:label:`fig_norm-taxonomy`

Post-norm is not doomed at depth. DeepNet :cite:`Wang.Ma.Dong.ea.2024`, met in
:numref:`sec_transformer-block`, trains post-norm stacks of a thousand layers
by scaling down the initial weights of each branch and scaling up the stream.
What fails at depth is the size of each branch relative to the stream, and
DeepNet shrinks it. Dynamic tanh :cite:`Zhu.Chen.He.ea.2025` takes the
opposite route and removes the normalization. It replaces every LayerNorm by
the elementwise
$\boldsymbol{\gamma} \odot \tanh(\alpha \mathbf{x}) + \boldsymbol{\beta}$,
with a learned scalar $\alpha$ and the per-channel scale $\boldsymbol{\gamma}$
and shift $\boldsymbol{\beta}$ that LayerNorm also has, so that no statistic
is computed over the token. At the scales reported it matches normalized
models in vision and language.

### Initialization of Deep Stacks

`GPT` in :numref:`sec_gpt` uses GPT-2's initialization, with the
projections that write to the stream scaled by $1/\sqrt{2L}$. SmallInit
:cite:`Nguyen.Salazar.2019` draws the attention weights with a smaller
standard deviation and, with pre-norm, trains translation models without
warmup. OLMo 2 went the other way: it dropped its depth-scaled scheme for a
plain normal with standard deviation 0.02 for every parameter, and reports
that activations and gradients then keep their scale better across layers.

The contradiction is smaller than it looks. OLMo 2 normalizes what each
branch writes, and a norm after the output projection cancels any constant
factor in that projection's weights. Scaling them by $1/\sqrt{2L}$ would
leave what the branch writes at initialization unchanged. The right
initialization depends on where the normalization sits, and a scheme taken
from one paper travels only together with its placement. muP
:cite:`Yang.Hu.Babuschkin.ea.2022`, taught in :numref:`sec_scaling`, is the
scheme to learn once hyperparameters tuned on a small model must carry over
to a large one.

### Modifications That Transfer

Between 2017 and 2020 many changes to the block were proposed, each with a
gain in its own paper. :citet:`Narang.Chung.Tay.ea.2021` reimplemented
dozens of them in one codebase and found that few improve on the original
transformer there; RMSNorm and the gated feed-forward networks of
:numref:`sec_transformer-block` are among the few. A gain reported in one
codebase is a hypothesis about the next one.

## The Training Recipe

The recipe of :numref:`sec_gpt` is a sound default at the scale of this
chapter. Two of its settings change when a run grows.

### Adam's Second Moment

`LMOptimizer` keeps AdamW's default $\beta_2 = 0.999$, and
:numref:`sec_gpt` notes that large pretraining runs use $0.95$. Both are
right for their batches. :citet:`Porian.Wortsman.Jitsev.ea.2024` measured
where the boundary lies: for models of 5M to 220M parameters trained with
batches of 128 sequences or fewer, $0.95$ is a poor choice, and values
between $0.99$ and $0.999$ work better. A setting copied from the report of
a large run can be wrong for a small one for this reason alone.

### Constant Rates and Cooldowns

Warmup--stable--decay (:numref:`sec_scheduler`) holds the peak learning
rate for most of the run and decays only at the end, so the length of
training can be decided late. Two results tie it to this chapter.
:citet:`Hagele.Bakouch.Kosson.ea.2024` find that a cooldown over the last
fifth of the steps, with the rate falling as $1 - \sqrt{t}$ while $t$ runs
from 0 to 1 over the cooldown, matches a tuned cosine schedule, which is
what `LMOptimizer` uses. And for a scaling study it changes the cost: one
long run with cooldowns branched off at several points yields the losses
of many data budgets, where :numref:`sec_scaling-laws` trains a separate
run for each.

## Generation and the Cache

This is the most practical group. The serving engines that batch, page
and share caches appeared in :numref:`sec_kv-cache`, and
:numref:`sec_model_serving` shows one at work. Speculative decoding is how
deployed models generate faster today; the compression of a trained
model's cache is an active field whose benchmarks still disagree.

### Speculative Decoding

Decoding reads all weights to produce one token
(:numref:`subsec_kv-cache-prefill-decode`), so a pass that scores five
tokens costs about as much as one that scores one. Speculative decoding
:cite:`Leviathan.Kalman.Matias.2023,Chen.Borgeaud.Irving.ea.2023` turns
this into speed: a small draft model proposes several tokens, which the
large model scores in one pass. Let $p$ and $q$ be the distributions of the
large and the draft model. Accepting each proposed token with probability
$\min(1, p(x)/q(x))$, and resampling a rejected one from the normalized
$\max(0, p - q)$, leaves the output distribution exactly that of the large
model. After a rejection the remaining drafts are discarded, and drafting
starts again from the resampled token; when every draft is accepted, the
same pass also yields the large model's next token. The gain
therefore rests on how often the large model accepts the drafts, which
depends on the task and the sampler as much as on the draft model. On
T5-XXL, Leviathan et al. report decoding two to three times faster with
identical outputs.

Later methods change where the drafts come from. Medusa
:cite:`Cai.Li.Geng.ea.2024` replaces the draft model by extra decoding
heads, EAGLE :cite:`Li.Wei.Zhang.ea.2024` drafts from the large model's
features, and multi-token prediction
:cite:`Gloeckle.Idrissi.Roziere.ea.2024` trains the model itself to predict
several tokens at once.

### Compression of a Trained Model's Cache

:numref:`sec_kv-cache-reduction` compressed GPT-2's cache with a low-rank
basis fitted once on calibration text and with a window of recent tokens,
and it mentioned storing fewer bits. Methods applied to a trained model go
further along all three lines. Palu :cite:`Chang.Lin.Lin.ea.2025`
factorizes the key and value projections and caches the latent. KVQuant
:cite:`Hooper.Kim.Mohammadzadeh.ea.2024` keeps three-bit caches within 0.1
perplexity of the full ones by quantizing keys before RoPE is applied,
using levels spaced unevenly, and keeping a few outliers exact. H2O
:cite:`Zhang.Sheng.Zhou.ea.2023` evicts tokens instead: it keeps the recent
ones plus those that have accumulated the most attention. SnapKV
:cite:`Li.Huang.Yang.ea.2024` chooses the positions to keep per head, from
the attention that the last few prompt tokens pay to each position.

A benchmark of single questions hides the cost of such evictions. SCBench
:cite:`Li.Jiang.Wu.ea.2025` finds that methods whose memory grows more
slowly than the context fail when later turns of a conversation need what
an earlier turn evicted.

## Wirings and Images

The first entry tests the wirings of :numref:`sec_transformer` at scale.
For images, the encoders trained without labels or with captions matter
most; the other image entries matter when the data are few, the
resolution is high, or the patch outputs themselves are used.

### Comparisons of the Wirings

Which wiring makes the best language model?
:citet:`Wang.Roberts.Hesslow.ea.2022` compared a decoder, an
encoder--decoder, and a prefix language model
(:numref:`subsec_encdec-choosing`) at more than 5B parameters, and found
that the best wiring depends on what follows pretraining. The plain decoder
was best for zero-shot use. The models that read their input in both
directions, trained to fill in masked spans, were best after multitask
fine-tuning. T5Gemma :cite:`Zhang.Moiseev.Ainslie.ea.2025` converts
pretrained decoders into encoder--decoders and finds them stronger at a
similar cost of inference, so the cheapest route to a good
encoder--decoder may now start from a decoder.

### Vision Transformers on Modest Data

A vision transformer lacks the locality that a convolution builds in, and
:numref:`sec_vision-transformer` showed what that costs on a small dataset.
Three recipes narrow the gap without more data. DeiT
:cite:`touvron2021training` adds a *distillation token*, a second learned
input beside the class token whose output is trained to match the
predictions of a convolutional teacher. :citet:`Beyer.Zhai.Kolesnikov.2022`
pooled the outputs instead of reading a class token, used fixed 2-D
sinusoidal positions, and reached about 80% top-1 on ImageNet-1k alone in
300 epochs. Compact transformers :cite:`Hassani.Walton.Shah.ea.2021`
replace the patch stem by a small convolutional tokenizer, which puts
locality back at the input. Of the three, the tokenizer is the smallest
change and the one aimed most directly at small datasets.

### Windows and Resolution

Detection and segmentation work at high resolution, where attention over
all patches is too expensive. Swin :cite:`liu2021swin` attends within local
windows that shift between blocks, which makes the cost linear in the
number of patches, and merges neighboring patches between stages into a
pyramid of resolutions, as a CNN has. RoPE in two dimensions
:cite:`Heo.Park.Han.ea.2024` extrapolates to resolutions above those of
training better than a learned table does.

### Registers

The *registers* of :numref:`sec_vision-transformer` are a few input-free
tokens that hold the global information which large ViTs otherwise store
in background patches. :citet:`Darcet.Oquab.Mairal.ea.2024` found those
patches by their outputs, about ten times larger in norm than the rest and
holding little information about their own patch; with registers, these outliers
disappear. Registers need not even be trained:
:citet:`Jiang.Dravid.Efros.ea.2025` trace the outliers to a few neurons and
move their activation into an extra token at test time, with nearly the
effect of trained registers.

### Self-Supervised and Image--Text Encoders

The ViT of the chapter learns from labels, and labels are the scarcest
thing in vision. Masked autoencoders :cite:`he2022masked` learn by
reconstructing masked patches from the quarter that remains visible.
DINOv2 :cite:`Oquab.Darcet.Moutakanni.ea.2024` uses self-distillation on a
curated image collection: a network is trained to agree with a slowly
updated copy of itself on different crops of the same image. CLIP
:cite:`radford2021learning` trains an image encoder and a text encoder
together with a contrastive loss: for each image, a softmax over all
captions in the batch must pick its own caption, and likewise for each
caption over the images. SigLIP :cite:`Zhai.Mustafa.Kolesnikov.ea.2023`
replaces these softmaxes by a sigmoid loss on each image--caption pair,
which decouples the loss from the batch size. Encoders of this kind are
the pretrained image encoders on which vision--language models build
(:numref:`sec_vision-transformer`).

## Experts

Mixtures of experts are settled practice in the largest released models.
How to route and how sparse to be are not: each entry below is one paper's
answer.

### Router Stability

Large sparse models are prone to loss spikes that start in the router,
whose logits can grow without bound. ST-MoE
:cite:`Zoph.Bello.Kumar.ea.2022` keeps them small with a penalty on
their squared log-sum-exp, the z-loss of "Bounds on the Logits" applied to
the router. With it, ST-MoE trains a sparse model of 269B parameters at the
computation per token of a 32B dense model.

### Expert-Choice Routing

In the layer of :numref:`sec_moe` each token chooses its experts, so loads
must be balanced afterwards. Expert-choice routing
:cite:`Zhou.Lei.Liu.ea.2022` reverses the roles: each expert picks the
tokens it scores highest, up to its capacity, so loads are balanced by
construction. A token may then be picked by no expert or by many, and its
experts depend on the other tokens of the batch, which complicates decoding
one token at a time. Released decoder-only mixtures route by token choice.

### Sigmoid Gates and Limited Groups

In the router of :numref:`sec_moe`, a softmax over experts makes each
expert's score depend on the scores of all the others. DeepSeek-V3
:cite:`Liu.Feng.Xue.ea.2024` scores each expert with its own sigmoid, so
that a score depends only on the token and that expert, and normalizes the
scores of the selected experts to obtain their weights. It also restricts
the experts of each token to a few groups of devices, which bounds the
communication per token. The layer of :numref:`sec_moe` keeps the softmax
and adopts from DeepSeek-V3 only the per-expert bias that steers selection
toward balance.

### Sparsity and Expert Width

The mixture of :numref:`sec_moe` ended below the dense model at the same
computation per token. Does the gain hold at other sizes?
:citet:`Clark.Casas.Guy.ea.2022` found gains at every size from 15M to 1.3B
parameters, shrinking as the model grows, but they held the tokens fixed.
:citet:`Krajewski.Ludziejewski.Adamczewski.ea.2024` scale the tokens with
the model and tune the width of the experts; the advantage of mixtures then
grows with compute, and experts as wide as the dense feed-forward network
are almost never optimal. Later work finds that, at a fixed compute budget,
the optimal sparsity grows with the total parameter count
:cite:`Abnar.Shah.Busbridge.ea.2025`. Released models have become sparser
accordingly: Qwen3-Next activates about 3B of its 80B parameters, with 512
experts per layer, and Kimi K2 :cite:`Kimi.Team.2025` combines the latent
attention of :numref:`sec_kv-cache-reduction` with 384 experts per layer.

## Scaling

Both entries add a variable that the laws of :numref:`sec_scaling-laws`
leave out: how often each token is read, and how many bits hold each
number. Neither overturns the laws.

### Repeated Data

:numref:`sec_scaling-laws` noted that a repeated token keeps its value for
about four passes. :citet:`Muennighoff.Rush.Barak.ea.2023` fit how it
loses value after that. By their fit, $U$ unique tokens read in $1 + R$
passes are worth

$$U + U R^* \left(1 - e^{-R/R^*}\right)$$

fresh tokens, with $R^* \approx 15$. For a few passes the bracket is close
to $R/R^*$, and every pass counts almost fully: four passes ($R = 3$) are
worth about $3.7\,U$. For many passes the bracket tends to one, so no
amount of repetition makes $U$ tokens worth more than about $16\,U$ fresh
ones.

### Precision as a Scaling Variable

:citet:`Kumar.Ankner.Spector.ea.2025` treat the precision of training as a
reduction of the effective parameter count. Quantizing a model after
training hurts more the longer it was trained, so, surprisingly, beyond
some point more pretraining data makes the quantized model worse. In every
other law of this chapter, more data only ever lowered the loss. They also
find the compute-optimal precision for training to lie around seven to
eight bits. Their fits stop at 1.7B parameters and 26B tokens, so these
numbers are a guide for planning, not yet a law.

[Discussions](https://d2l.discourse.group/t/36853)
