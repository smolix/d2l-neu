# Language Models

This part applies the sequence and transformer architectures to natural
language. It divides into two movements. *Pretraining* covers how text is turned
into something a model can learn from and how general-purpose representations are
built: tokenization and subword embeddings, word2vec and GloVe, and the
masked-language-model pretraining that gives BERT its bidirectional context.

*Applications* then fine-tunes those pretrained representations for concrete
tasks — sentiment analysis, natural language inference — and shows the
now-standard recipe of adapting a large pretrained model to a downstream problem
with comparatively little task-specific data. Together the two chapters trace the
arc from raw characters to a working, fine-tuned language system.
