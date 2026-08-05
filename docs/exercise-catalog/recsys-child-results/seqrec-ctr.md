# seqrec/ctr research (recsys sub-agent result, verified sources)

## seqrec
- Tang & Wang, Caser (WSDM 2018), https://jiaxit.github.io/resources/wsdm18caser.pdf (arXiv:1809.07426): paper's own ablation is the origin of the book's exercises — variants Caser-p/h/v/vh/ph/pv/pvh via MAP on MovieLens+Gowalla (p worst, vh/pvh best); Markov-order sweep Caser-1/2/3 vs window L (Fig. 6). Cite as "reproduce Table/Figure N" framing.
- Quadrana/Cremonesi/Jannach sars_tutorial (RecSys 2018 / TheWebConf 2019), https://github.com/mquad/sars_tutorial — 8 notebooks (TopPop, FreqSeqMining, MarkovChain, FPMC, Prod2Vec, SessionRNN, PersonalizedRNN, KNN); model for turning "session vs sequence-aware" reading prompt into hands-on comparison (notebooks 05 vs 06).
- NEGATIVES (verified): MMDS ch9 exercises static only; CS246 HW2 static MF/CF; Minnesota Coursera spec has no sequential assignment; caser_pytorch README has no ablation scaffolding.

## ctr
- Criteo Display Advertising Challenge (Kaggle 2014): logloss; Label + I1–I13 integer + C1–C26 categorical (32-bit hashed). Precedent for continuous-field handling + leakage pitfalls.
- Avazu CTR (Kaggle 2014): logloss; ~20+ all-categorical fields incl. temporal `hour` — contrast with Criteo's numeric/categorical split (motivates the book's continuous-path exercise).
- McMahan et al., "Ad Click Prediction: a View from the Trenches" (KDD 2013) — canonical engineering reference; no course exercise exists on it (verified).
- NEGATIVES: MMDS ch8 = AdWords matching/mechanism design, not CTR classification; CS246 HW4 = streaming count sketches, near-miss.

Summary: seqrec has no classroom tradition (paper ablations are the citable template); ctr has a strong competition tradition (Criteo/Avazu) justifying 2–4 added exercises on hashing collisions, calibration, cardinality, leakage-free binning.
