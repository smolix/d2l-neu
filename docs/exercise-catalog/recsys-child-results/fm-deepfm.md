# fm/deepfm research (recsys sub-agent result, verified sources)

## FM
- Rendle, libFM manual v1.4.2 (2014), http://www.libfm.org/libfm-1.42.manual.pdf — worked tutorial: MovieLens 1M → libFM format, `./libFM -task r -dim '1,1,8'` across SGD/ALS/MCMC/Adaptive-SGD; concrete RMSE metric. Template for fixing the "vary embedding size" exercise.
- Criteo Display Advertising Challenge (Kaggle 2014), logloss metric — https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview
- Avazu CTR Prediction (Kaggle 2014), logloss — https://www.kaggle.com/competitions/avazu-ctr-prediction/overview (winner detail unverified, flagged)
- "3 Idiots" Criteo winning FFM pipeline repo, ~0.445 logloss, exact repro commands — https://github.com/ycjuan/kaggle-2014-criteo
- "4 Idiots" Avazu repo (ensemble of ~20 FFM models, staged logloss scores) — https://github.com/ycjuan/kaggle-avazu
- Juan et al., "Field-aware Factorization Machines for CTR Prediction" (RecSys 2016) — controlled LM/Poly2/FM/FFM comparison on Criteo+Avazu via logloss; template for "implement FM, compare vs FFM on same split" — https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
- LibFFM repo quick-start with toy GBDT-preprocessed Criteo subset — https://github.com/ycjuan/libffm
- NEGATIVE: Stanford CS246 verified to have NO FM/FFM homework (classical MF only).

## DeepFM
- Guo et al., "DeepFM" (IJCAI 2017), §3.3 hyper-parameter study: activation (ReLU vs tanh), dropout, neurons/layer, #layers, network shape (constant/increasing/decreasing/diamond; constant wins) + Table 2 AUC/logloss comparisons — https://arxiv.org/abs/1703.04247. Mirror for the "vary MLP structure" exercise.
- DeepCTR (shenweichen/DeepCTR): examples/run_classification_criteo.py trains DeepFM on criteo_sample.txt (13 dense + 26 sparse), BCE, 80/20. CORRECTION: no standalone "FM" class exists — comparable classes are NFM/AFM/PNN/FNN/WDL/FwFM; frame comparison exercises accordingly.
- NEGATIVE: no university course has a documented DeepFM problem set — paper ablations + library examples only.
