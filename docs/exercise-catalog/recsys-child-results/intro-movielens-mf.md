# recsys-intro/movielens/mf research (recsys sub-agent result, verified sources)

## recsys-intro
- MMDS ch9.2.8 Ex 9.2.1–9.2.3 (http://infolab.stanford.edu/~ullman/mmds/ch9.pdf, verified): cosine similarity under feature scaling; normalized user profile — hand-computable content-based complement.
- MMDS ch9.3.4 Ex 9.3.1–9.3.2: 3-user × 8-item utility matrix; Jaccard vs cosine under binarization/normalization; hierarchical clustering then recompute — metric/preprocessing changes neighbor structure.
- Stanford CS246 W2018 PS2 Q4 (http://snap.stanford.edu/class/cs246-2018/homeworks/hw2/hw2.pdf, verified): derive user-user vs item-item CF score matrices in terms of R, P, Q (parts a–c pure derivation; d needs code).
- FINDING: no external homework on warm/cold-start split design — the book's existing framing is a rare original.

## movielens
- Minnesota Coursera intro course, Module 3 Assignment 1: top-by-mean, association with Toy Story, male-female rating differences (LensKit/spreadsheet).
- DataCamp Python recsys "Matrix sparsity" exercise (verified): count non-empty cells → sparsity.
- DataCamp PySpark course ch3 (PDF extracted): sparsity ≈99.8% on ML-20M; group ratings per user; exercise 8: 80/20 randomSplit + ALS config.
- FINDING: no verified course exercise uses leave-last-out/temporal split — random splits everywhere; temporal-split exercise must be original.

## mf
- CS246 W2018 PS2 Q3 "Latent Features" (verified): derive SGD updates for R≈QPᵀ + L2; implement streaming from disk; CHECKABLE criterion: E<65000 after 40 iters, k=20, λ=0.1, plot E vs iteration, report η. Exactly the well-specified pattern to emulate.
- MMDS ch9.4.6 Ex 9.4.1–9.4.5 (verified): UV-decomposition coordinate-descent derivation sequence, closed-form updates, normalization order.
- Minnesota Coursera "Matrix Factorization and Advanced Techniques" Module 3: graded MF + Programming SVD assignments.
- DataCamp PySpark ALS grid search (verified): rank∈{5,40,80,120}, maxIter∈{5,100,250,500}, regParam∈{.05,.1,1.5}, RMSE → turns "vary k" into a concrete grid with a reported winner.
- NOT verified: Aggarwal textbook exercises (paywalled); no homework built around Koren/Bell/Volinsky directly.

Raw PDFs cached in scratchpad root: mmds_ch9_book.pdf, cs246_2018_hw2.pdf, datacamp_ch3.pdf, cse258_assignment1.pdf.
