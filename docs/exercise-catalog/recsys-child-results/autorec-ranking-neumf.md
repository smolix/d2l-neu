# autorec/ranking/neumf research (recsys sub-agent result, verified vs raw PDFs)

METHOD WARNING (propagate to synthesis): WebFetch's summarizer FABRICATED
exercise content twice (claimed UCSD CSE 258 covered AutoRec/BPR/NeuMF — raw
PDF has none; invented a "Table 1" for a Rendle paper). All items below were
verified against raw source text.

## AutoRec — no external exercise tradition (verified negative)
- MMDS ch9 (via nerdai/MMDS_Exercises mirror; infolab host TLS error): similarity + UV decomposition only, no neural content.
- CS246 2020: no recsys-specific homework. UCSD CSE 158/258 (McAuley, fa25, both assignment PDFs read): open-ended Goodreads tasks, no AutoRec/BPR/NeuMF.
- Minnesota Coursera: no deep learning. BlueCourses (Baesens): AutoRec notebook as delivered content, not an assignment. Aggarwal textbook: paywalled, inconclusive (no unauthorized mirror used).
- Dacrema et al. reproduce CDAE/Mult-VAE/CVAE but not AutoRec.

## Ranking (BPR) — keep existing 3; one genuine addition
- Rendle et al., BPR (UAI 2009, arxiv 1205.2618): "Analogies to AUC optimization" section — BPR-Opt as differentiable AUC surrogate (Heaviside → ln σ). Natural NEW exercise: derive the correspondence. Not covered by current exercises.
- Rendle et al., "NCF vs MF Revisited" (RecSys 2020, arxiv 2005.09683): tuned dot-product MF beats learned-similarity across d∈{16..192} (Fig. 2) — embedding-dim sensitivity exercise.
- Dacrema et al. (TOIS 2021, arxiv 1911.07698): tuned BPR-MF as baseline in Tables 7/8 — optional grounding.
- Verified absent: MMDS/CS246 ranking homework; recsys-summer-school repo is bandits; Microsoft recommenders BPR notebook is a walkthrough.

## NeuMF — strong reproduce/verify/critique tradition (replace "vary X" filler)
1. Official repo hexiangnan/neural_collaborative_filtering: exact commands + hyperparams (num_factors=8, layers=[64,32,16,8], num_neg=4, lr=0.001 Adam) on ml-1m/pinterest-20; evaluate.py = leave-one-out HR/NDCG@K vs 100 sampled negatives → "reproduce these numbers" with real success criterion.
2. Rendle et al. 2020: implement tuned-MF baseline under NCF's own protocol; check Fig. 2 qualitative finding (dot product wins; pretrained NeuMF competitive only at large d on one dataset).
3. Dacrema et al. Appendix A.3: full NeuMF reproduction vs ItemKNN/BPR-MF/eALS; NeuMF inconsistently better (PureSVD beats it on Pinterest); DOCUMENTED BUG: original code early-stops by maximizing hit rate ON THE TEST SET → exercise: identify why this inflates Hit@10/NDCG@10 and fix with a validation split.
