# Review of Chapter 28: Information Theory and Divergences

## Scope

Reviewed `chapter_mdl-information-theory/index.md`, `mdl-information-theory.md`, `mdl-divergences-distances.md`, and `mdl-mutual-information.md`, including all prose, derivations, code, figures, exercises, summaries, and slides.

## Executive assessment

This chapter has a clear conceptual ambition and many excellent explanations: cross-entropy as excess code length, divergence families as objective choices, and mutual-information bounds as training objectives rather than trustworthy measurements. The main problems arise exactly where confident prose outruns mathematical scope. Continuous entropy is initially folded into discrete “surprise,” empirical KL is incorrectly generalized to densities in one passage, optimal transport is claimed to be uniquely informative on disjoint supports, and the order-level `O(log N)` limitation is repeatedly presented as an exact universal `log N` cap. The long modern-application arcs also make the chapter feel like a survey rather than a guided derivation.

Scores (0–10): **writing quality 7.7**, **explanation/pedagogy 8.3**, **technical/logical quality 7.6**.

## Architecture and logical order

Entropy/KL → discrepancy families → mutual information is sound. Each section, however, expands into many modern applications. Establish a core path containing definitions, coding, variational duality, data processing, and InfoNCE. Mark MDL, rate–distortion, channel coding, Stein discrepancies, Sinkhorn, and information bottleneck as advanced branches. This prevents applications from diluting the main logical dependencies.

## Detailed issues

| ID | Severity | Location | Problem and violated guide rule | Concrete revision direction |
|---|---|---|---|---|
| C28-01 | Moderate | `chapter_mdl-information-theory/index.md:4-10` | The index lists families and applications but does not state the common decision: different discrepancies encode different notions of error and produce different objectives. | Open with one contrasting example and a map from question to quantity: coding, distribution fitting, sample comparison, dependence. |
| C28-02 | Major | `mdl-information-theory.md:57-84` | The card example is effective, but “rare means surprising” can be read as semantic information or anomaly value. Self-information is relative to the chosen probabilistic model and event partition. | State explicitly that surprise is model-relative and a confidently wrong model assigns high information to ordinary events. |
| C28-03 | Critical | `mdl-information-theory.md:102-136` | The text first defines entropy for a p.m.f. “or p.d.f.” as expected self-information, says the minus sign makes it positive, and describes spread as more surprise; only afterward does it explain that differential entropy can be negative and is coordinate-dependent. The initial claims are false for densities. | Define discrete Shannon entropy first. Introduce differential entropy as a separate analogue with immediate warnings; do not call density values outcome probabilities or imply nonnegativity. |
| C28-04 | Major | `mdl-information-theory.md:149-162` | Using `nansum` to encode `0 log 0=0` can silently hide unrelated NaNs or invalid negative probabilities. This teaches a numerically permissive implementation. | Mask `p>0`, validate nonnegativity and normalization, and explain why targeted masking is safer than dropping all NaNs. |
| C28-05 | Critical | `mdl-information-theory.md:614-623` | The text says empirical NLL is cross-entropy and empirical-KL minimization “for … densities alike.” An atomic empirical distribution is singular with respect to a continuous density, so the empirical KL is generally infinite—the Chapter 27 likelihood section correctly says so. | Restrict the empirical-distribution identity to discrete outcomes. For densities, state the sample NLL and its population cross-entropy/KL interpretation under a data-generating density. |
| C28-06 | Major | `mdl-information-theory.md:1033-1046` | Scaling-law prose says loss approaches the language entropy rate predictably and that each compute multiple removes a constant fraction of the “remaining excess.” Empirical power laws with fitted irreducible terms do not identify the true language entropy rate, and the constant-fraction statement is not the generic meaning of a power law. | Describe the empirical fit actually supported by the citations, distinguish fitted asymptote from entropy rate, and remove the exponential-decay interpretation. |
| C28-07 | Moderate | `mdl-information-theory.md:1053-1140` | MDL is introduced as a corollary in a “Modern Uses” section, but model coding choices and universal codes are subtle. A short passage may leave readers thinking parameter count plus NLL is uniquely defined. | State code dependence, prefix/decodability requirements, and the invariance/complexity issue; mark this as an introduction rather than a derivation. |
| C28-08 | Major | `mdl-divergences-distances.md:101-117` | “Optimal transport distances … alone stay informative when P and Q do not overlap” is false. MMD with a characteristic geometry-sensitive kernel can vary on disjoint supports; total variation remains a valid (though often saturated) signal. | Say OT explicitly measures displacement and often supplies geometry-sensitive changes where bounded f-divergences saturate. Compare with MMD rather than claiming uniqueness. |
| C28-09 | Moderate | `mdl-divergences-distances.md:122-236` | The f-divergence template needs support/absolute-continuity conventions for `p/q`, especially where `q=0`. Those cases drive forward/reverse-KL behavior but are easy to miss in the main definition. | State the measure-theoretic or extended-value convention at definition time and revisit it in the mode-covering example. |
| C28-10 | Major | `mdl-divergences-distances.md:322-439` | The text says convex duality rewrites “every f-divergence as the value of a game,” but equality requires a sufficiently rich critic and integrability; restricted neural critics provide only a lower bound. The caveat appears later and should govern the first claim. | State unrestricted variational equality first, then restricted-class lower bound and optimization error as separate layers. |
| C28-11 | Major | `mdl-divergences-distances.md:1012-1017` | Scores are called “the only family” that avoids a normalizing constant. Other ratio/contrastive methods and Stein constructions can also cancel normalizers; moreover score access assumes differentiability and support conditions. | Say score matching is a principal normalizer-free route and state its regularity/support requirements. |
| C28-12 | Major | `mdl-divergences-distances.md:1284-1318` | The objective map turns idealized theoretical correspondences into deterministic behavior: original GAN → JS, WGAN “gradients survive,” MLE “mass-covering by construction.” These depend on optimal critics, model families, optimization, and support. | Add columns for idealized assumptions and optimization/critic limitations. Use tendencies, not inevitabilities. |
| C28-13 | Moderate | `mdl-mutual-information.md:78-92` | “Entropy extends … the obvious way” is a banned ease claim exactly where continuous joint laws require care. The caveat follows, but the opening invites overgeneralization. | Define the discrete case plainly, then treat continuous quantities via KL as the primary safe formulation. |
| C28-14 | Moderate | `mdl-mutual-information.md:314-320` | Calling a histogram estimator “obvious” adds no information. One grid and one seed do not separate discretization bias from sampling variance. | Name it the plug-in histogram estimator, vary bin count/sample size, and report both bias and variability. |
| C28-15 | Critical | `mdl-mutual-information.md:651-679` | An informal theorem giving an `O(log N)` ceiling for distribution-free high-confidence lower bounds is immediately converted into an exact statement: a batch of 256 “can certify at most ln 256 … no matter” the critic, and reported MI is `min(I,log N)+noise`. The exact `log N` cap holds for InfoNCE, not every possible lower-bound procedure with the theorem’s unspecified constants. | Keep the general theorem qualitative/order-level. Introduce the exact InfoNCE bound separately and restrict all numerical ceilings to InfoNCE. |
| C28-16 | Major | `mdl-mutual-information.md:644-649` | “Mutual information cares only about the copula” requires continuous variables and suitable transforms; discrete/mixed cases are more subtle. The claim is used to motivate a general impossibility result. | Scope the copula statement to continuous laws with continuous marginals and keep invariance as the general fact. |
| C28-17 | Major | `mdl-mutual-information.md:1208-1216` | “Batch size sets the resolution of the instrument” is a strong metaphor based on one trained critic and negative count. In-batch dependence, critic optimization, estimator bias, and evaluation sample size also determine the result. | State what is held fixed and call negative count one ceiling/variance control, not the sole resolution. |
| C28-18 | Moderate | `mdl-mutual-information.md:1221-1257` | The information-bottleneck extremes (“retains everything relevant,” “recovers sufficient statistics”) are stated broadly. Existence, encoder family, and tradeoff geometry matter, and deterministic continuous encoders can produce infinite MI. | State the finite/discrete or stochastic-encoder setting and qualify the sufficient-statistic limit. |
| C28-19 | Major | `mdl-mutual-information.md:1399-1432` | The guideline says the `log N` ceiling caps what histogram, kernel, nearest-neighbor, and neural estimators “can certify.” This again conflates a distribution-free confidence guarantee with point estimates and InfoNCE’s exact range. | Separate estimation, lower confidence bounds, and variational objectives in a three-column table; apply the exact cap only where proved. |
| C28-20 | Moderate | Slide blocks throughout | The decks contain many strong claim titles, but some encode the same overstatements (“Information only leaks,” “ceiling at log N,” “only at the right critic”) without conditions. | Keep memorable titles, add qualifiers in subtitles, and distinguish theorem slides from empirical illustrations. |

## Math and notation

Highest priorities are discrete versus differential entropy, empirical KL for densities, support conventions for f-divergences, ideal versus restricted critics, and general `O(log N)` versus exact InfoNCE `log N`. Use `H` only for discrete entropy and `h` for differential entropy from first introduction; distinguish population objectives, empirical estimates, and confidence guarantees typographically and verbally.

## Figures, captions, and slides

The visual taxonomy, transport comparison, information plane, and InfoNCE-ceiling figure are valuable. Captions are generally self-contained. Revise captions/titles where an illustration is used as evidence for a universal theorem or where the exact bound applies only to one estimator.

## Code and experiment pedagogy

Known-distribution simulations are strong. The entropy helper should validate inputs rather than hide NaNs. MI-estimator comparisons should vary seeds and tuning parameters and explicitly decompose critic error, finite-sample error, and estimator bias. Objective-map code/figures should state ideal-critic assumptions.

## Recurring artifacts

- “Obvious,” “simply,” “everything,” “only family,” and narrative “story” language.
- Exact-sounding conclusions derived from informal/order-level theorems.
- Idealized divergence behavior stated as inevitable training behavior.
- Advanced survey branches interrupting the core derivation.

## Strengths to preserve

- Coding interpretation of entropy, cross-entropy, and KL.
- Careful warning that neural MI bounds are objectives, not measurements.
- Geometry-based comparison of discrepancy families.
- Good figures and concrete closed-form anchors.
- Explicit discussion of critic restriction and estimator failure, once moved earlier.

## Prioritized revision plan

1. Correct the differential-entropy opening and empirical-density KL claim.
2. Separate the general `O(log N)` limitation from InfoNCE’s exact `log N` bound everywhere.
3. Qualify OT uniqueness and idealized objective-map behavior.
4. Mark a core path and move MDL/Stein/IB material into clearly advanced branches.
5. Harden numerical examples and revise slide claims to carry their assumptions.
