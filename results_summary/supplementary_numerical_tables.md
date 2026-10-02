## Tables

All ADE values are in meters. Positive Delta=ADE_Pairwise−ADE_DU favors DU. Positive prototype improvement=ADE_DU−ADE_filter favors the filter. Intervals are percentile 95% scene-bootstrap intervals, using 10,000 replicates and RNG seed 2026. Within-scene contrasts are paired; the cross-set rows in Table 5 independently resample the two scene sets. Values are transcribed at the archived display precision, not recalculated from rounded columns. NR means not reported here, not zero.

### Table 1. Main development scaling comparison

QCNet seven-condition development eval1000; five frozen seeds. The baseline ADE is 1.837619 m throughout. Pointwise intervals are unadjusted.

| Training scenes | Original Pairwise ADE | C1 ADE | DU ADE | Pairwise−DU | 95% CI |
|---:|---:|---:|---:|---:|---|
| 700 | 1.665339 | 1.682514 | 1.677334 | −0.011994 | [−0.055544, 0.032972] |
| 1500 | 1.657517 | 1.674611 | 1.641396 | +0.016121 | [−0.015833, 0.049708] |
| 3000 | 1.660783 | 1.669774 | 1.633431 | +0.027352 | [−0.003049, 0.058013] |
| 5000 | 1.660804 | 1.665204 | 1.625661 | +0.035142 | [0.008185, 0.063504] |

Only the N5000 pointwise interval has a positive lower bound. The individual-budget intervals alone do not establish an objective-by-budget interaction; the direct retrospective N5000−700 contrast is reported separately in Table 5. Data and updates change together.

### Table 2. Development reference and later confirmation outcomes

Rows have different evidence roles and, where indicated, different scene sets. No pooled effect is calculated, and the final rule is not retroactively applied to preceding checks.

| Evidence role | Setting / evaluation scope | Training scenes | Pairwise−DU | 95% CI |
|---|---|---:|---:|---|
| Development reference | QCNet seven-condition; development eval1000 | 5000 | +0.035142 | [0.008185, 0.063504] |
| Confirmatory cross-setting | QCNet clean-only; fixed eval1000, independence not established here | 5000 | +0.007381 | [−0.020911, 0.035660] |
| Confirmatory cross-setting | EMP-M clean-only; not fully matched | 5000 | −0.000148 | [−0.028610, 0.028301] |
| Confirmatory separate scenes | QCNet seven-condition; separate confirmatory eval1000 | 5000 | +0.002753 | [−0.026925, 0.030858] |
| Confirmatory separate scenes | QCNet seven-condition; same separate confirmatory eval1000 | 10000 | +0.005758 | [−0.020608, 0.032030] |
| Final holdout | QCNet seven-condition; fresh500 | 5000 | −0.004854 | [−0.041376, 0.030742] |

On the same separate confirmatory eval1000, DU N5000/N10000 ADE is 1.660981/1.657431 m. The saved same-test improvement is +0.003550 m [−0.008207, 0.015281]. No further improvement was detected; equivalence and saturation are not established. Fresh500 QCNet/Pairwise/DU ADE is 1.739430/1.588432/1.593286 m. Fresh500 fails the frozen final rule.

### Table 3. Descriptive post-confirmation characterization

Exploratory, using consumed outputs. Gain/loss masses average all weighted records, including no-switch zeros. Selected-utility means are conditional on DU interventions. AUC treats higher diagnostic scores as indicating harm. Approximate values retain the source's precision labels; no diagnostic intervals are inferred.

| Diagnostic | Development eval1000 | Fresh500 |
|---|---:|---:|
| Extra beneficial gain mass, DU−Pairwise (m) | approximately +0.121463 | approximately +0.098715 |
| Excess harmful loss mass, DU−Pairwise (m) | approximately +0.086320 | approximately +0.103569 |
| DU predicted selected utility mean (m) | 0.292981 | 0.278514 |
| DU realized selected utility mean (m) | 0.390592 | 0.278566 |
| DU decision-margin harmful-label AUC | NR | 0.493836 |
| Pairwise decision-margin harmful-label AUC | 0.470025 | 0.483094 |
| DU utility ensemble-std harmful-label AUC | approximately 0.4345 | approximately 0.4995 |

The saved approximately −0.039996 m point-estimate change comprises approximately −0.017248 m associated with increased excess harmful loss and −0.022748 m associated with reduced extra beneficial gain: rounded accounting shares of 43% and 57%, respectively. These are not causal shares.

### Table 4. Two exploratory frozen mitigation prototypes

Auxiliary train700; reused internal-validation150; seven conditions; five seeds; frozen N5000 DU upstream. References on this split are DU 1.627662 m, Pairwise 1.637878 m, and QCNet 1.715850 m. Neither prototype uses fresh500 or new holdout1000 for evaluation.

| Prototype | ADE | DU−filter | 95% CI | Harmful loss reduction | Beneficial gain retained | Switch retention | Advancement |
|---|---:|---:|---|---:|---:|---:|---|
| RACC-U | 1.704232 | −0.076570 | [−0.194274, 0.044079] | 96.72% | 5.96% | 1.16% | FAIL |
| RACC-Q | 1.710674 | −0.083011 | [−0.208240, 0.046763] | 100.00% | 1.59% | 0.76% | FAIL |

Both have 0/5 positive seed improvements and all-negative LOSO means. Their ADE intervals include zero. U risk AUC/ECE/Brier/NLL are 0.566920/0.046879/0.240613/0.674216. Q empirical u<q20 coverage is 25.56%, versus nominal 20%; coverage is diagnostic, not a hard criterion. These are two operating points, not an optimal frontier.

### Table 5. Retrospective effect contrasts

Five fixed seeds; equal condition weights; 10,000 scene-bootstrap replicates; RNG seed2026; nominal 95% percentile intervals. The two within-development contrasts pair the same 1000 scenes. Cross-set contrasts independently resample scenes and use the same frozen N5000 model pairs. These analyses were requested after the historical outcomes were known.

| Contrast | Effect difference (m) | 95% CI | Interpretation |
|---|---:|---|---|
| Δ_scale: N=5000−700 | +0.047137 | [0.018924, 0.074637] | interval-supported within development set |
| Δ_condition: corruption−clean | +0.041102 | [0.016640, 0.067666] | interval-supported within development set |
| Δ_set: dev−fresh500 | +0.039996 | [−0.004123, 0.085168] | not interval-supported |
| Secondary: dev−confirm1000 at same N=5000 | +0.032390 | [−0.006818, 0.073518] | not interval-supported |

The condition contrast is an evaluation-regime association, not training-diversity causality. Zero-crossing cross-set intervals do not establish equality. The primary designation applies only to the retrospective N5000−700 analysis; it does not change the original final holdout endpoint.

### Table S1. Frozen main training schedules

| Training scenes | Pairwise updates | C1 updates per head | DU updates |
|---:|---:|---:|---:|
| 700 | 400 | 1600 | 500 |
| 1500 | 857 | 3429 | 1071 |
| 3000 | 1714 | 6857 | 2143 |
| 5000 | 2857 | 11429 | 3571 |

C1 has five fixed scene-grouped heads and a frozen same-budget/same-seed Pairwise upstream. All model checkpoints are final-step choices.

### Table S2. Final holdout seed sensitivity and frozen decision

| Seed | Pairwise−DU on fresh500 (m) |
|---:|---:|
| 2026 | −0.014380 |
| 2027 | +0.007029 |
| 2028 | −0.016212 |
| 2029 | +0.003191 |
| 2030 | −0.003897 |

| Criterion frozen before performance access | Fresh500 value / summary | Outcome |
|---|---|---|
| Mean Pairwise−DU ≥ +0.01 m | −0.004854 m | FAIL |
| At least 4/5 positive seeds | 2/5 | FAIL |
| Every LOSO mean >0 | All five negative | FAIL |
| Overall scene-bootstrap CI lower >0 | −0.041376 m | FAIL |

Final confirmation: NOT CONFIRMED. LOSO magnitudes are not reconstructed from rounded seed summaries.

### Frozen prototype advancement rules

Both prototypes require DU-relative ADE improvement ≥0.01 m, a positive paired CI lower bound, at least four positive seeds, all-positive LOSO means, and harmful-loss reduction ≥20%. U additionally requires beneficial-gain retention ≥80%, switch retention ≥40%, risk AUC ≥0.65, and ECE ≤0.05. Q instead requires gain retention ≥70% and switch retention ≥25%; its quantile coverage is diagnostic. These are the recorded prototype-specific rules, not a common retrospectively selected criterion.


### Table S3. Additional retrospective scale contrasts

Original seven-condition development eval1000; same scenes paired across budgets; five fixed seeds. These secondary intervals are unadjusted and are not used to establish a stronger scaling claim.

| Contrast | Effect difference (m) | Nominal 95% CI |
|---|---:|---|
| N1500−N700 | +0.028116 | [0.008615, 0.047315] |
| N3000−N700 | +0.039347 | [0.012242, 0.066262] |
| N5000−N1500 | +0.019021 | [0.002787, 0.035300] |
