# RACC reranking stability

Public reproducibility materials for **How Stable Are Post-hoc Trajectory Reranking Gains? Scaling and Independent Confirmation in Automated Driving** by Luhang Xiong.

This release corresponds to the `IJTST_SUBMISSION_LOCKED` manuscript after the literature-only language revision. `manifests/locked_manuscript.json` records its identities. It is a submission version, not a claim of journal acceptance. The manuscript itself is not redistributed here.

## Scope and conclusions

The study compares Original Pairwise, C1 and Direct Utility (DU) with fixed generated candidates. At N=5000, the original seven-condition development evaluation supports a Pairwise−DU advantage of +0.035142 m, 95% CI [0.008185, 0.063504]. Independent setting checks did not confirm a stable advantage. Fresh500 did not confirm the development advantage: −0.004854 m [−0.041376, 0.030742], with all four frozen success criteria unmet.

Retrospective within-development scale and condition contrasts have interval support. Development-minus-fresh500 and development-minus-confirm1000 contrasts span zero; non-confirmation does not prove degradation, equality or a changed population effect. Condition diversity remains an unresolved explanation. The two exploratory U/Q prototypes rejected most switches and did not improve numerical ADE; their outcomes do not establish that conservative filtering generally fails. No road-safety endpoint was measured.

## Evidence roles

| Role | Setting | Status |
|---|---|---|
| Development | QCNet, fixed eval1000, seven conditions, N=700/1500/3000/5000 | Original objective-scaling observation; data and updates increase together |
| Confirmatory cross-setting | QCNet clean-only | Development-scene independence is not established |
| Confirmatory cross-setting | EMP-M clean-only, 1000 scenes | Unmatched second-backbone check, not fully matched replication |
| Confirmatory separate scenes | confirm1000, same N5000 models and N10000 extension | Same-set extension did not detect further improvement; no proof of saturation |
| Final holdout | fresh500, seven conditions, five frozen seeds | One-shot final confirmation completed; NOT CONFIRMED |
| Retrospective | Scale, condition and cross-set paired/effect contrasts | Requested after outcomes; nominal intervals, secondary contrasts unadjusted |
| Exploratory | Failure characterization and RACC-U/Q | Consumed outputs; prototypes reuse internal-validation150 sequentially |

**Fresh500 has been used for final confirmation and later diagnostics. It is no longer an untouched holdout. New holdout1000 was identity-frozen only and contributes no result to this paper.** This repository does not distribute its IDs or payload.

## What can be reproduced

1. **Without data or checkpoints:** inspect the archived result summaries, exact configurations and artifact identities; run the public-release verification and synthetic statistical self-check. These checks validate packaging/statistical code, not a fresh empirical evaluation.
2. **With privately held, authorized frozen outputs:** rerun scene-paired/independent-set bootstrap contrasts and condition decomposition. The required per-scene errors, predictions, caches and manifests are not bundled. CIs cannot be reconstructed from aggregate summaries alone.
3. **With additionally authorized weights and exact original RACC modules:** replay the frozen reranker scores for the existing dev/fresh failure analysis. The replay verifies identical choices and does not run QCNet or train.
4. **Exact end-to-end training/candidate generation is not runnable from this release alone.** Original RACC training/feature modules were not available in the local release source set and are not replaced by an invented implementation. Hashes and configurations are retained. AV2, upstream software/checkpoints and the exact original RACC sources are separate prerequisites. Upstream current code is not guaranteed to match frozen historical identities.

## Quick start

Python 3.11 is the common documented interpreter family. The NumPy-only statistical environment is pinned separately from the legacy Torch environment:

```sh
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/self_check.py
```

See [reproducibility instructions](docs/reproducibility.md) for private input layout and historical environments. No script automatically downloads data or weights. No new holdout is included in any execution path.

## Contents

- `configs/`: frozen architecture, schedules, seeds, endpoint, bootstrap and exploratory prototype settings.
- `manifests/`: relative artifact identities, hashes, split counts and source provenance; no payload or scene-ID lists.
- `scripts/`: portable versions of existing RACC statistical/diagnostic scripts plus release checks.
- `results_summary/`: published aggregate values and retrospective summary intervals, not prediction arrays.
- `docs/`: reproduction boundaries, access chronology, data/license instructions and release audit.

Public source files have new hashes after path adaptation. Historical digests still identify the original private bytes; they are not claimed to hash the sanitized public copy.

## Data, licenses and citation

AV2 must be obtained by each user under the [official terms](https://www.argoverse.org/about.html). Do not upload data, cache tensors, prediction arrays, scene lists, weights or private configuration into this repository. See [data access](docs/data_access.md).

The MIT license applies to the original RACC code and repository documentation released here. No AV2 dataset, upstream implementation or model weights are relicensed. Use `CITATION.cff`; no publication DOI or repository URL is invented. Generative AI assisted preparation of these materials; the author is responsible for verification.
