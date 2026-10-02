# Reproducibility summary

## Release scope

This package supports public inspection of aggregate results and reproduction of analyses conditional on privately available, authorized frozen inputs. It does not provide AV2 payloads, weights, per-scene prediction arrays or all original training/feature-extraction sources. No exact end-to-end training claim is made. No fresh model, feature substitute, threshold or experiment was introduced during packaging.

## Data and grouping

All paper results use AV2. Ordered development train prefixes contain 700/1500/3000/5000 scenes; the original fixed development evaluation contains 1000 scenes. Seven conditions are equally weighted and follow their parent scene in splits. Clean-only QCNet uses the original scene setting without established independent-scene provenance. EMP-M evaluates 1000 independently selected scenes with clean-only coverage and different representation. Confirm1000 is separate and shared by original N5000 and new N10000 models. Fresh500 contains 500 scenes and is consumed. The U/Q exploratory split reuses development1000 as train700/calibration150/internal-validation150; Q's validation follows U's use. New holdout1000 is metadata-only and contributes no paper result.

## Frozen methods and optimization

`configs/frozen_training.json` records the exact architectures, optimizer, lr, weight decay, batch units, schedules, actual seed transforms and C1 fold construction. Seeds are nominally 2026–2030; DU actual seeds are five times nominal. All checkpoints are final-step selections. C1 uses same-N/seed frozen Pairwise upstream and five scene-grouped heads; the whole upstream pipeline is not out-of-fold. Data and updates vary together and methods are not equal-compute comparisons.

Train-only normalization is frozen in the checkpoint container; normalizer entries identified by the container hash do not pretend to be independent tensor hashes. Inference has no GT/ADE/FDE/condition-ID features. GT errors supply supervised targets and evaluation only. Original pair/tie and feature semantics remain in the original source identities; their missing bytes are not reconstructed from descriptions.

## Statistical estimation

Primary effect is Pairwise ADE−DU ADE. Average five frozen seed effects within scene-condition, then seven conditions equally within scene. Resample scenes, not conditions or seeds. Use 10,000 replicates, NumPy default_rng(2026), percentile 95% CI. Within-development contrasts pair the same scenes. Cross-set contrasts independently resample each set and subtract means within each replicate. Historical effect analysis uses 10 batches of 1000 replicates, drawing A then B in each batch. Never average condition-level CIs to form an overall CI. LOSO omits one frozen model seed at a time; it is sensitivity analysis, not additional scene replication.

## Statistical reproduction with authorized private artifacts

The NumPy-only effect script expects these ignored relative archive paths:

```
inputs/development_results.zip
inputs/development_cache.zip
inputs/fresh500_results.zip
inputs/confirmation_results.zip
inputs/confirmation_cache.zip
```

Archive internal layout remains the historical layout. `manifests/private_input_identities.json` records original archive hashes and cache anchors. The script verifies record/tar/commit/prediction hashes, scene disjointness, fixed N5000 model identities and saved aggregate consistency. Run:

```sh
python scripts/analyze_effect_heterogeneity.py
```

Outputs go to ignored `results_local/effect_heterogeneity/`. Do not publish those per-scene outputs without a separate release review. The published aggregate summaries alone are insufficient to reconstruct the CIs. No such private assets were read during packaging.

For the legacy Torch condition decomposition, use a separate Python 3.11 environment matching `configs/historical_torch_environment.txt`. Place original selection, training lock, committed result directory and cache directory beneath `inputs/racc_artifacts/`, retaining the directory names used by the script. Run `python scripts/decompose_conditions.py`. It checks original hashes and only analyzes existing errors/choices; it does not run QCNet or train. Torch pickle-based legacy records require trusted, hash-verified inputs; never load arbitrary downloaded pickle files.

## Optional historical failure analysis

`scripts/failure_analysis/` preserves the original descriptive analysis and frozen score replay. It requires the authorized existing dev/fresh caches, frozen N5000 reranker weights, admission metadata and exact original RACC modules. The modules must be placed in ignored `inputs/external/racc_mf/` and upstream QCNet dependencies in `inputs/external/QCNet/`. These sources and weights are not supplied. A local-only copy of admission metadata must express each `locked_sources.*.path` relative to that RACC source root while preserving expected source digests; the copy is not presented as the original admission bytes. Original metadata/weight identities remain in the ledger. Do not edit or replace original private audit records.

Run `python scripts/failure_analysis/analyze_fresh_failure.py` only when those prerequisites are satisfied. It verifies exact agreement with stored selected modes and only recomputes frozen reranker scores; it does not run QCNet, fit models or tune. Diagnostic outputs are exploratory, not renewed independent confirmation. Changing path plumbing changes public script hashes, as recorded in source_provenance.json.

## Confirmation chronology and limitations

See `docs/data_access_chronology.md`. Final endpoint, model identities and success rule were locked before recorded fresh500 performance access. Some exact legacy first-read timestamps are NOT LOGGED. Hash locks establish recorded order, not a complete historical-access audit or an external preregistration. Public backbone upstream exposure is not excluded by the project's scene registry. Retrospective contrasts and post-hoc diagnostics are not prespecified confirmations.

## Available public checks

`python scripts/verify_release.py` checks file hashes, file policy and common privacy/credential patterns. `python scripts/self_check.py` uses synthetic constants to check scene-paired and independent bootstrap machinery and frozen configuration invariants. These are software checks, not new empirical experiments.
