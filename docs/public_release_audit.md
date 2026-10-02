# Public release audit

Status: **PUBLIC_REPOSITORY_READY** (local package; not uploaded).

| Check | Outcome | Evidence |
|---|---|---|
| Sensitive information | PASS | No account identifier, contact email, credential assignment, service token or private key found. Public author name is retained for attribution. |
| Local paths | PASS | Paths are relative aliases or documented placeholders; no absolute local paths. |
| Dataset payloads | PASS | Text-only allowlist; no AV2 raw data, maps, images, cache tensors, full scene-ID lists or per-scene prediction arrays. |
| Weights | PASS | No checkpoints or normalizer tensors; digests are metadata only. |
| License boundary | PASS | Original RACC scripts/documentation and published aggregates only. Upstream implementations, manuscripts, commercial reports and weights are excluded. External terms are linked, not replaced by MIT. |
| Scientific consistency | PASS | Four summary tables exactly match the locked manuscript; retrospective summary CSVs are copied unchanged. README retains all evidence distinctions and bounded conclusions. |
| Data consumption | PASS | Fresh500 explicitly consumed; new holdout1000 contributes no result or execution input. |
| Manuscript preservation | PASS | Manuscript identities match the scientific lock; no paper edits or empirical execution. |
| Software checks | PASS | Synthetic paired/independent bootstrap, deterministic RNG and configuration invariants pass without reading data or weights. |
| Citation metadata | PASS | Exact locked title and sole author; no invented publication DOI, contact address or repository URL. |

## Provenance and reproduction limits

Public path adaptations have new source hashes. Historical artifact digests identify original bytes rather than sanitized public copies. Embedded normalizer container hashes are not individual tensor hashes. Identity entries do not imply public availability.

The repository is ready for release as statistical-reproducibility and artifact-documentation materials. Exact original training/feature modules, authorized data, frozen outputs and weights remain external prerequisites. They are not replaced by guessed code. Full end-to-end empirical reproduction is not available from this package alone; README states this explicitly.

No licensing/privacy blocker was found in the packaged files. This does not authorize redistribution of excluded assets. The audit combines a file allowlist, common-pattern scans and manual source review; arbitrary future additions need a new audit.

No GitHub upload, new training, GPU execution, holdout opening or experiment change was performed. The manuscript uses a dynamic page-number field; its PDF pages are 1–19. Preview caching does not change the manuscript.
