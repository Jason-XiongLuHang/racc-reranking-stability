# Artifact identity conventions

`manifests/frozen_artifact_identities.csv` is a sanitized metadata inventory, not a distribution of the identified artifacts. All names are logical relative aliases. Historical SHA256 values identify original bytes before path redaction, not this CSV or adapted public scripts.

The inventory covers checkpoint containers, embedded normalizers, source/config/selection/cache manifests and recorded evaluation locks. An embedded normalizer's checkpoint-container digest is not an independent normalizer-tensor digest. The 10k combined training identity is SHA256 of the ASCII concatenation of old and new training-lock digests; it is not the digest of the new lock alone.

No scene-ID lists, cache tensors, prediction arrays or checkpoint weights are released. Metadata names do not imply public availability. Original RACC training/feature modules not locally available for packaging remain external prerequisites; historical source hashes are preserved without fabricated replacement code. Path-only adaptations are recorded in `source_provenance.json`; their public hashes differ from the originals.
