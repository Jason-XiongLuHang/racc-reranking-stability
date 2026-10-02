# Data access and redistribution boundaries

The repository contains original RACC scripts, metadata identities, configurations and published aggregate summaries. It contains no trajectory data, maps, images, per-scene predictions, cache records, original scene-ID lists or weights. The MIT license does not relicense upstream material.

Users must obtain Argoverse 2 Motion Forecasting independently and comply with the [official dataset page](https://www.argoverse.org/av2.html) and [Terms of Use](https://www.argoverse.org/about.html). The official pages identify AV2 data/documentation as CC BY-NC-SA 4.0 and its API as MIT. Dataset attribution: © 2021 Argo AI, LLC; cite Wilson et al., Argoverse 2. Nothing here grants commercial-data or checkpoint redistribution rights.

The original [QCNet repository](https://github.com/ZikangZhou/QCNet) identifies its implementation as Apache 2.0. The original [EMP repository](https://github.com/a-pru/emp) identifies its implementation as BSD-3-Clause. Neither implementation is vendored here. Obtain required weights independently, review all applicable artifact-specific terms, and verify the frozen SHA256 identities. A source-code license is not treated here as separate proof of unrestricted redistribution rights for every weight file. We do not distribute weights.

Official pages were reviewed on 2026-10-02. These statements record the sources checked, not an independent legal opinion. Publication summaries are attributed to the manuscript and AV2; no dataset payload is packaged.

Private inputs belong under ignored `inputs/`. Do not commit your data, tokens, contact information, private paths or reconstructed score/cache files. Fresh500 is already consumed; do not describe it as untouched. New holdout1000 is outside all reported results and execution inputs.
