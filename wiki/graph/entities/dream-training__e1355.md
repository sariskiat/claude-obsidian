---
type: entity
id: 1355
name: DREAM Training
super_type: Method
sub_type: diffusion regularization
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- dream training
---

Training strategy with λ balancing distortion vs perceptual quality. λ→∞ disables; small λ→smooth; large λ→high-freq artifacts. CatVTON optimal λ=10 (Zhou et al., 2024).
