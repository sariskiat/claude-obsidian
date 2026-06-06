---
type: entity
id: 347
name: Latent Decorrelation Loss
super_type: Method
sub_type: regularization
source_paper: optimizing-diffusion-noise-can-serve-as-universal-motion-priors
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- latent decorrelation loss
---

Penalizes dot product between consecutive frames of x_T at multiple temporal resolutions. Prevents foot skating by discouraging correlated latent samples.
