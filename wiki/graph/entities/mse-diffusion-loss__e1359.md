---
type: entity
id: 1359
name: MSE Diffusion Loss
super_type: Method
sub_type: training objective
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: true
canonical_id: null
merge_confidence: 0.95
metadata: null
aliases:
- mse diffusion loss
---

L = ||ε − ε_θ(z_t, conditions)||². Standard DDPM objective predicting added noise ε. CatVTON uses MSE for both mask-based and mask-free training.
