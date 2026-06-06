---
type: entity
id: 1455
name: MSE Diffusion Loss
super_type: Method
sub_type: training objective
source_paper: improving-virtual-try-on-with-garment-focused-diffusion-models
is_canonical: false
canonical_id: 1359
merge_confidence: 0.95
metadata: null
aliases: []
canonical: '[[mse-diffusion-loss__e1359]]'
---

L = E[||ϵ − ϵ_θ(z_t, t, cond)||²]. Standard DDPM/LDM training objective minimizing noise prediction error.
