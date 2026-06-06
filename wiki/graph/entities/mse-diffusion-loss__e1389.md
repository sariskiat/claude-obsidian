---
type: entity
id: 1389
name: MSE Diffusion Loss
super_type: Method
sub_type: denoising objective
source_paper: ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on
is_canonical: true
canonical_id: null
merge_confidence: 0.95
metadata: null
aliases: []
---

L = E[||ϵ − ϵ_θ(z_t, t, τ_θ(y))||²]. Standard LDM training objective minimizing noise prediction error.
