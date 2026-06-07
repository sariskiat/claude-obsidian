---

type: entity
id: 1389
name: MSE Diffusion Loss
super_type: Method
sub_type: denoising objective
source_paper: ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on
is_canonical: false
canonical_id: 1359
merge_confidence: 1.0
metadata: null
aliases: []
canonical: '[[mse-diffusion-loss__e1359]]'
---

L = E[||ϵ − ϵ_θ(z_t, t, τ_θ(y))||²]. Standard LDM training objective minimizing noise prediction error.
