---
type: entity
id: 1350
name: Denoising Diffusion Probabilistic Models (DDPM)
super_type: Method
sub_type: iterative denoising
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases: []
---

Generative framework reversing gradual noising. z_T∼N(0,1) denoised T steps: z_{t-1}=UNet(z_t,cond). Standard formulation used by CatVTON as the base generative process.
