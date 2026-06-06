---
type: entity
id: 1356
name: T-Step Iterative Denoising
super_type: Method
sub_type: generative sampling
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- t-step iterative denoising
---

z_T∼N(0,1) progressively denoised T timesteps: z_{t-1}=UNet(z_t ⊙ conditions). Output z_0 decoded by VAE to pixels. CatVTON input: concatenated noise+latent person+latent garment.
