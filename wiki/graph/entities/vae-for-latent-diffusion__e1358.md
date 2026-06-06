---
type: entity
id: 1358
name: VAE for Latent Diffusion
super_type: Method
sub_type: latent space encoder/decoder
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- vae for latent diffusion
---

Encoder: image→z∈R^{4×H/8×W/8} (8× down). Decoder: z→image. Standard LDM component. CatVTON encodes garment+person via VAE before spatial concat in latent space.
