---
type: entity
id: 1384
name: Denoising UNet
super_type: Method
sub_type: iterative noise predictor
source_paper: ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- denoising unet
---

UNet ϵ_θ predicting added noise with MSE objective. Alternating ResNet+transformer blocks. OOTDiffusion expands first conv to 8 input channels. Cross-attention for CLIP conditioning.
