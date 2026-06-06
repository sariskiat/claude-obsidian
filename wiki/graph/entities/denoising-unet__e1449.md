---
type: entity
id: 1449
name: Denoising UNet
super_type: Method
sub_type: noise predictor
source_paper: improving-virtual-try-on-with-garment-focused-diffusion-models
is_canonical: false
canonical_id: 1384
merge_confidence: 0.95
metadata: null
aliases: []
canonical: '[[denoising-unet__e1384]]'
---

UNet ϵ_θ predicting added noise. Alternating ResNet blocks (local features) + transformer blocks (global self-attention). Core diffusion backbone.
