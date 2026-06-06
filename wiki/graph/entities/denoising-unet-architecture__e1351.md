---
type: entity
id: 1351
name: Denoising UNet Architecture
super_type: Method
sub_type: diffusion backbone
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: false
canonical_id: 1384
merge_confidence: 0.9
metadata: null
aliases:
- denoising unet architecture
canonical: '[[denoising-unet__e1384]]'
---

Core LDM network. Alternating ResNet blocks (local conv features) + transformer blocks (global self-attention). CatVTON simplifies by removing cross-attention text conditioning.
