---
type: entity
id: 1346
name: CatVTON
super_type: Method
sub_type: lightweight diffusion VTON
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases: []
---

Simple efficient VTON. Spatially concatenates garment+person as UNet input. No text encoder/cross-attn/ReferenceNet. 899M total, 49.57M trainable (5.5%). ICLR 2025.
