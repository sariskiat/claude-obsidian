---
type: entity
id: 1353
name: Cross-Attention Text Conditioning
super_type: Method
sub_type: text-to-image mechanism
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- cross-attention text conditioning
---

Standard LDM mechanism injecting CLIP text embeddings into UNet via cross-attn. CatVTON removes entirely: text is redundant for image-based VTON, saves 167M params.
