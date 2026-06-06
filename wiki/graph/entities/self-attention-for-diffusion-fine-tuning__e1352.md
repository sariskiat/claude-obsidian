---
type: entity
id: 1352
name: Self-Attention for Diffusion Fine-tuning
super_type: Method
sub_type: critical adaptation component
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: false
canonical_id: 1410
merge_confidence: 0.9
metadata: null
aliases:
- self-attention for diffusion fine-tuning
canonical: '[[self-attention-for-diffusion__e1410]]'
---

Self-attention layers in UNet transformer blocks. CatVTON identifies these as the most critical for VTON adaptation, training only 49.57M out of 899.06M (5.5%) with comparable results to full fine-tuning.
