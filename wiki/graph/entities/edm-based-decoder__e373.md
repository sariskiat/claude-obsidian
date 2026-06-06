---
type: entity
id: 373
name: EDM-Based Decoder
super_type: Method
sub_type: decoder architecture
source_paper: diffsda-unsupervised-diffusion-sequential-disentanglement-across-modalities
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- edm-based decoder
---

Uses EDM framework with 63 NFEs at inference. Decoder D_θ takes noisy x_t^τ and factors z_0^τ=(s_0,d_0^τ), returns denoised estimate via skip connection + AdaGN conditioning.
