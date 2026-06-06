---
type: entity
id: 372
name: Sequential Semantic Encoder
super_type: Method
sub_type: encoder architecture
source_paper: diffsda-unsupervised-diffusion-sequential-disentanglement-across-modalities
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- sequential semantic encoder
---

Factorizes sequences: modality-specific backbone (U-Net/MLP) + LSTM. Last hidden→s_0 (static), all hiddens through second LSTM→d_0^{1:V} (dynamic).
