---
type: entity
id: 325
name: Heavy Ball Momentum
super_type: Method
sub_type: optimization technique
source_paper: diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- heavy ball momentum
---

Polyak's method applied to diffusion: v_{n+1}=(1-β)v_n+βΣb_i f(x_{n-i}), x_{n+1}=x_n+δv_{n+1}. Expands stability but drops to first-order.
