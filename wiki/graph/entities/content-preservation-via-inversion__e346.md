---
type: entity
id: 346
name: Content Preservation via Inversion
super_type: Method
sub_type: editing technique
source_paper: optimizing-diffusion-noise-can-serve-as-universal-motion-priors
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- content preservation via inversion
---

Inverts reference motion via DDIM inversion to get x_T_ref, uses it as optimization start, penalizes L_cont=||x_T_ref-x_T||₂ to stay close to reference during editing.
