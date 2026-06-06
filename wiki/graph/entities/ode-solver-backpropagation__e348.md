---
type: entity
id: 348
name: ODE Solver Backpropagation
super_type: Method
sub_type: gradient computation
source_paper: optimizing-diffusion-noise-can-serve-as-universal-motion-priors
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- ode solver backpropagation
---

Solves ODE forward to get motion, computes task loss, backpropagates through every solver step for gradient w.r.t. x_T. Memory-intensive but feasible with few DDIM steps.
