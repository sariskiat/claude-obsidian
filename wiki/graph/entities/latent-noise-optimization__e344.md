---
type: entity
id: 344
name: Latent Noise Optimization
super_type: Method
sub_type: optimization technique
source_paper: optimizing-diffusion-noise-can-serve-as-universal-motion-priors
is_canonical: false
canonical_id: 343
merge_confidence: 0.9
metadata: null
aliases:
- latent noise optimization
canonical: '[[diffusion-noise-optimization-dno__e343]]'
---

Rather than optimizing in raw motion space (implausible results), optimizes in diffusion latent space z=x_T where z*=argmin_z L(f(z)), decoded through ODE solver.
