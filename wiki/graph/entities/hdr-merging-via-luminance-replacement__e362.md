---
type: entity
id: 362
name: HDR Merging via Luminance Replacement
super_type: Method
sub_type: HDR technique
source_paper: diffusionlight-light-probes-for-free-by-painting-a-chrome-ball
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- hdr merging via luminance replacement
---

Replaces overexposed luminance (>0.9) with exposure-corrected values from lower EV images, retaining chroma from EV0. Avoids ghosting from standard HDR merging.
