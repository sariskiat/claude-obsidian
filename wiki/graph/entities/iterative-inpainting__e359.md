---
type: entity
id: 359
name: Iterative Inpainting
super_type: Method
sub_type: noise optimization
source_paper: diffusionlight-light-probes-for-free-by-painting-a-chrome-ball
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- iterative inpainting
---

Locates good noise maps: (1) generate N chrome balls, (2) compute pixel-wise median, (3) SDEdit on median composite, (4) repeat 2x. Leverages finding that noise maps encode semantics consistently across images.
