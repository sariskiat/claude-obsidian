---
type: entity
id: 363
name: Depth-Conditioned ControlNet
super_type: Method
sub_type: conditioning technique
source_paper: diffusionlight-light-probes-for-free-by-painting-a-chrome-ball
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- depth-conditioned controlnet
---

Uses off-the-shelf depth predictor; paints white circle at center (closest to camera) for chrome ball. Feeds depth+mask+image+prompt to SDXL for reliable ball insertion.
