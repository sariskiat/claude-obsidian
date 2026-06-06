---
type: entity
id: 358
name: DiffusionLight Pipeline
super_type: Method
sub_type: light estimation
source_paper: diffusionlight-light-probes-for-free-by-painting-a-chrome-ball
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- diffusionlight pipeline
---

In-paints chrome ball into input image via pre-trained diffusion model (SDXL+depth ControlNet), then unwraps to HDR environment map. Avoids training on limited HDR datasets.
