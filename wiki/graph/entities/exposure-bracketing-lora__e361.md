---
type: entity
id: 361
name: Exposure Bracketing LoRA
super_type: Method
sub_type: fine-tuning technique
source_paper: diffusionlight-light-probes-for-free-by-painting-a-chrome-ball
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- exposure bracketing lora
---

Continuous LoRA mapping text embedding interpolations to specific EV values. Single LoRA (not per-EV) preserves scene structure via weight sharing. Text embedding: ξ_ev=ξ_o+(ev/EV_min)(ξ_d-ξ_o).
