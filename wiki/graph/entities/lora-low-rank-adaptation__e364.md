---
type: entity
id: 364
name: LoRA (Low-Rank Adaptation)
super_type: Method
sub_type: fine-tuning
source_paper: diffusionlight-light-probes-for-free-by-painting-a-chrome-ball
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- lora
- lora (low-rank adaptation)
- low-rank adaptation
---

Optimizes low-rank residual ΔW_i=A_iB_i (rank d≪m,n). Final: W_i'=W_i+αΔW_i. Used to fine-tune SDXL for chrome ball appearance and exposure control with minimal parameters.
