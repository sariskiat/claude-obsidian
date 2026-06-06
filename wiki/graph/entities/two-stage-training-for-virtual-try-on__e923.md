---
type: entity
id: 923
name: Two-Stage Training for Virtual Try-On
super_type: Method
sub_type: training strategy
source_paper: enhanced-realism-in-virtual-try-on-tasks-using-diffusion-methods
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- two-stage training for virtual try-on
---

Stage1: U-Net inpainting 50ep lr=1e-7. Stage2: freeze U-Net, train ControlNet 20ep lr=1e-5.
