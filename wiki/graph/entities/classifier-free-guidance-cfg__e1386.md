---
type: entity
id: 1386
name: Classifier-Free Guidance (CFG)
super_type: Method
sub_type: conditional sampling
source_paper: ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases: []
---

ε = ε_uncond + s·(ε_cond − ε_uncond). Balances conditional and unconditional predictions via guidance scale s. OOTDiffusion enables garment-aware CFG through outfitting dropout (Ho & Salimans, 2022).
