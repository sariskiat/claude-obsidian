---

type: entity
id: 1386
name: Classifier-Free Guidance (CFG)
super_type: Method
sub_type: conditional sampling
source_paper: ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on
is_canonical: false
canonical_id: 730
merge_confidence: 0.833
metadata: null
aliases: []
canonical: '[[classifier-free-guidance__e730]]'
---

ε = ε_uncond + s·(ε_cond − ε_uncond). Balances conditional and unconditional predictions via guidance scale s. OOTDiffusion enables garment-aware CFG through outfitting dropout (Ho & Salimans, 2022).
