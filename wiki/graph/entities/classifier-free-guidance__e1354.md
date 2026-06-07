---

type: entity
id: 1354
name: Classifier-Free Guidance
super_type: Method
sub_type: conditional sampling
source_paper: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
is_canonical: false
canonical_id: 730
merge_confidence: 1.0
metadata: null
aliases: []
canonical: '[[classifier-free-guidance__e730]]'
---

ε_CFG = ε_uncond + s·(ε_cond − ε_uncond). CatVTON uses 10% conditional dropout, s=2.5 optimal at inference. s>3.5 causes color distortion and high-frequency noise (Ho & Salimans, 2022).
