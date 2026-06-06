---
type: entity
id: 1539
name: Classifier-Free Guidance (CFG)
super_type: Method
sub_type: conditional sampling
source_paper: catv2ton-taming-diffusion-transformers
is_canonical: false
canonical_id: 1386
merge_confidence: 0.95
metadata: null
aliases: []
canonical: '[[classifier-free-guidance-cfg__e1386]]'
---

ε = ε_uncond + s·(ε_cond − ε_uncond). Balances conditional/unconditional via guidance scale s. (Ho & Salimans, 2022)
