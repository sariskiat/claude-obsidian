---
type: entity
id: 1284
name: Task Token
super_type: Concept
sub_type: Conditioning Mechanism
source_paper: voost-a-unified-and-scalable-diffusion-transformer-for-bidirectional-virtual-try-on-and-try-off
is_canonical: true
canonical_id: null
merge_confidence: null
metadata: null
aliases:
- task token
---

Concatenation of a mode token (on/off) specifying generation direction and a category token (upper/lower/full) encoding garment type, passed to the transformer as additional conditioning.
