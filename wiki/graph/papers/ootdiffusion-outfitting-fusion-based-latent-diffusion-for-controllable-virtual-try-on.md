---
type: paper
slug: ootdiffusion-outfitting-fusion-based-latent-diffusion-for-controllable-virtual-try-on
title: 'OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on'
authors: Yuhao Xu, Tao Gu, Weifeng Chen, Chengcai Chen
source_path: inbox/vton-diffusion-pdfs/2403.01779.pdf
ingested_at: '2026-06-05 13:02:45'
authors_list: []
sections:
- id: 904
  heading: Abstract
  role: Introduction
  order_index: 0
  summary: LDM-based VTON with Outfitting UNet, Outfitting Fusion in self-attention, Outfitting Dropout for CFG. SOTA on VITON-HD/DressCode.
- id: 905
  heading: 1. Introduction
  role: Introduction
  order_index: 1
  summary: 'GAN VTON lacks realism. LDM methods warp or CLIP-invert, losing detail. OOTDiffusion: self-attn fusion, CFG controllability.'
- id: 906
  heading: Method
  role: Methods
  order_index: 2
  summary: Outfitting UNet (single-step garment features), Outfitting Fusion (self-attn alignment), Outfitting Dropout (CFG). Auxiliary CLIP embeddings.
- id: 907
  heading: Experiments
  role: Evaluation
  order_index: 3
  summary: VITON-HD/DressCode SOTA. CFG guidance scale analysis.
---

# OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on

## [Introduction] Abstract
LDM-based VTON with Outfitting UNet, Outfitting Fusion in self-attention, Outfitting Dropout for CFG. SOTA on VITON-HD/DressCode.

## [Introduction] 1. Introduction
GAN VTON lacks realism. LDM methods warp or CLIP-invert, losing detail. OOTDiffusion: self-attn fusion, CFG controllability.

## [Methods] Method
Outfitting UNet (single-step garment features), Outfitting Fusion (self-attn alignment), Outfitting Dropout (CFG). Auxiliary CLIP embeddings.

## [Evaluation] Experiments
VITON-HD/DressCode SOTA. CFG guidance scale analysis.
