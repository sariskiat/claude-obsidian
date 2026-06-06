---
type: paper
slug: catvton-concatenation-is-all-you-need-for-virtual-try-on-with-diffusion-models
title: 'CatVTON: Concatenation is All You Need for Virtual Try-On with Diffusion Models'
authors: Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, Dongmei Jiang, Xiaodan Liang
source_path: inbox/vton-diffusion-pdfs/2407.15886.pdf
ingested_at: '2026-06-05 12:56:21'
authors_list: []
sections:
- id: 896
  heading: Abstract
  role: Introduction
  order_index: 0
  summary: Lightweight VTON (899M/49.6M params). Spatial concatenation. No text encoder, cross-attn, ReferenceNet. ICLR 2025.
- id: 897
  heading: 1. Introduction
  role: Introduction
  order_index: 1
  summary: 'GAN warping fails complex poses. Diffusion adds overhead. CatVTON: concat-only input, 49% less memory.'
- id: 898
  heading: 2. Related Work
  role: Background
  order_index: 2
  summary: 'Subject-driven (LoRA, DreamBooth, IP-Adapter, AnyDoor). VTON: warping vs ReferenceNet/Dual-UNet diffusion.'
- id: 899
  heading: 3.1 Lightweight Network
  role: Methods
  order_index: 3
  summary: VAE→latent. Simplified UNet without text/cross-attn. 899M params. Spatial concat garment+person.
- id: 900
  heading: 3.2 Parameter-Efficient Training
  role: Methods
  order_index: 4
  summary: Only self-attn trained (49.57M, 5.5%). 10% dropout for CFG. DREAM λ=10. Mask-free via synthetic pairs.
- id: 901
  heading: 3.3 Simplified Inference
  role: Methods
  order_index: 5
  summary: 'No pose/parse/caption. Mask-free: person+garment only. T-step denoising, VAE decode.'
- id: 902
  heading: 4. Experiments
  role: Evaluation
  order_index: 6
  summary: VITON-HD/DressCode/DeepFashion. SSIM/LPIPS/FID/KID. SOTA results. CFG=2.5, DREAM λ=10.
- id: 903
  heading: 5. Conclusion
  role: Conclusion
  order_index: 7
  summary: Compact, efficient, SOTA quality. Opens new directions in VTON.
---

# CatVTON: Concatenation is All You Need for Virtual Try-On with Diffusion Models

## [Introduction] Abstract
Lightweight VTON (899M/49.6M params). Spatial concatenation. No text encoder, cross-attn, ReferenceNet. ICLR 2025.

## [Introduction] 1. Introduction
GAN warping fails complex poses. Diffusion adds overhead. CatVTON: concat-only input, 49% less memory.

## [Background] 2. Related Work
Subject-driven (LoRA, DreamBooth, IP-Adapter, AnyDoor). VTON: warping vs ReferenceNet/Dual-UNet diffusion.

## [Methods] 3.1 Lightweight Network
VAE→latent. Simplified UNet without text/cross-attn. 899M params. Spatial concat garment+person.

## [Methods] 3.2 Parameter-Efficient Training
Only self-attn trained (49.57M, 5.5%). 10% dropout for CFG. DREAM λ=10. Mask-free via synthetic pairs.

## [Methods] 3.3 Simplified Inference
No pose/parse/caption. Mask-free: person+garment only. T-step denoising, VAE decode.

## [Evaluation] 4. Experiments
VITON-HD/DressCode/DeepFashion. SSIM/LPIPS/FID/KID. SOTA results. CFG=2.5, DREAM λ=10.

## [Conclusion] 5. Conclusion
Compact, efficient, SOTA quality. Opens new directions in VTON.
