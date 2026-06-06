---
type: paper
slug: enhanced-realism-in-virtual-try-on-tasks-using-diffusion-methods
title: Enhanced Realism in Virtual Try-On Tasks Using Diffusion Methods
authors: Saris Kiattithapanayong, Suronapee Phoomvuthisarn
source_path: ~/.paper-scholar/enhanced-realism-in-virtual-try-on-tasks-using-diffusion-methods/virtualtry-on final.md
ingested_at: '2026-06-05 11:35:22'
authors_list: []
sections:
- id: 853
  heading: Abstract
  role: Introduction
  order_index: 0
  summary: VTON framework with VQ-VAE, CLIP, ControlNet. LPIPS 0.082, FID 7.782, KID 1.53.
- id: 854
  heading: I. Introduction
  role: Introduction
  order_index: 1
  summary: Motivates VTON for retail, GAN limitations, three-component framework.
- id: 855
  heading: II. Literature Review
  role: Background
  order_index: 2
  summary: GAN-based and diffusion-based VTON, DDPM fundamentals. Gaps in spatial info loss.
- id: 856
  heading: III. Method — Model Overview
  role: Methods
  order_index: 3
  summary: Person+clothing through autoencoder, diffusion U-Net, feature preservation blocks.
- id: 857
  heading: III. Method — Autoencoder
  role: Methods
  order_index: 4
  summary: 'VQ-VAE: RGB→latent z=E(x), f=8, VQ reg in decoder.'
- id: 858
  heading: III. Method — Visual Feature Inversion
  role: Methods
  order_index: 5
  summary: CLIP image encoder→token space, visual inversion adapter as conditioning.
- id: 859
  heading: III. Method — Diffusion UNET
  role: Methods
  order_index: 6
  summary: '9-channel U-Net: noisy+masked latent+mask. CLIP cross-attention. L_simple loss.'
- id: 860
  heading: III. Method — Additional Feature Preserving Block
  role: Methods
  order_index: 7
  summary: 'ControlNet-inspired: garment+pose encoders, residuals into frozen U-Net.'
- id: 861
  heading: III. Method — Experimental Setting
  role: Methods
  order_index: 8
  summary: VITON-HD, Stage1 50ep lr=1e-7, Stage2 20ep lr=1e-5, DDPM 50-step, A5000.
- id: 862
  heading: III. Method — Evaluation Metrics
  role: Evaluation
  order_index: 9
  summary: 'Paired: LPIPS+SSIM. Unpaired: FID+KID. vs VITON-HD, LADI-VTON, HR-VTON.'
- id: 863
  heading: III. Method — Results
  role: Results
  order_index: 10
  summary: LPIPS 0.082, FID 7.782, KID 1.53. SSIM 0.825 trade-off.
- id: 864
  heading: IV. Conclusion
  role: Conclusion
  order_index: 11
  summary: '3 contributions + future: multimodal cond, perceptual loss, paired settings.'
---

# Enhanced Realism in Virtual Try-On Tasks Using Diffusion Methods

## [Introduction] Abstract
VTON framework with VQ-VAE, CLIP, ControlNet. LPIPS 0.082, FID 7.782, KID 1.53.

## [Introduction] I. Introduction
Motivates VTON for retail, GAN limitations, three-component framework.

## [Background] II. Literature Review
GAN-based and diffusion-based VTON, DDPM fundamentals. Gaps in spatial info loss.

## [Methods] III. Method — Model Overview
Person+clothing through autoencoder, diffusion U-Net, feature preservation blocks.

## [Methods] III. Method — Autoencoder
VQ-VAE: RGB→latent z=E(x), f=8, VQ reg in decoder.

## [Methods] III. Method — Visual Feature Inversion
CLIP image encoder→token space, visual inversion adapter as conditioning.

## [Methods] III. Method — Diffusion UNET
9-channel U-Net: noisy+masked latent+mask. CLIP cross-attention. L_simple loss.

## [Methods] III. Method — Additional Feature Preserving Block
ControlNet-inspired: garment+pose encoders, residuals into frozen U-Net.

## [Methods] III. Method — Experimental Setting
VITON-HD, Stage1 50ep lr=1e-7, Stage2 20ep lr=1e-5, DDPM 50-step, A5000.

## [Evaluation] III. Method — Evaluation Metrics
Paired: LPIPS+SSIM. Unpaired: FID+KID. vs VITON-HD, LADI-VTON, HR-VTON.

## [Results] III. Method — Results
LPIPS 0.082, FID 7.782, KID 1.53. SSIM 0.825 trade-off.

## [Conclusion] IV. Conclusion
3 contributions + future: multimodal cond, perceptual loss, paired settings.
