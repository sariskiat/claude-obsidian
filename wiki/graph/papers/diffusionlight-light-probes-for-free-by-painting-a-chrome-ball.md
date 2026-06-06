---
type: paper
slug: diffusionlight-light-probes-for-free-by-painting-a-chrome-ball
title: 'DiffusionLight: Light Probes for Free by Painting a Chrome Ball'
authors: Pakkapon Phongthawee, Worameth Chinchuthakun, Nontaphat Sinsunthithet, Amit Raj, Varun Jampani, Pramook Khungurn, Supasorn Suwajanakorn
source_path: ~/.paper-scholar/diffusionlight-light-probes-for-free-by-painting-a-chrome-ball/paper.json
ingested_at: '2026-06-04 11:20:06'
authors_list:
- Pakkapon Phongthawee
- Worameth Chinchuthakun
- Nontaphat Sinsunthithet
- Amit Raj
- Varun Jampani
- Pramook Khungurn
- Supasorn Suwajanakorn
sections:
- id: 149
  heading: Abstract
  role: Abstract
  order_index: 0
  summary: Leverages pre-trained SDXL to render chrome balls for HDR lighting estimation. Iterative inpainting with noise averaging and LoRA exposure bracketing for HDR.
- id: 150
  heading: Introduction
  role: Introduction
  order_index: 1
  summary: 'Single-view lighting estimation is ill-posed. Existing methods rely on limited HDR panorama datasets. Key insight: use pre-trained T2I diffusion to inpaint chrome ball.'
- id: 151
  heading: Related Work
  role: Background
  order_index: 2
  summary: Lighting estimation methods, diffusion-based inpainting, personalized T2I with DreamBooth and LoRA.
- id: 152
  heading: Approach
  role: Methods
  order_index: 3
  summary: 'Three components: depth-conditioned SDXL, iterative inpainting via sample averaging + SDEdit, continuous LoRA for exposure bracketing with HDR luminance replacement.'
- id: 153
  heading: Experiments
  role: Evaluation
  order_index: 4
  summary: Laval Indoor + Poly Haven benchmarks. Outperforms StyleLight on angular error. Competitive with task-specific methods. Generalizes to random camera params and in-the-wild.
---

# DiffusionLight: Light Probes for Free by Painting a Chrome Ball

## [Abstract] Abstract
Leverages pre-trained SDXL to render chrome balls for HDR lighting estimation. Iterative inpainting with noise averaging and LoRA exposure bracketing for HDR.

## [Introduction] Introduction
Single-view lighting estimation is ill-posed. Existing methods rely on limited HDR panorama datasets. Key insight: use pre-trained T2I diffusion to inpaint chrome ball.

## [Background] Related Work
Lighting estimation methods, diffusion-based inpainting, personalized T2I with DreamBooth and LoRA.

## [Methods] Approach
Three components: depth-conditioned SDXL, iterative inpainting via sample averaging + SDEdit, continuous LoRA for exposure bracketing with HDR luminance replacement.

## [Evaluation] Experiments
Laval Indoor + Poly Haven benchmarks. Outperforms StyleLight on angular error. Competitive with task-specific methods. Generalizes to random camera params and in-the-wild.
