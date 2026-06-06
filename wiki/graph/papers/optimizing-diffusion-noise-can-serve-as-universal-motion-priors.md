---
type: paper
slug: optimizing-diffusion-noise-can-serve-as-universal-motion-priors
title: Optimizing Diffusion Noise Can Serve As Universal Motion Priors
authors: Korrawe Karunratanakul, Konpat Preechakul, Emre Aksan, Thabo Beeler, Supasorn Suwajanakorn, Siyu Tang
source_path: ~/.paper-scholar/optimizing-diffusion-noise-can-serve-as-universal-motion-priors/paper.json
ingested_at: '2026-06-04 11:19:17'
authors_list:
- Korrawe Karunratanakul
- Konpat Preechakul
- Emre Aksan
- Thabo Beeler
- Supasorn Suwajanakorn
- Siyu Tang
sections:
- id: 141
  heading: Abstract
  role: Abstract
  order_index: 0
  summary: DNO leverages motion diffusion models as universal motion priors by optimizing latent noise through backpropagation of task criteria through the full denoising process, without model retraining.
- id: 142
  heading: Introduction
  role: Introduction
  order_index: 1
  summary: Many motion tasks are finding plausible motions fulfilling criteria. DNO treats denoising as black box and optimizes latent noise.
- id: 143
  heading: Related Works
  role: Background
  order_index: 2
  summary: Motion synthesis, editing, completion; diffusion models and guidance; DOODL latent optimization for images.
- id: 144
  heading: Background
  role: Background
  order_index: 3
  summary: Motion generation with diffusion, deterministic DDIM sampling, diffusion inversion, relative-root motion representation.
- id: 145
  heading: Diffusion Noise Optimization
  role: Methods
  order_index: 4
  summary: Optimizes x_T via gradient descent through ODE solver. Normalized gradients, content preservation via inversion, latent decorrelation loss. Differs from guided diffusion by computing criteria on full-chain output.
- id: 146
  heading: Applications
  role: Methods
  order_index: 5
  summary: Motion editing (trajectory, pose, obstacle avoidance), refinement (denoising), completion (missing joints/frames) via composable loss functions.
- id: 147
  heading: Experiments
  role: Evaluation
  order_index: 6
  summary: 'HumanML3D + MDM/GMD/MLD: DNO outperforms GMD on editing; DNO-GMD best on refinement. Ablation validates normalized gradients, decorrelation loss.'
- id: 148
  heading: Discussion and Limitations
  role: Discussion
  order_index: 7
  summary: Plug-and-play but limited by base model. Works better with good body coverage. Speed limits interactive use.
---

# Optimizing Diffusion Noise Can Serve As Universal Motion Priors

## [Abstract] Abstract
DNO leverages motion diffusion models as universal motion priors by optimizing latent noise through backpropagation of task criteria through the full denoising process, without model retraining.

## [Introduction] Introduction
Many motion tasks are finding plausible motions fulfilling criteria. DNO treats denoising as black box and optimizes latent noise.

## [Background] Related Works
Motion synthesis, editing, completion; diffusion models and guidance; DOODL latent optimization for images.

## [Background] Background
Motion generation with diffusion, deterministic DDIM sampling, diffusion inversion, relative-root motion representation.

## [Methods] Diffusion Noise Optimization
Optimizes x_T via gradient descent through ODE solver. Normalized gradients, content preservation via inversion, latent decorrelation loss. Differs from guided diffusion by computing criteria on full-chain output.

## [Methods] Applications
Motion editing (trajectory, pose, obstacle avoidance), refinement (denoising), completion (missing joints/frames) via composable loss functions.

## [Evaluation] Experiments
HumanML3D + MDM/GMD/MLD: DNO outperforms GMD on editing; DNO-GMD best on refinement. Ablation validates normalized gradients, decorrelation loss.

## [Discussion] Discussion and Limitations
Plug-and-play but limited by base model. Works better with good body coverage. Speed limits interactive use.
