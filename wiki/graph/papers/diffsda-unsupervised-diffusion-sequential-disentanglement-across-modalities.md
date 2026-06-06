---
type: paper
slug: diffsda-unsupervised-diffusion-sequential-disentanglement-across-modalities
title: 'DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities'
authors: Hedi Zisling, Ilan Naiman, Nimrod Berman, Supasorn Suwajanakorn, Omri Azencot
source_path: ~/.paper-scholar/diffsda-unsupervised-diffusion-sequential-disentanglement-across-modalities/paper.json
ingested_at: '2026-06-04 11:20:58'
authors_list:
- Hedi Zisling
- Ilan Naiman
- Nimrod Berman
- Supasorn Suwajanakorn
- Omri Azencot
sections:
- id: 154
  heading: Abstract
  role: Abstract
  order_index: 0
  summary: Modal-agnostic framework for unsupervised sequential disentanglement using latent diffusion. Separates static/dynamic factors across time series, video, audio with single loss term.
- id: 155
  heading: Introduction
  role: Introduction
  order_index: 1
  summary: 'Existing methods rely on VAEs/GANs with complex multi-loss optimization. DiffSDA: diffusion-based, dependent static-dynamic factors, single loss, zero-shot transfer.'
- id: 156
  heading: Related Work
  role: Background
  order_index: 2
  summary: VAE/GAN disentanglement, diffusion-based non-sequential disentanglement, animation methods. DiffSDA = first diffusion-based sequential disentanglement.
- id: 157
  heading: Method
  role: Methods
  order_index: 3
  summary: Probabilistic framework with two diffusion models. Sequential semantic encoder (U-Net/MLP+LSTM), EDM decoder, single score-matching loss. Disentanglement via shared static factor and low-dim dynamic bottleneck.
- id: 158
  heading: Results
  role: Evaluation
  order_index: 4
  summary: Video (VoxCeleb, CelebV-HQ, TaiChi-HD, MUG), audio (TIMIT, LibriSpeech), time series (PhysioNet, ETTh1). Outperforms SPYL/DBSE on swap, reconstruction, speaker ID, prediction. First zero-shot swap. PCA multifactor disentanglement.
- id: 159
  heading: Conclusions
  role: Conclusion
  order_index: 5
  summary: 'Addresses key sequential disentanglement limitations for real-world data. Future: computational efficiency, sensor data, full multifactor disentanglement.'
---

# DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities

## [Abstract] Abstract
Modal-agnostic framework for unsupervised sequential disentanglement using latent diffusion. Separates static/dynamic factors across time series, video, audio with single loss term.

## [Introduction] Introduction
Existing methods rely on VAEs/GANs with complex multi-loss optimization. DiffSDA: diffusion-based, dependent static-dynamic factors, single loss, zero-shot transfer.

## [Background] Related Work
VAE/GAN disentanglement, diffusion-based non-sequential disentanglement, animation methods. DiffSDA = first diffusion-based sequential disentanglement.

## [Methods] Method
Probabilistic framework with two diffusion models. Sequential semantic encoder (U-Net/MLP+LSTM), EDM decoder, single score-matching loss. Disentanglement via shared static factor and low-dim dynamic bottleneck.

## [Evaluation] Results
Video (VoxCeleb, CelebV-HQ, TaiChi-HD, MUG), audio (TIMIT, LibriSpeech), time series (PhysioNet, ETTh1). Outperforms SPYL/DBSE on swap, reconstruction, speaker ID, prediction. First zero-shot swap. PCA multifactor disentanglement.

## [Conclusion] Conclusions
Addresses key sequential disentanglement limitations for real-world data. Future: computational efficiency, sensor data, full multifactor disentanglement.
