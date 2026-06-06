---
type: paper
slug: diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts
title: Diffusion Sampling with Momentum for Mitigating Divergence Artifacts
authors: Suttisak Wizadwongsa, Worameth Chinchuthakun, Pramook Khungurn, Amit Raj, Supasorn Suwajanakorn
source_path: ~/.paper-scholar/diffusion-sampling-with-momentum-for-mitigating-divergence-artifacts/paper.json
ingested_at: '2026-06-04 11:17:33'
authors_list:
- Suttisak Wizadwongsa
- Worameth Chinchuthakun
- Pramook Khungurn
- Amit Raj
- Supasorn Suwajanakorn
sections:
- id: 134
  heading: Abstract
  role: Abstract
  order_index: 0
  summary: Proposes HB momentum and GHVB to expand stability regions of diffusion samplers, reducing divergence artifacts in low-step sampling.
- id: 135
  heading: Introduction
  role: Introduction
  order_index: 1
  summary: Higher-order methods produce divergence artifacts at low step counts. Stability regions identified as root cause.
- id: 136
  heading: Background
  role: Background
  order_index: 2
  summary: Diffusion ODE form, guided sampling, splitting methods, and stability region concept.
- id: 137
  heading: Understanding Artifacts in Diffusion Sampling
  role: Methods
  order_index: 3
  summary: Analyzes artifacts via latent magnitudes; connects narrow stability regions to artifacts through eigenvalue analysis.
- id: 138
  heading: Methodology
  role: Methods
  order_index: 4
  summary: Proposes HB momentum (first-order) and GHVB (high-order) with damping coefficient β for accuracy-stability tradeoff.
- id: 139
  heading: Experiments
  role: Evaluation
  order_index: 5
  summary: Validates on Stable Diffusion, ADM, DiT. Both methods reduce artifacts and improve FID at low step counts.
- id: 140
  heading: Discussion
  role: Discussion
  order_index: 6
  summary: Training-free, computationally negligible, compatible with existing samplers.
---

# Diffusion Sampling with Momentum for Mitigating Divergence Artifacts

## [Abstract] Abstract
Proposes HB momentum and GHVB to expand stability regions of diffusion samplers, reducing divergence artifacts in low-step sampling.

## [Introduction] Introduction
Higher-order methods produce divergence artifacts at low step counts. Stability regions identified as root cause.

## [Background] Background
Diffusion ODE form, guided sampling, splitting methods, and stability region concept.

## [Methods] Understanding Artifacts in Diffusion Sampling
Analyzes artifacts via latent magnitudes; connects narrow stability regions to artifacts through eigenvalue analysis.

## [Methods] Methodology
Proposes HB momentum (first-order) and GHVB (high-order) with damping coefficient β for accuracy-stability tradeoff.

## [Evaluation] Experiments
Validates on Stable Diffusion, ADM, DiT. Both methods reduce artifacts and improve FID at low step counts.

## [Discussion] Discussion
Training-free, computationally negligible, compatible with existing samplers.
