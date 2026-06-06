---
type: paper
slug: inductive-moment-matching
title: Inductive Moment Matching
authors: '["Linqi Zhou", "Stefano Ermon", "Jiaming Song"]'
source_path: ~/.paper-scholar/inductive-moment-matching/2503.07565.md
ingested_at: '2026-06-05 05:49:57'
authors_list:
- Linqi Zhou
- Stefano Ermon
- Jiaming Song
sections:
- id: 558
  heading: Abstract
  role: intro
  order_index: 0
  summary: 'Propose IMM: a single-stage training procedure for few-step generative models using inductive moment matching'
- id: 559
  heading: 1. Introduction
  role: intro
  order_index: 1
  summary: Motivation for stable few-step generation and overview of IMM contributions
- id: 560
  heading: 2. Preliminaries
  role: background
  order_index: 2
  summary: Diffusion, Flow Matching, stochastic interpolants, and MMD background
- id: 561
  heading: 3. Inductive Moment Matching
  role: method
  order_index: 3
  summary: 'IMM framework overview: model construction via interpolants'
- id: 562
  heading: 3.1. Model Construction via Interpolants
  role: method
  order_index: 4
  summary: Definition of marginal-preserving interpolants and model distribution
- id: 563
  heading: 3.2. Learning via Inductive Bootstrapping
  role: method
  order_index: 5
  summary: Inductive learning algorithm with MMD objective and convergence guarantee
- id: 564
  heading: 4. Simplified Formulation and Practice
  role: method
  order_index: 6
  summary: DDIM interpolant, self-consistency, simplified objective, implementation choices
- id: 565
  heading: 4.1. Algorithmic Considerations
  role: method
  order_index: 7
  summary: Self-consistent interpolants, DDIM, stop-gradient, simplified MMD objective
- id: 566
  heading: 4.3. Sampling
  role: method
  order_index: 8
  summary: Pushforward and restart sampling with classifier-free guidance
- id: 567
  heading: 5. Connection with Prior Works
  role: background
  order_index: 9
  summary: CMs as single-particle special case of IMM, Diffusion GAN, GMMN connections
- id: 568
  heading: 7. Experiments
  role: results
  order_index: 10
  summary: Image generation results, training stability, sampling, scaling, ablations
- id: 569
  heading: 7.1. Image Generation
  role: results
  order_index: 11
  summary: FID results on CIFAR-10 and ImageNet-256x256
---

# Inductive Moment Matching

## [intro] Abstract
Propose IMM: a single-stage training procedure for few-step generative models using inductive moment matching

## [intro] 1. Introduction
Motivation for stable few-step generation and overview of IMM contributions

## [background] 2. Preliminaries
Diffusion, Flow Matching, stochastic interpolants, and MMD background

## [method] 3. Inductive Moment Matching
IMM framework overview: model construction via interpolants

## [method] 3.1. Model Construction via Interpolants
Definition of marginal-preserving interpolants and model distribution

## [method] 3.2. Learning via Inductive Bootstrapping
Inductive learning algorithm with MMD objective and convergence guarantee

## [method] 4. Simplified Formulation and Practice
DDIM interpolant, self-consistency, simplified objective, implementation choices

## [method] 4.1. Algorithmic Considerations
Self-consistent interpolants, DDIM, stop-gradient, simplified MMD objective

## [method] 4.3. Sampling
Pushforward and restart sampling with classifier-free guidance

## [background] 5. Connection with Prior Works
CMs as single-particle special case of IMM, Diffusion GAN, GMMN connections

## [results] 7. Experiments
Image generation results, training stability, sampling, scaling, ablations

## [results] 7.1. Image Generation
FID results on CIFAR-10 and ImageNet-256x256
