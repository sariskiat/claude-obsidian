---
type: paper
slug: span-id-page-0-0-span-exploring-diffusion-transformer-designs-via-grafting
title: Exploring Diffusion Transformer Designs via Grafting
authors: '["Keshigeyan Chandrasegaran", "Michael Poli", "Daniel Y. Fu", "Dongjun Kim", "Lea M. Hadzic", "Manling Li", "Agrim Gupta", "Stefano Massaroli", "Azalia Mirhoseini", "Juan Carlos Niebles", "Stefano Ermon", "Li Fei-Fei"]'
source_path: ~/.paper-scholar/span-id-page-0-0-span-exploring-diffusion-transformer-designs-via-grafting/2506.05340.md
ingested_at: '2026-06-05 05:49:08'
authors_list:
- Keshigeyan Chandrasegaran
- Michael Poli
- Daniel Y. Fu
- Dongjun Kim
- Lea M. Hadzic
- Manling Li
- Agrim Gupta
- Stefano Massaroli
- Azalia Mirhoseini
- Juan Carlos Niebles
- Stefano Ermon
- Li Fei-Fei
sections:
- id: 478
  heading: Abstract
  role: intro
  order_index: 0
  summary: 'Propose grafting: a two-stage approach to edit pretrained DiTs for new architectures under small compute budgets'
- id: 479
  heading: 1 Introduction
  role: intro
  order_index: 1
  summary: Motivation for architecture editing via grafting, key results summary
- id: 480
  heading: 2 Prerequisites
  role: background
  order_index: 2
  summary: Diffusion models, DiTs, datasets and evaluation metrics
- id: 481
  heading: 3 Grafting Diffusion Transformers
  role: method
  order_index: 3
  summary: Two-stage grafting approach overview
- id: 482
  heading: 3.1 Two-Stage Grafting Approach
  role: method
  order_index: 4
  summary: Activation distillation (Stage 1) and lightweight finetuning (Stage 2)
- id: 483
  heading: 3.2 Self-grafting Baseline
  role: method
  order_index: 5
  summary: Control setup replacing operators with random-weight identical operators
- id: 484
  heading: 3.3 Activation Behavior Analysis and Self-grafting Results
  role: results
  order_index: 6
  summary: Analysis of MHA/MLP activation variance and regression objective comparison
- id: 485
  heading: 3.4 Locality Analysis of Self-attention
  role: results
  order_index: 7
  summary: Band-k metric quantifying MHA locality to guide grafting decisions
- id: 486
  heading: '4 Experiments I: Hybrid Architectures via Grafting'
  role: results
  order_index: 8
  summary: Testbed for evaluating hybrid architectures via grafting
- id: 487
  heading: 4.1 Testbed and Experiment setup
  role: method
  order_index: 9
  summary: 'Design axes: operator type, replacement, layer selection, replacement ratio'
- id: 488
  heading: '5 Experiments II: Grafting Text-to-Image Diffusion Transformers'
  role: results
  order_index: 10
  summary: Grafting PixArt-Sigma with Hyena-X for 1.43x speedup
- id: 489
  heading: '6 Case Study: Converting Model Depth to Width via Grafting'
  role: results
  order_index: 11
  summary: Parallelizing sequential transformer blocks to reduce depth 2x
- id: 490
  heading: 8 Conclusion and Discussion
  role: conclusion
  order_index: 12
  summary: Summary, limitations, and future work
---

# Exploring Diffusion Transformer Designs via Grafting

## [intro] Abstract
Propose grafting: a two-stage approach to edit pretrained DiTs for new architectures under small compute budgets

## [intro] 1 Introduction
Motivation for architecture editing via grafting, key results summary

## [background] 2 Prerequisites
Diffusion models, DiTs, datasets and evaluation metrics

## [method] 3 Grafting Diffusion Transformers
Two-stage grafting approach overview

## [method] 3.1 Two-Stage Grafting Approach
Activation distillation (Stage 1) and lightweight finetuning (Stage 2)

## [method] 3.2 Self-grafting Baseline
Control setup replacing operators with random-weight identical operators

## [results] 3.3 Activation Behavior Analysis and Self-grafting Results
Analysis of MHA/MLP activation variance and regression objective comparison

## [results] 3.4 Locality Analysis of Self-attention
Band-k metric quantifying MHA locality to guide grafting decisions

## [results] 4 Experiments I: Hybrid Architectures via Grafting
Testbed for evaluating hybrid architectures via grafting

## [method] 4.1 Testbed and Experiment setup
Design axes: operator type, replacement, layer selection, replacement ratio

## [results] 5 Experiments II: Grafting Text-to-Image Diffusion Transformers
Grafting PixArt-Sigma with Hyena-X for 1.43x speedup

## [results] 6 Case Study: Converting Model Depth to Width via Grafting
Parallelizing sequential transformer blocks to reduce depth 2x

## [conclusion] 8 Conclusion and Discussion
Summary, limitations, and future work
