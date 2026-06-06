---
type: paper
slug: anthropic-kg-cookbook
title: Knowledge graph construction with Claude
authors: '["Anthropic"]'
source_path: https://platform.claude.com/cookbook/capabilities-knowledge-graph-guide
ingested_at: '2026-06-04 11:32:45'
authors_list: []
sections:
- id: 284
  heading: Introduction
  role: introduction
  order_index: 1
  summary: Build knowledge graphs from unstructured text using Claude
- id: 285
  heading: Setup
  role: method
  order_index: 2
  summary: Python setup with Anthropic API, NetworkX, Pydantic
- id: 286
  heading: Building a Corpus
  role: method
  order_index: 3
  summary: Fetching Wikipedia summaries for Apollo program
- id: 287
  heading: Entity and Relation Extraction
  role: method
  order_index: 4
  summary: Structured output extraction with Pydantic models
- id: 288
  heading: Entity Resolution
  role: method
  order_index: 5
  summary: Claude-driven entity resolution to collapse surface forms
- id: 289
  heading: Assembling the Graph
  role: method
  order_index: 6
  summary: Building a NetworkX MultiDiGraph with canonical entities
- id: 290
  heading: Entity Summarization
  role: method
  order_index: 7
  summary: Summarizing hub nodes with graph neighborhood context
- id: 291
  heading: Querying the Graph
  role: method
  order_index: 8
  summary: Multi-hop question answering via serialized subgraphs
- id: 292
  heading: Evaluation
  role: method
  order_index: 9
  summary: Precision/recall against gold set for extraction quality
- id: 293
  heading: Scaling Up
  role: conclusion
  order_index: 10
  summary: 'Notes on scaling: prompt caching, batch API, blocking, incremental updates'
- id: 294
  heading: Summary
  role: conclusion
  order_index: 11
  summary: Complete KG pipeline with nothing but prompts
---

# Knowledge graph construction with Claude

## [introduction] Introduction
Build knowledge graphs from unstructured text using Claude

## [method] Setup
Python setup with Anthropic API, NetworkX, Pydantic

## [method] Building a Corpus
Fetching Wikipedia summaries for Apollo program

## [method] Entity and Relation Extraction
Structured output extraction with Pydantic models

## [method] Entity Resolution
Claude-driven entity resolution to collapse surface forms

## [method] Assembling the Graph
Building a NetworkX MultiDiGraph with canonical entities

## [method] Entity Summarization
Summarizing hub nodes with graph neighborhood context

## [method] Querying the Graph
Multi-hop question answering via serialized subgraphs

## [method] Evaluation
Precision/recall against gold set for extraction quality

## [conclusion] Scaling Up
Notes on scaling: prompt caching, batch API, blocking, incremental updates

## [conclusion] Summary
Complete KG pipeline with nothing but prompts
