---
type: paper
slug: graphrag-from-local-to-global
title: 'From Local to Global: A GraphRAG Approach to Query-Focused Summarization'
authors: '["Darren Edge", "Ha Trinh", "Newman Cheng", "Joshua Bradley", "Alex Chao", "Apurva Mody", "Steven Truitt", "Dasha Metropolitansky", "Robert Osazuwa Ness", "Jonathan Larson"]'
source_path: /Users/saris.kia.adm/.paper-scholar/from-local-to-global-a-graphrag-approach-to-query-focused-summarization/paper.json
ingested_at: '2026-06-04 11:31:08'
authors_list: []
sections:
- id: 248
  heading: Abstract
  role: abstract
  order_index: 1
  summary: GraphRAG combines KG generation and QFS for sensemaking over entire corpora
- id: 249
  heading: 1 Introduction
  role: introduction
  order_index: 2
  summary: 'Problem: RAG fails on global questions; GraphRAG proposed solution'
- id: 250
  heading: 2 Background
  role: background
  order_index: 3
  summary: Background on RAG, knowledge graphs with LLMs, and evaluation
- id: 251
  heading: 2.1 RAG Approaches and Systems
  role: background
  order_index: 4
  summary: RAG systems overview and vector RAG limitations
- id: 252
  heading: 2.2 Using Knowledge Graphs with LLMs and RAG
  role: background
  order_index: 5
  summary: KG extraction using LLMs and graph modularity
- id: 253
  heading: 2.3 Adaptive benchmarking for RAG Evaluation
  role: background
  order_index: 6
  summary: Generating evaluation benchmarks for global sensemaking
- id: 254
  heading: 2.4 RAG evaluation criteria
  role: background
  order_index: 7
  summary: LLM-as-judge evaluation approach
- id: 255
  heading: 3 Methods
  role: method
  order_index: 8
  summary: GraphRAG workflow details
- id: 256
  heading: 3.1 GraphRAG Workflow
  role: method
  order_index: 9
  summary: End-to-end pipeline from source docs to global answers
- id: 257
  heading: 3.1.1 Source Documents to Text Chunks
  role: method
  order_index: 10
  summary: Document chunking strategy
- id: 258
  heading: 3.1.2 Text Chunks to Entities and Relationships
  role: method
  order_index: 11
  summary: LLM-based entity and relationship extraction
- id: 259
  heading: 3.1.5 Graph Communities to Community Summaries
  role: method
  order_index: 12
  summary: Report-like summaries of each community
- id: 260
  heading: 3.1.6 Community Summaries to Community Answers to Global Answer
  role: method
  order_index: 13
  summary: Multi-stage answer generation from community summaries
- id: 261
  heading: 4 Analysis
  role: method
  order_index: 14
  summary: Experimental design and datasets
- id: 262
  heading: 5 Results
  role: method
  order_index: 15
  summary: Experimental results
- id: 263
  heading: 6 Discussion
  role: conclusion
  order_index: 16
  summary: Limitations and future work
- id: 264
  heading: 7 Conclusion
  role: conclusion
  order_index: 17
  summary: GraphRAG conclusion and contributions
---

# From Local to Global: A GraphRAG Approach to Query-Focused Summarization

## [abstract] Abstract
GraphRAG combines KG generation and QFS for sensemaking over entire corpora

## [introduction] 1 Introduction
Problem: RAG fails on global questions; GraphRAG proposed solution

## [background] 2 Background
Background on RAG, knowledge graphs with LLMs, and evaluation

## [background] 2.1 RAG Approaches and Systems
RAG systems overview and vector RAG limitations

## [background] 2.2 Using Knowledge Graphs with LLMs and RAG
KG extraction using LLMs and graph modularity

## [background] 2.3 Adaptive benchmarking for RAG Evaluation
Generating evaluation benchmarks for global sensemaking

## [background] 2.4 RAG evaluation criteria
LLM-as-judge evaluation approach

## [method] 3 Methods
GraphRAG workflow details

## [method] 3.1 GraphRAG Workflow
End-to-end pipeline from source docs to global answers

## [method] 3.1.1 Source Documents to Text Chunks
Document chunking strategy

## [method] 3.1.2 Text Chunks to Entities and Relationships
LLM-based entity and relationship extraction

## [method] 3.1.5 Graph Communities to Community Summaries
Report-like summaries of each community

## [method] 3.1.6 Community Summaries to Community Answers to Global Answer
Multi-stage answer generation from community summaries

## [method] 4 Analysis
Experimental design and datasets

## [method] 5 Results
Experimental results

## [conclusion] 6 Discussion
Limitations and future work

## [conclusion] 7 Conclusion
GraphRAG conclusion and contributions
