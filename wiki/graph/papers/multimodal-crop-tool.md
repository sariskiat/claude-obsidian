---
type: paper
slug: multimodal-crop-tool
title: Giving Claude a crop tool for better image analysis
authors: '["Nadine Yasser"]'
source_path: https://platform.claude.com/cookbook/multimodal-crop-tool
ingested_at: '2026-06-04 11:33:21'
authors_list: []
sections:
- id: 295
  heading: Introduction
  role: introduction
  order_index: 1
  summary: Give Claude a crop tool to zoom into image regions for detailed analysis
- id: 296
  heading: When is a Crop Tool Useful
  role: introduction
  order_index: 2
  summary: Charts, documents, technical diagrams, dense images
- id: 297
  heading: Setup
  role: method
  order_index: 3
  summary: Python setup with Anthropic API, Pillow, datasets
- id: 298
  heading: Load an Example Chart
  role: method
  order_index: 4
  summary: Using FigureQA dataset for demonstration
- id: 299
  heading: Define the Crop Tool
  role: method
  order_index: 5
  summary: Tool definition using normalized coordinates 0-1
- id: 300
  heading: The Agentic Loop
  role: method
  order_index: 6
  summary: Connect image with crop tool in a tool-use loop
- id: 301
  heading: 'Demo: Chart Analysis'
  role: method
  order_index: 7
  summary: Demonstrating crop tool with pie chart and bar chart analysis
- id: 302
  heading: 'Alternative: Using the Claude Agent SDK'
  role: method
  order_index: 8
  summary: Cleaner tool definition using Python decorators
- id: 303
  heading: Summary
  role: conclusion
  order_index: 9
  summary: Crop tool pattern is simple but powerful for detailed image analysis
---

# Giving Claude a crop tool for better image analysis

## [introduction] Introduction
Give Claude a crop tool to zoom into image regions for detailed analysis

## [introduction] When is a Crop Tool Useful
Charts, documents, technical diagrams, dense images

## [method] Setup
Python setup with Anthropic API, Pillow, datasets

## [method] Load an Example Chart
Using FigureQA dataset for demonstration

## [method] Define the Crop Tool
Tool definition using normalized coordinates 0-1

## [method] The Agentic Loop
Connect image with crop tool in a tool-use loop

## [method] Demo: Chart Analysis
Demonstrating crop tool with pie chart and bar chart analysis

## [method] Alternative: Using the Claude Agent SDK
Cleaner tool definition using Python decorators

## [conclusion] Summary
Crop tool pattern is simple but powerful for detailed image analysis
