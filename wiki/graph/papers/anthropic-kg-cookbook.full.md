---
type: paper-fulltext
slug: anthropic-kg-cookbook
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/anthropic-kg-cookbook/anthropic-kg-cookbook.md
paper: "[[anthropic-kg-cookbook]]"
---
<!-- fetched from https://platform.claude.com/cookbook/capabilities-knowledge-graph-guide by afk (web cookbook, not an arXiv paper) -->

# Knowledge Graph Construction with Claude

## Overview

This cookbook guide demonstrates how to build knowledge graphs from unstructured text using Claude. It covers entity extraction, relation mining, deduplication (entity resolution), and multi-hop graph querying—all without training data.

## What You'll Learn

By following this guide, you'll be able to:

- Use **structured outputs** to extract typed entities and subject–predicate–object triples from text
- Apply **Claude-driven entity resolution** to collapse surface-form variants into canonical nodes
- Assemble and query an in-memory graph using NetworkX
- Run **multi-hop questions** by serializing subgraphs back to Claude
- Measure extraction quality with **precision/recall** and reason about cost/quality tradeoffs between Haiku and Sonnet

## Prerequisites

- Python 3.11+
- Anthropic API key
- Basic familiarity with graphs (nodes, edges, traversal)

## Key Components

### 1. Setup

The guide uses two Claude models:
- **Haiku** for high-volume, schema-constrained extraction (speed and cost-efficient)
- **Sonnet** for entity resolution and summarization (where nuance matters)

```python
EXTRACTION_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-sonnet-4-6"
```

### 2. Entity and Relation Extraction

Using **structured outputs** with Pydantic models:

```python
class Entity(BaseModel):
    name: str
    type: EntityType  # PERSON, ORGANIZATION, LOCATION, EVENT, ARTIFACT
    description: str

class Relation(BaseModel):
    source: str
    predicate: str
    target: str

class ExtractedGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
```

The extraction uses `client.messages.parse()` to guarantee schema-compliant output.

### 3. Entity Resolution

Rather than string similarity, Claude clusters entities using their descriptions as disambiguation context. For example, "Edwin Aldrin" and "Buzz Aldrin" (zero character overlap) are correctly unified as the same person using their contextual descriptions.

```python
class Cluster(BaseModel):
    canonical: str
    aliases: list[str]
```

### 4. Graph Assembly

Entities are loaded into a NetworkX `MultiDiGraph` where:
- **Nodes** carry entity type, source documents, and mention counts
- **Edges** carry predicates and source document information
- Multiple distinct predicates between the same pair of entities are preserved

### 5. Entity Summarization

Hub nodes (high-degree entities) get rich profiles synthesized across all documents mentioning them:

```python
class EntityProfile(BaseModel):
    summary: str
    key_facts: list[str]
    time_range: TimeRange
```

### 6. Multi-Hop Querying

Subgraphs are serialized as triples and returned to Claude for reasoning:

```
(node1) --[predicate]--> (node2)
(node2) --[predicate]--> (node3)
```

This enables questions like "Which locations are connected to people who were part of Apollo 11?"

## Evaluation

The guide includes evaluation against a gold standard set using precision, recall, and F1 scores:

```python
def prf(predicted: set, gold: set) -> tuple[float, float, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1
```

## Scaling Considerations

- **Extraction cost**: Use prompt caching and the Batch API for large corpora
- **Entity resolution**: Block candidates by cheap signals (name overlap, embeddings) before sending to Claude
- **Incremental updates**: Only re-summarize entities when their source documents materially change
- **Storage**: NetworkX works up to hundreds of thousands of edges; migrate to Neo4j, Neptune, or Postgres for larger graphs

## Key Advantages Over Traditional Approaches

- **No training data required** — prompts replace trained NER and relation classifiers
- **Semantic entity resolution** — catches non-obvious aliases (e.g., "Edwin" vs "Buzz")
- **Traceable answers** — graph-grounded responses cite specific edges rather than relying on pretraining
- **End-to-end pipeline** — extraction, resolution, summarization, and querying all use the same interface
