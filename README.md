# evapoContext™
> **Hardware-Aware Stateful Context Router & Model Context Protocol (MCP) Gateway**

`evapoContext` is an enterprise-grade, local background context router and Model Context Protocol (MCP) gateway. It prevents local Large Language Models (LLMs) from experiencing effective context window collapse and thermal throttling due to large JSON tool definitions ("Tools Tax") and chat history volumes.

Unlike traditional context managers that treat chat history as flat text, `evapoContext` calculates context retention using physical metaphors—temporal decay, memory pressure, and gravitational fields—to automatically evict low-value contexts under high load while preserving critical system instructions.

---

## Key Features

* **Adaptive Hardware-Aware Retrieval Gating (AHARG):** Dynamically scales vector search candidates under load. Automatically switches to BM25 sparse matching under extreme CPU stress ($\ge 90\%$ system pressure) to prevent execution thrashing.
* **Relativistic Gravitational Scoring:** Computes retention priority based on semantic relevance and temporal rank decay:
  $$F_g = \frac{M_{query} \cdot M_{context}}{r^2}$$
* **Systemic Flux Pinning:** Lock critical instructions or soft-boost specific context categories (system rules, tool schemas, user memories) using a three-tier pinning system.
* **Lazy-Loaded Tool Schema Compression (JIT):** Minimizes LLM tool tax by returning empty schema definitions on startup and hydrating them on demand only when called.
* **Dual-Layer Caching:** Integrates an exact lexical cache (~0.003 ms responses) and a semantic cache ($\ge 96\%$ cosine similarity) to completely bypass vector indexing overhead.
* **Zero-Compilation Persistence:** Deserializes embeddings instantly from standard SQLite binary BLOBs without re-embedding.

---

## Architecture

```
       ┌──────────────┐     stdio (JSON-RPC)     ┌────────────────────┐
       │  AI CLIENT   │ ◄──────────────────────► │ evapocontext daemon│
       │ (Claude/etc) │                          │     (server.py)    │
       └──────────────┘                          └─────────┬──────────┘
                                                           │
        ┌──────────────────────────────────────────────────┼──────────────────────────────────────────────────┐
        │                                                  │                                                  │
        ▼                                                  ▼                                                  ▼
┌───────────────────────────────┐                  ┌───────────────────────────────┐                  ┌───────────────────────────────┐
│     TELEMETRY MONITOR         │                  │      SCORING ENGINE           │                  │       SEMANTIC INDEX          │
│      (telemetry.py)           │                  │        (engine.py)            │                  │       (retrieval.py)          │
├───────────────────────────────┤                  ├───────────────────────────────┤                  ├───────────────────────────────┤
│ Monitor virtual RAM, CPU, and │                  │ Calculates context relevance  │                  │ Hybrid search (BM25 + Dense)  │
│ swap usage. Smooths transient │                  │ scores under real-time memory │                  │ utilizing local ONNX BGE      │
│ spikes using EMA filtering.   │                  │ pressure.                     │                  │ embeddings and SQLite.        │
└───────────────────────────────┘                  └───────────────────────────────┘                  └───────────────────────────────┘
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/evapocontext/evapocontext.git
cd evapocontext

# Install dependencies
pip install numpy psutil onnxruntime tokenizers huggingface_hub jsonschema
```

---

## Quick Start (Developer API)

Integrate `evapoContext` into your agent pipelines (LangChain, AutoGen, CrewAI) with less than 15 lines of code:

```python
import os
from evapocontext_connector import EvapoContextConnector

# Initialize connector
model_dir = "./model"
connector = EvapoContextConnector(model_dir=model_dir)

# Index background context
connector.add_document(
    doc_id="system_rules",
    text="CRITICAL REQUIREMENT: Always verify database SSL is active.",
    category="system_rule",
    is_pinned=True  # Infinitely pinned
)

connector.add_document(
    doc_id="memory_1",
    text="Deploying on AWS ECS using Docker containers.",
    category="memory"
)

# Query and get optimized context
context = connector.search_and_optimize(
    query="FastAPI connection settings",
    token_budget=500
)

print(context["optimized_text"])

# Cleanup resources
connector.close()
```

---

## Claude Desktop Integration

Add `evapoContext` as an MCP server in Claude Desktop:

* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "evapocontext": {
      "command": "python",
      "args": [
        "/path/to/evapocontext/src/server.py"
      ]
    }
  }
}
```

---

### Copyright & Trademark

Copyright (c) 2026 yashs. All rights reserved.

`evapoContext`™ and its related mechanics (Adaptive Hardware-Aware Retrieval Gating, Relativistic Gravitational Scoring, Systemic Flux Pinning) are trademarks of yashs.
