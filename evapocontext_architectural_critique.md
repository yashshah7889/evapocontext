# EvapoContext: Master Architectural Critique & Drawbacks Log

This document consolidates the active architectural drawbacks, design flaws, and performance bottlenecks identified in the **Project EvapoContext** codebase. It serves as our live brainstorm roadmap for refactoring and enhancement.

---

## 1. Algorithmic and Database Bottlenecks

### A. $O(N)$ Python-Level Search Loop & RAM Bloat
*   **Target Code:** 
    *   [HybridRetrievalStore._load_from_sqlite](file:///C:/Users/yashs/Desktop/weekend/src/retrieval.py#L353-L389)
    *   [BM25Index.search](file:///C:/Users/yashs/Desktop/weekend/src/retrieval.py#L264-L297)
*   **The Issue:** On startup, the retrieval store queries the entire SQLite database (`SELECT * FROM chunks`) and loads every document, token count, category weight, and raw 384-dimensional embedding float array into memory under `self.chunks`.
*   **The Impact:** As the corpus scales to tens of thousands of documents, loading everything into RAM exhausts local host memory. This causes the telemetry monitor in [telemetry.py](file:///C:/Users/yashs/Desktop/weekend/src/telemetry.py) to flag high system pressure ($P_{sys}$), forcing the gateway into permanent, degraded "Vector Bypass" mode. Furthermore, scoring matches by iterating over `self.chunks` in a raw Python loop inside [BM25Index.search](file:///C:/Users/yashs/Desktop/weekend/src/retrieval.py#L276-L294) scales at $O(N)$ complexity, bottlenecks the CPU, and causes query response times to balloon.

### B. Naive Whitespace Chunking
*   **Target Code:** 
    *   [TextSplitter.split_text](file:///C:/Users/yashs/Desktop/weekend/src/retrieval.py#L50-L68)
*   **The Issue:** The text splitter segments documents based on simple whitespace splitting: `text.strip().split()`. It does not parse punctuation, line breaks, Markdown structural tags (e.g., `#` headers), or syntax boundaries for code blocks, JSON strings, or Python scripts.
*   **The Impact:** Slicing passages mid-clause or mid-code block breaks logical context. This degrades dense vector representation, leading to poor cosine similarity alignment and reduced retrieval recall.

---

## 2. Logic Loop in Lazy-Loaded Schema Gateway

*   **Target Code:** 
    *   [server.py](file:///C:/Users/yashs/Desktop/weekend/src/server.py) (specifically the JSON-RPC message loop routing `tools/list`)
*   **The Issue:** To bypass the "Tools Tax" (bloat) in the initial prefill, the daemon registers tools with empty parameters schemas: `{"type": "object", "properties": {}}`.
*   **The Impact:** The client LLM is misled into believing the tool requires zero arguments. When it triggers a `tools/call` command, it generates an empty parameters payload. The server's validation layer catches this missing payload and errors out, or runs the tool handler with invalid/nil variables, rendering complex tools unusable.

---

## 3. Mathematical and Scoring Engine Vulnerabilities

### A. Relevance-Agnostic Pruning Threshold (Evaporation into Emptiness)
*   **Target Code:** 
    *   [DynamicContextReRanker.calculate_pruning_threshold](file:///C:/Users/yashs/Desktop/weekend/src/engine.py#L77-L105)
*   **The Issue:** The pruning threshold rises strictly according to system telemetry pressure: $\text{Threshold} = 0.20 + (P_{sys}^2 \cdot 0.55)$. It has no awareness of the range of relevance scores in the active retrieved set.
*   **The Impact:** If the host machine is under heavy load ($P_{sys} = 0.85$, threshold $= 0.60$), and the most relevant documents in the database score between $0.40$ and $0.55$, **every single unpinned document is evaporated**. The context manager feeds an empty payload to the LLM, causing a factual accuracy failure, even if the model's token budget has plenty of space.

### B. Conceptual Misalignment: "Temporal Decay" vs. "Relevance Rank Decay"
*   **Target Code:** 
    *   [DynamicContextReRanker.calculate_retention_score](file:///C:/Users/yashs/Desktop/weekend/src/engine.py#L107-L174)
*   **The Issue:** The decay factor is calculated using the variable `rank` (the index in the similarity-sorted search results, 1-indexed), not the actual conversational age of the turn (turns offset or timestamp decay).
*   **The Impact:** The decay of a message fluctuates randomly depending on what the user currently queries. A highly relevant old turn (Turn 1) matched #1 in search results receives zero decay penalty, whereas a newer turn (Turn 3) matched #2 is decayed. This double-penalizes semantic similarity while failing to prune chronologically outdated context.

---

## 4. Concurrency and Thread Safety Issues

### A. Lock Release Race Conditions in Caches
*   **Target Code:** 
    *   [HybridRetrievalStore.retrieve](file:///C:/Users/yashs/Desktop/weekend/src/retrieval.py#L517-L708)
*   **The Issue:** The lock `self.lock` is acquired and released multiple times during a single retrieval call. The lock is released during query embedding generation and matrix multiplications, and re-acquired at the end to commit results to cache.
*   **The Impact:** If another thread executes `add_chunks` (which calls `_invalidate_caches` to wipe all cache records) during the time the search thread has released the lock to run ONNX, the search thread will finish, re-acquire the lock, and write its **stale search results** back into the newly invalidated cache. Subsequent queries will hit this stale cache, returning outdated records.

---

## 5. Brainstorming Notes & Proposed Solutions

### A. Resolution for 3B: Segment-Aware Stateful Turn Decay
*   **The Design:**
    Transition the gateway into a Stateful Turn-Aware Router. Tag each indexed chunk with a `turn_index` and maintain a `current_turn` counter in the session. Route the decay calculations dynamically based on chunk category to ensure static knowledge never decays, while dynamic session data is pruned chronologically.
*   **Segment-Aware Policies (Plain Text):**
    *   `system_rule` / `tool_schema` (Static Rules): Turn Offset = 0 (No Decay)
    *   `memory` (Static RAG documents / PDFs / Code Files): Turn Offset = 0 (No Decay)
    *   `conversation` (Chat History turns): Turn Offset = Current Turn - Turn Indexed
    *   `tool_output` (MCP Tool Outputs): Turn Offset = (Current Turn - Turn Indexed) * 2.0 (Aggressive Decay)
*   **Decay and Retention Score Math (Plain Text):**
    ```text
    Temporal Decay = Log2(1 + Turn Offset)
    Retention Score = (Similarity * Token Normalization Factor) / Temporal Decay
    ```
*   **Expected Behavior:**
    1. **Dynamic vs Static Isolation:** Static reference items and tool definitions never decay.
    2. **Freshness Preservation:** Chat turns and MCP tool outputs age naturally, preventing the stale-topic bug.
    3. **Aggressive Temporary Garbage Collection:** Massive outputs from tools are evicted twice as fast.

### B. Resolution for 3A: Top-Score Capped Adaptive Thresholding
*   **The Design:**
    Cap the telemetry-driven threshold dynamically based on the maximum score of the retrieved chunks. This ensures that when match quality is low, the threshold does not rise so high that it evaporates all unpinned context, while maintaining aggressive pruning when match quality is high.
*   **Math Formulation (Plain Text):**
    ```text
    Telemetry Threshold = Base Threshold + (System Pressure^2 * Pressure Factor)
    Threshold Cap = Max(Base Threshold, Max Score * Threshold Ratio)
    Pruning Threshold = Min(Telemetry Threshold, Threshold Cap)
    ```
    Where `Threshold Ratio` defaults to `0.80`.
*   **Expected Behavior:**
    1. **Relevance-Aware Scaling:** If `Max Score` is low (e.g., `0.40`), the threshold is capped (e.g., `0.32`), preserving the top match and related supporting chunks.
    2. **Telemetry-Driven Pruning:** If `Max Score` is high (e.g., `0.95`), the threshold is governed by telemetry, pruning lower-value chunks aggressively.
    3. **Junk Suppression:** If `Max Score` is below `Base Threshold`, the threshold defaults to `Base Threshold`, evaporating irrelevant noise.

---

## 6. Implementation & Verification of Solutions 3A & 3B

The solutions for Vulnerabilities 3A and 3B have been fully integrated into the EvapoContext runtime core.

### A. Adaptive Pruning Threshold (3A) Specification

To prevent context collapse under high system stress when retrieved documents have low relevance, we implement a dynamic cap on the pruning threshold.

1. **Mathematical Formulations (Plain Text):**
   * Telemetry_Threshold = Base_Threshold + (System_Pressure^2 * Pressure_Factor)
   * Threshold_Cap = Max(Base_Threshold, Max_Score * Threshold_Ratio)
   * Pruning_Threshold = Min(Telemetry_Threshold, Threshold_Cap)
2. **Parameters:**
   * Base_Threshold = 0.20 (Default)
   * Pressure_Factor = 0.55 (Default)
   * Threshold_Ratio = 0.80 (Default)
3. **Behavioral Justification:**
   * Under low pressure (System_Pressure = 0.0), Pruning_Threshold defaults to 0.20.
   * Under critical pressure (System_Pressure = 1.0) with a highly relevant match (Max_Score = 0.95), Pruning_Threshold is Min(0.75, Max(0.20, 0.76)) = 0.75, enabling aggressive pruning.
   * Under critical pressure (System_Pressure = 1.0) with weak matches (Max_Score = 0.40), Pruning_Threshold is Min(0.75, Max(0.20, 0.32)) = 0.32. This preserves the top-scoring matches (similarity 0.40) rather than evaporating them.

### B. Segment-Aware Stateful Turn Decay (3B) Specification

To resolve the conceptual flaw of ranking-based decay (which double-penalizes similarity and fluctuates randomly based on query strings), we transition EvapoContext to a stateful turn-based decay system with segment-aware routing policies.

1. **Stateful Turn tracking:**
   * A persistent `turn_index` column is added to the SQLite database schema with a dynamic PRAGMA migration check for compatibility with legacy database files.
   * A stateful session counter `current_turn` is maintained by the EvapoContext connector and daemon, incremented automatically on query retrieval operations.
2. **Segment-Aware Decay Policies (Plain Text):**
   * **RAG (memory category):** Turn_Offset = 0. Static document pages or manual chapters loaded at startup do not decay.
   * **MCP (system_rule / tool_schema categories):** Turn_Offset = 0. Tool schema definitions and critical instructions are immune to temporal decay.
   * **NO RAG (conversation category):** Turn_Offset = Current_Turn - Turn_Indexed. Chat message turns decay naturally based on chronological age.
   * **Hybrid (tool_output category):** Turn_Offset = (Current_Turn - Turn_Indexed) * 2.0. Dynamic tool results (e.g. command outputs, file views) decay at double speed to prevent RAM/context bloat.
3. **Decay and Retention Score Formulas (Plain Text):**
   * Age = Max(1, Turn_Offset) [or Max(1, Retrieval_Rank) if Turn_Offset is None]
   * Decay_Factor = Log2(Age + 1)
   * Retention_Score = Weight_Product / Decay_Factor

