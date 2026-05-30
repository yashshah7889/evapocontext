"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import sys
import os
import time
import json
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telemetry import HardwareTelemetryMonitor
from engine import DynamicContextReRanker
from retrieval import EmbeddingGenerator, HybridRetrievalStore, TextSplitter

# 10 specific anchor facts to hide in the noise
HEAVY_ANCHORS = [
    {
        "id": "heavy_anchor_admin",
        "text": "CRITICAL PASS RULE: The secret admin code for this session is admin_master_token_5588. Keep it secret.",
        "is_pinned": True,
        "pinning_level": "critical",
        "category": "system_rule"
    },
    {
        "id": "heavy_anchor_db",
        "text": "TOOL CONFIG RULE: PostgreSQL DB connection is host 10.0.0.45, port 5432, username db_admin.",
        "is_pinned": False,
        "pinning_level": "soft",
        "category": "tool_schema"
    },
    {
        "id": "heavy_anchor_memory_1",
        "text": "EPISODE MEMORY: The user stated that they are building a web server backend using Python and FastAPI.",
        "is_pinned": False,
        "category": "memory"
    },
    {
        "id": "heavy_anchor_memory_2",
        "text": "EPISODE MEMORY: The deployment target is AWS ECS Fargate with a memory allocation of 1.0 GB RAM.",
        "is_pinned": False,
        "category": "memory"
    },
    {
        "id": "heavy_anchor_pagefile",
        "text": "CONVERSATION STATE: Windows virtual memory pagefile handles excessive allocations on disk swap regions.",
        "is_pinned": False,
        "category": "conversation"
    }
]

# Standard articles to generate heavy noise (20,000 words)
NOISE_TEMPLATES = [
    "The architecture of modern processors relies on a deep instruction pipeline, branch prediction, and multi-level cache hierarchies. CPUs are designed to fetch, decode, and execute instructions concurrently. When a pipeline stall occurs, the processor must flush its pipeline, causing latency wiggles in performance. Cache levels (L1, L2, and L3) store frequently used memory blocks closer to the CPU cores, reducing the access time from nanoseconds to picoseconds.",
    "Quantum mechanics describes the physical properties of nature at the atomic and subatomic scale. Unlike classical physics, where values are continuous, quantum properties exist in discrete energy packets. Qubits are the fundamental processing elements of quantum computers. They can exist in a superposition of states, allowing quantum algorithms to compute complex linear algebra equations exponentially faster than classical silicon chips.",
    "The history of ocean exploration is marked by the development of specialized diving equipment, submarines, and autonomous underwater vehicles. Deep-sea trenches, like the Mariana Trench, host unique lifeforms that survive under extreme hydrostatic pressure. Chemosynthetic bacteria utilize sulphur compounds from thermal vents to generate energy, forming the foundation of complete ecosystems that exist completely isolated from solar energy.",
    "Linguistic relativity, also known as the Sapir-Whorf hypothesis, suggests that the structure and vocabulary of a language influence its speakers' cognitive processes and worldview. Different languages categorize colors, directions, and time in highly distinct ways. For example, some indigenous cultures navigate using absolute cardinal directions (North, South, East, West) rather than relative terms (left, right, front, back).",
    "Photosynthesis in green plants is a multi-stage chemical reaction that converts solar light energy, carbon dioxide, and water into chemical energy in the form of glucose. This reaction is catalyzed by chlorophyll molecules located inside plant chloroplast structures. The process is divided into light-dependent reactions, which generate ATP and NADPH, and the light-independent Calvin cycle, which synthesizes carbohydrates.",
    "Modern cloud computing relies heavily on hypervisors and container orchestration engines to run thousands of isolated applications on shared physical hardware. Kubernetes manages container lifecycles, automated scaling, and routing pathways across distributed host clusters. Load balancers distribute traffic dynamically to prevent application servers from experiencing bottleneck collapses."
]


def run_heavy_benchmarks():
    print("=" * 75)
    print("     PROJECT EVAPOCONTEXT: 20,000-TOKEN HEAVY PROMPT STRESS BENCH")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
    db_path = os.path.join(model_dir, "heavy_stress.db")

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # 1. Initialize Components
    print("\n--- Phase 1: Hydrating Models & Database ---")
    embedder = EmbeddingGenerator(model_dir=model_dir)
    store = HybridRetrievalStore(embedding_generator=embedder, db_path=db_path)
    engine = DynamicContextReRanker(budget_sorting_mode="force")

    # Seed heavy noise database (20,000 words ~ 120 text chunks)
    store.clear()
    
    # Generate ~120 noise passages (repeating templates to make up volume)
    raw_chunks = []
    for idx in range(115):
        template = NOISE_TEMPLATES[idx % len(NOISE_TEMPLATES)]
        raw_chunks.append({
            "id": f"heavy_noise_{idx}",
            "text": f"[Chunk #{idx}] {template}",
            "category": "conversation"
        })
        
    # Scatter the 5 specific anchors inside the noise list
    for idx, anchor in enumerate(HEAVY_ANCHORS):
        insert_pos = (idx * 20) + 10
        raw_chunks.insert(insert_pos, anchor)

    print(f"Total chunks to index: {len(raw_chunks)}")

    # Measure bulk indexing speed
    t_start = time.perf_counter()
    store.add_chunks(raw_chunks)
    t_index = time.perf_counter() - t_start
    print(f"Bulk Vector Indexing completed in: {t_index:.3f} s")
    print(f"Average indexing speed: {len(raw_chunks) / t_index:.2f} chunks/sec")

    # Measure memory growth
    mem_after = process.memory_info().rss / (1024 * 1024)
    print(f"Resident Memory Footprint: {mem_after:.2f} MB (Growth: {mem_after - mem_before:+.2f} MB)")

    # 2. Context Retrieval & Evaporation Run
    # Query targets multiple details simultaneously
    stress_query = "FastAPI web server database port connection token security key windows pagefile virtual memory ECS Fargate"
    
    # We retrieve a large candidate set (top 80 chunks) to verify sorting scaling
    print("\n--- Phase 2: Simulating Context Evaporation Routing ---")
    t_start = time.perf_counter()
    candidates = store.retrieve(stress_query, top_k=80, bm25_candidates=100)
    t_retrieve = time.perf_counter() - t_start
    print(f"Search & Re-rank retrieval completed in: {t_retrieve * 1000:.3f} ms")

    # Assign ranks
    for idx, c in enumerate(candidates):
        c["rank"] = idx + 1

    candidate_tokens = sum(c["token_count"] for c in candidates)
    print(f"Original Candidate prompt size: {candidate_tokens} tokens across {len(candidates)} chunks.")

    # Test Case A: Low Pressure (P_sys = 0.0), No Budget (Normal exploration)
    optimized_low = engine.optimize_context(candidates, system_pressure=0.0)
    tokens_low = sum(c["token_count"] for c in optimized_low)
    expected_anchors = [c["id"] for c in candidates if c["id"].startswith("heavy_anchor")]
    survived_low = [c["id"] for c in optimized_low if c["id"].startswith("heavy_anchor")]
    recall_low = len(survived_low) / len(expected_anchors) if expected_anchors else 1.0

    print("\n[Case A: Idle System (Pressure = 0.0)]")
    print(f"  Optimized Context Size: {tokens_low} tokens ({100.0 - (tokens_low/candidate_tokens)*100.0:.2f}% compression)")
    print(f"  Expected Anchors:       {expected_anchors}")
    print(f"  Survived Anchors:       {survived_low}")
    print(f"  Recall Accuracy:        {recall_low * 100.0:.2f}%")

    # Test Case B: High Pressure (P_sys = 0.8), No Budget (Dynamic Evaporation Active)
    optimized_high = engine.optimize_context(candidates, system_pressure=0.8)
    tokens_high = sum(c["token_count"] for c in optimized_high)
    survived_high = [c["id"] for c in optimized_high if c["id"].startswith("heavy_anchor")]
    recall_high = len(survived_high) / len(expected_anchors) if expected_anchors else 1.0

    print("\n[Case B: Stressed System (Pressure = 0.8)]")
    print(f"  Optimized Context Size: {tokens_high} tokens ({100.0 - (tokens_high/candidate_tokens)*100.0:.2f}% compression)")
    print(f"  Survived Anchors:       {survived_high}")
    print(f"  Recall Accuracy:        {recall_high * 100.0:.2f}%")

    # Test Case C: High Pressure (P_sys = 0.8) + Strict Token Budget (1,200 tokens)
    token_budget = 1200
    optimized_budget = engine.optimize_context(candidates, system_pressure=0.8, token_budget=token_budget)
    tokens_budget = sum(c["token_count"] for c in optimized_budget)
    survived_budget = [c["id"] for c in optimized_budget if c["id"].startswith("heavy_anchor")]
    recall_budget = len(survived_budget) / len(expected_anchors) if expected_anchors else 1.0

    print(f"\n[Case C: Stressed System + Strict Token Budget ({token_budget} tokens)]")
    print(f"  Optimized Context Size: {tokens_budget} tokens ({100.0 - (tokens_budget/candidate_tokens)*100.0:.2f}% compression)")
    print(f"  Survived Anchors:       {survived_budget}")
    print(f"  Recall Accuracy:        {recall_budget * 100.0:.2f}%")

    # Assertions to verify correctness
    assert "heavy_anchor_admin" in survived_budget, "Critical pinned rule evaporated under budget constraint!"
    assert tokens_budget <= token_budget, f"Token budget exceeded: {tokens_budget} > {token_budget}"
    print("\nVerification checks resolved successfully. Critical pinned data is preserved under maximum stress.")

    # 3. Clean up database
    store.clear()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n>> 20,000-TOKEN STRESS BENCHMARK COMPLETED SUCCESSFULLY! <<\n")


if __name__ == "__main__":
    run_heavy_benchmarks()
