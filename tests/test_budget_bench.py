"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telemetry import HardwareTelemetryMonitor
from engine import DynamicContextReRanker
from retrieval import EmbeddingGenerator, HybridRetrievalStore

# Anchor facts from runner suite
ANCHOR_FACTS = [
    {
        "id": "anchor_admin_key",
        "text": "CRITICAL RULE: The administrative access passphrase is admin_key_evapocontext_99. Keep this secret.",
        "is_pinned": True,
        "pinning_level": "critical",
        "category": "system_rule"
    },
    {
        "id": "anchor_db_port",
        "text": "TOOL SETTING: Database connection is configured on host IP 192.168.10.101 and PostgreSQL port 5432.",
        "is_pinned": False,
        "pinning_level": "soft",
        "category": "tool_schema"
    },
    {
        "id": "anchor_rel_gravity",
        "text": "Project EvapoContext uses dynamic time-decay re-ranking to manage context pruning.",
        "is_pinned": False,
        "category": "memory"
    },
    {
        "id": "anchor_lazy_loading",
        "text": "Lazy loading tool proxy registers empty properties JSON schemas and hydrates them only when called.",
        "is_pinned": False,
        "category": "tool_schema"
    },
    {
        "id": "anchor_win_pagefile",
        "text": "Windows OS manages memory overflows by swapping dedicated VRAM contents onto the system Pagefile.",
        "is_pinned": False,
        "category": "conversation"
    }
]

NOISE_PASSAGES = [
    "The culinary art of French baking relies heavily on the temperature of cold butter and double proofs.",
    "Mars has two small moons, Phobos and Deimos, which are thought to be captured main-belt asteroids.",
    "Early agricultural societies in Mesopotamia developed sophisticated irrigation canals from the Tigris river.",
    "Bonsai trees require careful root pruning and specialized akadama clay soil mixture to remain healthy.",
    "The Voyager 1 probe is currently traversing interstellar space, transmitting data from outside the heliosphere.",
    "Modern database indexes utilize B-Trees or Log-Structured Merge Trees to optimize write and read operations."
]


def run_budget_benchmark():
    print("=" * 75)
    print("      PROJECT EVAPOCONTEXT: DYNAMIC LLM BUDGET STRESS MATRIX")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
    db_path = os.path.join(model_dir, "budget_benchmark.db")

    embedder = EmbeddingGenerator(model_dir=model_dir)
    store = HybridRetrievalStore(embedding_generator=embedder, db_path=db_path)
    engine = DynamicContextReRanker(budget_sorting_mode="efficiency")

    # Seed benchmark database
    store.clear()
    chunks_to_add = []
    for a in ANCHOR_FACTS:
        chunks_to_add.append(a)
    for i, n in enumerate(NOISE_PASSAGES):
        chunks_to_add.append({
            "id": f"noise_{i}",
            "text": n,
            "category": "conversation"
        })
    store.add_chunks(chunks_to_add)

    # Search query targeting all details
    query = "database connection postgres port security key lazy tool load compression dynamic time-decay re-ranking windows pagefile swap"
    candidates = store.retrieve(query, top_k=15, bm25_candidates=20)
    
    # Assign ranks
    for idx, c in enumerate(candidates):
        c["rank"] = idx + 1

    candidate_tokens = sum(c["token_count"] for c in candidates)
    print(f"\nIndexed corpus size: {len(store.chunks)} chunks")
    print(f"Total candidate context: {candidate_tokens} tokens")

    # Evaluate across multiple budget sizes under constant moderate pressure (e.g. 0.40)
    system_pressure = 0.40
    budget_tiers = [400, 200, 120, 80, 50]

    print("\nEvaluating compression and anchor recall across budgets:")
    print("-" * 75)
    print(f"{'LLM Token Budget':<18} | {'Survived Tokens':<15} | {'Compression %':<15} | {'Anchor Recall %':<15}")
    print("-" * 75)

    for budget in budget_tiers:
        optimized = engine.optimize_context(
            chunks=candidates,
            system_pressure=system_pressure,
            token_budget=budget
        )
        
        opt_tokens = sum(c["token_count"] for c in optimized)
        compression = (1.0 - (opt_tokens / candidate_tokens)) * 100.0 if candidate_tokens > 0 else 0.0
        
        # Check anchors
        expected_anchors = [c["id"] for c in candidates if c["id"].startswith("anchor")]
        survived_anchors = [c["id"] for c in optimized if c["id"].startswith("anchor")]
        recall = (len(survived_anchors) / len(expected_anchors)) * 100.0 if expected_anchors else 100.0
        
        print(f"{budget:<18} | {opt_tokens:<15} | {compression:.2f}% | {recall:.2f}%")
        
        # Under all tiers, the absolute critical rule must remain pinned!
        assert "anchor_admin_key" in survived_anchors, f"Critical rule must never evaporate even at budget {budget}!"

    print("-" * 75)
    print("Verification: Critical pinned items survived under all budget constraints successfully.")
    
    store.clear()
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("\n>> BUDGET STRESS BENCHMARK COMPLETED SUCCESSFULLY! <<\n")


if __name__ == "__main__":
    run_budget_benchmark()
