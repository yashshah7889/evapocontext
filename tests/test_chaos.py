"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import sys
import os
import time
import concurrent.futures
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retrieval import EmbeddingGenerator, HybridRetrievalStore, TextSplitter
from engine import DynamicContextReRanker


def run_chaos_tests():
    print("=" * 75)
    print("      PROJECT EVAPOCONTEXT: CHAOS & BOUNDARY STRESS TEST SUITE")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
    
    # Hydrate dependencies
    embedder = EmbeddingGenerator(model_dir=model_dir)
    store = HybridRetrievalStore(embedding_generator=embedder)
    engine = DynamicContextReRanker()

        # Chaos Test 1: Nil / Empty Input Resiliency
        print("\n--- Chaos Test 1: Empty and Nil Payloads ---")
    
    # Query with empty query
    res_empty_q = store.retrieve("", top_k=5)
    print(f"Empty query returned {len(res_empty_q)} results.")
    assert isinstance(res_empty_q, list), "Should return list"
    
    # Try indexing empty lists or corrupt chunks
    store.add_chunks([])
    print("Empty raw chunks handled safely.")
    
    corrupt_chunks = [
        {"id": "corrupt_1", "category": "system_rule"},  # Missing 'text' key
        {"id": "corrupt_2", "text": "", "category": "memory"},  # Empty text
        {"id": "corrupt_3", "text": "   ", "category": "conversation"},  # Whitespace only
    ]
    store.add_chunks(corrupt_chunks)
    print("Corrupt/empty text chunks bypassed safely.")
    assert len(store.chunks) == 0, "No corrupt items should be indexed"

        # Chaos Test 2: Invalid Categories & Weights
        print("\n--- Chaos Test 2: Unregistered Categories & Custom Weights ---")
    
    valid_chunk = {
        "id": "valid_1",
        "text": "This is a legitimate context chunk talking about database security.",
        "category": "unregistered_custom_category"  # Custom category name
    }
    store.add_chunks([valid_chunk])
    print(f"Custom category '{store.chunks[0]['category']}' indexed successfully.")
    
    # Retrieve with custom category weights that do not define the custom category
    res_custom_weights = store.retrieve(
        "database security", 
        category_weights={"system_rule": 2.0}  # Does not specify the unregistered_custom_category
    )
    print(f"Search with missing category weights returns score: {res_custom_weights[0]['similarity']:.4f}")
    assert res_custom_weights[0]["similarity"] > 0.0, "Expected a valid score (fallback weight 1.0 should be applied)"
    
    # Clean store for next test
    store.clear()

        # Chaos Test 3: Zero & Negative Ranks
        print("\n--- Chaos Test 3: Division-by-Zero / Zero & Negative Ranks ---")
    
    # Feeding zero, negative, or floats to calculate_force
    force_zero = engine.calculate_retention_score(similarity=0.8, token_count=100, rank=0)
    force_neg = engine.calculate_retention_score(similarity=0.8, token_count=100, rank=-10)
    force_float = engine.calculate_retention_score(similarity=0.8, token_count=100, rank=1.5)
    
    print(f"Force calculated at rank 0:  {force_zero:.4f}")
    print(f"Force calculated at rank -10: {force_neg:.4f}")
    print(f"Force calculated at rank 1.5: {force_float:.4f}")
    
    assert force_zero > 0.0 and force_neg > 0.0, "Ranks below 1 should safely fallback to rank 1 properties"
    assert force_zero == force_neg, "Rank 0 and negative ranks should yield identical limits"

        # Chaos Test 4: Giant Payloads & Split bounds
        print("\n--- Chaos Test 4: Giant Payload Bounds ---")
    
    # Large word segment to stress tokenizer and Splitter
    giant_text = "word " * 20000
    split_chunks = TextSplitter.split_text(giant_text, chunk_size=200, chunk_overlap=50)
    print(f"Split giant text into {len(split_chunks)} chunks.")
    assert len(split_chunks) > 0
    
    # Index one giant chunk to verify tokenizer limits ( BGE model limit is 512 tokens )
    store.add_chunks([{"id": "giant_1", "text": giant_text, "category": "conversation"}])
    assert len(store.chunks) == 1
    print(f"Indexed giant chunk successfully. Cached token count: {store.chunks[0]['token_count']}")
    
    # Clean store for thread stress test
    store.clear()

        # Chaos Test 5: Concurrent Multithreaded Stress
        print("\n--- Chaos Test 5: Concurrent Multi-Threaded Stress Querying ---")
    
    # Fill database with 20 distinct records
    test_data = [
        {"id": f"stress_{i}", "text": f"This is context sentence index {i} containing details for stress testing.", "category": "memory"}
        for i in range(20)
    ]
    store.add_chunks(test_data)
    
    num_threads = 8
    num_queries_per_thread = 15
    queries = [
        "details for stress testing",
        "context sentence index",
        "sentence index 5",
        "non-existent match details"
    ]
    
    def run_queries(thread_idx):
        success_count = 0
        for i in range(num_queries_per_thread):
            q = queries[(thread_idx + i) % len(queries)]
            # Dynamic wiggles of pressure parameters to stress cache keys lock safety
            p = (thread_idx * 0.1) % 1.0
            res = store.retrieve(q, top_k=3, system_pressure=p)
            if len(res) > 0 or q == "non-existent match details":
                success_count += 1
        return success_count

    print(f"Spawning {num_threads} threads, executing {num_queries_per_thread} searches each concurrently...")
    
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_queries, i) for i in range(num_threads)]
        results = [f.result() for f in futures]
        
    t_elapsed = time.perf_counter() - t_start
    total_calls = num_threads * num_queries_per_thread
    print(f"Successfully processed {sum(results)}/{total_calls} queries.")
    print(f"Concurrently queried at: {total_calls / t_elapsed:.2f} queries/sec")
    
    assert sum(results) == total_calls, "All concurrent queries must resolve safely without deadlocks"
    print("Multi-threaded query stress test resolved with 100% lock safety.")

    # Clean up database
    store.clear()
    print("\n>> ALL CHAOS & BOUNDARY STRESS TESTS COMPLETED SUCCESSFULLY! <<\n")


if __name__ == "__main__":
    run_chaos_tests()
