"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retrieval import EmbeddingGenerator, BM25Index, HybridRetrievalStore, TextSplitter
from engine import GravitationalScoringEngine


def run_retrieval_tests():
    print("=" * 75)
    print("     PROJECT EVAPOCONTEXT: LOCALIZED RETRIEVAL PIPELINE TEST SUITE")
    print("=" * 75)

    # 1. Test Embedding Generator
    print("\n--- Test 1: Embedding Generator ---")
    # Point model dir to src/model/ relative to tests location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
    
    embedder = EmbeddingGenerator(model_dir=model_dir)
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "System authentication logs are stored in /var/log/auth.log.",
        "Model Context Protocol utilizes JSON-RPC 2.0 over standard I/O."
    ]
    
    embeddings = embedder.generate_embeddings(sentences, is_query=False)
    print(f"Generated embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (3, 384), f"Expected shape (3, 384), got {embeddings.shape}"
    
    for idx, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        print(f"Sentence {idx} embedding L2 Norm: {norm:.6f}")
        assert abs(norm - 1.0) < 1e-5, f"Norm is {norm}, expected 1.0"

    # 2. Test BM25 Indexing
    print("\n--- Test 2: BM25 Indexing (with Stopword Filtering) ---")
    bm25 = BM25Index()
    
    documents = [
        "Standard JSON-RPC 2.0 servers listen on standard input and output streams.",
        "Authentication logs track failed SSH connection attempts in Unix systems.",
        "Context routers reduce LLM token overhead during long conversations.",
        "Authentication endpoints require signed JSON Web Tokens for validation."
    ]
    
    for i, doc in enumerate(documents):
        bm25.add_document(doc_id=f"doc_{i}", text=doc)
        
    print(f"Indexed documents count: {bm25.doc_count}")
    assert bm25.doc_count == 4, f"Expected 4 documents, got {bm25.doc_count}"
    
    results = bm25.search("authentication logs", top_k=2)
    print("Search results for 'authentication logs':")
    for doc_idx, score in results:
        print(f"  Doc Index: {doc_idx} | Text: '{documents[doc_idx]}' | BM25 Score: {score:.4f}")
        
    assert results[0][0] == 1, "Expected doc_1 to be the highest scoring match for 'authentication logs'"
    assert results[0][1] > 0.0, "Expected positive score for matching keywords"

    # Verify that stopword filtering excludes words like "and"
    tokens = bm25.tokenize("authentication and logs")
    print(f"Tokenized query 'authentication and logs': {tokens}")
    assert "and" not in tokens, "Stopwords should be filtered out from tokenized output"
    assert "authentication" in tokens and "logs" in tokens

    # 3. Test Hybrid Retrieval Store
    print("\n--- Test 3: Hybrid Retrieval Store (Two-Tiered Recall) ---")
    store = HybridRetrievalStore(embedding_generator=embedder)
    
    test_chunks = [
        {
            "id": "mcp_spec",
            "text": "The Model Context Protocol (MCP) establishes a JSON-RPC 2.0 communication layer between client interfaces and server processes.",
            "is_pinned": True,
            "pinning_level": "critical",
            "category": "system_rule",
            "metadata": {"source": "mcp_master_spec"}
        },
        {
            "id": "auth_policy",
            "text": "All user sessions expire after 15 minutes of inactivity. Access keys must be rotated every 30 days.",
            "is_pinned": False,
            "category": "system_rule",
            "metadata": {"source": "security_guide"}
        },
        {
            "id": "rel_gravity",
            "text": "Relativistic gravitational scoring calculates retention force pull based on chronological rank decay and semantic mass.",
            "is_pinned": False,
            "category": "memory",
            "metadata": {"source": "evapocontext_docs"}
        },
        {
            "id": "lazy_loading",
            "text": "Lazy-loading tool registry proxy only hydrates full JSON schemas when a specific tool identifier is explicitly called.",
            "is_pinned": False,
            "category": "tool_schema",
            "metadata": {"source": "mcp_study_guide"}
        },
        {
            "id": "hardware_telemetry",
            "text": "Hardware telemetry sensor loop collects virtual memory percentage, swap spaces, and CPU load averages every 2 seconds.",
            "is_pinned": False,
            "category": "conversation",
            "metadata": {"source": "telemetry_docs"}
        }
    ]
    
    store.add_chunks(test_chunks)
    
    print(f"Store chunks count: {len(store.chunks)}")
    assert len(store.chunks) == 5, "Expected 5 chunks in store"
    for chunk in store.chunks:
        print(f"  Chunk ID: {chunk['id']:20} | Category: {chunk['category']:12} | Tokens: {chunk['token_count']}")
        assert chunk["token_count"] > 10, "Expected realistic cached token counts"
        assert "embedding" in chunk, "Expected cached dense embedding array"
        
    query = "how does lazy loading schemas work?"
    retrieved = store.retrieve(query, top_k=3, bm25_candidates=5, hybrid_weight=0.3)
    
    print(f"\nRetrieved top 3 results for query: '{query}'")
    for idx, c in enumerate(retrieved):
        print(
            f"  Rank {idx+1}: ID={c['id']:18} | Rank={c['retrieval_rank']} | "
            f"Score={c['similarity']:.4f} | CosSim={c['cosine_similarity']:.4f} | BM25={c['bm25_score']:.4f}"
        )
        assert "retrieval_rank" in c
        assert "bm25_score" in c
        assert "cosine_similarity" in c
        assert "similarity" in c
        
    assert retrieved[0]["id"] == "lazy_loading", "Expected 'lazy_loading' chunk to rank first for lazy loading query"

    # 4. Test Category Boosting
    print("\n--- Test 4: Category-based Metadata Boosting ---")
    boost_test_chunks = [
        {
            "id": "rule_logging",
            "text": "Verify system logs periodically for unauthorized root logins.",
            "category": "system_rule"
        },
        {
            "id": "conv_logging",
            "text": "Verify system logs periodically for unauthorized root logins.",
            "category": "conversation"
        }
    ]
    store.add_chunks(boost_test_chunks)
    
    boost_results = store.retrieve("Verify system logs", top_k=2)
    print("Default Boosting weights (system_rule=1.5, conversation=1.0):")
    for r in boost_results:
        print(f"  ID: {r['id']:15} | Category: {r['category']:15} | Score: {r['similarity']:.4f} | CosSim: {r['cosine_similarity']:.4f}")
        
    assert boost_results[0]["id"] == "rule_logging", "System rule category should outrank conversation category due to weight boost"
    assert boost_results[0]["similarity"] > boost_results[1]["similarity"], "Boosted score should be strictly greater"

    custom_weights = {"system_rule": 0.5, "conversation": 2.0}
    custom_results = store.retrieve("Verify system logs", top_k=2, category_weights=custom_weights)
    print("\nCustom Boosting weights (system_rule=0.5, conversation=2.0):")
    for r in custom_results:
        print(f"  ID: {r['id']:15} | Category: {r['category']:15} | Score: {r['similarity']:.4f} | CosSim: {r['cosine_similarity']:.4f}")
        
    assert custom_results[0]["id"] == "conv_logging", "Conversation category should now outrank system rule due to custom weights"

    # 5. Test SQLite Persistence
    print("\n--- Test 5: SQLite Database Persistence ---")
    test_db_path = os.path.join(model_dir, "test_evapocontext.db")
    
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    db_store = HybridRetrievalStore(embedding_generator=embedder, db_path=test_db_path)
    print(f"Created persistent store at: {test_db_path}")
    
    db_store.add_chunks([
        {"id": "persist_1", "text": "This text is saved to disk inside a local SQLite table.", "category": "memory"},
        {"id": "persist_2", "text": "This is a rule cached inside the database.", "category": "system_rule"}
    ])
    
    del db_store
    
    print("Reloading store from SQLite database...")
    restored_store = HybridRetrievalStore(embedding_generator=embedder, db_path=test_db_path)
    
    print(f"Restored store chunks count: {len(restored_store.chunks)}")
    assert len(restored_store.chunks) == 2, f"Expected 2 chunks to be loaded, got {len(restored_store.chunks)}"
    
    c1 = restored_store.chunks[0]
    c2 = restored_store.chunks[1]
    print(f"  Loaded Chunk 1: ID={c1['id']}, Tokens={c1['token_count']}, Category={c1['category']}")
    print(f"  Loaded Chunk 2: ID={c2['id']}, Tokens={c2['token_count']}, Category={c2['category']}")
    
    assert c1["id"] in ["persist_1", "persist_2"]
    assert c2["id"] in ["persist_1", "persist_2"]
    assert c1["token_count"] > 0
    assert c1["embedding"].shape == (384,)
    
    restored_results = restored_store.retrieve("saved to disk", top_k=2)
    print("Search results on restored store:")
    for r in restored_results:
        print(f"  ID: {r['id']:12} | Rank: {r['retrieval_rank']} | Score: {r['similarity']:.4f}")
        
    assert restored_results[0]["id"] == "persist_1"
    
    try:
        os.remove(test_db_path)
        print("Cleaned up temporary test database file.")
    except Exception as e:
        print(f"Could not clean up file {test_db_path}: {e}")

    # 6. Test Text Splitter Chunking
    print("\n--- Test 6: Text Splitter ---")
    long_document = (
        "The Model Context Protocol connects client hosts to servers. "
        "It provides standard Resources, Tools, and Prompts. "
        "This decouples the system integrations from the central LLM engine. "
        "We can deploy it locally over standard output and input streams."
    )
    doc_chunks = TextSplitter.split_text(long_document, chunk_size=10, chunk_overlap=2)
    print(f"Document split into {len(doc_chunks)} chunks:")
    for idx, segment in enumerate(doc_chunks):
        print(f"  Chunk {idx}: '{segment}'")
    
    assert len(doc_chunks) > 1
    assert "Model" in doc_chunks[0]
    assert "streams" in doc_chunks[-1]

    # 7. Test Embedding Generator Bypass (Deduplication)
    print("\n--- Test 7: Embedding Generator Bypass (Deduplication) ---")
    dedup_store = HybridRetrievalStore(embedding_generator=embedder)
    
    t_start = time.perf_counter()
    dedup_store.add_chunks([{"id": "c_orig", "text": "This is a highly specific sentence about deduplication math.", "category": "memory"}])
    t_orig = time.perf_counter() - t_start
    print(f"Time to insert & embed original chunk: {t_orig * 1000:.2f} ms")
    
    t_start = time.perf_counter()
    dedup_store.add_chunks([{"id": "c_dup", "text": "This is a highly specific sentence about deduplication math.", "category": "conversation"}])
    t_dup = time.perf_counter() - t_start
    print(f"Time to insert duplicate chunk (should bypass ONNX model): {t_dup * 1000:.2f} ms")
    
    print(f"Speedup ratio: {t_orig / (t_dup + 1e-9):.2f}x")
    assert len(dedup_store.chunks) == 2, "Expected both chunks to exist"
    assert np.array_equal(dedup_store.chunks[0]["embedding"], dedup_store.chunks[1]["embedding"]), "Both embeddings should be mathematically identical"

    # 8. Test Integration with GravitationalScoringEngine
    print("\n--- Test 8: Integration with GravitationalScoringEngine ---")
    engine = GravitationalScoringEngine(base_threshold=0.20, pressure_factor=0.55)
    
    search_query = "scoring engine and gravitational decay"
    retrieved_for_scoring = store.retrieve(search_query, top_k=5, bm25_candidates=5)
    
    print("Retrieved chunks for scoring evaluation:")
    for chunk in retrieved_for_scoring:
        print(f"  ID: {chunk['id']:20} | Similarity Score: {chunk['similarity']:.4f} | Rank: {chunk['retrieval_rank']}")

    print("\n[Case A: Low System Pressure (P_sys = 0.0)]")
    optimized_low = engine.optimize_context(retrieved_for_scoring, system_pressure=0.0)
    low_ids = [c["id"] for c in optimized_low]
    print(f"  Surviving chunks: {low_ids}")
    
    assert "mcp_spec" in low_ids, "Critical pinned chunk must survive"
    assert "rel_gravity" in low_ids, "High-relevance chunk should survive"
    
    print("\n[Case B: Critical System Pressure (P_sys = 0.9)]")
    optimized_high = engine.optimize_context(retrieved_for_scoring, system_pressure=0.9)
    high_ids = [c["id"] for c in optimized_high]
    print(f"  Surviving chunks: {high_ids}")
    
    assert "mcp_spec" in high_ids, "Critical pinned chunk must survive under critical pressure"
    assert len(high_ids) <= len(low_ids), "Evaporation should evict more chunks under higher pressure"

    # 9. Test Exact Query Caching
    print("\n--- Test 9: Exact Query Caching ---")
    
    store.retrieve("warmup query search", top_k=2)
    
    noise_chunks = [
        {
            "id": f"noise_{i}",
            "text": f"This is a dummy noise paragraph number {i} to populate the retrieval index database and increase the search search space.",
            "category": "conversation"
        }
        for i in range(50)
    ]
    store.add_chunks(noise_chunks)
    
    query_exact = "lazy loading schema configuration"
    
    t_first = float("inf")
    res_1 = None
    for _ in range(5):
        store._invalidate_caches()
        t_start = time.perf_counter()
        res_1 = store.retrieve(query_exact, top_k=2)
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed < t_first:
            t_first = t_elapsed
            
    print(f"Time for first query search: {t_first * 1000:.3f} ms")
    
    t_second = float("inf")
    res_2 = None
    for _ in range(5):
        t_start = time.perf_counter()
        res_2 = store.retrieve(query_exact, top_k=2)
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed < t_second:
            t_second = t_elapsed
            
    print(f"Time for identical query search (exact cache hit): {t_second * 1000:.3f} ms")
    print(f"Exact cache speedup ratio: {t_first / (t_second + 1e-9):.2f}x")
    
    assert [c["id"] for c in res_1] == [c["id"] for c in res_2]
    assert t_second < 0.001 or t_second < t_first / 10.0, "Exact cache hit must be significantly faster"

    # 10. Test Semantic Query Caching
    print("\n--- Test 10: Semantic Query Caching ---")
    # This query is semantically 98% similar to "lazy loading schema configuration"
    query_semantic = "lazy load schemas configuration"
    
    # Measure semantic cache hit (take minimum of multiple runs, clearing exact cache each time)
    t_sem = float("inf")
    res_semantic = None
    for _ in range(5):
        with store.lock:
            store.exact_query_cache.clear()
        t_start = time.perf_counter()
        res_semantic = store.retrieve(query_semantic, top_k=2)
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed < t_sem:
            t_sem = t_elapsed
            
    print(f"Time for similar query search (semantic cache hit): {t_sem * 1000:.3f} ms")
    print(f"Semantic cache speedup ratio (relative to first query): {t_first / (t_sem + 1e-9):.2f}x")
    
    assert [c["id"] for c in res_semantic] == [c["id"] for c in res_1]
    assert t_sem <= t_first, "Semantic cache hit must be faster or equal to original model execution + hybrid search"

    # 11. Test Adaptive Hardware-Aware Retrieval Gating (AHARG)
    print("\n--- Test 11: Adaptive Hardware-Aware Retrieval Gating (AHARG) ---")
    
    # Measure low pressure search (runs ONNX model)
    res_low = store.retrieve("lazy loading schemas", system_pressure=0.2, top_k=2)
    assert any("cosine_similarity" in c and c["cosine_similarity"] > 0.0 for c in res_low), "Expected cosine similarity calculations to be active at low pressure"
    
    # Measure critical pressure search (Vector Bypass active)
    t_start = time.perf_counter()
    res_high = store.retrieve("lazy loading schemas", system_pressure=0.95, top_k=2)
    t_bypass = time.perf_counter() - t_start
    print(f"Time for high pressure retrieval (Vector Bypass active): {t_bypass * 1000:.3f} ms")
    print(f"Vector Bypass speedup vs standard search: {t_first / (t_bypass + 1e-9):.2f}x")
    
    assert all(c.get("vector_bypass_active") is True for c in res_high), "Expected vector_bypass_active to be True at critical pressure"
    assert all(c["cosine_similarity"] == 0.0 for c in res_high), "Expected cosine_similarity calculations to be bypassed (set to 0.0)"
    assert t_bypass < 0.001 or t_bypass < t_first / 10.0, "Vector Bypass retrieval must be significantly faster than standard model search"
    print("Vector Bypass verified successfully!")

    print("\n>> ALL RETRIEVAL PIPELINE TESTS PASSED SUCCESSFULLY! <<\n")


if __name__ == "__main__":
    run_retrieval_tests()
