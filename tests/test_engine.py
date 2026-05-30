"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from engine import DynamicContextReRanker


def run_validation_suite():
    print("=" * 65)
    print("     PROJECT EVAPOCONTEXT: DYNAMIC CONTEXT RE-RANKER TEST SUITE")
    print("=" * 65)
    
    # 1. Verify Pruning Threshold Scaling
    print("\n--- Test 1: Pruning Threshold Scaling ---")
    engine_default = DynamicContextReRanker()
    pressures = [0.0, 0.2, 0.5, 0.8, 1.0]
    expected_thresholds = [0.20, 0.222, 0.3375, 0.552, 0.75]
    
    for p, expected in zip(pressures, expected_thresholds):
        calc = engine_default.calculate_pruning_threshold(p)
        status = "PASS" if abs(calc - expected) < 1e-4 else "FAIL"
        print(f"Pressure: {p:.1f} | Expected Threshold: {expected:.4f} | Calculated: {calc:.4f} | [{status}]")
        assert abs(calc - expected) < 1e-4, f"Threshold mismatch for pressure {p}"
        
    # 2. Synthetic Chunk Setup
    synthetic_chunks = [
        {
            "id": "chunk_old_low_sim",
            "text": "This is some old, low-relevance background info.",
            "similarity": 0.4,
            "token_count": 120,
            "is_pinned": False
        },
        {
            "id": "chunk_old_high_sim",
            "text": "This is old but contains very high topical relevance.",
            "similarity": 0.9,
            "token_count": 150,
            "is_pinned": False
        },
        {
            "id": "chunk_pinned_system",
            "text": "CRITICAL: System instruction. You are a helpful AI assistant.",
            "similarity": 0.1,
            "token_count": 50,
            "is_pinned": True
        },
        {
            "id": "chunk_new_mid_sim",
            "text": "This is recent conversational context with average relevance.",
            "similarity": 0.6,
            "token_count": 80,
            "is_pinned": False
        },
        {
            "id": "chunk_new_high_sim",
            "text": "This is the latest user query context, extremely relevant.",
            "similarity": 0.95,
            "token_count": 100,
            "is_pinned": False
        }
    ]

    # 3. Test Configurable Decay Modes
    print("\n--- Test 2: Decay Mode Comparison ---")
    modes = ["inverse_square", "inverse", "sqrt", "log"]
    for mode in modes:
        engine = DynamicContextReRanker(decay_mode=mode)
        score = engine.calculate_retention_score(similarity=0.9, token_count=150, rank=4)
        print(f"Decay Mode: {mode:15} | Rank 4 Score: {score:.4f}")
        if mode != "inverse_square":
            legacy_score = DynamicContextReRanker(decay_mode="inverse_square").calculate_retention_score(similarity=0.9, token_count=150, rank=4)
            assert score > legacy_score, f"Gentler decay mode {mode} failed to produce a higher score than inverse_square"

    # 4. Pressure Independent retention score values
    print("\n--- Test 3: Pressure Independent Score ---")
    engine = DynamicContextReRanker()
    score_p0 = engine.calculate_retention_score(similarity=0.95, token_count=100, rank=1, system_pressure=0.0)
    score_p8 = engine.calculate_retention_score(similarity=0.95, token_count=100, rank=1, system_pressure=0.8)
    print(f"Score at P=0.0: {score_p0:.4f} | Score at P=0.8: {score_p8:.4f}")
    assert abs(score_p0 - score_p8) < 1e-5, "Pressure should not affect retention score calculation"

    # 5. Token Weight Influence Verification
    print("\n--- Test 4: Token Weight Influence ---")
    engine_w1 = DynamicContextReRanker(token_weight=1.0)
    engine_w3 = DynamicContextReRanker(token_weight=3.0)
    
    score_small_w1 = engine_w1.calculate_retention_score(similarity=0.5, token_count=50, rank=1)
    score_large_w1 = engine_w1.calculate_retention_score(similarity=0.5, token_count=500, rank=1)
    
    score_small_w3 = engine_w3.calculate_retention_score(similarity=0.5, token_count=50, rank=1)
    score_large_w3 = engine_w3.calculate_retention_score(similarity=0.5, token_count=500, rank=1)
    
    ratio_w1 = score_large_w1 / score_small_w1
    ratio_w3 = score_large_w3 / score_small_w3
    
    print(f"Weight 1.0 -> Small Chunks Score: {score_small_w1:.4f} | Large Chunk Score: {score_large_w1:.4f} | Ratio: {ratio_w1:.4f}")
    print(f"Weight 3.0 -> Small Chunks Score: {score_small_w3:.4f} | Large Chunk Score: {score_large_w3:.4f} | Ratio: {ratio_w3:.4f}")
    
    assert ratio_w3 > ratio_w1, "Token weight increase should amplify the token size influence on the retention score"

    # 6. Three-Tier Pinning System with Multiplier
    print("\n--- Test 5: Three-Tier Pinning System ---")
    engine = DynamicContextReRanker(soft_pin_multiplier=2.5)
    pinning_test_chunks = [
        {"id": "c1", "similarity": 0.5, "token_count": 100, "pinning_level": "critical", "rank": 1},
        {"id": "c2", "similarity": 0.5, "token_count": 100, "pinning_level": "soft", "rank": 1},
        {"id": "c3", "similarity": 0.5, "token_count": 100, "pinning_level": "none", "rank": 1}
    ]
    optimized = engine.optimize_context(pinning_test_chunks, system_pressure=0.5)
    
    for c in optimized:
        print(f"Chunk ID: {c['id']} | Pinning Level: {c.get('pinning_level', 'none')} | Retention Score: {c['retention_score']}")
        
    assert optimized[0]["retention_score"] == float("inf"), "Critical pinned must have infinite retention score"
    assert abs(optimized[1]["retention_score"] - 2.5 * optimized[2]["retention_score"]) < 1e-4, "Soft pinned score must be scaled by soft_pin_multiplier"

    # 7. Token Budget Enforcement & Overflow Warning
    print("\n--- Test 6: Token Budget Enforcement ---")
    budget_chunks = [
        {"id": "critical_sys", "similarity": 0.1, "token_count": 1000, "pinning_level": "critical"},
        {"id": "high_val_1", "similarity": 0.9, "token_count": 2000, "pinning_level": "none", "rank": 1},
        {"id": "high_val_2", "similarity": 0.85, "token_count": 3000, "pinning_level": "none", "rank": 2},
        {"id": "low_val", "similarity": 0.4, "token_count": 1500, "pinning_level": "none", "rank": 3}
    ]
    
    optimized_budget = engine_default.optimize_context(budget_chunks, system_pressure=0.0, token_budget=4000)
    kept_ids = [c["id"] for c in optimized_budget]
    print(f"With Budget 4000: Kept Chunks: {kept_ids} | Total tokens: {sum(c['token_count'] for c in optimized_budget)}")
    assert "critical_sys" in kept_ids
    assert "high_val_1" in kept_ids
    assert "high_val_2" not in kept_ids
    assert "low_val" not in kept_ids

    # Budget Warning trigger
    print("\n--- Test 6b: Critical Budget Warning Trigger ---")
    optimized_overflow = engine_default.optimize_context(budget_chunks, system_pressure=0.0, token_budget=500)
    assert len(optimized_overflow) == 1 and optimized_overflow[0]["id"] == "critical_sys"

    # 8. Constant Normalization Mode
    print("\n--- Test 7: Constant Normalization Mode ---")
    engine_const = DynamicContextReRanker(default_normalization="constant", decay_mode="inverse")
    score_small = engine_const.calculate_retention_score(similarity=0.75, token_count=10, rank=1)
    score_large = engine_const.calculate_retention_score(similarity=0.75, token_count=10000, rank=1)
    print(f"Constant Mode -> Small Chunk Score: {score_small:.4f} | Large Chunk Score: {score_large:.4f}")
    assert score_small == score_large == 0.75, "In constant mode, token counts must have no influence on retention score"

    # 9. Budget Sorting Modes (Score vs. Efficiency)
    print("\n--- Test 8: Budget Sorting Modes (Score vs. Efficiency) ---")
    efficiency_chunks = [
        {"id": "chunk_big_heavy", "similarity": 0.9, "token_count": 600, "pinning_level": "none", "rank": 1},
        {"id": "chunk_small_efficient", "similarity": 0.7, "token_count": 50, "pinning_level": "none", "rank": 1}
    ]
    
    opt_force = engine_default.optimize_context(efficiency_chunks, system_pressure=0.0, token_budget=600, budget_sorting_mode="force")
    ids_force = [c["id"] for c in opt_force]
    print(f"Sorting Mode 'force' kept: {ids_force}")
    assert "chunk_big_heavy" in ids_force
    assert "chunk_small_efficient" not in ids_force

    opt_eff = engine_default.optimize_context(efficiency_chunks, system_pressure=0.0, token_budget=600, budget_sorting_mode="efficiency")
    ids_eff = [c["id"] for c in opt_eff]
    print(f"Sorting Mode 'efficiency' kept: {ids_eff}")
    assert "chunk_small_efficient" in ids_eff
    assert "chunk_big_heavy" not in ids_eff

    # 10. Dense Vector Embedding Scoring
    print("\n--- Test 9: Dense Vector Embedding Scoring ---")
    query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    chunk_vector_1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    chunk_vector_2 = np.array([-0.4, -0.3, 0.2, 0.1], dtype=np.float32) 
    
    score_1 = engine_default.calculate_retention_score_vector(
        query_vector=query_vector,
        chunk_vector=chunk_vector_1,
        token_count=100,
        rank=1,
        system_pressure=0.2
    )
    
    score_2 = engine_default.calculate_retention_score_vector(
        query_vector=query_vector,
        chunk_vector=chunk_vector_2,
        token_count=100,
        rank=1,
        system_pressure=0.2
    )
    
    print(f"Score with exact match chunk: {score_1:.4f}")
    print(f"Score with orthogonal chunk:  {score_2:.4f}")
    assert score_1 > score_2, "Exact match chunk should have higher score than orthogonal chunk"
    
    print("\n>> ALL MATH VALIDATION TESTS PASSED SUCCESSFULLY! <<\n")


if __name__ == "__main__":
    run_validation_suite()
