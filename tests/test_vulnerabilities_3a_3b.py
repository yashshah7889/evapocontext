"""
Project EvapoContext: Vulnerabilities 3A & 3B Validation Test Suite
File: tests/test_vulnerabilities_3a_3b.py
"""

import sys
import os
import time
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from engine import DynamicContextReRanker
from retrieval import EmbeddingGenerator, HybridRetrievalStore
from evapocontext_connector import EvapoContextConnector


def test_adaptive_pruning_3a():
    print("\n==================================================")
    print("Test 3A: Relevance-Capped Adaptive Pruning Threshold")
    print("==================================================")

    # Initialize ReRanker with standard default configuration
    # Base = 0.20, Pressure Factor = 0.55, Ratio = 0.80
    reranker = DynamicContextReRanker(base_threshold=0.20, pressure_factor=0.55, threshold_ratio=0.80)

    # Case 1: Low Pressure, Low Relevance Match
    # Telemetry threshold should be 0.20 + 0.0 * 0.55 = 0.20
    # Threshold cap = max(0.20, 0.40 * 0.80) = 0.32
    # Output should be min(0.20, 0.32) = 0.20
    t1 = reranker.calculate_pruning_threshold(system_pressure=0.0, max_score=0.40)
    print(f"Case 1: Low Pressure (0.0), Low Score (0.40) -> Threshold: {t1:.4f} (Expected: 0.2000)")
    assert abs(t1 - 0.2000) < 1e-4, f"Expected 0.2000, got {t1}"

    # Case 2: High Pressure, High Relevance Match
    # Telemetry threshold should be 0.20 + 1.0^2 * 0.55 = 0.75
    # Threshold cap = max(0.20, 0.95 * 0.80) = 0.76
    # Output should be min(0.75, 0.76) = 0.75
    t2 = reranker.calculate_pruning_threshold(system_pressure=1.0, max_score=0.95)
    print(f"Case 2: High Pressure (1.0), High Score (0.95) -> Threshold: {t2:.4f} (Expected: 0.7500)")
    assert abs(t2 - 0.7500) < 1e-4, f"Expected 0.7500, got {t2}"

    # Case 3: High Pressure, Low Relevance Match (Collapse Prevention Case!)
    # Telemetry threshold should be 0.75
    # Threshold cap = max(0.20, 0.40 * 0.80) = 0.32
    # Output should be min(0.75, 0.32) = 0.32
    # Without 3A cap, this would be 0.75, causing ALL chunks to be evaporated!
    t3 = reranker.calculate_pruning_threshold(system_pressure=1.0, max_score=0.40)
    print(f"Case 3: High Pressure (1.0), Low Score (0.40) -> Threshold: {t3:.4f} (Expected: 0.3200)")
    assert abs(t3 - 0.3200) < 1e-4, f"Expected 0.3200, got {t3}"

    # Case 4: High Pressure, Zero Relevance (Junk Suppression Case!)
    # Telemetry threshold should be 0.75
    # Threshold cap = max(0.20, 0.0 * 0.80) = 0.20
    # Output should be min(0.75, 0.20) = 0.20
    t4 = reranker.calculate_pruning_threshold(system_pressure=1.0, max_score=0.0)
    print(f"Case 4: High Pressure (1.0), Zero Score (0.0) -> Threshold: {t4:.4f} (Expected: 0.2000)")
    assert abs(t4 - 0.2000) < 1e-4, f"Expected 0.2000, got {t4}"

    print(">>> 3A ADAPTIVE THRESHOLD TEST PASSED <<<")


def test_turn_decay_policies_3b():
    print("\n==================================================")
    print("Test 3B: Segment-Aware Stateful Turn Decay Policies")
    print("==================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
    embedder = EmbeddingGenerator(model_dir=model_dir)

    # Use in-memory temporary database for retrieval tests
    temp_db = os.path.join(model_dir, "temp_test_3b.db")
    if os.path.exists(temp_db):
        os.remove(temp_db)

    store = HybridRetrievalStore(embedding_generator=embedder, db_path=temp_db)

    try:
        # Index document chunks at Turn 1
        chunks = [
            {"id": "rag_doc", "text": "This is a RAG document with static specification parameters for database configurations.", "category": "memory", "turn_index": 1},
            {"id": "mcp_schema", "text": "This is an MCP tool schema configuration specification mapping user query intents.", "category": "tool_schema", "turn_index": 1},
            {"id": "conv_history", "text": "This is a chat conversation message discussing general coding topics.", "category": "conversation", "turn_index": 1},
            {"id": "tool_res", "text": "This is a temporary tool output returning execution logs of pytest runs.", "category": "tool_output", "turn_index": 1}
        ]

        store.add_chunks(chunks, turn_index=1)

        # Retrieve at Turn 10 (9 turns later)
        # We query for general keyword overlaps to match everything
        results = store.retrieve("configuration specification general topics execution logs", top_k=10, current_turn=10)

        # Map results by ID
        res_map = {r["id"]: r for r in results}

        # Case 1: RAG (memory) -> Turn Offset must be 0
        assert "rag_doc" in res_map
        offset_rag = res_map["rag_doc"]["turn_offset"]
        print(f"RAG Document: Category: {res_map['rag_doc']['category']:15} | Turn Offset: {offset_rag} (Expected: 0)")
        assert offset_rag == 0, f"RAG should not decay, got offset {offset_rag}"

        # Case 2: MCP (tool_schema) -> Turn Offset must be 0
        assert "mcp_schema" in res_map
        offset_mcp = res_map["mcp_schema"]["turn_offset"]
        print(f"MCP Schema:   Category: {res_map['mcp_schema']['category']:15} | Turn Offset: {offset_mcp} (Expected: 0)")
        assert offset_mcp == 0, f"MCP Schema should not decay, got offset {offset_mcp}"

        # Case 3: Conversation history -> Turn Offset must be 10 - 1 = 9
        assert "conv_history" in res_map
        offset_conv = res_map["conv_history"]["turn_offset"]
        print(f"Conversation: Category: {res_map['conv_history']['category']:15} | Turn Offset: {offset_conv} (Expected: 9)")
        assert offset_conv == 9, f"Conversation history should decay naturally, got offset {offset_conv}"

        # Case 4: Tool output -> Turn Offset must be (10 - 1) * 2 = 18 (Double Penalty)
        assert "tool_res" in res_map
        offset_tool = res_map["tool_res"]["turn_offset"]
        print(f"Tool Output:  Category: {res_map['tool_res']['category']:15} | Turn Offset: {offset_tool} (Expected: 18)")
        assert offset_tool == 18, f"Tool outputs should decay at double speed, got offset {offset_tool}"

        # Verify decay impact on retention scores using re-ranker
        reranker = DynamicContextReRanker(decay_mode="log")
        
        # We will compute the retention scores using optimize_context
        # P_sys = 0.0 (pruning threshold 0.20)
        optimized = reranker.optimize_context(results, system_pressure=0.0)
        opt_map = {o["id"]: o for o in optimized}

        # Verify retention scores ordered by decay policies:
        # rag_doc and mcp_schema should have minimal decay (factor = log2(0 + 1 + 1) = 1.0)
        # conv_history should have decay factor = log2(9 + 1) = 3.32
        # tool_res should have decay factor = log2(18 + 1) = 4.25
        print("\nCalculated Retention Scores (Similarity weight adjusted by Temporal Decay):")
        for oid, c in opt_map.items():
            print(f"  ID: {oid:15} | Sim: {c['similarity']:.4f} | Turn Offset: {c['turn_offset']} | Score: {c['retention_score']:.4f}")

        # Assert retention score of RAG/MCP is not decayed (factor = 1.0)
        # For rag_doc: score = weight / log2(0 + 1 + 1) = weight / 1.0 = weight
        # If similarity = 0.60 (say), score should match the raw similarity-weight product.
        assert opt_map["rag_doc"]["retention_score"] > opt_map["conv_history"]["retention_score"] * 2.0, "Decayed conversation score should be significantly lower than RAG document"

        print(">>> 3B STATEFUL TURN DECAY TEST PASSED <<<")

    finally:
        store.clear()
        if os.path.exists(temp_db):
            try:
                os.remove(temp_db)
            except Exception:
                pass


def test_rag_mcp_hybrid_integration():
    print("\n==================================================")
    print("Test Integration: RAG, MCP, and Hybrid Scenarios")
    print("==================================================")

    # Instantiate the EvapoContext Connector which wraps telemetry, DB, and reranker
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
    db_path = os.path.join(model_dir, "temp_integration_3a_3b.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    connector = EvapoContextConnector(model_dir=model_dir, db_path=db_path)

    try:
        # Turn 0: Load static context (RAG reference manuals and MCP tool schemas)
        # RAG Manual
        connector.add_document(
            doc_id="rag_security_manual",
            text="SECURITY STANDARD: All PostgreSQL connections must use SSL mode require. Port defaults to 5432.",
            category="memory"
        )
        # MCP Tool Schema
        connector.add_document(
            doc_id="mcp_database_executor",
            text="TOOL SCHEMA: execute_sql_query accepts query string arguments to select database fields.",
            category="tool_schema"
        )

        # Turn 1: User conversation starts
        connector.current_turn = 1
        connector.add_document(
            doc_id="user_chat_t1",
            text="User says: Let's setup our postgres database connections with security parameters.",
            category="conversation"
        )

        # Turn 2: LLM executes a tool (Hybrid RAG + MCP scenario) and tool output is logged
        connector.current_turn = 2
        connector.add_document(
            doc_id="tool_output_t2",
            text="Tool output: SSL active, connection established on port 5432.",
            category="tool_output"
        )

        # Turn 5: User continues chat after a few messages
        connector.current_turn = 5

        # We query evapocontext to optimize context for Turn 5 under low pressure
        response = connector.search_and_optimize(
            query="setup postgres ssl connection and execute query",
            top_k=10,
            token_budget=500
        )

        survived = {s["id"]: s for s in response["survived_chunks"]}
        print("Surviving Chunks at Turn 5:")
        for sid, s in survived.items():
            print(f"  ID: {sid:25} | Category: {s['category']:15} | Score: {s['retention_score']:.4f}")

        # Assert static elements survived and have high priority
        assert "rag_security_manual" in survived, "RAG reference manual should survive"
        assert "mcp_database_executor" in survived, "MCP tool schema should survive"
        assert "user_chat_t1" in survived, "Recent conversation context should survive"

        # Now simulate 30 conversation turns passing to see evaporation in action
        connector.current_turn = 35

        # Query again. The conversation from turn 1 and tool output from turn 2 should have decayed away,
        # but the RAG reference manuals and MCP schemas (turn offset = 0) must remain!
        response_t35 = connector.search_and_optimize(
            query="setup postgres ssl connection and execute query",
            top_k=10,
            token_budget=500
        )

        survived_t35 = {s["id"]: s for s in response_t35["survived_chunks"]}
        print("\nSurviving Chunks at Turn 35:")
        for sid, s in survived_t35.items():
            print(f"  ID: {sid:25} | Category: {s['category']:15} | Score: {s['retention_score']:.4f}")

        # Assertions
        assert "rag_security_manual" in survived_t35, "RAG Manual must NOT evaporate"
        assert "mcp_database_executor" in survived_t35, "MCP Schema must NOT evaporate"
        assert "user_chat_t1" not in survived_t35, "Old conversation history from turn 1 should have evaporated by turn 35!"
        assert "tool_output_t2" not in survived_t35, "Old tool output from turn 2 should have evaporated by turn 35!"

        print(">>> INTEGRATION SCENARIOS TEST PASSED <<<")

    finally:
        connector.clear_database()
        connector.close()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


if __name__ == "__main__":
    test_adaptive_pruning_3a()
    test_turn_decay_policies_3b()
    test_rag_mcp_hybrid_integration()
    print("\n==================================================")
    print("ALL VULNERABILITIES 3A & 3B VALIDATION TESTS PASSED!")
    print("==================================================")
