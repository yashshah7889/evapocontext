"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import os
import logging
from typing import List, Dict, Any, Optional

# Import core EvapoContext modules
from telemetry import HardwareTelemetryMonitor
from engine import DynamicContextReRanker
from retrieval import EmbeddingGenerator, HybridRetrievalStore, DEFAULT_CATEGORY_WEIGHTS

logger = logging.getLogger("EvapoContextConnector")


class EvapoContextConnector:
    """
    Unified manager that coordinates telemetry monitoring, hybrid vector searches,
    and adaptive context re-ranking.
    """

    def __init__(self, model_dir: str = "./model", db_path: Optional[str] = None):
        """
        Initializes EvapoContext components and starts the hardware telemetry background monitor.
        
        Args:
            model_dir: Path to directory containing tokenizer.json and onnx/model.onnx.
            db_path: Path to SQLite database file. If None, defaults to evapocontext.db in model_dir.
        """
        self.model_dir = os.path.abspath(model_dir)
        
        if db_path is None:
            self.db_path = os.path.join(self.model_dir, "evapocontext.db")
        else:
            self.db_path = os.path.abspath(db_path)

        # 1. Start Telemetry Sensor Loop
        self.telemetry = HardwareTelemetryMonitor()
        self.telemetry.start()

        # 2. Hydrate Embedding Generator & Hybrid Database Store
        self.embedder = EmbeddingGenerator(model_dir=self.model_dir)
        self.store = HybridRetrievalStore(embedding_generator=self.embedder, db_path=self.db_path)

        # 3. Setup Dynamic Context Re-Ranker
        self.engine = DynamicContextReRanker()
        
        # Session state turn counter for turn-based chronological decay
        self.current_turn = 0
        
        logger.info(f"EvapoContext Connector loaded successfully. DB Path: {self.db_path}")

    def add_document(
        self, 
        doc_id: str, 
        text: str, 
        category: str = "conversation", 
        is_pinned: bool = False
    ):
        """
        Indexes a single text segment into the database.
        
        Args:
            doc_id: Unique string identifier for the segment.
            text: Raw string content.
            category: Category (system_rule, tool_schema, memory, conversation).
            is_pinned: If True, locks the segment inside primary memory (critical pinning).
        """
        chunk = {
            "id": doc_id,
            "text": text,
            "category": category,
            "is_pinned": is_pinned,
            "pinning_level": "critical" if is_pinned else "none",
            "turn_index": self.current_turn
        }
        self.store.add_chunks([chunk], turn_index=self.current_turn)
        logger.info(f"Document '{doc_id}' indexed successfully under category '{category}'.")

    def add_documents_batch(self, chunks: List[Dict[str, Any]]):
        """
        Indexes multiple text segments in parallel using batch vector calculations.
        
        Args:
            chunks: A list of dictionaries. Each dictionary must contain a 'text' key
                    and optionally 'id', 'category', 'is_pinned', 'pinning_level', and 'metadata'.
        """
        processed_chunks = []
        for c in chunks:
            is_pinned = c.get("is_pinned", False)
            processed_chunks.append({
                "id": c.get("id"),
                "text": c["text"],
                "category": c.get("category", "conversation"),
                "is_pinned": is_pinned,
                "pinning_level": c.get("pinning_level", "critical" if is_pinned else "none"),
                "metadata": c.get("metadata", {}),
                "turn_index": c.get("turn_index", self.current_turn)
            })
        self.store.add_chunks(processed_chunks, turn_index=self.current_turn)
        logger.info(f"Indexed batch of {len(processed_chunks)} documents successfully.")

    def search_and_optimize(
        self, 
        query: str, 
        top_k: int = 15, 
        token_budget: Optional[int] = None,
        category_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves matching segments, measures real-time hardware stress, and filters
        context to return a compressed, highly focused prompt payload.
        
        Args:
            query: The user prompt or intent.
            top_k: Maximum candidates to retrieve before scoring evaluation.
            token_budget: Optional strict token limit to enforce for the output context.
            category_weights: Optional custom boosts for categories.
            
        Returns:
            A dictionary containing:
              - 'optimized_text': The final concatenated prompt text ready for the LLM.
              - 'survived_chunks': Detailed records of chunks that passed eviction boundaries.
              - 'telemetry': Memory and CPU utilization diagnostic details.
        """
        self.current_turn += 1
        pressure = self.telemetry.get_pressure()

        candidates = self.store.retrieve(
            query=query,
            top_k=top_k,
            category_weights=category_weights,
            system_pressure=pressure,
            current_turn=self.current_turn
        )

        for idx, c in enumerate(candidates):
            c["rank"] = idx + 1

        optimized_chunks = self.engine.optimize_context(
            chunks=candidates,
            system_pressure=pressure,
            token_budget=token_budget
        )

        optimized_chunks.sort(key=lambda x: x.get("rank_assigned", 0), reverse=True)

        return {
            "optimized_text": "\n\n".join(c["text"] for c in optimized_chunks),
            "survived_chunks": [
                {
                    "id": c["id"],
                    "category": c["category"],
                    "token_count": c["token_count"],
                    "is_pinned": c["is_pinned"],
                    "retention_score": c["retention_score"]
                }
                for c in optimized_chunks
            ],
            "telemetry": {
                "system_pressure": round(pressure, 4),
                "original_candidates": len(candidates),
                "optimized_candidates": len(optimized_chunks),
                "token_budget": token_budget
            }
        }

    def clear_database(self):
        """Flushes the database table and resets index caches."""
        self.store.clear()
        logger.info("Context database reset complete.")

    def close(self):
        """Clean up resources and stops background telemetry monitoring thread."""
        self.telemetry.stop()
        logger.info("EvapoContext background telemetry listener shut down.")


# QUICK-START USAGE EXAMPLE
if __name__ == "__main__":
    import time
    
    # Configure logging to console for testing
    logging.basicConfig(level=logging.INFO)

    print("--- Starting EvapoContext Connector Integration Demo ---")
    
    # 1. Initialize EvapoContext (model path points relative to src folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model")
    db_path = os.path.join(model_path, "connector_demo.db")
    
    # Initialize the connector wrapper
    connector = EvapoContextConnector(model_dir=model_path, db_path=db_path)

    try:
        # 2. Add some test data
        connector.add_document(
            doc_id="sec_rules", 
            text="CRITICAL REQUIREMENT: Always verify database SSL is active.", 
            category="system_rule", 
            is_pinned=True
        )
        
        connector.add_documents_batch([
            {"id": "doc_devops", "text": "Deploy App using Docker containers on AWS ECS.", "category": "tool_schema"},
            {"id": "doc_chitchat", "text": "Hey assistant, it is a beautiful morning, let's work on coding.", "category": "conversation"}
        ])

        # 3. Simulate Query retrieval & context optimization
        time.sleep(1.0) # Wait a second for telemetry loops
        
        print("\nSearching and Evaporating Context...")
        response = connector.search_and_optimize(
            query="FastAPI database connection docker security settings",
            top_k=5,
            token_budget=500
        )

        print("\n[Optimized Context Result for LLM]")
        print("-" * 50)
        print(response["optimized_text"])
        print("-" * 50)
        print("Diagnostic Telemetry:", response["telemetry"])

    finally:
        # 4. Clean up database and close telemetry thread
        connector.clear_database()
        if os.path.exists(db_path):
            os.remove(db_path)
        connector.close()
        print("\n--- Demo finished. Resources released. ---")
