"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional
import jsonschema

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telemetry import HardwareTelemetryMonitor
from engine import DynamicContextReRanker
from retrieval import EmbeddingGenerator, HybridRetrievalStore, TextSplitter, DEFAULT_CATEGORY_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("EvapoContextDaemon")

# ----------------- FULL SCHEMAS (CACHED FOR JIT HYDRATION) -----------------
FULL_TOOL_SCHEMAS = {
    "index_context": {
        "name": "index_context",
        "description": "Indexes new text chunks into the persistent hybrid retrieval store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "description": "List of text chunks to index.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The raw text content."},
                            "id": {"type": "string", "description": "Optional unique identifier."},
                            "category": {"type": "string", "description": "Category (system_rule, tool_schema, memory, conversation).", "default": "conversation"},
                            "is_pinned": {"type": "boolean", "description": "If True, pins chunk in attention layer.", "default": False},
                            "pinning_level": {"type": "string", "description": "Pinning level (critical, soft, none).", "default": "none"},
                            "metadata": {"type": "object", "description": "Optional metadata key-value dictionary."}
                        },
                        "required": ["text"]
                    }
                }
            },
            "required": ["chunks"]
        }
    },
    "retrieve_optimized_context": {
        "name": "retrieve_optimized_context",
        "description": "Retrieves semantically relevant chunks, scores them using dynamic time-decay and hardware-stress scaling, and returns the optimized chronological context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user search query or intent."},
                "top_k": {"type": "integer", "description": "Max chunks to retrieve before scoring.", "default": 10},
                "hybrid_weight": {"type": "number", "description": "Weight of BM25 vs cosine similarity (0.0 to 1.0).", "default": 0.3},
                "token_budget": {"type": "integer", "description": "Optional strict token budget to enforce."}
            },
            "required": ["query"]
        }
    },
    "get_system_telemetry": {
        "name": "get_system_telemetry",
        "description": "Retrieves the current operating system telemetry report and moving average pressure index.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    "clear_all_context": {
        "name": "clear_all_context",
        "description": "Deletes all indexed text chunks and clears the persistent database.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
}


class EvapoContextDaemon:
    """
    Handles JSON-RPC 2.0 stdio communications, lazy schema loading,
    and maps execution calls to the underlying engine and database store.
    """

    def __init__(self, telemetry_monitor: HardwareTelemetryMonitor):
        self.telemetry = telemetry_monitor
        self.engine = DynamicContextReRanker()
        self.current_turn = 0
        
        # Configure model paths relative to server file location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "model")
        db_path = os.path.join(model_dir, "evapocontext.db")
        
        self.embedder = EmbeddingGenerator(model_dir=model_dir)
        self.store = HybridRetrievalStore(embedding_generator=self.embedder, db_path=db_path)
        
        logger.info(f"EvapoContextDaemon initialized (SQL Store: {db_path})")

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Routes incoming JSON-RPC requests to appropriate protocol handlers.
        """
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")

        logger.info(f"Processing request method: {method} (ID: {req_id})")

        # Route methods
        if method == "initialize":
            return self._handle_initialize(req_id, params)
        elif method in ("initialized", "notifications/initialized"):
            # Notifications do not receive responses
            logger.info("Connection handshake completed.")
            return None
        elif method == "ping":
            return {"jsonrpc": "2.0", "result": {}, "id": req_id}
        elif method == "tools/list":
            return self._handle_tools_list(req_id)
        elif method == "tools/call":
            return self._handle_tools_call(req_id, params)
        elif method == "resources/list":
            return self._handle_resources_list(req_id)
        elif method == "resources/read":
            return self._handle_resources_read(req_id, params)
        else:
            logger.warning(f"Unsupported method called: {method}")
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    },
                    "id": req_id
                }
            return None

    def _handle_initialize(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Responds to initialization handshakes and registers protocol capabilities.
        """
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "evapocontext",
                    "version": "1.0.0"
                }
            }
        }

    def _handle_tools_list(self, req_id: Any) -> Dict[str, Any]:
        """
        Returns skeletal schemas to the client (Lazy-Loaded Tool Schema Compression).
        Properties are kept completely empty to save context space.
        """
        logger.info("Executing tools/list (compressing output schemas)...")
        skeletal_tools = []
        
        for name, spec in FULL_TOOL_SCHEMAS.items():
            # Truncate description to 30 characters
            short_desc = f"{spec['description'][:30]}... [Lazy-Hydration Active]"
            
            skeletal_tool = {
                "name": name,
                "description": short_desc,
                "inputSchema": {
                    "type": "object",
                    "properties": {}  # Bypasses flat schema prompt tax!
                }
            }
            skeletal_tools.append(skeletal_tool)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
            "tools": skeletal_tools
            }
        }

    def _handle_tools_call(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercepts tool executions, performs JIT schema validation, and executes targets.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"Executing tools/call for: {tool_name}")

        if tool_name not in FULL_TOOL_SCHEMAS:
            return self._build_tool_error(req_id, f"Tool '{tool_name}' not found.")

        # JIT Schema Hydration & Validation
        full_schema = FULL_TOOL_SCHEMAS[tool_name]
        try:
            jsonschema.validate(instance=arguments, schema=full_schema["inputSchema"])
        except jsonschema.ValidationError as ve:
            logger.warning(f"JIT schema validation failed for '{tool_name}': {ve.message}")
            return self._build_tool_error(req_id, f"Schema validation error: {ve.message}")

        # Execute Handlers
        try:
            if tool_name == "index_context":
                return self._execute_index_context(req_id, arguments)
            elif tool_name == "retrieve_optimized_context":
                return self._execute_retrieve_optimized_context(req_id, arguments)
            elif tool_name == "get_system_telemetry":
                return self._execute_get_system_telemetry(req_id)
            elif tool_name == "clear_all_context":
                return self._execute_clear_all_context(req_id)
        except Exception as e:
            logger.exception(f"Handler failure executing '{tool_name}': {e}")
            return self._build_tool_error(req_id, f"Internal execution failure: {str(e)}")

    def _build_tool_error(self, req_id: Any, message: str) -> Dict[str, Any]:
        """
        Formats a structured JSON-RPC error response for tool calls.
        """
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": message
                    }
                ],
                "isError": True
            }
        }

    def _execute_index_context(self, req_id: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        chunks = args["chunks"]
        
        # Format chunks appropriately before indexing
        processed = []
        for c in chunks:
            processed.append({
                "id": c.get("id"),
                "text": c["text"],
                "category": c.get("category", "conversation"),
                "is_pinned": c.get("is_pinned", False),
                "pinning_level": c.get("pinning_level", "critical" if c.get("is_pinned") else "none"),
                "metadata": c.get("metadata", {}),
                "turn_index": c.get("turn_index", self.current_turn)
            })
            
        self.store.add_chunks(processed, turn_index=self.current_turn)
        response_text = f"Successfully indexed {len(processed)} chunks in persistent database."
        
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
            "content": [{"type": "text", "text": response_text}]
            }
        }

    def _execute_retrieve_optimized_context(self, req_id: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args["query"]
        top_k = args.get("top_k", 10)
        hybrid_weight = args.get("hybrid_weight", 0.3)
        token_budget = args.get("token_budget")

        # Increment current turn session state
        self.current_turn += 1

        # 1. Extract current real-time hardware pressure
        pressure = self.telemetry.get_pressure()

        # 2. Retrieve raw candidate segments with dynamic hardware-aware scaling
        candidates = self.store.retrieve(
            query,
            top_k=top_k,
            hybrid_weight=hybrid_weight,
            system_pressure=pressure,
            current_turn=self.current_turn
        )

        # 3. Score and prune chunks using the re-ranking engine
        optimized_chunks = self.engine.optimize_context(
            chunks=candidates,
            system_pressure=pressure,
            token_budget=token_budget
        )

        # 4. CHRONOLOGICAL RE-SORTING REQUIREMENT:
        # optimize_context sorts descending by score. We must re-sort them oldest-to-newest
        # (chronological) before injecting them. Older messages have higher rank indices.
        optimized_chunks.sort(key=lambda x: x.get("rank_assigned", 0), reverse=True)

        # 5. Build structured benchmark output payload
        result_payload = {
            "optimized_text": "\n\n".join(c["text"] for c in optimized_chunks),
            "chunks": [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "category": c["category"],
                    "is_pinned": c["is_pinned"],
                    "pinning_level": c["pinning_level"],
                    "token_count": c["token_count"],
                    "retention_score": c["retention_score"],
                    "bm25_score": c["bm25_score"],
                    "cosine_similarity": c["cosine_similarity"],
                    "similarity": c["similarity"],
                    "retrieval_rank": c["retrieval_rank"],
                    "rank_assigned": c.get("rank_assigned")
                }
                for c in optimized_chunks
            ],
            "telemetry": {
                "system_pressure": pressure,
                "token_budget": token_budget,
                "original_candidates": len(candidates),
                "optimized_candidates": len(optimized_chunks)
            }
        }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
            "content": [{"type": "text", "text": json.dumps(result_payload, indent=2)}]
            }
        }

    def _execute_get_system_telemetry(self, req_id: Any) -> Dict[str, Any]:
        report = self.telemetry.get_full_report()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
            "content": [{"type": "text", "text": json.dumps(report, indent=2)}]
            }
        }

    def _execute_clear_all_context(self, req_id: Any) -> Dict[str, Any]:
        self.store.clear()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
            "content": [{"type": "text", "text": "Context database index cleared successfully."}]
            }
        }

    def _handle_resources_list(self, req_id: Any) -> Dict[str, Any]:
        """
        Lists available read-only diagnostic resources.
        """
        resources = [
            {
                "uri": "evapocontext://telemetry/live",
                "name": "Live Hardware Telemetry Feed",
                "description": "Exposes real-time memory usage, swap file metrics, and moving average system pressure.",
                "mimeType": "application/json"
            },
            {
                "uri": "evapocontext://config/system",
                "name": "EvapoContext Configuration Parameters",
                "description": "Exposes engine threshold settings, decay coefficients, and category boosting weights.",
                "mimeType": "application/json"
            }
        ]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
            "resources": resources
            }
        }

    def _handle_resources_read(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the content of a requested diagnostic resource.
        """
        uri = params.get("uri")
        text = ""

        if uri == "evapocontext://telemetry/live":
            text = json.dumps(self.telemetry.get_full_report(), indent=2)
        elif uri == "evapocontext://config/system":
            config = {
                "base_threshold": self.engine.base_threshold,
                "pressure_factor": self.engine.pressure_factor,
                "default_normalization": self.engine.default_normalization,
                "decay_mode": self.engine.decay_mode,
                "token_weight": self.engine.token_weight,
                "soft_pin_multiplier": self.engine.soft_pin_multiplier,
                "budget_sorting_mode": self.engine.budget_sorting_mode,
                "default_category_weights": DEFAULT_CATEGORY_WEIGHTS
            }
            text = json.dumps(config, indent=2)
        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32602,
                    "message": f"Resource not found: {uri}"
                },
                "id": req_id
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": text
                    }
                ]
            }
        }


def run_daemon():
    """
    Bootstraps the telemetry background worker and spins up the stdio reading loop.
    """
    logger.info("Initializing background telemetry monitor...")
    with HardwareTelemetryMonitor() as monitor:
        # Instantiate daemon engine
        daemon = EvapoContextDaemon(telemetry_monitor=monitor)
        logger.info("EvapoContext Daemon stdio handler active. Reading requests...")

        # Process lines from stdin until EOF is reached (socket closed)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                
                # Check JSON-RPC formatting compliance
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                    error_resp = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32600, "message": "Invalid Request"},
                        "id": request.get("id") if isinstance(request, dict) else None
                    }
                    sys.stdout.write(json.dumps(error_resp) + "\n")
                    sys.stdout.flush()
                    continue

                # Process message
                response = daemon.handle_request(request)
                if response:
                    # Write response packet to stdout
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()

            except json.JSONDecodeError:
                # Handle malformed packets
                error_resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None
                }
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                # Catch-all safety boundary to prevent crash
                logger.exception(f"Unexpected exception processing input stream: {e}")

    logger.info("EvapoContext Daemon shutdown complete.")


if __name__ == "__main__":
    run_daemon()
