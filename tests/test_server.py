"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import sys
import os
import json
import subprocess
import time


def print_test_header(title):
    print("\n" + "=" * 65)
    print(f"  TEST: {title}")
    print("=" * 65)


def send_rpc(proc, message):
    """Sends a line-delimited JSON-RPC packet to the daemon process stdin."""
    payload = json.dumps(message) + "\n"
    proc.stdin.write(payload.encode("utf-8"))
    proc.stdin.flush()


def read_rpc(proc, timeout=5.0):
    """Reads a line-delimited JSON-RPC packet from the daemon process stdout."""
    line = proc.stdout.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8").strip())


def run_daemon_tests():
    print("=" * 70)
    print("      PROJECT EVAPOCONTEXT: MCP DAEMON INTEGRATION TEST SUITE")
    print("=" * 70)

    # Point server script location to src/server.py relative to tests location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.abspath(os.path.join(base_dir, "..", "src", "server.py"))
    
    proc = subprocess.Popen(
        [sys.executable, server_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        # 1. Test Initialize Handshake
        print_test_header("1. Initialize Handshake")
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "TestClient", "version": "1.0.0"}
            }
        }
        send_rpc(proc, init_req)
        response = read_rpc(proc)
        print("Initialize response:", json.dumps(response, indent=2))
        
        assert response is not None
        assert response.get("id") == 1
        assert "result" in response
        assert "capabilities" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "evapocontext"

        init_notif = {
            "jsonrpc": "2.0",
            "method": "initialized"
        }
        send_rpc(proc, init_notif)
        time.sleep(0.1)

        # 2. Test Tools Listing (Skeletal Schemas)
        print_test_header("2. Skeletal Tools List (Lazy Tool Schema Compression)")
        list_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        send_rpc(proc, list_req)
        response = read_rpc(proc)
        print("Tools list response:", json.dumps(response, indent=2))
        
        assert response is not None
        assert response.get("id") == 2
        tools_list = response["result"]["tools"]
        assert len(tools_list) == 4
        
        for tool in tools_list:
            print(f"  Tool: {tool['name']:25} | Desc: {tool['description']:40} | Empty Properties: {tool['inputSchema']['properties'] == {}}")
            assert tool["inputSchema"]["properties"] == {}, f"Skeletal tool {tool['name']} properties must be empty"
            assert "Lazy-Hydration Active" in tool["description"]

        # 3. Test JIT Validation failure
        print_test_header("3. JIT Schema Validation Check")
        invalid_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "index_context",
                "arguments": {
                "invalid_param": 123
                }
            }
        }
        send_rpc(proc, invalid_req)
        response = read_rpc(proc)
        print("Validation failure response:", json.dumps(response, indent=2))
        
        assert response is not None
        assert response.get("id") == 3
        assert response["result"]["isError"] is True
        assert "validation error" in response["result"]["content"][0]["text"].lower()

        # 4. Test Indexing Context
        print_test_header("4. Index Chunks (tools/call: index_context)")
        index_req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "index_context",
                "arguments": {
                    "chunks": [
                        {
                            "id": "daemon_rule",
                            "text": "All JSON-RPC communications must be line-delimited to prevent parser timeouts.",
                            "is_pinned": True,
                            "category": "system_rule"
                        },
                        {
                            "id": "daemon_desc",
                            "text": "The daemon acts as the communications gateway of EvapoContext, translating raw stdout streams.",
                            "is_pinned": False,
                            "category": "conversation"
                        }
                    ]
                }
            }
        }
        send_rpc(proc, index_req)
        response = read_rpc(proc)
        print("Index context response:", json.dumps(response, indent=2))
        
        assert response is not None
        assert response.get("id") == 4
        assert response["result"].get("isError") is not True
        assert "Successfully indexed" in response["result"]["content"][0]["text"]

        # 5. Test Context Retrieval with Chronological Sorting
        print_test_header("5. Context Retrieval (tools/call: retrieve_optimized_context)")
        retrieve_req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "retrieve_optimized_context",
                "arguments": {
                    "query": "JSON-RPC line-delimited communication and stdout gateway",
                    "top_k": 5
                }
            }
        }
        send_rpc(proc, retrieve_req)
        response = read_rpc(proc)
        
        assert response is not None
        assert response.get("id") == 5
        assert response["result"].get("isError") is not True
        
        result_text = response["result"]["content"][0]["text"]
        result_data = json.loads(result_text)
        print("Retrieve result metadata:", json.dumps(result_data["telemetry"], indent=2))
        print(f"Retrieve optimized text output:\n{result_data['optimized_text']}")
        
        chunks = result_data["chunks"]
        ranks = [c["rank_assigned"] for c in chunks]
        print(f"Ranks assigned in returned list order: {ranks}")
        
        assert all(ranks[i] >= ranks[i+1] for i in range(len(ranks)-1)), "Ranks must be ordered descending for chronological display"
        
        for c in chunks:
            assert "bm25_score" in c
            assert "cosine_similarity" in c
            assert "similarity" in c
            assert "retrieval_rank" in c

        # 6. Test Resources Listing and Reading
        print_test_header("6. Resource Listing and Live Read")
        
        res_list_req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "resources/list"
        }
        send_rpc(proc, res_list_req)
        response = read_rpc(proc)
        print("Resources list response:", json.dumps(response, indent=2))
        assert response is not None
        assert response.get("id") == 6
        assert len(response["result"]["resources"]) == 2
        
        res_read_req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "resources/read",
            "params": {
            "uri": "evapocontext://telemetry/live"
            }
        }
        send_rpc(proc, res_read_req)
        response = read_rpc(proc)
        assert response is not None
        assert response.get("id") == 7
        
        telemetry_text = response["result"]["contents"][0]["text"]
        telemetry_data = json.loads(telemetry_text)
        print("Telemetry resource content:", json.dumps(telemetry_data, indent=2))
        assert "system_pressure" in telemetry_data

        # Resource Read (System Config Parameters)
        config_read_req = {
            "jsonrpc": "2.0",
            "id": 75,
            "method": "resources/read",
            "params": {
            "uri": "evapocontext://config/system"
            }
        }
        send_rpc(proc, config_read_req)
        response = read_rpc(proc)
        assert response is not None
        assert response.get("id") == 75
        
        config_text = response["result"]["contents"][0]["text"]
        config_data = json.loads(config_text)
        print("Config resource content:", json.dumps(config_data, indent=2))
        assert "base_threshold" in config_data
        assert "default_category_weights" in config_data
        assert "soft_pin_multiplier" in config_data

        # 7. Test Parse and Invalid Request Errors
        print_test_header("7. Standard JSON-RPC Error Responses")
        
        proc.stdin.write(b"{\n")
        proc.stdin.flush()
        response = read_rpc(proc)
        print("Parse error response:", json.dumps(response, indent=2))
        assert response is not None
        assert response.get("error", {}).get("code") == -32700

        not_found_req = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "invalid_api_call"
        }
        send_rpc(proc, not_found_req)
        response = read_rpc(proc)
        print("Method not found response:", json.dumps(response, indent=2))
        assert response is not None
        assert response.get("id") == 8
        assert response.get("error", {}).get("code") == -32601

    finally:
        print_test_header("8. Server Subprocess Teardown")
        print("Closing subprocess stdin stream...")
        proc.stdin.close()
        
        stderr_output = proc.stderr.read().decode("utf-8")
        print("Stderr logs output from server:\n", stderr_output)
        
        ret_code = proc.wait(timeout=5.0)
        print(f"Process return code (should be 0): {ret_code}")
        assert ret_code == 0, f"Expected return code 0, got {ret_code}"
        
        # Clean up database file generated relative to src/model/
        db_path = os.path.join(base_dir, "..", "src", "model", "evapocontext.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print("Removed temporary test database: evapocontext.db")
            
        print("\n>> ALL MCP DAEMON INTEGRATION TESTS PASSED SUCCESSFULLY! <<\n")


if __name__ == "__main__":
    run_daemon_tests()
