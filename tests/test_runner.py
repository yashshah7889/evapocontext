"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import os
import sys
import json
import time
import re
import logging
import threading
import psutil
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from telemetry import HardwareTelemetryMonitor
from engine import DynamicContextReRanker
from retrieval import EmbeddingGenerator, HybridRetrievalStore, TextSplitter

# Setup basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("EvapoContextBenchmark")

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
    "Modern database indexes utilize B-Trees or Log-Structured Merge Trees to optimize write and read operations.",
    "Coffee beans develop their complex flavor profiles during the chemical Maillard reaction of roasting.",
    "Deep sea hydrothermal vents host unique ecosystems fueled by chemosynthesis rather than solar energy.",
    "The invention of the printing press by Johannes Gutenberg in 1440 accelerated the spread of literacy in Europe.",
    "Quantum computing uses qubits that exist in superpositions of states, enabling high-dimensional calculations.",
    "Organic polymers such as cellulose provide structural integrity to plant cell walls and woody tissues.",
    "The Great Barrier Reef in Australia is the largest living structure on Earth, visible from outer space.",
    "Standard chess engines use alpha-beta pruning algorithms combined with deep evaluation neural nets.",
    "Traditional Japanese calligraphy uses ink made from soot and animal glue, rubbed on a Suzuri stone.",
    "The core temperature of the Sun reaches approximately 15 million degrees Celsius due to nuclear fusion.",
    "Classical economics models assume agents act with rational self-interest and possess perfect information.",
    "Photosynthesis in green plants converts carbon dioxide and water into glucose using light energy.",
    "The architecture of Gothic cathedrals is characterized by pointed arches, ribbed vaults, and flying buttresses.",
    "Linguistic relativity suggests that the structure of a language affects its speakers' world view.",
    "The Sahara desert is expanding southward into the Sahel region due to cyclical climate variations.",
    "Acoustic guitars use wooden soundboards to amplify the vibrations of steel or nylon strings.",
    "The Doppler effect explains the shift in frequency of sound or light waves relative to a moving observer.",
    "Peptides are short chains of amino acids linked by peptide bonds, acting as hormone messengers.",
    "Ancient Greek philosophy laid the foundations of formal logic, metaphysics, and early political science.",
    "The standard model of particle physics classifies fundamental forces and subatomic matter particles."
]


class LoadStressor:
    """
    Simulates hardware load (RAM and CPU stress) to trigger EvapoContext's
    telemetry-driven evaporation thresholds.
    """

    def __init__(self):
        self.stop_event = threading.Event()
        self.cpu_threads: List[threading.Thread] = []
        self.allocated_memory: Optional[bytearray] = None

    def start_cpu_stress(self, thread_count: int = 4):
        """
        Spawns worker threads running float math to consume CPU cycles.
        """
        self.stop_event.clear()
        self.cpu_threads = []
        logger.info(f"Starting {thread_count} CPU stress threads...")
        
        def stress_worker():
            x = 0.0001
            while not self.stop_event.is_set():
                x = (x + 0.0001) * 1.0000001
                if x > 1e10:
                    x = 0.0001

        for i in range(thread_count):
            t = threading.Thread(target=stress_worker, name=f"EvapoContextCPUStressor-{i}")
            t.daemon = True
            t.start()
            self.cpu_threads.append(t)

    def start_ram_stress(self, gigabytes: float = 1.0):
        """
        Allocates a large memory buffer to simulate system RAM pressure.
        """
        logger.info(f"Allocating {gigabytes:.2f} GB to simulate RAM pressure...")
        try:
            size_bytes = int(gigabytes * 1024 * 1024 * 1024)
            self.allocated_memory = bytearray(size_bytes)
            for i in range(0, len(self.allocated_memory), 4096):
                self.allocated_memory[i] = 1
            logger.info("RAM allocation committed successfully.")
        except MemoryError:
            logger.error("Failed to allocate RAM stress block due to Out-Of-Memory conditions.")

    def stop_all(self):
        """
        Halts threads and releases allocated buffers.
        """
        logger.info("Releasing hardware stress loads...")
        self.stop_event.set()
        for t in self.cpu_threads:
            t.join(timeout=2.0)
        self.cpu_threads.clear()
        
        self.allocated_memory = None
        logger.info("Stress loads released successfully.")


class EvapoContextBenchmark:
    """
    Instruments timings and recalls to evaluate EvapoContext performance metrics.
    """

    def __init__(self):
        # Initialize components
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Save benchmark db in src/model/ relative to tests location
        model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
        db_path = os.path.join(model_dir, "benchmark_evapocontext.db")
        
        self.embedder = EmbeddingGenerator(model_dir=model_dir)
        self.store = HybridRetrievalStore(embedding_generator=self.embedder, db_path=db_path)
        self.engine = DynamicContextReRanker()
        
        # Clear database and index test corpus
        self.store.clear()
        logger.info("Indexing 5 anchor facts and 25 noise passages...")
        
        test_chunks = []
        for a in ANCHOR_FACTS:
            test_chunks.append(a)
        for idx, n in enumerate(NOISE_PASSAGES):
            test_chunks.append({
                "id": f"noise_{idx}",
                "text": n,
                "category": "conversation"
            })
            
        self.store.add_chunks(test_chunks)
        self.process = psutil.Process(os.getpid())

    def simulate_llm_ttft(self, prompt_tokens: int) -> float:
        """
        Simulates Time-to-First-Token based on prompt token processing time.
        """
        base_overhead = 0.50
        time_per_token = 0.00015
        return round(base_overhead + (prompt_tokens * time_per_token), 3)

    def run_eval(self, query: str, system_pressure: float) -> Dict[str, Any]:
        """
        Executes a single RAG-evaporation routing cycle and profiles measurements.
        """
        start_time = time.perf_counter()
        
        candidates = self.store.retrieve(query, top_k=20, bm25_candidates=30)
        
        for idx, c in enumerate(candidates):
            c["rank"] = idx + 1
            
        original_tokens = sum(c["token_count"] for c in candidates)
        
        optimized = self.engine.optimize_context(
            chunks=candidates,
            system_pressure=system_pressure
        )
        
        optimized_tokens = sum(c["token_count"] for c in optimized)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        rss_mb = self.process.memory_info().rss / (1024 * 1024)
        
        expected_anchors = []
        for c in candidates:
            if c["id"].startswith("anchor"):
                expected_anchors.append(c["id"])
                
        survived_anchors = [c["id"] for c in optimized if c["id"].startswith("anchor")]
        
        if expected_anchors:
            recall_rate = len(survived_anchors) / len(expected_anchors)
        else:
            recall_rate = 1.0

        sim_ttft_sec = self.simulate_llm_ttft(optimized_tokens)
        baseline_ttft_sec = self.simulate_llm_ttft(original_tokens)

        return {
            "query": query,
            "system_pressure": round(system_pressure, 4),
            "latency_ms": round(latency_ms, 3),
            "memory_rss_mb": round(rss_mb, 2),
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "compression_ratio": round((1.0 - (optimized_tokens / original_tokens)) * 100.0, 2) if original_tokens > 0 else 0.0,
            "simulated_ttft_sec": sim_ttft_sec,
            "baseline_ttft_sec": baseline_ttft_sec,
            "ttft_saving_sec": round(baseline_ttft_sec - sim_ttft_sec, 3),
            "expected_anchors": expected_anchors,
            "survived_anchors": survived_anchors,
            "recall_accuracy": round(recall_rate * 100.0, 2)
        }

    def cleanup(self):
        """Removes test database files."""
        self.store.clear()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.abspath(os.path.join(base_dir, "..", "src", "model"))
        db_path = os.path.join(model_dir, "benchmark_evapocontext.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception as e:
                logger.warning(f"Could not delete benchmark database: {e}")


def execute_benchmarks():
    print("=" * 65)
    print("      PROJECT EVAPOCONTEXT: SYSTEM BENCHMARKING INTERROGATOR")
    print("=" * 65)

    benchmark = EvapoContextBenchmark()
    stressor = LoadStressor()
    
    # Establish a live telemetry connection to monitor pressure values
    monitor = HardwareTelemetryMonitor()
    monitor.start()
    
    results = {}

    try:
        test_query = "database connection postgres port and administration passphrase security key and lazy tool load compression and dynamic time-decay re-ranking and windows pagefile memory swap"

        # ----------------- BASELINE TEST (IDLE RUN) -----------------
        print("\n>>> RUNNING BASELINE BENCHMARK (IDLE SYSTEM) <<<")
        time.sleep(1.0)
        base_pressure = monitor.get_pressure()
        print(f"Current Base System Pressure Index: {base_pressure:.4f}")
        
        # Force system_pressure to 0.0 for baseline to simulate a pristine idle system
        base_report = benchmark.run_eval(test_query, 0.0)
        results["baseline"] = base_report
        
        print("\n[Baseline Results Summary]")
        print(f"  System Pressure:   {base_report['system_pressure'] * 100:.2f}%")
        print(f"  Memory Footprint:  {base_report['memory_rss_mb']:.2f} MB")
        print(f"  Candidate Tokens:  {base_report['original_tokens']} tokens")
        print(f"  Optimized Tokens:  {base_report['optimized_tokens']} tokens")
        print(f"  Token Compression: {base_report['compression_ratio']:.2f}% reduction")
        print(f"  Simulated TTFT:    {base_report['simulated_ttft_sec']:.3f} s (vs Baseline: {base_report['baseline_ttft_sec']:.3f} s)")
        print(f"  TTFT Saved:        {base_report['ttft_saving_sec']:.3f} s")
        print(f"  Expected Anchors:  {base_report['expected_anchors']}")
        print(f"  Survived Anchors:  {base_report['survived_anchors']}")
        print(f"  Recall Accuracy:   {base_report['recall_accuracy']:.2f}%")

        # Assert context recall requirement is met (> 90%)
        assert base_report["recall_accuracy"] >= 90.0, "Recall accuracy failed to meet 90% benchmark under normal pressure."

        # ----------------- STRESS TEST (HARDWARE LOAD) -----------------
        print("\n>>> SPINNING UP HARDWARE STRESSORS (SIMULATING CRITICAL LOAD) <<<")
        available_mem_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        print(f"Available System memory: {available_mem_gb:.2f} GB")
        
        ram_alloc_gb = min(2.0, max(0.5, available_mem_gb - 2.0))
        stressor.start_ram_stress(gigabytes=ram_alloc_gb)
        stressor.start_cpu_stress(thread_count=4)
        
        # Wait for telemetry to update moving average pressure (usually takes 4-6 seconds)
        print("Waiting 6 seconds for moving average pressure loops to spike...")
        for i in range(6):
            time.sleep(1.0)
            print(f"  Pressure Index: {monitor.get_pressure():.4f}")
            
        stressed_pressure = monitor.get_pressure()
        print(f"\nStressed System Pressure Index: {stressed_pressure:.4f}")
        
        print("\n>>> RUNNING STRESSED BENCHMARK (EVAPORATION ACTIVE) <<<")
        # Ensure we evaluate at least at 0.85 to trigger eviction threshold
        stressed_report = benchmark.run_eval(test_query, max(stressed_pressure, 0.85))
        results["stressed"] = stressed_report
        
        print("\n[Stressed Results Summary]")
        print(f"  System Pressure:   {stressed_report['system_pressure'] * 100:.2f}%")
        print(f"  Memory Footprint:  {stressed_report['memory_rss_mb']:.2f} MB")
        print(f"  Candidate Tokens:  {stressed_report['original_tokens']} tokens")
        print(f"  Optimized Tokens:  {stressed_report['optimized_tokens']} tokens")
        print(f"  Token Compression: {stressed_report['compression_ratio']:.2f}% reduction")
        print(f"  Simulated TTFT:    {stressed_report['simulated_ttft_sec']:.3f} s (vs Baseline: {stressed_report['baseline_ttft_sec']:.3f} s)")
        print(f"  TTFT Saved:        {stressed_report['ttft_saving_sec']:.3f} s")
        print(f"  Expected Anchors:  {stressed_report['expected_anchors']}")
        print(f"  Survived Anchors:  {stressed_report['survived_anchors']}")
        print(f"  Recall Accuracy:   {stressed_report['recall_accuracy']:.2f}%")

        print(f"\n[EvapoContext Evaporation Curve Comparison]")
        print(f"  Idle Chunks Saved:   {len(base_report['survived_anchors'])}/{len(base_report['expected_anchors'])}")
        print(f"  Stressed Chunks Saved: {len(stressed_report['survived_anchors'])}/{len(stressed_report['expected_anchors'])}")
        print(f"  Evaporation Rate:      {base_report['compression_ratio']}% -> {stressed_report['compression_ratio']}%")
        
        # Verify pinned items survived even under high pressure
        assert "anchor_admin_key" in stressed_report["survived_anchors"], "Critical pinned anchor evaporated under pressure!"
        print("  Verification: Critical pinned context survived successfully.")

    finally:
        stressor.stop_all()
        monitor.stop()
        benchmark.cleanup()
        
        # Save log report in parent directory of tests folder (workspace root)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.abspath(os.path.join(base_dir, "..", "benchmark_results.json"))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark run complete. Saved log report: {out_path}")
        print("=" * 65)


if __name__ == "__main__":
    execute_benchmarks()
