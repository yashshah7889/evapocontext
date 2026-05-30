"""
evapoContext: Hardware-Aware Stateful Context Router
"""

import time
import platform
import threading
import logging
from typing import Dict, Any, Optional
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] [Thread: %(threadName)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EvapoContextTelemetry")


class TelemetryConfig:
    """
    Configuration data class for Telemetry parameters.
    Allows easy customization of weights, intervals, and smoothing coefficients.
    """
    def __init__(
        self,
        update_interval: float = 2.0,
        smoothing_factor: float = 0.3,
        weight_ram: float = 0.50,
        weight_swap: float = 0.40,
        weight_cpu: float = 0.10
    ):
        self.update_interval = update_interval
        # Exponential Moving Average smoothing factor (alpha). Must be in (0, 1].
        # Lower values make the pressure score smoother (filters spikes).
        # Higher values make it react quicker to changes.
        self.smoothing_factor = max(0.01, min(1.0, smoothing_factor))
        
        # Verify metric weights sum up to 1.0
        total_weight = weight_ram + weight_swap + weight_cpu
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(f"Telemetry weights sum to {total_weight}, normalizing to 1.0")
            self.weight_ram = weight_ram / total_weight
            self.weight_swap = weight_swap / total_weight
            self.weight_cpu = weight_cpu / total_weight
        else:
            self.weight_ram = weight_ram
            self.weight_swap = weight_swap
            self.weight_cpu = weight_cpu


class HardwareTelemetryMonitor:
    """
    An industry-grade system hardware monitor running as a background thread.
    
    Features:
        - Thread-safe value sharing using reentrant lock synchronization.
        - Exponential Moving Average (EMA) filtering to prevent execution thrashing.
        - Context manager interface for clean initialization and shutdown lifecycle.
        - Robust fallback error recovery for sandboxed or virtualized operating systems.
    """

    def __init__(self, config: Optional[TelemetryConfig] = None):
        """
        Initializes the telemetry monitor with thread safety constraints.
        """
        self.config = config or TelemetryConfig()
        
        self._lock = threading.RLock()
        
        self._current_pressure = 0.0
        self._raw_ram = 0.0
        self._raw_swap = 0.0
        self._raw_cpu = 0.0
        self._is_first_run = True
        
        self.system_os = platform.system()
        
        self._running = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info(
            f"Initialized Enterprise Telemetry Monitor (OS: {self.system_os}, "
            f"Interval: {self.config.update_interval}s, EMA alpha: {self.config.smoothing_factor})"
        )

    def __enter__(self) -> 'HardwareTelemetryMonitor':
        """
        Enables Python Context Manager support (e.g., `with HardwareTelemetryMonitor() as monitor:`).
        Guarantees that background threads start and stop cleanly.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Guarantees resource teardown when exiting the `with` block scope.
        """
        self.stop()

    def get_system_metrics(self) -> Dict[str, float]:
        """
        Queries raw telemetry data points from the operating system kernel.
        Includes error handling policies to recover if OS queries fail.
        """
        metrics = {"ram": 0.0, "swap": 0.0, "cpu": 0.0}
        
        try:
            # RAM Utilization
            metrics["ram"] = psutil.virtual_memory().percent
        except Exception as e:
            logger.debug(f"Unable to read physical RAM metrics: {e}")
            metrics["ram"] = 50.0  # Fallback: assume average load

        try:
            # Swap space metrics (Pagefile on Windows, Swap space on macOS)
            metrics["swap"] = psutil.swap_memory().percent
        except Exception as e:
            # Containerized virtual environments (e.g. Docker, WSL, isolated sandboxes) 
            # often block swap profile reads.
            logger.debug(f"Swap metrics unavailable: {e}. Defaulting to RAM usage profile.")
            metrics["swap"] = metrics["ram"]

        try:
            # CPU Utilization (Non-blocking query)
            metrics["cpu"] = psutil.cpu_percent(interval=None)
        except Exception as e:
            logger.debug(f"CPU telemetry reading failure: {e}")
            metrics["cpu"] = 10.0

        return metrics

    def update_metrics(self):
        """
        Calculates immediate raw pressure and applies Exponential Moving Average (EMA) smoothing.
        This mitigates context routing instability caused by instant, brief spikes in CPU activity.
        """
        raw_stats = self.get_system_metrics()
        
        # 1. Compute raw pressure score mapped to a [0.0, 1.0] scale
        raw_pressure = (
            (raw_stats["ram"] / 100.0) * self.config.weight_ram +
            (raw_stats["swap"] / 100.0) * self.config.weight_swap +
            (raw_stats["cpu"] / 100.0) * self.config.weight_cpu
        )
        # Verify range constraint
        raw_pressure = max(0.0, min(1.0, raw_pressure))

        with self._lock:
            self._raw_ram = raw_stats["ram"]
            self._raw_swap = raw_stats["swap"]
            self._raw_cpu = raw_stats["cpu"]
            
            if self._is_first_run:
                self._current_pressure = raw_pressure
                self._is_first_run = False
            else:
                alpha = self.config.smoothing_factor
                self._current_pressure = (alpha * raw_pressure) + ((1.0 - alpha) * self._current_pressure)

    def _run_loop(self):
        """
        Internal loop targeting execution in the background thread.
        """
        psutil.cpu_percent(interval=None)
        
        while not self._stop_event.is_set():
            self.update_metrics()
            
            if self._stop_event.wait(timeout=self.config.update_interval):
                break

    def start(self):
        """
        Starts the telemetry monitor in a non-blocking background thread.
        Uses lock protections to prevent duplicate threads.
        """
        with self._lock:
            if self._running:
                logger.warning("Telemetry monitor is already running.")
                return

            self._running = True
            self._stop_event.clear()
            
            self._current_pressure = 0.0
            self._is_first_run = True
            self.update_metrics()             
            self._monitor_thread = threading.Thread(
                target=self._run_loop, 
                name="EvapoContextTelemetryWorker"
            )
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Telemetry Monitor worker thread running in background.")

    def stop(self):
        """
        Halts the background thread and releases resources.
        """
        thread_to_join = None
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            thread_to_join = self._monitor_thread
            self._monitor_thread = None

        if thread_to_join:
            logger.info("Signaled telemetry thread to stop. Joining thread resources...")
            thread_to_join.join(timeout=2.0)
            logger.info("Telemetry Monitor shutdown complete.")

    def get_pressure(self) -> float:
        """
        Thread-safe getter for the System Pressure Index.
        
        Returns:
            Calculated System Pressure Index (0.0 to 1.0)
        """
        with self._lock:
            return round(self._current_pressure, 4)

    def get_full_report(self) -> Dict[str, Any]:
        """
        Thread-safe getter that returns a dictionary containing 
        the unified pressure score alongside all raw metric readings.
        """
        with self._lock:
            return {
                "system_pressure": round(self._current_pressure, 4),
                "raw_metrics": {
                    "ram_percentage": round(self._raw_ram, 2),
                    "swap_percentage": round(self._raw_swap, 2),
                    "cpu_percentage": round(self._raw_cpu, 2)
                },
                "operating_system": self.system_os
            }


# --- DEMONSTRATION RUNNER ---
if __name__ == "__main__":
    print("=" * 60)
    print("      PROJECT EVAPOCONTEXT: ENTERPRISE TELEMETRY RUNNER")
    print("=" * 60)
    print("Executing visual validation matrix. Press Ctrl+C to terminate.")
    print("Notice how the index moves smoothly during resource spikes due to EMA.\n")

    with HardwareTelemetryMonitor() as monitor:
        try:
            while True:
                report = monitor.get_full_report()
                idx = report["system_pressure"]
                metrics = report["raw_metrics"]
                
                bar = '#' * int(round(20 * idx)) + '-' * (20 - int(round(20 * idx)))
                
                print(
                    f"[{report['operating_system']}] "
                    f"Pressure Index: {idx:.4f} [{bar}] | "
                    f"RAM: {metrics['ram_percentage']:5.1f}% | "
                    f"Swap: {metrics['swap_percentage']:5.1f}% | "
                    f"CPU: {metrics['cpu_percentage']:5.1f}%",
                    end="\r"
                )
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n\nCtrl+C detected. Exiting context scope...")

    print("Successfully exited context manager scope. Telemetry threads joined cleanly.")
