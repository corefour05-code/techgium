"""Live Telemetry Simulation & Empirical Performance Benchmarking Suite (Phase 6).

Simulates real-time 1 Hz telemetry streaming from an Infusion Pump Digital Twin to the
ML backend. Measures empirical latency, throughput, memory consumption, and stability.
Outputs empirical benchmarks without fabrication or hardcoded numbers.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import psutil

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.app import (
    load_production_models,
    observe_stream,
    predict_fault,
    predict_sensor_error,
    predict_sensor_health,
    predict_batch,
    reset_session,
    get_session_status,
)
from api.schemas.payloads import (
    SequentialObservationRequest,
    SessionResetRequest,
    FaultPredictionRequest,
    SensorErrorRequest,
    SensorHealthRequest,
    BatchPredictionRequest,
)
from src.data.synthetic_generator import generate_infusion_run, SimulatorConfig
from src.utils.config import FaultClass
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def run_coro(coro):
    """Execute coroutine synchronously without event loop overhead."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


def get_current_memory_mb() -> float:
    """Return current process Resident Set Size (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def benchmark_model_loading() -> float:
    """Measure exact wall-clock time required to load all models into memory."""
    t0 = time.perf_counter()
    load_production_models()
    load_time_sec = time.perf_counter() - t0
    logger.info(f"Loaded all production models in {load_time_sec:.3f} s")
    return round(load_time_sec, 4)


def benchmark_single_endpoints(n_trials: int = 50) -> Dict[str, Any]:
    """Benchmark single-window prediction endpoints."""
    # Warmup runs
    dummy_p = [80.0 + (i % 3) * 0.1 for i in range(30)]
    fault_req = FaultPredictionRequest(sensor_id="PRESS_01", sensor_pressure_mmhg=dummy_p)
    err_req = SensorErrorRequest(sensor_id="PRESS_01", sensor_pressure_mmhg=dummy_p, independent_reference_value=80.0)
    health_req = SensorHealthRequest(sensor_id="PRESS_01", sensor_pressure_mmhg=dummy_p)

    run_coro(predict_fault(fault_req))
    run_coro(predict_sensor_error(err_req))
    run_coro(predict_sensor_health(health_req))

    # Benchmark predict_fault
    latencies_fault = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        run_coro(predict_fault(fault_req))
        latencies_fault.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark predict_sensor_error
    latencies_error = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        run_coro(predict_sensor_error(err_req))
        latencies_error.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark predict_sensor_health
    latencies_health = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        run_coro(predict_sensor_health(health_req))
        latencies_health.append((time.perf_counter() - t0) * 1000.0)

    def stats(arr: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }

    return {
        "predict_fault": stats(latencies_fault),
        "predict_sensor_error": stats(latencies_error),
        "predict_sensor_health": stats(latencies_health),
    }


def benchmark_streaming_pipeline(n_steps: int = 500) -> Dict[str, Any]:
    """Simulate extended 1 Hz live streaming and record observation processing latencies."""
    session_id = "benchmark_streaming_session"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    # Generate synthetic telemetry sequence
    cfg = SimulatorConfig(duration_seconds=n_steps + 100, random_seed=42)
    df_stream = generate_infusion_run(scenario=FaultClass.NORMAL, config=cfg)

    latencies_ms = []
    warmup_latencies_ms = []
    steady_latencies_ms = []

    mem_before = get_current_memory_mb()

    for idx, row in df_stream.iloc[:n_steps].iterrows():
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_STREAM_01",
            timestamp=float(row["timestamp_sec"]),
            sensor_pressure_mmhg=float(row["sensor_pressure_mmhg"]),
            sensor_force_n=float(row["sensor_force_n"]),
            sensor_flow_ml_h=float(row["sensor_flow_ml_h"]),
            sensor_temperature_c=float(row["sensor_temperature_c"]),
            commanded_rate_ml_h=float(row["commanded_rate_ml_h"]),
            dt_predicted_pressure_mmhg=float(row["dt_predicted_pressure_mmhg"]),
        )
        t0 = time.perf_counter()
        res = run_coro(observe_stream(req))
        dt_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(dt_ms)
        if res.warmup_status == "WARMUP":
            warmup_latencies_ms.append(dt_ms)
        else:
            steady_latencies_ms.append(dt_ms)

    mem_after = get_current_memory_mb()
    status = run_coro(get_session_status(session_id))

    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {}
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }

    return {
        "total_samples": n_steps,
        "overall_latency": stats(latencies_ms),
        "warmup_phase_latency": stats(warmup_latencies_ms),
        "steady_state_latency": stats(steady_latencies_ms),
        "memory_before_mb": round(mem_before, 2),
        "memory_after_mb": round(mem_after, 2),
        "memory_growth_mb": round(mem_after - mem_before, 2),
        "session_final_buffer_size": status.buffered_samples,
    }


def benchmark_batch_throughput(batch_sizes: List[int] = [20, 50, 100]) -> Dict[str, Any]:
    """Measure batch evaluation throughput in samples per second."""
    dummy_p = [80.0 + (i % 3) * 0.1 for i in range(30)]
    single_req = FaultPredictionRequest(sensor_id="PRESS_01", sensor_pressure_mmhg=dummy_p)

    throughput_results = {}
    for b_size in batch_sizes:
        batch_req = BatchPredictionRequest(items=[single_req] * b_size)
        t0 = time.perf_counter()
        run_coro(predict_batch(batch_req))
        elapsed = time.perf_counter() - t0
        throughput = b_size / elapsed
        throughput_results[f"batch_{b_size}"] = {
            "batch_size": b_size,
            "elapsed_sec": round(elapsed, 4),
            "throughput_samples_per_sec": round(throughput, 1),
            "avg_per_sample_ms": round((elapsed / b_size) * 1000.0, 2),
        }

    return throughput_results


def run_benchmark():
    """Execute complete benchmarking suite and serialize results."""
    logger.info("=== Starting Phase 6 Empirical Benchmarking Suite ===")

    gc.collect()
    initial_mem = get_current_memory_mb()

    # 1. Model Loading
    loading_time = benchmark_model_loading()
    post_load_mem = get_current_memory_mb()

    # 2. Single Endpoint Latency
    single_results = benchmark_single_endpoints(n_trials=50)

    # 3. Streaming Sequential Benchmark (500 steps = 500 seconds of simulated 1 Hz telemetry)
    streaming_results = benchmark_streaming_pipeline(n_steps=500)

    # 4. Batch Throughput
    batch_results = benchmark_batch_throughput(batch_sizes=[20, 50, 100])

    final_mem = get_current_memory_mb()

    benchmark_data = {
        "benchmark_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": "SYNTHETIC_DATA",
        "device_target": "Infusion Pump ML Integration",
        "system_specs": {
            "os": sys.platform,
            "python_version": sys.version.split()[0],
            "cpu_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024.0**3), 2),
        },
        "model_loading": {
            "load_time_sec": loading_time,
            "memory_initial_mb": round(initial_mem, 2),
            "memory_post_load_mb": round(post_load_mem, 2),
            "model_memory_footprint_mb": round(post_load_mem - initial_mem, 2),
        },
        "single_request_latencies": single_results,
        "sequential_streaming_benchmarks": streaming_results,
        "batch_throughput_benchmarks": batch_results,
        "memory_summary": {
            "initial_mb": round(initial_mem, 2),
            "post_load_mb": round(post_load_mem, 2),
            "post_streaming_mb": round(final_mem, 2),
            "net_growth_during_stream_mb": round(final_mem - post_load_mem, 2),
        },
    }

    out_file = Path("reports/benchmarks_phase6.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    logger.info(f"Empirical benchmark results persisted to {out_file}")
    print(json.dumps(benchmark_data, indent=2))
    return benchmark_data


if __name__ == "__main__":
    run_benchmark()

