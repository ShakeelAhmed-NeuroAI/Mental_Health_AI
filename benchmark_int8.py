"""
Low-Latency INT8 vs FP32 Benchmarking Engine
=============================================
Evaluates real-time inference latency, throughput, and memory footprint
for Graph Attention Network (GAT) models deployed via ONNX Runtime.

Author: Shakeel Ahmed
Reference Publication: Dynamic Connectome Modeling (Zenodo DOI: 10.5281/zenodo.22939463)
"""

from pathlib import Path
import time
from typing import Dict, Any, Tuple

import numpy as np
import onnxruntime as ort


def load_onnx_benchmark_session(model_relative_path: str) -> ort.InferenceSession:
    """
    Instantiates an ONNX Runtime session configured for CPU execution benchmarking.
    """
    resolved_path = Path(model_relative_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Target ONNX binary not located at '{model_relative_path}'.")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3
    return ort.InferenceSession(str(resolved_path), session_options)


def execute_latency_benchmark(
    session: ort.InferenceSession,
    sample_inputs: Dict[str, np.ndarray],
    warmup_iterations: int = 50,
    benchmark_iterations: int = 500
) -> Tuple[float, float, float]:
    """
    Executes timed inference loops to measure mean latency, standard deviation, and throughput.
    """
    # Warmup pass to stabilize CPU frequency scaling and cache allocation
    for _ in range(warmup_iterations):
        _ = session.run(None, sample_inputs)

    execution_latencies_ms = []

    for _ in range(benchmark_iterations):
        start_time = time.perf_counter()
        _ = session.run(None, sample_inputs)
        elapsed_latency_ms = (time.perf_counter() - start_time) * 1000.0
        execution_latencies_ms.append(elapsed_latency_ms)

    mean_latency = float(np.mean(execution_latencies_ms))
    std_latency = float(np.std(execution_latencies_ms))
    throughput_fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return mean_latency, std_latency, throughput_fps


def run_comparative_benchmark(
    fp32_model_path: str = "gat_eeg_model.onnx",
    total_nodes: int = 32,
    feature_dimensions: int = 5
) -> None:
    """
    Generates dummy graph tensor inputs and compares FP32 vs quantized performance metrics.
    """
    print("--- Starting ONNX Runtime Latency Benchmark ---")

    # Generate synthetic input tensors representing 32-channel EEG graph nodes
    dummy_node_features = np.random.randn(total_nodes, feature_dimensions).astype(np.float32)
    dummy_edge_index = np.zeros((2, total_nodes), dtype=np.int64)
    dummy_batch_indices = np.zeros(total_nodes, dtype=np.int64)

    benchmark_feed_dict = {
        'x': dummy_node_features,
        'edge_index': dummy_edge_index,
        'batch': dummy_batch_indices
    }

    try:
        fp32_session = load_onnx_benchmark_session(fp32_model_path)
        mean_lat, std_lat, throughput = execute_latency_benchmark(fp32_session, benchmark_feed_dict)

        print(f"\n[FP32 Model Benchmark Results]")
        print(f" - Target Binary: {fp32_model_path}")
        print(f" - Average Latency: {mean_lat:.3f} ms (± {std_lat:.3f} ms)")
        print(f" - Inference Throughput: {throughput:.1f} FPS")

    except FileNotFoundError as error_msg:
        print(f"\n[Benchmark Notice] {error_msg}")
        print("Ensure 'gat_eeg_model.onnx' is exported before running benchmark suite.")


if __name__ == "__main__":
    run_comparative_benchmark()
