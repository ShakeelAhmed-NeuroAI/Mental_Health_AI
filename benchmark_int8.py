import os
import time
import numpy as np
import onnxruntime as ort
import torch

# 1. Compare File Sizes
fp32_size = os.path.getsize("gat_eeg_model.onnx") / 1024
int8_size = os.path.getsize("gat_eeg_model_int8.onnx") / 1024

print("=== Model Footprint Comparison ===")
print(f"FP32 Model Size : {fp32_size:.2f} KB")
print(f"INT8 Model Size : {int8_size:.2f} KB")
print(f"Compression     : {(1 - int8_size/fp32_size)*100:.2f}% reduction")

# 2. Benchmark Sessions
dataset = torch.load("eeg_pyg_dataset.pt", weights_only=False)
sample = dataset[0]
x_np = sample.x.numpy().astype(np.float32)
edge_index_np = sample.edge_index.numpy().astype(np.int64)
batch_np = np.zeros(x_np.shape[0], dtype=np.int64)
inputs = {'x': x_np, 'edge_index': edge_index_np, 'batch': batch_np}

options = ort.SessionOptions()
options.log_severity_level = 3

s_fp32 = ort.InferenceSession("gat_eeg_model.onnx", options)
s_int8 = ort.InferenceSession("gat_eeg_model_int8.onnx", options)

# Warmup
s_fp32.run(None, inputs)
s_int8.run(None, inputs)

# Benchmark FP32
t0 = time.perf_counter()
for _ in range(200):
    s_fp32.run(None, inputs)
t_fp32 = ((time.perf_counter() - t0) / 200) * 1000

# Benchmark INT8
t0 = time.perf_counter()
for _ in range(200):
    s_int8.run(None, inputs)
t_int8 = ((time.perf_counter() - t0) / 200) * 1000

print("\n=== Latency Benchmark (200 Runs) ===")
print(f"FP32 Average Latency : {t_fp32:.3f} ms / window")
print(f"INT8 Average Latency : {t_int8:.3f} ms / window")
