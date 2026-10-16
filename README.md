# Mental_Health_AI: Real-Time Edge Neuro-AI Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Opset_17-green.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, ultra-low-latency Graph Attention Network (GAT) inference engine designed for real-time electroencephalography (EEG) functional connectivity decoding and embedded microcontroller deployment.

---

## Technical Architecture & Deployment Metrics

- **Dynamic Graph Construction:** Converts 32-channel EEG streams into 5-band spectral Welch PSD node features ($\delta, \theta, \alpha, \beta, \gamma$) paired with Pearson functional connectivity matrices ($32 \times 32$).
- **ONNX Acceleration:** Serialized PyTorch Geometric models to ONNX Opset 17, resolving dynamic `aten::scatter_reduce` graph operators to achieve **0.834 ms/window inference** (**2.03x speedup** over native PyTorch).
- **Real-Time Streaming:** Implements a 50% overlapping ring buffer streaming pipeline delivering live state predictions with end-to-end processing latencies between **12 ms and 20 ms**.
- **Bare-Metal Deployment:** Extracts trained parameters into static C float arrays (`gat_weights.h`) for zero-dependency execution on low-power microcontrollers (e.g., RP2040, ARM Cortex-M).
- **Monitoring Dashboard:** Features an interactive real-time interface built with PyQt6 and PyQtGraph (`dashboard.py`).

---

## Performance Benchmarks

| Metric | PyTorch Eager (FP32) | ONNX Runtime (Opset 17 FP32) |
| :--- | :--- | :--- |
| **Inference Latency** | 1.694 ms / window | **0.834 ms / window** |
| **Speedup Ratio** | 1.00x | **2.03x** |
| **Total Stream Latency** | — | **12.31 – 20.98 ms** |
| **Model Footprint** | 42 KB | **40.45 KB** |

---

## Repository Structure

```text
Mental_Health_AI/
├── gat_weights.h         # Static C float header for bare-metal MCU targets
├── benchmark_int8.py     # ONNX vs PyTorch performance & latency benchmark
├── dashboard.py          # Real-time PyQt6 streaming interface
├── export_c_header.py    # Parameter extraction script to C arrays
├── train_model.py        # Offline GAT PyG training script
└── README.md
---

## Citation & Author

**Shakeel Ahmed**  
*Electrical & Neuro-AI Engineer*  
- **PEC Registration:** Elect/75690  
- **ORCID:** [0009-0006-2566-1616](https://orcid.org/0009-0006-2566-1616)  
- **Preprint DOI:** [10.5281/zenodo.22939463](https://doi.org/10.5281/zenodo.22939463)  
- **Journal Under Review:** *Biomedical Signal Processing and Control* (Ref: BSPC-D-26-19431)
