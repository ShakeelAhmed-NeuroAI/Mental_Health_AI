"""
Real-Time Neuro-AI EEG GAT Inference Monitor
=============================================
PyQt6 / PyQtGraph real-time dashboard for low-latency ONNX model inference, 
32-channel Welch PSD feature extraction, and continuous state tracking.

Author: Shakeel Ahmed
Reference Publication: Dynamic Connectome Modeling (Zenodo DOI: 10.5281/zenodo.22939463)
"""

from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from scipy.signal import welch
import torch

SAMPLING_FREQUENCY_HZ: int = 500
WINDOW_SIZE_SAMPLES: int = 2 * SAMPLING_FREQUENCY_HZ
STEP_SIZE_SAMPLES: int = SAMPLING_FREQUENCY_HZ
TOTAL_EEG_CHANNELS: int = 32

FREQUENCY_BANDS: List[Tuple[float, float]] = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 30.0),
    (30.0, 45.0)
]

trapezoidal_integration = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def initialize_onnx_inference_engine(model_path: str = "gat_eeg_model.onnx") -> ort.InferenceSession:
    file_path = Path(model_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at '{model_path}'.")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    return ort.InferenceSession(str(file_path), session_options)


def load_graph_topology(dataset_path: str = "eeg_pyg_dataset.pt") -> Tuple[np.ndarray, np.ndarray]:
    file_path = Path(dataset_path)
    if not file_path.exists():
        edge_index = np.zeros((2, TOTAL_EEG_CHANNELS), dtype=np.int64)
    else:
        dataset_records = torch.load(str(file_path), weights_only=False)
        edge_index = dataset_records[0].edge_index.numpy().astype(np.int64)

    batch_indices = np.zeros(TOTAL_EEG_CHANNELS, dtype=np.int64)
    return edge_index, batch_indices


class EEGDashboard(QMainWindow):

    def __init__(self, onnx_session: ort.InferenceSession, edge_index: np.ndarray, batch_index: np.ndarray) -> None:
        super().__init__()
        self.onnx_session: ort.InferenceSession = onnx_session
        self.edge_index_np: np.ndarray = edge_index
        self.batch_np: np.ndarray = batch_index

        self.setWindowTitle("Real-Time Neuro-AI EEG GAT Inference Monitor")
        self.setGeometry(100, 100, 900, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.status_label = QLabel("Initializing Real-Time EEG Ingestion Engine...")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2b2b2b;")
        main_layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget(title="Class 0 Real-Time State Confidence (%)")
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.setLabel('left', 'Confidence', units='%')
        self.plot_widget.setLabel('bottom', 'Time Steps')
        self.confidence_curve = self.plot_widget.plot(pen=pg.mkPen(color='#1f77b4', width=2))
        main_layout.addWidget(self.plot_widget)

        self.ring_buffer: np.ndarray = np.zeros((TOTAL_EEG_CHANNELS, WINDOW_SIZE_SAMPLES), dtype=np.float32)
        self.confidence_history: List[float] = []

        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self.process_streaming_frame)
        self.stream_timer.start(200)

    def compute_power_spectral_density_features(self, eeg_chunk: np.ndarray) -> np.ndarray:
        extracted_channel_features: List[List[float]] = []

        for channel_idx in range(TOTAL_EEG_CHANNELS):
            frequencies, psd_power = welch(eeg_chunk[channel_idx], fs=SAMPLING_FREQUENCY_HZ, nperseg=256)
            band_powers: List[float] = []

            for low_freq, high_freq in FREQUENCY_BANDS:
                frequency_mask = np.logical_and(frequencies >= low_freq, frequencies <= high_freq)
                integrated_power = trapezoidal_integration(psd_power[frequency_mask], frequencies[frequency_mask])
                band_powers.append(float(integrated_power))

            extracted_channel_features.append(band_powers)

        feature_matrix = np.array(extracted_channel_features, dtype=np.float32)
        normalized_features = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
        return normalized_features

    def process_streaming_frame(self) -> None:
        start_time = time.perf_counter()

        incoming_stream = np.random.randn(TOTAL_EEG_CHANNELS, STEP_SIZE_SAMPLES).astype(np.float32)
        self.ring_buffer[:, :-STEP_SIZE_SAMPLES] = self.ring_buffer[:, STEP_SIZE_SAMPLES:]
        self.ring_buffer[:, -STEP_SIZE_SAMPLES:] = incoming_stream

        eeg_node_features = self.compute_power_spectral_density_features(self.ring_buffer)

        model_outputs = self.onnx_session.run(
            None,
            {
                'x': eeg_node_features,
                'edge_index': self.edge_index_np,
                'batch': self.batch_np
            }
        )[0]

        shifted_logits = model_outputs - np.max(model_outputs, axis=-1, keepdims=True)
        class_probabilities = np.exp(shifted_logits) / np.sum(np.exp(shifted_logits), axis=-1, keepdims=True)

        primary_class_confidence = float(class_probabilities[0][0] * 100.0)
        execution_latency_ms = (time.perf_counter() - start_time) * 1000.0

        self.confidence_history.append(primary_class_confidence)
        if len(self.confidence_history) > 50:
            self.confidence_history.pop(0)

        self.confidence_curve.setData(self.confidence_history)
        self.status_label.setText(
            f"Predicted Class: 0 | Confidence: {primary_class_confidence:.2f}% | Latency: {execution_latency_ms:.2f} ms"
        )


def launch_dashboard_app() -> None:
    application = QApplication(sys.argv)

    try:
        onnx_session = initialize_onnx_inference_engine("gat_eeg_model.onnx")
        edge_index, batch_index = load_graph_topology("eeg_pyg_dataset.pt")
    except Exception as error_msg:
        print(f"[Warning] Initialization fallback activated: {error_msg}")
        sys.exit(1)

    dashboard_window = EEGDashboard(onnx_session, edge_index, batch_index)
    dashboard_window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    launch_dashboard_app()
