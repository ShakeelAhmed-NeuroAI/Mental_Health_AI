import sys
import time
import numpy as np
import onnxruntime as ort
import torch
from scipy.signal import welch
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

# 1. Initialize ONNX
options = ort.SessionOptions()
options.log_severity_level = 3
session = ort.InferenceSession("gat_eeg_model.onnx", options)

sfreq = 500
window_size = 2 * sfreq
step_size = sfreq
num_channels = 32
bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

dataset = torch.load("eeg_pyg_dataset.pt", weights_only=False)
edge_index_np = dataset[0].edge_index.numpy().astype(np.int64)
batch_np = np.zeros(num_channels, dtype=np.int64)

class EEGDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Neuro-AI EEG GAT Inference Monitor")
        self.setGeometry(100, 100, 900, 600)

        main_widget = QWidget()
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.status_label = QLabel("Initializing Real-Time EEG Stream...")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2b2b2b;")
        layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget(title="Class 0 Real-Time State Confidence (%)")
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.setLabel('left', 'Confidence', units='%')
        self.plot_widget.setLabel('bottom', 'Time Steps')
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='b', width=2))
        layout.addWidget(self.plot_widget)

        self.ring_buffer = np.zeros((num_channels, window_size), dtype=np.float32)
        self.history = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stream)
        self.timer.start(200)

    def extract_psd(self, raw_chunk):
        features = []
        for ch in range(num_channels):
            freqs, psd = welch(raw_chunk[ch], fs=sfreq, nperseg=256)
            ch_bands = []
            for low, high in bands:
                idx = np.logical_and(freqs >= low, freqs <= high)
                ch_bands.append(trapz_fn(psd[idx], freqs[idx]))
            features.append(ch_bands)
        feats = np.array(features, dtype=np.float32)
        return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

    def update_stream(self):
        t0 = time.perf_counter()
        incoming = np.random.randn(num_channels, step_size)
        self.ring_buffer[:, :-step_size] = self.ring_buffer[:, step_size:]
        self.ring_buffer[:, -step_size:] = incoming

        x_feats = self.extract_psd(self.ring_buffer)
        logits = session.run(None, {'x': x_feats, 'edge_index': edge_index_np, 'batch': batch_np})[0]

        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        conf = probs[0][0] * 100.0
        latency = (time.perf_counter() - t0) * 1000

        self.history.append(conf)
        if len(self.history) > 50:
            self.history.pop(0)

        self.curve.setData(self.history)
        self.status_label.setText(f"Predicted Class: 0 | Confidence: {conf:.2f}% | Latency: {latency:.2f} ms")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGDashboard()
    window.show()
    sys.exit(app.exec())
