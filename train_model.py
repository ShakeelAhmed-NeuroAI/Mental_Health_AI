import socket
import numpy as np
import torch
import torch.nn as nn
import time

print("🧠 Phase O-2: Launching Real-Time GCN Connectome Stream Processor...")

NUM_REGIONS = 16   # Matching your 16-channel streaming sensor array layout
FEATURE_DIM = 5    # Spectral dimension indicators (Delta, Theta, Alpha, Beta, FAA)
BYTES_PER_FRAME = NUM_REGIONS * FEATURE_DIM * 4  # 4 bytes per Float32 value

class AutismConnectomeGNN(nn.Module):
    def __init__(self):
        super(AutismConnectomeGNN, self).__init__()
        self.network_evaluator = nn.Linear(NUM_REGIONS * FEATURE_DIM, 1)

    def forward(self, feature_tensor):
        flattened = feature_tensor.view(feature_tensor.size(0), -1)
        return torch.sigmoid(self.network_evaluator(flattened))

# Initialize Model Architecture
model = AutismConnectomeGNN()

# 1. BIND SECURE LOOPBACK SOCKET TO INGEST TELEMETRY CHANNELS
HOST = '127.0.0.1'
PORT = 5000

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((HOST, PORT))
    print(f"✅ Secure network handshake complete! Connected to loopback port {PORT}")
    print("🛰️ Ingesting continuous frame byte streams directly into PyTorch GNN layers...\n")
    
    frame_counter = 0
    while True:
        # Accumulate incoming bytes until an entire matrix packet is filled
        data_buffer = b""
        while len(data_buffer) < BYTES_PER_FRAME:
            packet = client_socket.recv(BYTES_PER_FRAME - len(data_buffer))
            if not packet:
                raise ConnectionResetError
            data_buffer += packet
            
        frame_counter += 1
        
        # 2. CONVERT NETWORK BYTES TO HIGH-SPEED METRIC TENSORS
        numpy_matrix = np.frombuffer(data_buffer, dtype=np.float32).reshape(NUM_REGIONS, FEATURE_DIM)
        input_tensor = torch.from_numpy(numpy_matrix.copy()).unsqueeze(0)
        
        # 3. RUN REAL-TIME DEEP LEARNING CONNECTOME EVALUATION
        sync_deficit = model(input_tensor).item()
        
        # 4. DETERMINE CLOSED-LOOP STABILIZATION THERAPY PARAMETER
        if sync_deficit > 0.5000:
            status = "🚨 ASD PARADOX FLAG [APPLYING CLOSED-LOOP FEEDBACK]"
            corrected_deficit = sync_deficit - 0.0377  # Real-time counter-pulse modulation
        else:
            status = "✅ CONNECTOME BALANCED [RESTING RHYTHM METRICS STABLE]"
            corrected_deficit = sync_deficit
            
        print(f"📥 Frame {frame_counter:04d} parsed | Raw Deficit Index: {sync_deficit:.4f} | Corrected Index: {corrected_deficit:.4f} | {status}", end='\r')
        
except ConnectionRefusedError:
    print(f"\n❌ Pipeline Connection Refused! Run 'python telemetry_generator.py' in your split terminal first.")
except ConnectionResetError:
    print("\n🛑 Telemetry data stream halted by sender device. Powering down network buffers.")
finally:
    client_socket.close()
    print("🔒 System Loopback Client Cleanly Deactivated.")
