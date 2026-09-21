import socket
import time
import numpy as np

print("📡 Phase N: Initializing Real-Time TCP/IP Telemetry Streaming Hardware Simulator...")

HOST = '127.0.0.1'  # Localhost loopback network adapter
PORT = 5000         # High-availability communications port
NUM_REGIONS = 16    # Matching your 16-channel sensor array layout
FEATURE_DIM = 5     # Spectral dimension indicators (Delta, Theta, Alpha, Beta, FAA)

# 1. ESTABLISH BOUND NETWORK SOCKET CONNECTION
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"🛰️ Hardware Telemetry broadcast active at tcp://{HOST}:{PORT}")
print("⏳ Waiting for your deep learning model to connect and initialize ingestion stream...")

conn, addr = server_socket.accept()
print(f"✅ Connection verified from model client endpoint: {addr}")

# 2. ENDLESSLY STREAM ACTIVE VOLTAGE FREQUENCY ARRAYS FRAME-BY-FRAME
try:
    frame_counter = 0
    while True:
        frame_counter += 1
        # Generating a simulated frame of live microvolt activations
        simulated_frame = np.random.randn(NUM_REGIONS, FEATURE_DIM).astype(np.float32)
        
        # Pack raw numerical arrays straight into standard byte arrays for network transmission
        byte_data = simulated_frame.tobytes()
        conn.sendall(byte_data)
        
        print(f"⚡ Frame {frame_counter:04d} transmitted: {NUM_REGIONS} channels x {FEATURE_DIM} parameters sent over port {PORT}", end='\r')
        time.sleep(0.1)  # Simulating a live 10 Hz biological sampling ingestion frequency rate
        
except ConnectionResetError:
    print("\n🛑 Client pipeline closed or disconnected from stream. Powering down telemetry arrays.")
finally:
    conn.close()
    server_socket.close()
    print("🔒 Socket loop adapter cleanly deactivated.")
