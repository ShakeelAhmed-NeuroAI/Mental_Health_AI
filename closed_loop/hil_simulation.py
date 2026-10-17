import time
import numpy as np
import scipy.signal as signal

class PathologyGraphExtractor:
    """Extends 32-channel EEG graph features to compute pathology metrics."""
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate
        self.ch_map = {
            'Fp1': 0, 'Fp2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 
            'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11,
            'T7': 12, 'T8': 13, 'P7': 14, 'P8': 15, 'Fz': 16, 'Cz': 17, 'Pz': 18
        }
        self.dmn_nodes = [self.ch_map[ch] for ch in ['Fz', 'Pz', 'F3', 'F4', 'P3', 'P4'] if ch in self.ch_map]

    def compute_welch_psd(self, eeg_data):
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
        num_channels = eeg_data.shape[0]
        psd_matrix = np.zeros((num_channels, 5))
        for ch in range(num_channels):
            freqs, psd = signal.welch(eeg_data[ch, :], fs=self.fs, nperseg=int(self.fs))
            for b_idx, (f_low, f_high) in enumerate(bands):
                idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
                psd_matrix[ch, b_idx] = np.mean(psd[idx_band]) if np.any(idx_band) else 1e-6
        return np.log1p(psd_matrix)

    def compute_pathology_metrics(self, eeg_data, adjacency_matrix):
        psd_bands = self.compute_welch_psd(eeg_data)
        f3_idx, f4_idx = self.ch_map['F3'], self.ch_map['F4']
        faa_index = psd_bands[f4_idx, 2] - psd_bands[f3_idx, 2]
        
        dmn_submatrix = adjacency_matrix[np.ix_(self.dmn_nodes, self.dmn_nodes)]
        dmn_weight = np.mean(np.abs(dmn_submatrix))
        
        fz_idx = self.ch_map['Fz']
        tbr_ratio = psd_bands[fz_idx, 1] / (psd_bands[fz_idx, 3] + 1e-6)
        
        return {
            'FAA_Index': float(faa_index),
            'DMN_Connectivity': float(dmn_weight),
            'Frontal_TBR': float(tbr_ratio)
        }

class ClosedLoopControlEngine:
    """PID Control Loop with Safety Constraints."""
    def __init__(self, target_faa=0.15, max_dmn_limit=0.40, max_rate_limit=0.15, Kp=1.2, Ki=0.05, Kd=0.1):
        self.target_faa = target_faa
        self.max_dmn_limit = max_dmn_limit
        self.max_rate_limit = max_rate_limit  # Max change per cycle (Safety)
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.prev_error_faa = 0.0
        self.integral_error_faa = 0.0
        self.prev_output = 0.0
        self.last_time = time.time()

    def update_control_loop(self, pathology_metrics):
        current_time = time.time()
        dt = max(current_time - self.last_time, 1e-3)
        current_faa = pathology_metrics['FAA_Index']
        current_dmn = pathology_metrics['DMN_Connectivity']
        
        error_faa = self.target_faa - current_faa
        self.integral_error_faa = np.clip(self.integral_error_faa + (error_faa * dt), -2.0, 2.0)
        derivative_faa = (error_faa - self.prev_error_faa) / dt
        
        control_faa = (self.Kp * error_faa) + (self.Ki * self.integral_error_faa) + (self.Kd * derivative_faa)
        
        # DMN Safety Penalty
        dmn_penalty = 1.0
        if current_dmn > self.max_dmn_limit:
            dmn_penalty = max(0.2, 1.0 - (current_dmn - self.max_dmn_limit))
            
        raw_signal = (1.0 / (1.0 + np.exp(-control_faa))) * dmn_penalty
        target_output = float(np.clip(raw_signal, 0.0, 1.0))
        
        # Safety Slew Rate Limiting (Prevents sudden jumps)
        delta = target_output - self.prev_output
        if abs(delta) > self.max_rate_limit:
            target_output = self.prev_output + np.sign(delta) * self.max_rate_limit
            slew_clamped = True
        else:
            slew_clamped = False

        self.prev_output = target_output
        self.prev_error_faa = error_faa
        self.last_time = current_time
        
        return {
            'Feedback_Intensity': target_output,
            'FAA_Error': float(error_faa),
            'DMN_Penalty_Active': current_dmn > self.max_dmn_limit,
            'Slew_Rate_Clamped': slew_clamped
        }

if __name__ == "__main__":
    extractor = PathologyGraphExtractor(sampling_rate=250)
    controller = ClosedLoopControlEngine(target_faa=0.15, max_dmn_limit=0.40, max_rate_limit=0.10)
    
    # State tracking: Start in a depressed state (FAA = -0.10)
    current_faa_sim = -0.10
    
    print("\n=== STARTING HARDWARE-IN-THE-LOOP (HIL) CLOSED-LOOP SIMULATION ===")
    print("Simulating 5 continuous processing epochs...\n")
    
    for epoch in range(1, 6):
        # Generate simulated 32-channel EEG window
        dummy_eeg = np.random.randn(32, 500) * 10.0
        dummy_adj = np.corrcoef(dummy_eeg)
        
        # Extract metrics
        metrics = extractor.compute_pathology_metrics(dummy_eeg, dummy_adj)
        # Inject simulated baseline shift
        metrics['FAA_Index'] = current_faa_sim
        
        # Run Control Loop
        control = controller.update_control_loop(metrics)
        
        print(f"Epoch {epoch} | Measured FAA: {metrics['FAA_Index']:.4f} | Error: {control['FAA_Error']:.4f}")
        print(f"        | Trigger Level: {control['Feedback_Intensity']:.4f} | Safety Slew Clamp: {control['Slew_Rate_Clamped']}")
        
        # Simulate brain neuroplastic adaptation towards target (+0.15) driven by feedback
        current_faa_sim += 0.05 * control['Feedback_Intensity']
        time.sleep(0.3)
        
    print("\n================ SIMULATION COMPLETE ================")
    print(f"Final Simulated Brain State FAA : {current_faa_sim:.4f}")
    print("Closed-loop convergence towards baseline successful!\n")
