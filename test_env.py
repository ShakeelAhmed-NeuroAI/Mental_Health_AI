import mne
import numpy as np

print("🧪 Testing your Neuro-AI laboratory environment...")

# Create 2 seconds of mock brain wave data for 4 channels
data = np.random.randn(4, 500) * 1e-6
info = mne.create_info(ch_names=['F3', 'F4', 'O1', 'O2'], sfreq=250, ch_types='eeg')
raw = mne.io.RawArray(data, info)

print("✅ Success! Your MNE package and AI environment are working perfectly.")