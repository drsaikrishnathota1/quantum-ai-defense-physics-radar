import numpy as np
import pandas as pd
from scipy.special import erfc
from pathlib import Path

np.random.seed(42)

N = 50000

# Radar/physics input ranges
frequency_ghz = np.random.uniform(1, 20, N)          # radar frequency
signal_power_db = np.random.uniform(-20, 20, N)      # input signal power
noise_power_db = np.random.uniform(-30, 10, N)       # noise power
range_km = np.random.uniform(1, 100, N)              # target range
rcs = np.random.uniform(0.1, 10, N)                  # radar cross section proxy

# Convert dB to linear
signal_power = 10 ** (signal_power_db / 10)
noise_power = 10 ** (noise_power_db / 10)

# Simple physics: wavelength
c = 3e8
frequency_hz = frequency_ghz * 1e9
wavelength_m = c / frequency_hz

# Simplified radar loss proxy
range_m = range_km * 1000
path_loss = (4 * np.pi * range_m / wavelength_m) ** 2

received_power = (signal_power * rcs) / path_loss
snr_linear = received_power / noise_power
snr_db = 10 * np.log10(snr_linear + 1e-12)

# Simple detection error model
# Higher SNR => lower error
detection_error = 0.5 * erfc(np.sqrt(np.maximum(snr_linear, 0)))

# Quantum-inspired features using sinusoidal encoding
q_feature_1 = np.sin(np.pi * frequency_ghz / 20)
q_feature_2 = np.cos(np.pi * snr_db / 40)
q_feature_3 = np.sin(np.pi * range_km / 100)

df = pd.DataFrame({
    "frequency_ghz": frequency_ghz,
    "signal_power_db": signal_power_db,
    "noise_power_db": noise_power_db,
    "range_km": range_km,
    "rcs": rcs,
    "wavelength_m": wavelength_m,
    "received_power": received_power,
    "snr_db": snr_db,
    "q_feature_1": q_feature_1,
    "q_feature_2": q_feature_2,
    "q_feature_3": q_feature_3,
    "detection_error": detection_error
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/radar_detection_dataset.csv", index=False)

print("Dataset created: data/radar_detection_dataset.csv")
print(df.head())
print(df.describe())
