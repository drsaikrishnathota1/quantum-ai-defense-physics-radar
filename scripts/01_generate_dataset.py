import numpy as np
import pandas as pd
from scipy.special import erfc
from pathlib import Path

np.random.seed(42)

N = 60000
c = 3e8
k_b = 1.380649e-23
T0 = 290.0
bandwidth_hz = 1e6

frequency_ghz = np.random.uniform(1.0, 18.0, N)
frequency_hz = frequency_ghz * 1e9
wavelength_m = c / frequency_hz
angular_frequency_rad_s = 2 * np.pi * frequency_hz

transmit_power_dbm = np.random.uniform(35.0, 85.0, N)
antenna_gain_db = np.random.uniform(18.0, 42.0, N)
noise_figure_db = np.random.uniform(2.0, 10.0, N)
range_km = np.random.uniform(1.0, 80.0, N)
rcs_m2 = 10 ** np.random.uniform(-1.0, 1.7, N)
system_loss_db = np.random.uniform(1.0, 8.0, N)
external_noise_db = np.random.uniform(-6.0, 10.0, N)

scenario = np.random.choice(
    ["low_noise", "medium_noise", "high_noise"],
    size=N,
    p=[0.34, 0.33, 0.33]
)

scenario_noise_penalty_db = np.select(
    [scenario == "low_noise", scenario == "medium_noise", scenario == "high_noise"],
    [0.0, 8.0, 16.0]
)

pt_w = 10 ** ((transmit_power_dbm - 30) / 10)
gain_linear = 10 ** (antenna_gain_db / 10)
loss_linear = 10 ** (system_loss_db / 10)
range_m = range_km * 1000

# Simplified monostatic radar equation:
# Pr = Pt * G^2 * lambda^2 * sigma / ((4*pi)^3 * R^4 * L)
received_power_w = (
    pt_w * (gain_linear ** 2) * (wavelength_m ** 2) * rcs_m2
) / (((4 * np.pi) ** 3) * (range_m ** 4) * loss_linear)

thermal_noise_w = k_b * T0 * bandwidth_hz
noise_power_w = thermal_noise_w * 10 ** (
    (noise_figure_db + external_noise_db + scenario_noise_penalty_db) / 10
)

snr_linear = received_power_w / (noise_power_w + 1e-30)
snr_db = 10 * np.log10(snr_linear + 1e-30)

# Physics-based detection-error proxy
detection_error = 0.5 * erfc(np.sqrt(np.maximum(snr_linear, 0) / 2))
detection_probability = 1.0 - detection_error

def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)

f_norm = minmax(frequency_ghz)
r_norm = minmax(range_km)
snr_norm = minmax(np.clip(snr_db, -40, 60))
rcs_norm = minmax(np.log10(rcs_m2))

# Quantum-inspired angle-encoding features
q_feature_1 = np.sin(np.pi * f_norm)
q_feature_2 = np.cos(np.pi * snr_norm)
q_feature_3 = np.sin(np.pi * r_norm)
q_feature_4 = np.cos(np.pi * rcs_norm)
q_feature_5 = np.sin(np.pi * f_norm * snr_norm)
q_feature_6 = np.cos(np.pi * r_norm * snr_norm)

quantum_ai_detection_index = (
    q_feature_1 + q_feature_2 + q_feature_3 + q_feature_4 + q_feature_5 + q_feature_6
) / 6.0

df = pd.DataFrame({
    "scenario": scenario,
    "frequency_ghz": frequency_ghz,
    "frequency_hz": frequency_hz,
    "angular_frequency_rad_s": angular_frequency_rad_s,
    "wavelength_m": wavelength_m,
    "transmit_power_dbm": transmit_power_dbm,
    "antenna_gain_db": antenna_gain_db,
    "noise_figure_db": noise_figure_db,
    "external_noise_db": external_noise_db,
    "scenario_noise_penalty_db": scenario_noise_penalty_db,
    "range_km": range_km,
    "rcs_m2": rcs_m2,
    "system_loss_db": system_loss_db,
    "received_power_w": received_power_w,
    "noise_power_w": noise_power_w,
    "snr_linear": snr_linear,
    "snr_db": snr_db,
    "q_feature_1": q_feature_1,
    "q_feature_2": q_feature_2,
    "q_feature_3": q_feature_3,
    "q_feature_4": q_feature_4,
    "q_feature_5": q_feature_5,
    "q_feature_6": q_feature_6,
    "quantum_ai_detection_index": quantum_ai_detection_index,
    "detection_error": detection_error,
    "detection_probability": detection_probability,
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/radar_detection_dataset.csv", index=False)

print("Dataset created: data/radar_detection_dataset.csv")
print("Rows:", len(df))
print(df.head())
print("\nScenario summary:")
print(df.groupby("scenario")[["snr_db", "detection_error", "detection_probability"]].mean())
