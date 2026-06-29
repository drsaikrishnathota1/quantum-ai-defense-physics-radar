import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

Path("results").mkdir(exist_ok=True)

df = pd.read_csv("data/radar_detection_dataset.csv")

features = [
    "frequency_ghz",
    "wavelength_m",
    "transmit_power_dbm",
    "antenna_gain_db",
    "noise_figure_db",
    "external_noise_db",
    "scenario_noise_penalty_db",
    "range_km",
    "rcs_m2",
    "system_loss_db",
    "received_power_w",
    "noise_power_w",
    "snr_db",
    "q_feature_1",
    "q_feature_2",
    "q_feature_3",
    "q_feature_4",
    "q_feature_5",
    "q_feature_6",
    "quantum_ai_detection_index"
]

target = "detection_error"

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42))
])

start_train = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_train

start_infer = time.time()
_ = model.predict(X_test)
infer_time = time.time() - start_infer

runtime_table = pd.DataFrame([
    {
        "operation": "Quantum-AI MLP training",
        "samples": len(X_train),
        "time_seconds": train_time
    },
    {
        "operation": "Quantum-AI MLP inference",
        "samples": len(X_test),
        "time_seconds": infer_time
    },
    {
        "operation": "Inference per sample",
        "samples": 1,
        "time_seconds": infer_time / len(X_test)
    }
])

runtime_table.to_csv("results/runtime_comparison.csv", index=False)

print("Runtime comparison saved.")
print(runtime_table)
