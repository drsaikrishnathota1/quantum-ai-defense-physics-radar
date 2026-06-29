import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

df = pd.read_csv("data/radar_detection_dataset.csv")

target = "detection_error"

physics_features = [
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
    "snr_db"
]

quantum_features = physics_features + [
    "q_feature_1",
    "q_feature_2",
    "q_feature_3",
    "q_feature_4",
    "q_feature_5",
    "q_feature_6",
    "quantum_ai_detection_index"
]

def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def train_model(name, model, features):
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.clip(pred, 0, 0.5)

    return evaluate(name, y_test, pred), y_test, pred

models = [
    (
        "Random Forest Physics",
        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        physics_features
    ),
    (
        "Gradient Boosting Physics",
        GradientBoostingRegressor(random_state=42),
        physics_features
    ),
    (
        "MLP Physics",
        Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42))
        ]),
        physics_features
    ),
    (
        "Quantum-AI MLP",
        Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42))
        ]),
        quantum_features
    )
]

results = []
saved_y = None
saved_pred = None

for name, model, features in models:
    metric, y_test, pred = train_model(name, model, features)
    results.append(metric)
    if name == "Quantum-AI MLP":
        saved_y = y_test
        saved_pred = pred

results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
results_df.to_csv("results/model_comparison.csv", index=False)

plt.figure(figsize=(6, 6))
plt.scatter(saved_y, saved_pred, s=8, alpha=0.35)
plt.plot([0, 0.5], [0, 0.5], linestyle="--", linewidth=2)
plt.xlabel("Physics-Computed Detection Error")
plt.ylabel("AI-Predicted Detection Error")
plt.title("Quantum-AI Model Validation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig4_actual_vs_predicted_quantum_ai.png", dpi=300)
plt.savefig("figures/fig4_actual_vs_predicted_quantum_ai.pdf")
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(results_df["model"], results_df["r2"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("R² Score")
plt.title("Model Comparison for Detection-Error Prediction")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig5_model_r2_comparison.png", dpi=300)
plt.savefig("figures/fig5_model_r2_comparison.pdf")
plt.close()

print("Model comparison saved: results/model_comparison.csv")
print(results_df)
