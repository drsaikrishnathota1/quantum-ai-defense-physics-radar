import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

df = pd.read_csv("data/radar_detection_dataset.csv")

target = "detection_error"

physics_features = [
    "frequency_ghz",
    "signal_power_db",
    "noise_power_db",
    "range_km",
    "rcs",
    "wavelength_m",
    "received_power",
    "snr_db"
]

quantum_features = physics_features + [
    "q_feature_1",
    "q_feature_2",
    "q_feature_3"
]

def evaluate_model(name, y_true, y_pred):
    return {
        "model": name,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

# -------------------------
# Random Forest baseline
# -------------------------
X = df[physics_features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

results = []
results.append(evaluate_model("Random Forest", y_test, rf_pred))

# -------------------------
# PyTorch DNN helper
# -------------------------
class DNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_dnn(feature_list, model_name, epochs=150):
    X = df[feature_list].values
    y = df[target].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = DNNRegressor(input_dim=X_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()

    y_test_flat = y_test.flatten()

    metric = evaluate_model(model_name, y_test_flat, y_pred)

    return metric, y_test_flat, y_pred, loss_history

standard_metric, y_test_standard, pred_standard, loss_standard = train_dnn(
    physics_features,
    "Standard DNN"
)
results.append(standard_metric)

quantum_metric, y_test_quantum, pred_quantum, loss_quantum = train_dnn(
    quantum_features,
    "Quantum-Inspired DNN"
)
results.append(quantum_metric)

# Save model comparison table
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nModel comparison:")
print(results_df)

# -------------------------
# Figure 4: Actual vs predicted
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test_quantum, pred_quantum, s=8, alpha=0.35)
plt.plot([0, 0.5], [0, 0.5], linestyle="--", linewidth=2)
plt.xlabel("Physics-Computed Detection Error")
plt.ylabel("AI-Predicted Detection Error")
plt.title("Quantum-Inspired DNN Prediction Validation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig4_actual_vs_predicted_quantum_dnn.png", dpi=300)
plt.savefig("figures/fig4_actual_vs_predicted_quantum_dnn.pdf")
plt.close()

# -------------------------
# Figure 5: Training loss comparison
# -------------------------
plt.figure(figsize=(7, 5))
plt.plot(loss_standard, label="Standard DNN")
plt.plot(loss_quantum, label="Quantum-Inspired DNN")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig5_training_loss.png", dpi=300)
plt.savefig("figures/fig5_training_loss.pdf")
plt.close()

print("\nSaved:")
print("results/model_comparison.csv")
print("figures/fig4_actual_vs_predicted_quantum_dnn.png")
print("figures/fig5_training_loss.png")
