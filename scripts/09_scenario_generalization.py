import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

train_df = df[df["scenario"].isin(["low_noise", "medium_noise"])]
test_df = df[df["scenario"] == "high_noise"]

X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

generalization = pd.DataFrame([{
    "train_scenarios": "low_noise + medium_noise",
    "test_scenario": "high_noise",
    "mse": mean_squared_error(y_test, pred),
    "mae": mean_absolute_error(y_test, pred),
    "r2": r2_score(y_test, pred)
}])

generalization.to_csv("results/scenario_generalization.csv", index=False)

print("Scenario generalization saved.")
print(generalization)
