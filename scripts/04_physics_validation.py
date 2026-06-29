import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

df = pd.read_csv("data/radar_detection_dataset.csv")

scenario_table = df.groupby("scenario").agg(
    samples=("scenario", "count"),
    mean_snr_db=("snr_db", "mean"),
    mean_detection_error=("detection_error", "mean"),
    mean_detection_probability=("detection_probability", "mean"),
    mean_noise_power_w=("noise_power_w", "mean")
).reset_index()

scenario_order = {"low_noise": 1, "medium_noise": 2, "high_noise": 3}
scenario_table["order"] = scenario_table["scenario"].map(scenario_order)
scenario_table = scenario_table.sort_values("order").drop(columns=["order"])

scenario_table.to_csv("results/physics_scenario_validation.csv", index=False)

plt.figure(figsize=(7, 5))
plt.plot(
    scenario_table["scenario"],
    scenario_table["mean_detection_error"],
    marker="o",
    linewidth=2
)
plt.xlabel("Noise Scenario")
plt.ylabel("Mean Detection Error")
plt.title("Physics Validation: Detection Error Increases Under Noise Stress")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig6_noise_scenario_validation.png", dpi=300)
plt.savefig("figures/fig6_noise_scenario_validation.pdf")
plt.close()

corr = df[[
    "frequency_ghz",
    "range_km",
    "rcs_m2",
    "received_power_w",
    "noise_power_w",
    "snr_db",
    "detection_error"
]].corr()

corr.to_csv("results/physics_correlation_matrix.csv")

print("Physics validation saved.")
print(scenario_table)
