import pandas as pd
from pathlib import Path

Path("results").mkdir(exist_ok=True)

dataset_variables = pd.DataFrame([
    ["frequency_ghz", "Radar carrier frequency", "Input physics variable"],
    ["wavelength_m", "Radar wavelength computed from c/f", "Derived physics variable"],
    ["transmit_power_dbm", "Transmitted radar power", "Input system variable"],
    ["antenna_gain_db", "Antenna gain", "Input system variable"],
    ["range_km", "Target range", "Input physics variable"],
    ["rcs_m2", "Radar cross-section proxy", "Input target variable"],
    ["received_power_w", "Received power from simplified radar equation", "Derived physics output"],
    ["noise_power_w", "Thermal and scenario-dependent noise power", "Derived noise output"],
    ["snr_db", "Signal-to-noise ratio", "Derived detection variable"],
    ["q_feature_1 to q_feature_6", "Quantum-inspired angle-encoding features", "AI feature map"],
    ["detection_error", "Physics-computed detection error probability", "Prediction target"]
], columns=["Variable", "Meaning", "Role"])

dataset_variables.to_csv("results/table1_dataset_variables.csv", index=False)

model_comparison = pd.read_csv("results/model_comparison.csv")
scenario_validation = pd.read_csv("results/physics_scenario_validation.csv")
ablation = pd.read_csv("results/quantum_feature_ablation.csv")

with open("results/publication_tables.md", "w") as f:
    f.write("# Publication Tables\n\n")
    f.write("## Table 1. Dataset variables\n\n")
    f.write(dataset_variables.to_markdown(index=False))
    f.write("\n\n## Table 2. Model comparison\n\n")
    f.write(model_comparison.to_markdown(index=False))
    f.write("\n\n## Table 3. Physics scenario validation\n\n")
    f.write(scenario_validation.to_markdown(index=False))
    f.write("\n\n## Table 4. Quantum-inspired feature ablation\n\n")
    f.write(ablation.to_markdown(index=False))

print("Publication tables saved in results/")
