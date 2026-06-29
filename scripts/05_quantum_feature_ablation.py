import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

model_df = pd.read_csv("results/model_comparison.csv")

physics_mlp = model_df[model_df["model"] == "MLP Physics"].iloc[0]
quantum_full = model_df[model_df["model"] == "Quantum-AI MLP Full-Physics"].iloc[0]
quantum_operational = model_df[model_df["model"] == "Operational Quantum-AI MLP"].iloc[0]

ablation = pd.DataFrame([
    {
        "feature_set": "Physics only",
        "model": "MLP Physics",
        "mse": physics_mlp["mse"],
        "mae": physics_mlp["mae"],
        "r2": physics_mlp["r2"]
    },
    {
        "feature_set": "Full physics + quantum-inspired features",
        "model": "Quantum-AI MLP Full-Physics",
        "mse": quantum_full["mse"],
        "mae": quantum_full["mae"],
        "r2": quantum_full["r2"]
    },
    {
        "feature_set": "Operational inputs + quantum-inspired features",
        "model": "Operational Quantum-AI MLP",
        "mse": quantum_operational["mse"],
        "mae": quantum_operational["mae"],
        "r2": quantum_operational["r2"]
    }
])

ablation["r2_gain_vs_physics_mlp"] = ablation["r2"] - physics_mlp["r2"]
ablation.to_csv("results/quantum_feature_ablation.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(ablation["feature_set"], ablation["r2"])
plt.ylabel("R² Score")
plt.title("Ablation: Quantum-Inspired Features and Operational Inputs")
plt.xticks(rotation=20, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig7_quantum_feature_ablation.png", dpi=300)
plt.savefig("figures/fig7_quantum_feature_ablation.pdf")
plt.close()

print("Quantum-feature ablation saved.")
print(ablation)
