import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

model_df = pd.read_csv("results/model_comparison.csv")

physics_mlp = model_df[model_df["model"] == "MLP Physics"].iloc[0]
quantum_mlp = model_df[model_df["model"] == "Quantum-AI MLP"].iloc[0]

ablation = pd.DataFrame([
    {
        "feature_set": "Physics only",
        "model": "MLP Physics",
        "mse": physics_mlp["mse"],
        "mae": physics_mlp["mae"],
        "r2": physics_mlp["r2"]
    },
    {
        "feature_set": "Physics + quantum-inspired features",
        "model": "Quantum-AI MLP",
        "mse": quantum_mlp["mse"],
        "mae": quantum_mlp["mae"],
        "r2": quantum_mlp["r2"]
    }
])

ablation["r2_gain"] = ablation["r2"] - ablation.loc[0, "r2"]
ablation.to_csv("results/quantum_feature_ablation.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(ablation["feature_set"], ablation["r2"])
plt.ylabel("R² Score")
plt.title("Ablation: Effect of Quantum-Inspired Features")
plt.xticks(rotation=15, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig7_quantum_feature_ablation.png", dpi=300)
plt.savefig("figures/fig7_quantum_feature_ablation.pdf")
plt.close()

print("Quantum-feature ablation saved.")
print(ablation)
