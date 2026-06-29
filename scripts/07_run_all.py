import subprocess
import sys

scripts = [
    "scripts/01_generate_dataset.py",
    "scripts/02_generate_figures.py",
    "scripts/03_train_models.py",
    "scripts/04_physics_validation.py",
    "scripts/05_quantum_feature_ablation.py",
    "scripts/08_runtime_comparison.py",
    "scripts/09_scenario_generalization.py",
    "scripts/06_generate_publication_tables.py",
]

for script in scripts:
    print(f"\nRunning {script}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(f"Failed: {script}")

print("\nAll scripts completed successfully.")
