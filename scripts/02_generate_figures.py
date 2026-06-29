import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

df = pd.read_csv("data/radar_detection_dataset.csv")

# Figure 1: SNR vs Detection Error
plt.figure(figsize=(7, 5))
sample = df.sample(5000, random_state=42)
plt.scatter(sample["snr_db"], sample["detection_error"], s=8, alpha=0.35)
plt.xlabel("SNR (dB)")
plt.ylabel("Detection Error Probability")
plt.title("Radar Detection Error Under Noisy Defense Channel Conditions")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig1_snr_vs_detection_error.png", dpi=300)
plt.savefig("figures/fig1_snr_vs_detection_error.pdf")
plt.close()

# Figure 2: Frequency vs Range colored by SNR
plt.figure(figsize=(7, 5))
sc = plt.scatter(
    sample["frequency_ghz"],
    sample["range_km"],
    c=sample["snr_db"],
    s=8,
    alpha=0.55
)
plt.colorbar(sc, label="SNR (dB)")
plt.xlabel("Radar Frequency (GHz)")
plt.ylabel("Range (km)")
plt.title("Physics-Based SNR Variation Across Frequency and Range")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig2_frequency_range_snr_map.png", dpi=300)
plt.savefig("figures/fig2_frequency_range_snr_map.pdf")
plt.close()

# Figure 3: Average detection error by SNR bins
df["snr_bin"] = pd.cut(df["snr_db"], bins=30)
bin_stats = df.groupby("snr_bin", observed=True).agg(
    snr_mean=("snr_db", "mean"),
    error_mean=("detection_error", "mean")
).dropna()

plt.figure(figsize=(7, 5))
plt.plot(bin_stats["snr_mean"], bin_stats["error_mean"], marker="o", linewidth=2)
plt.xlabel("Mean SNR (dB)")
plt.ylabel("Mean Detection Error Probability")
plt.title("Mean Detection Error Trend with Increasing SNR")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3_mean_error_trend.png", dpi=300)
plt.savefig("figures/fig3_mean_error_trend.pdf")
plt.close()

print("Figures created in figures/ folder:")
print("fig1_snr_vs_detection_error.png")
print("fig2_frequency_range_snr_map.png")
print("fig3_mean_error_trend.png")
