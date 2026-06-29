# Publication Tables

## Table 1. Dataset variables

| Variable                   | Meaning                                          | Role                         |
|:---------------------------|:-------------------------------------------------|:-----------------------------|
| frequency_ghz              | Radar carrier frequency                          | Input physics variable       |
| wavelength_m               | Radar wavelength computed from c/f               | Derived physics variable     |
| transmit_power_dbm         | Transmitted radar power                          | Input system variable        |
| antenna_gain_db            | Antenna gain                                     | Input system variable        |
| range_km                   | Target range                                     | Input physics variable       |
| rcs_m2                     | Radar cross-section proxy                        | Input target variable        |
| received_power_w           | Received power from simplified radar equation    | Derived physics output       |
| noise_power_w              | Thermal and scenario-dependent noise power       | Derived noise output         |
| snr_db                     | Signal-to-noise ratio                            | Derived detection variable   |
| q_feature_1 to q_feature_6 | Quantum-inspired angle-encoding features         | AI feature map               |
| quantum_ai_detection_index | Bounded aggregate quantum-inspired feature index | AI interpretability variable |
| detection_error            | Physics-computed detection error probability     | Prediction target            |
| detection_probability      | One minus detection-error probability            | Derived output               |

## Table 2. Model comparison

| model                       | feature_mode                |         mse |         mae |       r2 |
|:----------------------------|:----------------------------|------------:|------------:|---------:|
| Random Forest Physics       | Random Forest Physics       | 3.28086e-10 | 7.0545e-06  | 1        |
| Gradient Boosting Physics   | Gradient Boosting Physics   | 4.69931e-07 | 0.000377817 | 0.999986 |
| Operational Quantum-AI MLP  | Operational Quantum-AI MLP  | 4.87057e-05 | 0.00439961  | 0.998563 |
| Quantum-AI MLP Full-Physics | Quantum-AI MLP Full-Physics | 5.21073e-05 | 0.00490845  | 0.998462 |
| MLP Physics                 | MLP Physics                 | 0.000106987 | 0.00603068  | 0.996843 |

## Table 3. Physics scenario validation

| scenario     |   samples |   mean_snr_db |   mean_detection_error |   mean_detection_probability |   mean_noise_power_w |
|:-------------|----------:|--------------:|-----------------------:|-----------------------------:|---------------------:|
| low_noise    |     20266 |      -16.01   |               0.325961 |                     0.674039 |          4.83778e-14 |
| medium_noise |     19973 |      -24.3411 |               0.373798 |                     0.626202 |          3.09873e-13 |
| high_noise   |     19761 |      -32.0139 |               0.410577 |                     0.589423 |          1.90117e-12 |

## Table 4. Quantum-inspired feature ablation

| feature_set                                    | model                       |         mse |        mae |       r2 |   r2_gain_vs_physics_mlp |
|:-----------------------------------------------|:----------------------------|------------:|-----------:|---------:|-------------------------:|
| Physics only                                   | MLP Physics                 | 0.000106987 | 0.00603068 | 0.996843 |               0          |
| Full physics + quantum-inspired features       | Quantum-AI MLP Full-Physics | 5.21073e-05 | 0.00490845 | 0.998462 |               0.00161954 |
| Operational inputs + quantum-inspired features | Operational Quantum-AI MLP  | 4.87057e-05 | 0.00439961 | 0.998563 |               0.00171993 |

## Table 5. Runtime comparison

| operation                |   samples |   time_seconds |
|:-------------------------|----------:|---------------:|
| Quantum-AI MLP training  |     48000 |    2.98867     |
| Quantum-AI MLP inference |     12000 |    0.00920033  |
| Inference per sample     |         1 |    7.66695e-07 |

## Table 6. Scenario generalization

| train_scenarios          | test_scenario   |        mse |       mae |       r2 |
|:-------------------------|:----------------|-----------:|----------:|---------:|
| low_noise + medium_noise | high_noise      | 0.00329049 | 0.0501606 | 0.868047 |