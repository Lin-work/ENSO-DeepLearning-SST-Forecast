# ENSO-DeepLearning-SST-Forecast

## Executive Summary
This project builds a PyTorch-based Convolutional Neural Network (CNN) framework designed to forecast the **Niño 3.4 Index (El Niño–Southern Oscillation)** using global **Sea Surface Temperature (SST) anomaly** fields parsed from NetCDF datasets. By incorporating multi-scale convolutional operations, the model captures micro-to-macro thermal gradients across the Pacific Basin to predict ENSO variability and evaluate out-of-sample climate indices.

---

## Physical Insights & Multi-Scale CNN Architecture

### 1. Seasonal SST Complexity & Feature Spatial Dynamics
- **Seasonal Variability:** Comparative geospatial analysis (e.g., comparing winter vs. summer global SST distributions) shows localized sea surface temperature spikes alongside unexpected regional cooling trends. This non-linear spatial behavior underscores the inherent complexity of atmospheric-oceanic forecasting.
- **Geospatial Processing:** Built with `netCDF4` and `Cartopy` under a `PlateCarree` projection to parse and map global grid anomalies.

### 2. Multi-Scale Convolutional Design Philosophy
The 3-layer CNN architecture is explicitly designed to mirror multi-scale atmospheric fluid dynamics:
- **`Conv1` (Small Scale / Fine Grid):** Captures localized sea surface temperature gradients and micro-scale thermal fluxes.
- **`Conv2` (Regional Scale / Medium Grid):** Resolves synoptic regional SST variations across oceanic basins.
- **`Conv3` (Basin Scale / Large Grid):** Extracts macro-scale coupled ocean-atmosphere patterns across the broader Pacific Ocean.
- **Fully Connected (FC) Layers:** Maps multi-scale spatial representations down to a single continuous prediction of the Niño 3.4 Index ($64 \times 15 \times 30 \to 200 \to 50 \to 1$).

---

## Performance & Out-of-Sample Results Analysis

| Global SST Anomaly Map | Model Training Loss Curve | Predicted vs. Actual Niño 3.4 Index |
| :---: | :---: | :---: |
| ![SST Map](hw9-0.png) | ![Loss Curve](hw9-1.png) | ![Loss Curve (Logarithmic Scale)](hw9-2.png) | ![Niño 3.4 Prediction](hw9-3.png) |
| *Geospatial contour plot of global SST anomalies parsed via Cartopy & NetCDF4.* | *Convergence of MSE Loss across 100 training/validation epochs.* | *Model out-of-sample forecast vs. ground truth Niño 3.4 Index time-series.* |

### Model Evaluation & Findings
* **Correlation Coefficient ($r$):** Achieved an impressive out-of-sample correlation of **$r = 0.702$** between CNN predictions and actual Niño 3.4 observation series.
* **Temporal Cyclicity & Phase Alignment:** The CNN successfully captures the underlying **phase transitions and periodicity** of ENSO cycles. 
* **Outlier Analysis:** While the model accurately aligns with the temporal phase across most test periods, noticeable amplitude deviations (underestimation/overestimation) occur during extreme ENSO events (e.g., 2009, 2011, and 2015). 
* **Key Takeaway:** The hierarchical multi-scale feature extraction (`Conv1` $\to$ `Conv3`) allows the neural network to capture non-linear SST interactions far more effectively than traditional single-scale statistical regressions.
