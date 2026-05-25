# ORION–AURORA: Dual-Model Solar Power Forecasting System
### Project Plan — Browser-Based Interactive Demo

---

## 1. Project Summary

This project implements a browser-based interactive demonstration of a dual-stage deep learning framework for solar PV power forecasting in smart grid environments. The system is built on the research paper *"Dual-Model Deep Learning Framework with ORION and AURORA for Robust and Accurate Solar Power Prediction in Smart Grid Systems"* and faithfully mirrors its two-model pipeline architecture in a fully self-contained, offline-capable web application — no server, no API key, no backend required.

The core idea: solar energy output is inherently unpredictable due to weather variability, cloud cover, temperature shifts, and seasonal changes. This makes accurate forecasting critical for grid stability. The proposed system uses two cooperating models — **ORION** for feature extraction and initial prediction, and **AURORA** for adaptive refinement — to achieve lower forecasting errors than any single-model approach.

---

## 2. Background & Motivation

Solar PV power generation is one of the most volatile renewable energy sources. Even short-duration cloud cover can drop a PV installation's output by over 20%, creating sudden supply imbalances in the grid. Grid operators must balance supply and demand in real-time, which is impossible without reliable short-term forecasting.

Traditional statistical models (ARIMA, linear regression) fail to capture the non-linear, non-stationary, multi-scale nature of solar data. While deep learning models like LSTM and CNN improve on this, they still struggle with:
- Rapid oscillatory fluctuations in solar irradiance
- Multi-scale temporal dependencies (minute-level spikes + seasonal trends)
- Generalization across varying weather and geographic conditions

The ORION–AURORA framework addresses these gaps directly through a hybrid reservoir-transformer pipeline followed by adaptive activation refinement.

---

## 3. System Architecture

The system has two distinct model stages that operate sequentially.

### Stage 1 — ORION (Oscillatory Reservoir-Infused Orchestration Network)

ORION's job is to take raw solar input features and produce an initial forecast. Its novelty lies in combining reservoir computing with transformer-based attention.

**Components:**
- **Input Embedding Layer** — Encodes raw features (irradiance, temperature, humidity, wind speed, hour of day) into a latent representation
- **Echo State Network (ESN) Reservoir** — A fixed, high-dimensional random recurrent network that expands input into a rich dynamical feature space. The reservoir captures oscillatory and cyclic patterns in solar data without the computational cost of full backpropagation through time. Key parameter: spectral radius constrained to [0, 1] to maintain echo-state property
- **Reservoir Activation Adaptation** — A gating mechanism (gain vector) that scales reservoir states to control sensitivity
- **Vanilla Transformer Encoder** — Multi-head self-attention over the reservoir-enriched features, capturing global temporal correlations across meteorological variables
- **Temporal Fusion Transformer (TFT)** — Handles multivariate, non-stationary inputs through variable selection networks and gating. Decomposes output into trend, seasonal, and residual components
- **Prediction Layer** — Concatenates reservoir states and transformer output; applies a readout weight matrix to produce the initial solar PV power forecast

**What ORION captures:**
- Short-term: sudden irradiance drops from cloud cover
- Long-term: daily solar arcs and seasonal generation trends
- Cross-variable: interaction between temperature and output efficiency

---

### Stage 2 — AURORA (Adaptive Unified Reservoir Optimization with Radiant Adaptation)

AURORA receives ORION's initial prediction as context and performs adaptive refinement. Its novelty is that it does not use fixed activation functions — instead, it dynamically estimates and updates reservoir activations based on current environmental conditions.

**Components:**
- **Adaptive Reservoir Optimization Unit** — Continuously tunes reservoir activation functions on-the-fly based on incoming data characteristics. Unlike ORION's fixed reservoir, AURORA's reservoir responds to data drift
- **Radiant Adaptation Module** — Computes correction signals based on the gap between ORION's prediction and real-time feature patterns, applying weighted adjustments that scale with prediction uncertainty
- **Neural Potential Layer** — A sequential processing layer (LSTM or lightweight transformer) that captures residual temporal dependencies missed by ORION
- **Prediction Decision Node** — Produces the final refined solar PV power forecast with error confidence bounds

**What AURORA adds:**
- Corrects for non-stationarity in solar data across different weather regimes
- Reduces overfitting by dynamically adjusting model sensitivity
- Improves stability during edge cases: cloud burst onset, sensor noise, dawn/dusk transition

---

### Combined Pipeline Flow

```
Raw Input Features
    │
    ▼
[Normalization + Feature Scaling]
    │
    ▼
┌─────────────────────────────────┐
│           ORION                 │
│  ESN Reservoir → Transformer   │
│  → TFT → Initial Prediction    │
└─────────────────────────────────┘
    │
    ▼ (Initial Forecast + Reservoir States)
┌─────────────────────────────────┐
│           AURORA                │
│  Adaptive Reservoir → Neural   │
│  Potential Layer → Refinement   │
└─────────────────────────────────┘
    │
    ▼
Final Solar PV Power Forecast
+ Error Metrics (RMSE, MAE, MAPE)
```

---

## 4. Implementation Plan

### Tech Stack
- **Framework:** React (single-file, browser-runnable)
- **Math:** Pure JavaScript (no ML library needed — ESN is just matrix ops)
- **Charts:** Recharts for actual vs predicted visualization
- **Data:** UCI Solar Energy Dataset (subset, hardcoded — no file upload needed for demo)
- **Styling:** Tailwind CSS
- **Deployment:** Fully self-contained — runs in any browser, zero dependencies to install

---

### Phase 1 — Data Layer (Day 1)
- [ ] Embed a 200-point subset of the UCI solar dataset as a JS constant
- [ ] Implement min-max normalization for all features
- [ ] Build feature vector constructor: `[irradiance, temp, humidity, wind, hour_sin, hour_cos]`
- [ ] Add optional AEP dataset subset for the second benchmark

---

### Phase 2 — ORION Engine (Day 1–2)
- [ ] Implement Echo State Network:
  - Random reservoir weight matrix W_res (100×100), scaled to spectral radius 0.9
  - Input weight matrix W_in (100×6), random uniform
  - State update: `S_t = (1-a)*S_{t-1} + a*tanh(W_res*S_{t-1} + W_in*x_t + β*sin(ωt+φ))`
  - Oscillatory modulation term `β*sin(ωt+φ)` included as per paper
- [ ] Collect reservoir states matrix over full input sequence
- [ ] Train readout weights via ridge regression (closed-form, fast) on training split
- [ ] Produce ORION initial prediction array

---

### Phase 3 — AURORA Engine (Day 2)
- [ ] Implement adaptive gain correction layer:
  - Compute residuals between ORION prediction and training targets
  - Fit a lightweight linear correction: `y_final = α*y_orion + β*features + γ`
  - Parameters re-estimated per data window (simulates online adaptation)
- [ ] Apply radiant adaptation scaling: error-weighted adjustment that amplifies correction during high-variance windows
- [ ] Produce AURORA final prediction array

---

### Phase 4 — Metrics & Evaluation (Day 2)
- [ ] Compute RMSE, MAE, MAPE for:
  - Baseline (mean prediction)
  - ORION only
  - ORION + AURORA (full system)
- [ ] Replicate paper's benchmark comparisons (VGG16, ResNet50, Hybrid CNN-LSTM etc.) as static reference values from the paper
- [ ] Build comparison bar chart

---

### Phase 5 — UI & Visualization (Day 3)
- [ ] **Live Input Panel** — Sliders for irradiance, temperature, cloud cover, hour of day; model runs in real-time on slider change
- [ ] **Pipeline Visualization** — Animated data flow: Input → ORION stage → AURORA stage → Output. Each stage lights up as computation runs
- [ ] **Actual vs Predicted Chart** — Line chart with actual (UCI/AEP data) and both ORION-only and ORION+AURORA predictions overlaid
- [ ] **Metrics Dashboard** — RMSE / MAE / MAPE displayed with comparison to paper's benchmarks
- [ ] **Reservoir State Heatmap** — Visual display of ESN reservoir activations (the most visually impressive part — looks like a neural activity monitor)
- [ ] **Dataset Toggle** — Switch between UCI Solar and AEP datasets
- [ ] **Solar Grid Contribution Panel** — Pie/bar showing estimated % of grid demand met by predicted solar output

---

### Phase 6 — Polish (Day 3)
- [ ] Responsive layout
- [ ] Loading animation when model "runs"
- [ ] Tooltips explaining ORION and AURORA stages in plain language
- [ ] Export button to download predictions as CSV

---

## 5. Key Metrics (Target — mirroring paper results)

| Dataset | Metric | Target Value |
|---------|--------|-------------|
| UCI Solar | RMSE | 3.50 |
| UCI Solar | MAE | 2.00 |
| UCI Solar | MAPE | 1.50% |
| AEP | RMSE | 4.00 |
| AEP | MAE | 2.50 |
| AEP | MAPE | 2.00% |

The demo will display these values alongside live-computed approximations from the JS model, showing the dual-stage improvement from ORION alone to ORION+AURORA.

---

## 6. What Makes This Stand Out

1. **No black box** — Every stage of the pipeline is visible. The user can literally watch ORION produce an initial prediction and AURORA correct it
2. **Real math** — The ESN is a genuine reservoir computing implementation, not a fake. The spectral radius constraint, oscillatory modulation, and ridge regression readout are all implemented correctly
3. **Dual-dataset benchmarking** — Both UCI Solar and AEP datasets are demonstrated, exactly matching the paper's evaluation
4. **Interactive** — Adjusting input sliders updates predictions in real-time, making the model's behavior immediately understandable
5. **Self-contained** — Entire demo runs in a single browser tab. No installs, no server, no API key

---

## 7. Scope Boundaries (What This Is NOT)

- Not a production forecasting system — this is a research demonstration
- Not trained on the full UCI/AEP datasets — uses a representative subset for browser performance
- Not a replacement for industrial SCADA or grid management software
- The AURORA "adaptive" behavior is approximated via windowed linear correction, not full online gradient descent — the behavior is faithful to the paper's description but simplified for browser execution

---

## 8. Deliverables

| # | Deliverable | Format |
|---|-------------|--------|
| 1 | Working interactive demo | React single-file app |
| 2 | ORION ESN + Transformer pipeline | In-browser JS |
| 3 | AURORA adaptive refinement layer | In-browser JS |
| 4 | Actual vs Predicted visualization | Recharts line chart |
| 5 | Benchmark comparison table | Bar chart + table |
| 6 | Reservoir state heatmap | SVG grid viz |
| 7 | UCI + AEP dataset results | Both datasets supported |

---

## 9. Timeline

| Day | Work |
|-----|------|
| Day 1 | Data layer + ORION ESN implementation + ridge regression readout |
| Day 2 | AURORA adaptive layer + metrics computation + dataset toggle |
| Day 3 | Full UI, pipeline animation, charts, polish, export |

**Total estimated time: 3 days**

---

*Plan prepared for: ORION–AURORA Solar Forecasting Demo*
*Based on: "Dual-Model Deep Learning Framework with ORION and AURORA for Robust and Accurate Solar Power Prediction in Smart Grid Systems"*
