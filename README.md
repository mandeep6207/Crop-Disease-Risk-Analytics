# 🌿 CropGuard AI — Crop Disease Risk Analytics System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest%20%7C%20LogReg-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Predict crop disease risk levels (Low / Medium / High) using environmental conditions,
> soil health indicators, and agronomic features — powered by ensemble machine learning.**

---

## 🌾 Agriculture Problem Statement

Crop diseases cause **10–40% yield losses globally**, costing farmers billions annually.
Early detection and risk assessment are critical but require expertise that is scarce in
rural areas. CropGuard AI bridges this gap by providing data-driven disease risk predictions
from readily measurable field parameters — enabling proactive intervention before outbreaks occur.

---

## 📋 Project Overview

| Item            | Details                                      |
|-----------------|----------------------------------------------|
| **Target**      | `disease_risk` (Low Risk / Medium Risk / High Risk) |
| **Dataset**     | 5,000 records × 19 features (14 raw + 5 engineered) |
| **Best Model**  | Random Forest — **96.40% accuracy**          |
| **Notebook**    | `notebooks/cropguard_analysis.ipynb`         |
| **Artifacts**   | Trained model, JSON metrics, Markdown report |

---

## 📂 Dataset Overview

**Source:** Synthetic agricultural dataset generated with domain-realistic profiles.

| Feature               | Type       | Description                              |
|-----------------------|------------|------------------------------------------|
| `crop_type`           | Categorical | Crop species (Wheat, Rice, Maize, etc.) |
| `temperature`         | Numeric    | Air temperature (°C)                     |
| `humidity`            | Numeric    | Relative humidity (%)                    |
| `rainfall`            | Numeric    | Rainfall (mm)                            |
| `soil_moisture`       | Numeric    | Soil moisture content (%)                |
| `soil_ph`             | Numeric    | Soil pH level                            |
| `nitrogen`            | Numeric    | Nitrogen level (kg/ha)                   |
| `phosphorus`          | Numeric    | Phosphorus level (kg/ha)                 |
| `potassium`           | Numeric    | Potassium level (kg/ha)                  |
| `sunlight_hours`      | Numeric    | Daily sunlight hours                     |
| `pest_activity`       | Binary     | Pest presence (0/1)                      |
| `leaf_discoloration`  | Binary     | Leaf discoloration observed (0/1)        |
| `growth_rate`         | Numeric    | Crop growth rate (0–1)                   |
| `previous_disease_history` | Binary | Prior disease occurrence (0/1)        |
| `disease_risk`        | Target     | **Low Risk / Medium Risk / High Risk**   |

### Engineered Features

| Feature                    | Description                                              |
|----------------------------|----------------------------------------------------------|
| `crop_health_score`        | Composite of growth rate, soil pH, and soil moisture     |
| `nutrient_balance_score`   | Weighted NPK balance index                               |
| `disease_pressure_index`   | Humidity + pest + leaf symptoms + history                |
| `environmental_stress_score` | Temperature deviation + rainfall deficit + UV stress  |
| `pest_risk_score`          | Pest activity + humidity + temperature interaction       |

---

## 🔄 Workflow

```
Dataset Loading → Exploration → Missing Value Analysis → Data Cleaning
      ↓
Feature Engineering (5 new features)
      ↓
EDA & Visualizations (9 PNG charts)
      ↓
Encoding → Train/Test Split (80/20, stratified)
      ↓
Logistic Regression → Random Forest → XGBoost
      ↓
Model Comparison → Best Model Selection → Export
      ↓
Reports: model_metrics.json + project_report.md
```

---

## 📊 Visualizations

All charts saved to `visuals/`:

| Visualization                    | Description                              |
|----------------------------------|------------------------------------------|
| `disease_risk_distribution.png`  | Class distribution bar chart             |
| `temperature_impact.png`         | Temperature vs risk level analysis       |
| `humidity_analysis.png`          | Humidity box plots + scatter             |
| `soil_health_analysis.png`       | Soil pH, moisture, and health score      |
| `rainfall_impact.png`            | Rainfall patterns across risk classes    |
| `nutrient_analysis.png`          | NPK nutrient box plots by risk level     |
| `correlation_heatmap.png`        | Feature correlation matrix               |
| `feature_importance.png`         | RF and XGBoost feature importances       |
| `confusion_matrix.png`           | Confusion matrices for all three models  |

---

## 🤖 Model Comparison

| Model                | Accuracy | F1-Score | Precision | Recall |
|----------------------|----------|----------|-----------|--------|
| Logistic Regression  | 96.20%   | 0.9622   | 0.9626    | 0.9620 |
| **Random Forest**    | **96.40%** | **0.9641** | **0.9642** | **0.9640** |
| XGBoost              | 95.70%   | 0.9571   | 0.9573    | 0.9570 |

**Winner: Random Forest** — Best F1-score with balanced precision/recall.

---

## 🏆 Results

- **Best Model:** Random Forest Classifier
- **Accuracy:** 96.40%
- **F1-Score:** 0.9641 (weighted)
- **Key Predictors:** `disease_pressure_index`, `pest_risk_score`, `pest_activity`,
  `previous_disease_history`, `leaf_discoloration`, `environmental_stress_score`
- **Model File:** `models/crop_disease_predictor.pkl` (3 MB)

---

## 🌱 Soil Health Analysis Findings

Soil parameters play a critical role in disease susceptibility:

| Soil Parameter | Low Risk Optimal | High Risk Indicator |
|---------------|-----------------|---------------------|
| `soil_ph` | 6.0 – 7.0 | < 5.5 or > 8.0 |
| `soil_moisture` | 55 – 70% | < 30% or > 85% |
| `nitrogen` | > 60 kg/ha | < 30 kg/ha |
| `phosphorus` | > 40 kg/ha | < 20 kg/ha |
| `potassium` | > 50 kg/ha | < 25 kg/ha |
| `crop_health_score` | > 0.70 | < 0.40 |

Nutrient-depleted soils with extreme pH dramatically increase fungal and bacterial disease incidence.

---

## 🔭 Future Scope

1. **Real field data integration** via IoT sensors and satellite imagery
2. **Time-series modeling** for seasonal disease trend forecasting
3. **Crop-specific sub-models** for fine-grained predictions
4. **Mobile app** for farmer-accessible risk scoring
5. **API deployment** (FastAPI/Flask) for third-party integrations
6. **Explainability layer** with SHAP values for per-prediction reasoning
7. **Geospatial risk mapping** using GPS coordinates

---

## 🛠️ Installation Guide

```bash
# 1. Clone repository
git clone https://github.com/mandeep6207/Crop-Disease-Risk-Analytics-.git
cd Crop-Disease-Risk-Analytics-

# 2. Create virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

---

## 📖 Usage Guide

```python
# Open the notebook
# notebooks/cropguard_analysis.ipynb

# Run all cells (Kernel → Restart & Run All)
# Outputs generated automatically:
#   visuals/*.png     — 9 visualization charts
#   models/*.pkl      — trained model bundle
#   reports/*.json    — model metrics
#   reports/*.md      — project report
```

**Load & use the saved model:**

```python
import joblib
import numpy as np

bundle = joblib.load('models/crop_disease_predictor.pkl')
model  = bundle['model']
scaler = bundle['scaler']
le     = bundle['label_encoder_target']
features = bundle['feature_cols']

# Example prediction (must include engineered features)
sample = np.array([[...]])  # 19 feature values
pred = model.predict(sample)
print(le.inverse_transform(pred))  # 'Low Risk', 'Medium Risk', or 'High Risk'
```

---

## 📁 Project Structure

```
cropguard-ai/
├── data/
│   └── crop_disease_dataset.csv      # 5,000-record agricultural dataset
├── notebooks/
│   └── cropguard_analysis.ipynb      # Complete ML workflow notebook
├── models/
│   └── crop_disease_predictor.pkl    # Trained model bundle (RF + scaler + encoders)
├── visuals/
│   ├── disease_risk_distribution.png
│   ├── temperature_impact.png
│   ├── humidity_analysis.png
│   ├── soil_health_analysis.png
│   ├── rainfall_impact.png
│   ├── nutrient_analysis.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
├── reports/
│   ├── model_metrics.json            # All model accuracy/F1/precision/recall
│   └── project_report.md            # Executive project report
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 👨‍💻 Author

**Mandeep Kumar**
- Email: mk6207114453@gmail.com
- GitHub: [@mandeep6207](https://github.com/mandeep6207)

---

*CropGuard AI — Empowering precision agriculture with intelligent disease risk prediction.*
