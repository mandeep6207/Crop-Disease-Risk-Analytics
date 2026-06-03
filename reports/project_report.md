# CropGuard AI - Project Report

## Executive Summary
CropGuard AI is a machine learning system that predicts crop disease risk
levels (Low, Medium, High) using environmental and agronomic data.

## Dataset Overview
- Total Records   : 5,000
- Original Features: 14
- Engineered Features: 5 (crop_health_score, nutrient_balance_score,
  disease_pressure_index, environmental_stress_score, pest_risk_score)
- Total Features Used: 19
- Target Classes  : ['High Risk', 'Low Risk', 'Medium Risk']

## Class Distribution
disease_risk
Low Risk       1667
Medium Risk    1667
High Risk      1666

## Model Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 96.20% | 0.9622 | 0.9626 | 0.9620 |
| Random Forest       | 96.40% | 0.9641 | 0.9642 | 0.9640 |
| XGBoost             | 95.70% | 0.9571 | 0.9573 | 0.9570 |

## Best Model
**Random Forest**
- Accuracy : 96.40%
- F1-Score : 0.9641

## Key Findings
1. Disease pressure index and pest risk score are top predictors.
2. High humidity (>75%) combined with elevated temperatures increases risk.
3. Nutrient imbalance significantly correlates with high disease risk.
4. Previous disease history is a strong predictor of future risk.
5. Crop health score effectively distinguishes Low vs High risk cases.

## Visualizations Generated
- disease_risk_distribution.png
- temperature_impact.png
- humidity_analysis.png
- soil_health_analysis.png
- rainfall_impact.png
- nutrient_analysis.png
- correlation_heatmap.png
- feature_importance.png
- confusion_matrix.png

## Files
- models/crop_disease_predictor.pkl
- reports/model_metrics.json
- reports/project_report.md
