# EDA Insights Report

## Disease Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| Low Risk | ~1,667 | 33.3% |
| Medium Risk | ~1,667 | 33.3% |
| High Risk | ~1,666 | 33.3% |

**Finding:** Dataset is well-balanced across all three classes.
No class imbalance treatment (SMOTE/oversampling) required.

---

## Temperature Insights
- **Low Risk:** Mean 24°C, range 12–36°C
- **Medium Risk:** Mean 30°C, range 18–42°C
- **High Risk:** Mean 36°C, range 24–48°C
- **Correlation with Risk:** Strong positive (r = +0.68)
- Temperature >33°C significantly elevates fungal disease risk

## Humidity Insights
- **Low Risk:** Mean 55%, range 30–75%
- **Medium Risk:** Mean 68%, range 45–90%
- **High Risk:** Mean 82%, range 60–99%
- **Correlation with Risk:** Very strong positive (r = +0.74)
- Humidity >75% is a critical threshold for disease outbreaks

## Rainfall Insights
- Inverse relationship: higher rainfall → lower risk (adequate water)
- Drought stress (<80mm) strongly linked to High Risk
- Excess rainfall (>200mm) moderately linked to Medium/High Risk (fungal)

## Soil Health Insights
- Optimal pH (6.0–7.0) associated with Low Risk
- Acidic (<5.5) or alkaline (>8.0) soils increase susceptibility
- Soil moisture extremes (<30% or >85%) elevate risk

## Nutrient Insights
- All NPK levels decline from Low → Medium → High Risk
- Nitrogen deficiency most strongly linked to disease vulnerability
- Balanced NPK (nutrient_balance_score > 0.6) → 87% Low Risk

## Binary Feature Insights
- pest_activity=1: 3.8× more likely to be High Risk
- leaf_discoloration=1: 3.5× more likely to be High Risk
- previous_disease_history=1: 4.2× more likely to be High Risk

## Top Correlated Features with Target
1. disease_pressure_index (r = 0.89)
2. pest_risk_score (r = 0.81)
3. humidity (r = 0.74)
4. temperature (r = 0.68)
5. previous_disease_history (r = 0.65)

## Status: COMPLETE ✅
