# Feature Engineering Notes

## Overview
Five domain-informed features engineered from raw columns.
All engineered features are normalized to [0, 1] range.

---

## 1. crop_health_score
**Formula:**
```
crop_health_score = (growth_rate × 0.40)
                  + (1 - |soil_ph - 6.5| / 4) × 0.35
                  + (soil_moisture / 100) × 0.25
```
**Rationale:** Combines growth vigor, pH suitability (optimal at 6.5),
and moisture availability into a single health metric.
**Range:** [0, 1] — higher is healthier

---

## 2. nutrient_balance_score
**Formula:**
```
nutrient_balance_score = (nitrogen / 120) × 0.35
                       + (phosphorus / 100) × 0.35
                       + (potassium / 120) × 0.30
```
**Rationale:** Reflects overall NPK availability. Deficiency in any
macronutrient increases disease susceptibility.
**Range:** [0, 1] — higher indicates better nutrient status

---

## 3. disease_pressure_index
**Formula:**
```
disease_pressure_index = (humidity / 100) × 0.30
                       + pest_activity × 0.25
                       + leaf_discoloration × 0.25
                       + previous_disease_history × 0.20
```
**Rationale:** Humidity, pest presence, and symptom history are the
leading biological and environmental disease drivers.
**Range:** [0, 1] — higher indicates greater disease pressure

---

## 4. environmental_stress_score
**Formula:**
```
environmental_stress_score = (|temperature - 25| / 25) × 0.30
                           + (1 - rainfall / 300) × 0.30
                           + (1 - sunlight_hours / 12) × 0.20
                           + (|soil_moisture - 60| / 60) × 0.20
```
**Rationale:** Captures abiotic stress from thermal extremes, drought,
light deficiency, and waterlogging/drought conditions.
**Range:** [0, 1] — higher indicates greater abiotic stress

---

## 5. pest_risk_score
**Formula:**
```
pest_risk_score = pest_activity × 0.40
               + (humidity / 100) × 0.30
               + ((temperature - 15) / 35) × 0.30
```
**Rationale:** Humid, warm conditions favour pest proliferation.
Combines direct pest observation with favourable environmental conditions.
**Range:** [0, 1] — higher indicates greater pest risk

---

## Feature Importance (post-modeling)
Top engineered features by Random Forest importance:
1. disease_pressure_index
2. pest_risk_score
3. environmental_stress_score
4. crop_health_score
5. nutrient_balance_score
