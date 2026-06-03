# Dataset Validation Report

## crop_disease_dataset.csv

| Property | Value |
|----------|-------|
| Rows | 5,000 |
| Columns | 15 |
| Missing Values | 0 |
| Duplicates | 0 |
| Target Classes | Low Risk, Medium Risk, High Risk |
| Class Balance | ~33% each class |
| File Size | ~400 KB |

## Column Schema

| Column | Dtype | Min | Max |
|--------|-------|-----|-----|
| crop_type | object | — | — |
| temperature | float64 | 5.0 | 50.0 |
| humidity | float64 | 20.0 | 100.0 |
| rainfall | float64 | 0.0 | 300.0 |
| soil_moisture | float64 | 10.0 | 90.0 |
| soil_ph | float64 | 4.5 | 9.0 |
| nitrogen | float64 | 0.0 | 120.0 |
| phosphorus | float64 | 0.0 | 100.0 |
| potassium | float64 | 0.0 | 120.0 |
| sunlight_hours | float64 | 2.0 | 14.0 |
| pest_activity | int64 | 0 | 1 |
| leaf_discoloration | int64 | 0 | 1 |
| growth_rate | float64 | 0.1 | 1.0 |
| previous_disease_history | int64 | 0 | 1 |
| disease_risk | object | — | — |

## Validation Status: PASSED ✅
