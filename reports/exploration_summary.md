# Dataset Exploration Summary

## Overview
- **Total Records:** 5,000
- **Total Features:** 15 (14 input + 1 target)
- **Numeric Features:** 11
- **Categorical Features:** 1 (crop_type)
- **Binary Features:** 3 (pest_activity, leaf_discoloration, previous_disease_history)

## Target Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Low Risk | ~1,667 | ~33.3% |
| Medium Risk | ~1,667 | ~33.3% |
| High Risk | ~1,666 | ~33.3% |

## Key Statistical Insights

### Temperature
- Low Risk mean: ~24°C
- Medium Risk mean: ~30°C
- High Risk mean: ~36°C
- Clear upward trend with risk level

### Humidity
- Low Risk mean: ~55%
- Medium Risk mean: ~68%
- High Risk mean: ~82%
- Strong positive correlation with risk

### Rainfall
- Low Risk mean: ~150mm (adequate water)
- High Risk mean: ~75mm (drought stress)
- Inverse relationship with risk level

### Nutrient Levels (NPK)
- Declining trend: Low Risk > Medium Risk > High Risk
- Nutrient depletion linked to disease susceptibility

### Binary Features
- pest_activity: 10% Low / 40% Medium / 75% High Risk
- leaf_discoloration: 10% Low / 40% Medium / 75% High Risk
- previous_disease_history: 15% Low / 45% Medium / 80% High Risk

## Crop Type Distribution
10 crop types: Wheat, Rice, Maize, Soybean, Cotton,
Sugarcane, Potato, Tomato, Barley, Sunflower (~500 each)

## Exploration Status: COMPLETE ✅
