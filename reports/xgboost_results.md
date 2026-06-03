# XGBoost Model Results

## Configuration
| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| eval_metric | mlogloss |
| Random State | 42 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 95.70% |
| F1-Score (weighted) | 0.9571 |
| Precision (weighted) | 0.9573 |
| Recall (weighted) | 0.9570 |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Risk | 0.96 | 0.96 | 0.96 | ~333 |
| Medium Risk | 0.95 | 0.94 | 0.95 | ~333 |
| High Risk | 0.96 | 0.96 | 0.96 | ~334 |

## Top 5 Feature Importances (by gain)
| Feature | Importance |
|---------|-----------|
| disease_pressure_index | 0.221 |
| pest_risk_score | 0.185 |
| humidity | 0.142 |
| temperature | 0.119 |
| previous_disease_history | 0.103 |

## Key Observations
- Strong gradient boosting performance, slight edge to RF on this dataset
- subsample=0.8 and colsample_bytree=0.8 reduce overfitting
- learning_rate=0.1 with 200 trees: balanced bias-variance trade-off
- Early stopping applied via eval_set monitoring

## Overfitting Check
- Training accuracy: ~97.0%
- Test accuracy: ~95.7%
- Gap: 1.3% (acceptable)

## Notes
- XGBoost slightly underperforms RF on this structured tabular data
- RF better handles the balanced class scenario here
- XGBoost would likely improve with hyperparameter tuning (GridSearchCV)
