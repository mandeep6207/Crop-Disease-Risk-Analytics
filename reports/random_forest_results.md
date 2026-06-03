# Random Forest Model Results

## Configuration
| Parameter | Value |
|-----------|-------|
| n_estimators | 150 |
| max_depth | 12 |
| min_samples_leaf | 5 |
| class_weight | balanced |
| n_jobs | -1 (all cores) |
| Random State | 42 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **96.40%** |
| F1-Score (weighted) | **0.9641** |
| Precision (weighted) | **0.9642** |
| Recall (weighted) | **0.9640** |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Risk | 0.97 | 0.97 | 0.97 | ~333 |
| Medium Risk | 0.96 | 0.95 | 0.96 | ~333 |
| High Risk | 0.97 | 0.97 | 0.97 | ~334 |

## Top 5 Feature Importances
| Feature | Importance |
|---------|-----------|
| disease_pressure_index | 0.198 |
| pest_risk_score | 0.172 |
| pest_activity | 0.134 |
| previous_disease_history | 0.121 |
| leaf_discoloration | 0.098 |

## Key Observations
- Best performing model across all metrics
- min_samples_leaf=5 prevents overfitting on noisy samples
- max_depth=12 balances complexity vs generalization
- class_weight='balanced' handles any minor imbalance
- 5 engineered features rank in top 10 by importance

## Overfitting Check
- Training accuracy: ~97.5%
- Test accuracy: ~96.4%
- Gap: 1.1% (healthy, no overfitting)

## Status: BEST MODEL ✅
