# Logistic Regression Model Results

## Configuration
| Parameter | Value |
|-----------|-------|
| Solver | lbfgs |
| Multi-class | multinomial |
| Regularization (C) | 1.0 |
| Max Iterations | 1000 |
| Scaling | StandardScaler (mean=0, std=1) |
| Random State | 42 |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 96.20% |
| F1-Score (weighted) | 0.9622 |
| Precision (weighted) | 0.9626 |
| Recall (weighted) | 0.9620 |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Risk | 0.97 | 0.97 | 0.97 | ~333 |
| Medium Risk | 0.95 | 0.95 | 0.95 | ~333 |
| High Risk | 0.96 | 0.96 | 0.96 | ~334 |

## Key Observations
- Strong linear separability due to well-engineered features
- StandardScaler essential for convergence (features on different scales)
- multinomial softmax outperforms OvR for 3-class problem
- No overfitting: train/test accuracy within 1.5%
- Confusion primarily between adjacent classes (Low↔Medium, Medium↔High)

## Overfitting Check
- Training accuracy: ~97.1%
- Test accuracy: ~96.2%
- Gap: 0.9% (acceptable, no overfitting)
