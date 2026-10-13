# Housing Price Prediction with SHAP Explainability

Predictive model for Ames, Iowa housing prices using ensemble ML models with SHAP-based explainability to identify key price drivers across 2,930 properties and 80+ features.

## Key Results
Test set: 20% holdout (`random_state=42`). Tuned models use GridSearchCV with 3-fold CV.

| Model | Baseline R² | Tuned R² | Tuned MAE |
|---|---|---|---|
| **XGBoost** | 0.920 | **0.932** | **$14,461** |
| Random Forest | 0.911 | 0.910 | $15,933 |
| LightGBM | 0.912 | 0.908 | $15,423 |

- Best model: tuned XGBoost (R² 0.93, MAE about $14.5K)
- Tuning helped XGBoost but slightly lowered Random Forest and LightGBM test R², so their default settings were already near their best for this data
- Applied SHAP to explain individual predictions and identify top price drivers (Overall Quality, Gr Liv Area, Neighborhood)

## Tools & Libraries
- Python, Pandas, Scikit-learn, XGBoost, LightGBM
- SHAP for model explainability
- Matplotlib, Seaborn for visualization

## Dataset
Ames Housing dataset (2,930 properties, 82 features)
Source: Journal of Statistics Education

## Setup
```bash
pip install -r requirements.txt
jupyter notebook House_Prediction_project.ipynb
```
