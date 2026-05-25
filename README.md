# Housing Price Prediction with SHAP Explainability

Predictive model for Ames, Iowa housing prices using ensemble ML models with SHAP-based explainability to identify key price drivers across 2,930 properties and 80+ features.

## Key Results
- Best model: Optimized XGBoost with 0.93 R²
- Baseline XGBoost: 0.92 R²
- Random Forest: 0.91 R²
- Applied SHAP to explain individual predictions and identify top price drivers (Overall Quality, Gr Liv Area, Neighborhood)
- Compared Random Forest and XGBoost with hyperparameter tuning via GridSearchCV

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
