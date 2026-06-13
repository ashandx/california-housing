# California Housing: Linear Regression Baseline

End-to-end regression notebook predicting median house values across California districts using the 1990 census dataset.

---

## Business Problem

Property investors and valuers need reliable estimates of median house values across California districts. This notebook builds a baseline linear regression model using demographic and geographic features to predict median house value, identifies the strongest predictors, and quantifies model performance with a full diagnostic evaluation.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.fetch_california_housing` |
| Samples | 20,640 districts |
| Features | 8 (income, house age, rooms, bedrooms, population, occupancy, lat/lon) |
| Target | `MedHouseVal`: median house value in $100,000s |
| Note | Values above $500,000 are censored at 5.0 (known data artefact that affects high-value predictions) |

---

## Setup

```bash
conda activate learning-data-science
```

Dependencies (all included in the conda environment):

```
numpy  pandas  matplotlib  seaborn  scikit-learn
```

---

## How to Run

```bash
jupyter notebook california_housing.ipynb
```

Run all cells top-to-bottom. The notebook is self-contained; data is loaded directly from sklearn with no external files required.

To verify a clean top-to-bottom execution:

```bash
jupyter nbconvert --to notebook --execute california_housing.ipynb --output california_housing_executed.ipynb
```

---

## Key Findings

- **RMSE ~0.73 ($73,000), R² ~0.60**: the model explains 60% of variance in median house values with no feature engineering
- **Median income is the dominant predictor**: its coefficient is substantially larger than all other features, consistent with a correlation of 0.69 with the target
- **Heteroscedasticity present**: prediction error grows with house value, and the $500k data ceiling causes systematic underprediction at the high end; a log transformation of the target is the recommended next step

---

## Skills Demonstrated

`EDA` `correlation analysis` `train/test split` `data leakage prevention` `StandardScaler` `LinearRegression` `OLS` `RMSE` `MAE` `R²` `residual analysis` `actual vs predicted` `multicollinearity` `heteroscedasticity` `scikit-learn` `matplotlib` `seaborn`
