import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
"""
modelling.py
------------
OLS regression model for the FAO EU27 crop yield analysis.

I used Ordinary Least Squares (OLS) from statsmodels 
"""
def get_ols_features(X: pd.DataFrame) -> pd.DataFrame:

    core_features = [
        "LOG_NitrogenUse",
        "LOG_AreaHarvested",
        "LOG_PesticideUse",
    ]
    selected = [f for f in core_features if f in X.columns]
    return X[selected]

def fit_ols(X: pd.DataFrame, y: pd.Series):
    """

    Parameters
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector

    """

    # Add 1s in X dataset to allow model to not intercept the origin
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    return model


def evaluate_model(y_true: pd.Series,
                   y_pred: np.ndarray,
                   model_name: str = "OLS",
                   log_target: bool = False) -> dict:
    """
    Compute RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² on the test set.
    """
    if log_target:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    return {
        "Model":       model_name,
        "RMSE (t/ha)": round(rmse, 4),
        "MAE (t/ha)":  round(mae,  4),
        "R²":          round(r2,   3),
    }
