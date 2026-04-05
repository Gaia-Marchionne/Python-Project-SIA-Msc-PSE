"""
test_data_loader.py
-------------------
Unit tests for data_loader.py
"""

import pytest
import pandas as pd
import numpy as np

def test_get_feature_target_split(func, TARGET):
    # --- Case 1 : Test of default features ---

    # Initialisation of a synthetic dataset
    input = {
        "Year": [2020, 2021, 2022],
        "Temperature": [15.5, 16.2, 14.8],
        "Rainfall": [100, 120, 110],
        "Region": ["North", "South", "East"],
        "Yield_t_ha": [2500, 2700, 2600] 
    }

    output1 = ["Temperature", "Rainfall"]
    output2 = ["Temperature"]

    df = pd.DataFrame(input)

    X_auto, y_auto = func(df)
    
    assert list(X_auto.columns) == output1
    assert X_auto.shape == (3, 2)
    assert y_auto.name == TARGET

    # --- Case 2 : Test of custom features ---
    custom_features = ["Temperature"]
    X_custom, y_custom = func(df, features=custom_features)
    
    assert list(X_custom.columns) == output2
    assert X_custom.shape == (3, 1)
    assert y_custom.name == TARGET

    print("Test complete successfully")


