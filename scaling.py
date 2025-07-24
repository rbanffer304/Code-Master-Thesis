# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def scaling(X_train, X_test, technique):
    """Apply different types of scaling"""

    if technique == 'standard_scaling':
        # Apply standard scaling
        standard_scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            standard_scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            standard_scaler.fit_transform(X_test),
            columns=X_test.columns
        )
    elif technique == 'minmax_scaling':
        # Apply min max scaling
        minmax_scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            minmax_scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            minmax_scaler.transform(X_test),
            columns= X_test.columns
        )
    else:
        raise ValueError(f"Unknown scaling technique: {technique}")

    return X_train_scaled, X_test_scaled

