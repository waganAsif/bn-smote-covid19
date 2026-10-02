"""Class balancing with SMOTE."""

import pandas as pd
from imblearn.over_sampling import SMOTE

from . import config


def smote_balance(df: pd.DataFrame, target: str = config.TARGET,
                  random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Oversample the minority class with SMOTE and return a balanced DataFrame."""
    X = df.drop(columns=[target])
    y = df[target]
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(X, y)
    balanced = pd.DataFrame(X_res, columns=X.columns)
    balanced[target] = y_res
    return balanced.astype(int)
