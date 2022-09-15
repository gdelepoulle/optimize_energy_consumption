from typing import Tuple

import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_appliance_gains(
    y_pred: pd.Series, y_true: pd.Series, y_opt: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    gains = (y_opt / y_true).mean()
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    gain_error = rmse / y_true.mean()
    return gains, gain_error
